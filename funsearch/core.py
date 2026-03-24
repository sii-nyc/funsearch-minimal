"""FunSearch 的核心搜索循环、问题定义和候选评估逻辑。

这是整个项目最重要的文件。可以把它理解为三部分：
1. `ProblemSpecification`：描述“要搜索什么问题”
2. `evaluate_program_detailed`：执行候选程序并打分
3. `FunSearchRunner.run()`：实现最小版的生成-评估-入库循环
"""

from __future__ import annotations

import contextlib
import math
import random
import signal
import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

from funsearch.database import ProgramDatabase, ProgramRecord
from funsearch.llm import LLMClient
from funsearch.prompting import (
    build_prompt,
    extract_function_source,
    extract_generated_function,
    replace_function,
)

if TYPE_CHECKING:
    from funsearch.tracing import TraceWriter


@dataclass(frozen=True)
class ProblemSpecification:
    """描述一个搜索问题需要的全部信息。

    字段含义：
    - `seed_program`：固定程序骨架，也是搜索的起点
    - `target_function`：唯一允许被进化的函数名
    - `entrypoint`：评估时调用的入口函数名，通常是 `main`
    - `inputs`：一批固定评测输入；同一次 run 中所有候选都用这批输入比较
    - `aggregate_scores`：把 per-input 分数组合成总分
    - `evaluation_timeout`：单次执行的超时秒数，避免坏程序卡死搜索
    """

    seed_program: str
    target_function: str
    entrypoint: str
    inputs: tuple[Any, ...]
    aggregate_scores: Callable[[tuple[float, ...]], float]
    evaluation_timeout: float = 10.0


@dataclass(frozen=True)
class EvaluationResult:
    """一次成功评估后的结果。"""

    signature: tuple[float, ...]
    aggregate_score: float


@dataclass(frozen=True)
class SearchConfig:
    """控制最小同步搜索循环的参数。"""

    iterations: int = 20
    islands: int = 4
    reset_interval: int = 10
    prompt_versions: int = 2
    cluster_temperature: float = 1.0
    program_temperature: float = 0.2
    random_seed: int = 0


@dataclass(frozen=True)
class SearchResult:
    """一次搜索结束后返回的最佳结果。"""

    best_program: str
    best_function: str
    best_signature: tuple[float, ...]
    best_score: float
    evaluated_candidates: int
    trace_dir: str | None = None


def evaluate_program(
    program_source: str, specification: ProblemSpecification
) -> EvaluationResult | None:
    """执行候选程序并返回合法分数。

    这是一个简化封装：如果程序被拒绝，则只返回 `None`；
    如果需要失败原因，调用下面的 `evaluate_program_detailed`。
    """

    result, _ = evaluate_program_detailed(program_source, specification)
    return result


def evaluate_program_detailed(
    program_source: str,
    specification: ProblemSpecification,
) -> tuple[EvaluationResult | None, str | None]:
    """执行候选程序，并返回“成功结果或拒绝原因”。

    拒绝的常见原因包括：
    - 语法错误 / 顶层执行错误
    - 找不到入口函数
    - 运行时报错
    - 返回值不是有限数值
    - 超时
    """

    namespace: dict[str, Any] = {}
    try:
        # 先把完整程序作为一个模块执行出来，得到其中定义的函数。
        with _evaluation_timeout(specification.evaluation_timeout):
            exec(program_source, namespace)
    except TimeoutError:
        return None, "exec_timeout"
    except Exception as exc:
        return None, f"exec_error: {type(exc).__name__}: {exc}"

    if specification.entrypoint not in namespace:
        return None, f"missing_entrypoint: {specification.entrypoint}"
    entrypoint = namespace[specification.entrypoint]

    signature = []
    for problem_input in specification.inputs:
        try:
            # 每个输入都会独立跑一次入口函数，组成最终的 signature。
            with _evaluation_timeout(specification.evaluation_timeout):
                score = entrypoint(problem_input)
        except TimeoutError:
            return None, f"runtime_timeout[{problem_input!r}]"
        except Exception as exc:
            return (
                None,
                f"runtime_error[{problem_input!r}]: {type(exc).__name__}: {exc}",
            )
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            return (
                None,
                f"invalid_score_type[{problem_input!r}]: {type(score).__name__}",
            )
        score = float(score)
        if not math.isfinite(score):
            return None, f"non_finite_score[{problem_input!r}]"
        signature.append(score)

    signature_tuple = tuple(signature)
    return (
        EvaluationResult(
            signature=signature_tuple,
            # 聚合规则由具体问题决定，核心循环并不知道“高分”具体是什么意思。
            aggregate_score=float(specification.aggregate_scores(signature_tuple)),
        ),
        None,
    )


@contextlib.contextmanager
def _evaluation_timeout(timeout_seconds: float) -> Any:
    """在可用时，用一个很小的 SIGALRM 超时保护包住执行。

    这是一个刻意保持简单的实现：
    - 不引入额外 worker 或沙箱
    - 足够阻止明显的死循环把整个 demo 卡住
    - 如果当前平台或线程不支持，就直接退化为“不加超时”
    """

    if timeout_seconds <= 0:
        yield
        return
    if threading.current_thread() is not threading.main_thread():
        yield
        return
    if not hasattr(signal, "setitimer"):
        yield
        return

    def _raise_timeout(signum: int, frame: Any) -> None:
        del signum, frame
        raise TimeoutError("Candidate evaluation exceeded the time limit.")

    previous_handler = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, _raise_timeout)
    signal.setitimer(signal.ITIMER_REAL, timeout_seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous_handler)


class FunSearchRunner:
    """最小同步版 FunSearch 主循环。"""

    def __init__(
        self,
        specification: ProblemSpecification,
        llm: LLMClient,
        config: SearchConfig,
        trace_writer: TraceWriter | None = None,
    ) -> None:
        self.specification = specification
        self.llm = llm
        self.config = config
        # 所有随机性统一从这里来，方便复现。
        self.rng = random.Random(config.random_seed)
        self.trace_writer = trace_writer

    def run(self) -> SearchResult:
        """执行完整搜索。

        主循环非常朴素：
        1. 先验证 seed program 可运行
        2. 用 seed 初始化每个 island
        3. 反复从某个 island 采样历史程序构建 prompt
        4. 让 LLM 只补全下一个版本的目标函数
        5. 把补全函数塞回固定骨架后执行评分
        6. 合法程序写入数据库
        7. 定期重置最差的一半 island
        """

        seed_result, seed_error = evaluate_program_detailed(
            self.specification.seed_program, self.specification
        )
        if seed_result is None:
            raise ValueError(
                f"Seed program is invalid and cannot initialize the search: {seed_error}"
            )

        seed_function = extract_function_source(
            self.specification.seed_program,
            self.specification.target_function,
        )
        seed_record = ProgramRecord(
            source=self.specification.seed_program,
            function_source=seed_function,
            signature=seed_result.signature,
            aggregate_score=seed_result.aggregate_score,
            source_length=len(self.specification.seed_program),
            created_at=0,
        )
        database = ProgramDatabase.from_seed(
            seed_record=seed_record,
            island_count=self.config.islands,
            rng=self.rng,
            cluster_temperature=self.config.cluster_temperature,
            program_temperature=self.config.program_temperature,
        )
        if self.trace_writer is not None:
            self.trace_writer.write_run_metadata(
                specification=self.specification,
                config=self.config,
                llm_backend=type(self.llm).__name__,
            )
            self.trace_writer.log_event(
                "run_started",
                seed_signature=list(seed_result.signature),
                seed_score=seed_result.aggregate_score,
            )
            initial_snapshot = self.trace_writer.write_snapshot(
                "initial_database", database
            )
            self.trace_writer.log_event(
                "database_snapshot",
                name="initial_database",
                snapshot_path=initial_snapshot,
            )

        for iteration in range(self.config.iterations):
            # 从某个岛抽样历史程序，然后构造一个“best-shot” prompt。
            island_index, sampled_programs = database.sample_prompt_records(
                self.config.prompt_versions
            )
            prompt = build_prompt(
                seed_program=self.specification.seed_program,
                target_function=self.specification.target_function,
                sampled_programs=sampled_programs,
            )
            prompt_path = None
            if self.trace_writer is not None:
                prompt_path = self.trace_writer.write_prompt(iteration, prompt)
                self.trace_writer.log_event(
                    "prompt_sampled",
                    iteration=iteration,
                    island_index=island_index,
                    prompt_path=prompt_path,
                    sampled_programs=[
                        {
                            "program_id": record.program_id,
                            "aggregate_score": record.aggregate_score,
                            "signature": list(record.signature),
                            "source_path": self.trace_writer.write_program(record),
                        }
                        for record in sampled_programs
                    ],
                )
            # LLM 只负责提出下一个目标函数版本。
            completion = self.llm.generate(prompt)
            completion_path = None
            if self.trace_writer is not None:
                completion_path = self.trace_writer.write_completion(
                    iteration, completion
                )
                self.trace_writer.log_event(
                    "completion_received",
                    iteration=iteration,
                    island_index=island_index,
                    completion_path=completion_path,
                )
            generated_name = (
                f"{self.specification.target_function}_v{len(sampled_programs)}"
            )
            generated_function = extract_generated_function(
                completion,
                function_name=generated_name,
                renamed_to=self.specification.target_function,
            )
            if generated_function is not None:
                candidate_program = replace_function(
                    self.specification.seed_program,
                    self.specification.target_function,
                    generated_function,
                )
                candidate_path = None
                if self.trace_writer is not None:
                    candidate_path = self.trace_writer.write_candidate_program(
                        iteration, candidate_program
                    )
                    self.trace_writer.log_event(
                        "candidate_extracted",
                        iteration=iteration,
                        island_index=island_index,
                        candidate_program_path=candidate_path,
                    )
                candidate_result, rejection_reason = evaluate_program_detailed(
                    candidate_program, self.specification
                )
                if candidate_result is not None:
                    # 只有“可执行且分数合法”的程序才会进入数据库。
                    record = database.add_program(
                        island_index=island_index,
                        source=candidate_program,
                        function_source=generated_function,
                        signature=candidate_result.signature,
                        aggregate_score=candidate_result.aggregate_score,
                    )
                    if self.trace_writer is not None:
                        self.trace_writer.log_event(
                            "candidate_accepted",
                            iteration=iteration,
                            island_index=island_index,
                            program_id=record.program_id,
                            aggregate_score=record.aggregate_score,
                            signature=list(record.signature),
                            source_path=self.trace_writer.write_program(record),
                        )
                elif self.trace_writer is not None:
                    self.trace_writer.log_event(
                        "candidate_rejected",
                        iteration=iteration,
                        island_index=island_index,
                        reason=rejection_reason,
                        candidate_program_path=candidate_path,
                    )
            elif self.trace_writer is not None:
                self.trace_writer.log_event(
                    "candidate_rejected",
                    iteration=iteration,
                    island_index=island_index,
                    reason=f"generated_function_not_found: {generated_name}",
                    completion_path=completion_path,
                )

            # reset_interval 是按“已评估候选数”触发的简化版 reset。
            reset_actions = database.record_evaluation(self.config.reset_interval)
            if self.trace_writer is not None and reset_actions:
                self.trace_writer.log_event(
                    "islands_reset",
                    iteration=iteration,
                    actions=[
                        {
                            "island_index": action.island_index,
                            "donor_island_index": action.donor_island_index,
                            "donor_program_id": action.donor_program_id,
                            "new_program_id": action.new_program_id,
                        }
                        for action in reset_actions
                    ],
                )
            if self.trace_writer is not None:
                snapshot_path = self.trace_writer.write_snapshot(
                    f"iteration_{iteration:04d}", database
                )
                current_best = database.best_program()
                self.trace_writer.log_event(
                    "iteration_completed",
                    iteration=iteration,
                    best_program_id=current_best.program_id,
                    best_score=current_best.aggregate_score,
                    snapshot_path=snapshot_path,
                )

        best = database.best_program()
        if self.trace_writer is not None:
            final_snapshot = self.trace_writer.write_snapshot(
                "final_database", database
            )
            self.trace_writer.log_event(
                "run_completed",
                best_program_id=best.program_id,
                best_score=best.aggregate_score,
                best_signature=list(best.signature),
                final_snapshot_path=final_snapshot,
            )
        return SearchResult(
            best_program=best.source,
            best_function=best.function_source,
            best_signature=best.signature,
            best_score=best.aggregate_score,
            evaluated_candidates=database.evaluated_candidates,
            trace_dir=str(self.trace_writer.root_dir)
            if self.trace_writer is not None
            else None,
        )
