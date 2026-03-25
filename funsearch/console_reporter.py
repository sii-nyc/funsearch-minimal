"""把搜索过程直接打印到命令行。

默认输出面向“边跑边看”的场景：
- 运行开始时说明固定问题上下文
- 每轮只展示抽样、生成结果和当前数据库状态
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, TextIO


class ConsoleRunReporter:
    """在搜索运行过程中持续打印轮次记录。"""

    def __init__(self, output: TextIO | None = None) -> None:
        self.output = output if output is not None else sys.stdout

    def report_run_started(
        self,
        *,
        target_function: str,
        entrypoint: str,
        input_count: int,
        llm_backend: str,
        iterations: int,
        islands: int,
        reset_interval: int,
        prompt_versions: int,
        random_seed: int,
        trace_dir: str | None,
        problem_summary: list[str],
        seed_score: float,
        seed_signature: tuple[float, ...],
    ) -> None:
        lines = [
            "=== FunSearch Run Started ===",
            f"problem: target={target_function} entrypoint={entrypoint} inputs={input_count}",
            (
                f"config: llm={llm_backend} iterations={iterations} islands={islands} "
                f"reset_interval={reset_interval} prompt_versions={prompt_versions} seed={random_seed}"
            ),
            f"seed: score={seed_score} signature={list(seed_signature)}",
            f"trace_dir: {trace_dir or 'disabled'}",
            "prompt: fixed skeleton and instructions are reused every iteration; only the sampled versions change",
            "problem_summary:",
            *[f"- {line}" for line in problem_summary],
            "",
        ]
        self._write_lines(lines)

    def report_iteration(
        self,
        *,
        iteration: int,
        total_iterations: int,
        current_best_score: float,
        current_best_program_id: int,
        iteration_item: dict[str, Any],
    ) -> None:
        status = iteration_item.get("status", "unknown")
        score_info = iteration_item.get("score_info") or {}
        lines = [
            f"=== Iteration {iteration + 1}/{total_iterations} ===",
            (
                f"status: {status} | selected_island={iteration_item.get('selected_island_index')} "
                f"| current_best_score={current_best_score} | current_best_program_id={current_best_program_id}"
            ),
            f"sampled_program_ids: {iteration_item.get('sampled_program_ids', [])}",
            f"selected_island_clusters: {iteration_item.get('selected_island_clusters', [])}",
            "",
            "[Generated Function]",
        ]

        generated_function = iteration_item.get("generated_function_text")
        if generated_function:
            lines.extend(generated_function.splitlines())
        else:
            lines.append("No valid target function extracted from the completion.")

        lines.extend(["", "[Result]"])
        if status == "accepted":
            lines.extend(
                [
                    (
                        f"accepted: program_id={score_info.get('program_id')} "
                        f"aggregate_score={score_info.get('aggregate_score')} "
                        f"signature={score_info.get('signature')}"
                    ),
                    f"source_path: {score_info.get('source_path')}",
                ]
            )
        else:
            lines.append(f"rejected: {score_info.get('reason')}")

        if iteration_item.get("reset_actions"):
            lines.extend(["", "[Reset Actions]"])
            for action in iteration_item["reset_actions"]:
                lines.append(
                    (
                        f"island={action.get('island_index')} <- donor_island={action.get('donor_island_index')} "
                        f"donor_program_id={action.get('donor_program_id')} new_program_id={action.get('new_program_id')}"
                    )
                )

        lines.extend(["", "[Islands]"])
        lines.extend(_build_database_lines(iteration_item.get("post_snapshot") or {}))
        lines.append("")
        self._write_lines(lines)

    def report_run_completed(
        self,
        *,
        best_score: float,
        best_signature: tuple[float, ...],
        evaluated_candidates: int,
        trace_dir: str | None,
    ) -> None:
        lines = [
            "=== FunSearch Run Completed ===",
            f"best_score: {best_score}",
            f"best_signature: {list(best_signature)}",
            f"evaluated_candidates: {evaluated_candidates}",
            f"trace_dir: {trace_dir or 'disabled'}",
            "",
        ]
        self._write_lines(lines)

    def _write_lines(self, lines: list[str]) -> None:
        self.output.write("\n".join(lines))
        if not lines or lines[-1] != "":
            self.output.write("\n")
        self.output.flush()


def _format_cluster_program_ids(clusters: list[dict[str, Any]]) -> list[list[int]]:
    cluster_groups = []
    for cluster in clusters:
        program_ids = [program["program_id"] for program in cluster.get("programs", [])]
        cluster_groups.append(program_ids)
    return cluster_groups


def _build_database_lines(snapshot: dict[str, Any]) -> list[str]:
    if not snapshot:
        return ["Database snapshot is not available."]

    lines = [
        f"evaluated_candidates: {snapshot.get('evaluated_candidates')}",
        f"best_program_id: {snapshot.get('best_program_id')}",
    ]
    for island in snapshot.get("islands", []):
        lines.append(
            (
                f"island={island.get('index')} best={island.get('best_program_id')} "
                f"clusters={_format_cluster_program_ids(island.get('clusters', []))}"
            )
        )
    return lines


def build_selected_island_summary(
    *,
    island_index: int,
    island: Any,
    program_source_paths: dict[int, str] | None = None,
) -> dict[str, Any]:
    """把被选中的 island 转成统一的展示结构。"""

    program_source_paths = program_source_paths or {}
    programs = []
    for program in island.all_programs():
        programs.append(
            {
                "program_id": program.program_id,
                "aggregate_score": program.aggregate_score,
                "signature": list(program.signature),
                "source_length": program.source_length,
                "source_path": program_source_paths.get(program.program_id),
            }
        )
    programs.sort(key=lambda item: (item["aggregate_score"], -item["program_id"]), reverse=True)
    clusters = []
    for signature, records in island.clusters.items():
        clusters.append(
            {
                "signature": list(signature),
                "programs": [
                    {
                        "program_id": record.program_id,
                        "aggregate_score": record.aggregate_score,
                    }
                    for record in records
                ],
            }
        )
    clusters.sort(
        key=lambda cluster: (
            max(program["aggregate_score"] for program in cluster["programs"]),
            -min(program["program_id"] for program in cluster["programs"]),
        ),
        reverse=True,
    )
    return {
        "index": island_index,
        "best_program_id": island.best_program().program_id,
        "program_count": len(programs),
        "cluster_count": len(island.clusters),
        "programs": programs,
        "clusters": clusters,
    }


def build_sampled_programs_summary(
    sampled_programs: list[Any],
    *,
    program_source_paths: dict[int, str] | None = None,
) -> list[dict[str, Any]]:
    """把 prompt 采样到的程序转成统一的展示结构。"""

    program_source_paths = program_source_paths or {}
    return [
        {
            "program_id": record.program_id,
            "aggregate_score": record.aggregate_score,
            "signature": list(record.signature),
            "source_path": program_source_paths.get(record.program_id),
        }
        for record in sampled_programs
    ]


def build_database_snapshot_summary(
    database: Any,
    *,
    program_source_paths: dict[int, str] | None = None,
) -> dict[str, Any]:
    """把当前数据库压成一份简短摘要。"""

    program_source_paths = program_source_paths or {}
    islands = []
    for index, island in enumerate(database.islands):
        clusters = []
        for signature, records in island.clusters.items():
            clusters.append(
                {
                    "signature": list(signature),
                    "programs": [
                        {
                            "program_id": record.program_id,
                            "aggregate_score": record.aggregate_score,
                        }
                        for record in records
                    ],
                }
            )
        clusters.sort(
            key=lambda cluster: (
                max(program["aggregate_score"] for program in cluster["programs"]),
                -min(program["program_id"] for program in cluster["programs"]),
            ),
            reverse=True,
        )
        islands.append(
            {
                "index": index,
                "best_program_id": island.best_program().program_id,
                "program_count": len(island.all_programs()),
                "cluster_count": len(island.clusters),
                "best_program_path": program_source_paths.get(island.best_program().program_id),
                "clusters": clusters,
            }
        )
    return {
        "evaluated_candidates": database.evaluated_candidates,
        "best_program_id": database.best_program().program_id,
        "islands": islands,
    }


def build_program_source_path(program_id: int, trace_dir: str | Path | None) -> str | None:
    """根据 trace 目录推导 program 文件路径。"""

    if trace_dir is None:
        return None
    return str(Path(trace_dir) / "programs" / f"program_{program_id:06d}.py")
