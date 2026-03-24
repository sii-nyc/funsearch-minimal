"""Core FunSearch loop, specification, and evaluation."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

from funsearch.database import ProgramDatabase, ProgramRecord
from funsearch.llm import LLMClient
from funsearch.prompting import build_prompt, extract_function_source, extract_generated_function, replace_function

if TYPE_CHECKING:
    from funsearch.tracing import TraceWriter


@dataclass(frozen=True)
class ProblemSpecification:
    """Everything FunSearch needs to define and score a search problem."""

    seed_program: str
    target_function: str
    entrypoint: str
    inputs: tuple[Any, ...]
    aggregate_scores: Callable[[tuple[float, ...]], float]


@dataclass(frozen=True)
class EvaluationResult:
    """A valid evaluator output for a program."""

    signature: tuple[float, ...]
    aggregate_score: float


@dataclass(frozen=True)
class SearchConfig:
    """Configuration for the minimal synchronous search loop."""

    iterations: int = 20
    islands: int = 4
    reset_interval: int = 10
    prompt_versions: int = 2
    cluster_temperature: float = 1.0
    program_temperature: float = 0.2
    random_seed: int = 0


@dataclass(frozen=True)
class SearchResult:
    """Final best program discovered by the search loop."""

    best_program: str
    best_function: str
    best_signature: tuple[float, ...]
    best_score: float
    evaluated_candidates: int
    trace_dir: str | None = None


def evaluate_program(program_source: str, specification: ProblemSpecification) -> EvaluationResult | None:
    """Executes a candidate program and returns its valid score signature."""

    result, _ = evaluate_program_detailed(program_source, specification)
    return result


def evaluate_program_detailed(
    program_source: str,
    specification: ProblemSpecification,
) -> tuple[EvaluationResult | None, str | None]:
    """Executes a candidate program and returns either a score or a rejection reason."""

    namespace: dict[str, Any] = {}
    try:
        exec(program_source, namespace)
    except Exception as exc:
        return None, f"exec_error: {type(exc).__name__}: {exc}"

    if specification.entrypoint not in namespace:
        return None, f"missing_entrypoint: {specification.entrypoint}"
    entrypoint = namespace[specification.entrypoint]

    signature = []
    for problem_input in specification.inputs:
        try:
            score = entrypoint(problem_input)
        except Exception as exc:
            return None, f"runtime_error[{problem_input!r}]: {type(exc).__name__}: {exc}"
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            return None, f"invalid_score_type[{problem_input!r}]: {type(score).__name__}"
        score = float(score)
        if not math.isfinite(score):
            return None, f"non_finite_score[{problem_input!r}]"
        signature.append(score)

    signature_tuple = tuple(signature)
    return (
        EvaluationResult(
            signature=signature_tuple,
            aggregate_score=float(specification.aggregate_scores(signature_tuple)),
        ),
        None,
    )


class FunSearchRunner:
    """Minimal synchronous reproduction of the FunSearch core loop."""

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
        self.rng = random.Random(config.random_seed)
        self.trace_writer = trace_writer

    def run(self) -> SearchResult:
        seed_result, seed_error = evaluate_program_detailed(self.specification.seed_program, self.specification)
        if seed_result is None:
            raise ValueError(f"Seed program is invalid and cannot initialize the search: {seed_error}")

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
            initial_snapshot = self.trace_writer.write_snapshot("initial_database", database)
            self.trace_writer.log_event("database_snapshot", name="initial_database", snapshot_path=initial_snapshot)

        for iteration in range(self.config.iterations):
            island_index, sampled_programs = database.sample_prompt_records(self.config.prompt_versions)
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
            completion = self.llm.generate(prompt)
            completion_path = None
            if self.trace_writer is not None:
                completion_path = self.trace_writer.write_completion(iteration, completion)
                self.trace_writer.log_event(
                    "completion_received",
                    iteration=iteration,
                    island_index=island_index,
                    completion_path=completion_path,
                )
            generated_name = f"{self.specification.target_function}_v{len(sampled_programs)}"
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
                    candidate_path = self.trace_writer.write_candidate_program(iteration, candidate_program)
                    self.trace_writer.log_event(
                        "candidate_extracted",
                        iteration=iteration,
                        island_index=island_index,
                        candidate_program_path=candidate_path,
                    )
                candidate_result, rejection_reason = evaluate_program_detailed(candidate_program, self.specification)
                if candidate_result is not None:
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
                snapshot_path = self.trace_writer.write_snapshot(f"iteration_{iteration:04d}", database)
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
            final_snapshot = self.trace_writer.write_snapshot("final_database", database)
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
            trace_dir=str(self.trace_writer.root_dir) if self.trace_writer is not None else None,
        )
