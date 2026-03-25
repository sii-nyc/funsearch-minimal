"""把搜索过程直接打印到命令行。

这个模块不依赖额外进程或浏览器，而是在主搜索循环运行时
按轮次把关键记录直接输出到 stdout。
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, TextIO

from funsearch.trace_formatting import (
    SECTION_ORDER,
    SECTION_TITLES,
    build_iteration_section_lines,
)


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
        seed_score: float,
        seed_signature: tuple[float, ...],
    ) -> None:
        lines = [
            "=== FunSearch Run Started ===",
            (
                f"problem: target={target_function} entrypoint={entrypoint} "
                f"inputs={input_count}"
            ),
            (
                f"config: llm={llm_backend} iterations={iterations} islands={islands} "
                f"reset_interval={reset_interval} prompt_versions={prompt_versions} "
                f"seed={random_seed}"
            ),
            f"seed: score={seed_score} signature={list(seed_signature)}",
            f"trace_dir: {trace_dir or 'disabled'}",
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
        lines = [
            f"=== Iteration {iteration + 1}/{total_iterations} ===",
            (
                f"status: {status} | selected_island={iteration_item.get('selected_island_index')} "
                f"| current_best_score={current_best_score} | current_best_program_id={current_best_program_id}"
            ),
            "",
        ]
        for section in SECTION_ORDER:
            lines.append(f"[{SECTION_TITLES[section]}]")
            lines.extend(build_iteration_section_lines(iteration_item, section))
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
    return {
        "index": island_index,
        "best_program_id": island.best_program().program_id,
        "program_count": len(programs),
        "cluster_count": len(island.clusters),
        "programs": programs,
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
        islands.append(
            {
                "index": index,
                "best_program_id": island.best_program().program_id,
                "program_count": len(island.all_programs()),
                "cluster_count": len(island.clusters),
                "best_program_path": program_source_paths.get(island.best_program().program_id),
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
