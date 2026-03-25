"""把搜索过程直接打印到命令行。

默认优先使用 rich 做紧凑的终端展示；如果环境里没有 rich，
则退回普通文本输出。
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, TextIO

try:
    from rich.console import Console, Group
    from rich.panel import Panel
    from rich.syntax import Syntax
    from rich.table import Table

    _HAS_RICH = True
except ImportError:  # pragma: no cover - rich 缺失时只走回退路径
    Console = None
    Group = None
    Panel = None
    Syntax = None
    Table = None
    _HAS_RICH = False


class ConsoleRunReporter:
    """在搜索运行过程中持续打印轮次记录。"""

    def __init__(self, output: TextIO | None = None) -> None:
        self.output = output if output is not None else sys.stdout
        self._console = (
            Console(file=self.output, force_terminal=False, highlight=False, soft_wrap=True)
            if _HAS_RICH
            else None
        )

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
        if self._console is None:
            self._write_lines(
                [
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
            )
            return

        summary_table = Table.grid(expand=True)
        summary_table.add_column(style="cyan", ratio=1)
        summary_table.add_column(ratio=5)
        summary_table.add_row("Problem", f"target={target_function} entrypoint={entrypoint} inputs={input_count}")
        summary_table.add_row(
            "Config",
            (
                f"llm={llm_backend} iterations={iterations} islands={islands} "
                f"reset_interval={reset_interval} prompt_versions={prompt_versions} seed={random_seed}"
            ),
        )
        summary_table.add_row("Seed", f"score={seed_score} signature={list(seed_signature)}")
        summary_table.add_row("Trace", trace_dir or "disabled")
        summary_table.add_row(
            "Prompt",
            "fixed skeleton and instructions are reused every iteration; only the sampled versions change",
        )

        problem_table = Table.grid(padding=(0, 1))
        problem_table.add_column()
        for line in problem_summary:
            problem_table.add_row(f"- {line}")

        self._console.print(
            Panel(
                Group(summary_table, "", "problem_summary:", problem_table),
                title="FunSearch Run Started",
                border_style="blue",
            )
        )

    def report_iteration(
        self,
        *,
        iteration: int,
        total_iterations: int,
        current_best_score: float,
        current_best_program_id: int,
        iteration_item: dict[str, Any],
    ) -> None:
        if self._console is None:
            self._report_iteration_plain(
                iteration=iteration,
                total_iterations=total_iterations,
                current_best_score=current_best_score,
                current_best_program_id=current_best_program_id,
                iteration_item=iteration_item,
            )
            return

        status = iteration_item.get("status", "unknown")
        score_info = iteration_item.get("score_info") or {}
        border_style = "green" if status == "accepted" else "red"

        meta_table = Table.grid(expand=True)
        meta_table.add_column(style="cyan", ratio=1)
        meta_table.add_column(ratio=5)
        meta_table.add_row(
            "Status",
            (
                f"{status} | selected_island={iteration_item.get('selected_island_index')} "
                f"| current_best_score={current_best_score} | current_best_program_id={current_best_program_id}"
            ),
        )
        meta_table.add_row("Sampled", str(iteration_item.get("sampled_program_ids", [])))
        meta_table.add_row("Selected Clusters", str(iteration_item.get("selected_island_clusters", [])))

        function_block = iteration_item.get("generated_function_text") or "No valid target function extracted from the completion."
        generated_panel = Panel(
            Syntax(function_block, "python", word_wrap=True, theme="monokai"),
            title="Generated Function",
            border_style="magenta",
        )

        result_table = Table.grid(expand=True)
        result_table.add_column(style="cyan", ratio=1)
        result_table.add_column(ratio=5)
        if status == "accepted":
            result_table.add_row("Result", f"accepted program_id={score_info.get('program_id')}")
            result_table.add_row("Score", str(score_info.get("aggregate_score")))
            result_table.add_row("Signature", str(score_info.get("signature")))
            result_table.add_row("Path", str(score_info.get("source_path")))
        else:
            result_table.add_row("Result", f"rejected: {score_info.get('reason')}")

        panels: list[Any] = [meta_table, generated_panel, Panel(result_table, title="Result", border_style=border_style)]

        reset_actions = iteration_item.get("reset_actions") or []
        if reset_actions:
            reset_table = Table(title="Reset Actions", expand=True)
            reset_table.add_column("Island", style="cyan")
            reset_table.add_column("Donor Island", style="cyan")
            reset_table.add_column("Donor Program", style="cyan")
            reset_table.add_column("New Program", style="cyan")
            for action in reset_actions:
                reset_table.add_row(
                    str(action.get("island_index")),
                    str(action.get("donor_island_index")),
                    str(action.get("donor_program_id")),
                    str(action.get("new_program_id")),
                )
            panels.append(reset_table)

        panels.append(_build_islands_table(iteration_item.get("post_snapshot") or {}))

        self._console.print(
            Panel(
                Group(*panels),
                title=f"Iteration {iteration + 1}/{total_iterations}",
                border_style=border_style,
            )
        )

    def report_run_completed(
        self,
        *,
        best_score: float,
        best_signature: tuple[float, ...],
        evaluated_candidates: int,
        trace_dir: str | None,
    ) -> None:
        if self._console is None:
            self._write_lines(
                [
                    "=== FunSearch Run Completed ===",
                    f"best_score: {best_score}",
                    f"best_signature: {list(best_signature)}",
                    f"evaluated_candidates: {evaluated_candidates}",
                    f"trace_dir: {trace_dir or 'disabled'}",
                    "",
                ]
            )
            return

        table = Table.grid(expand=True)
        table.add_column(style="cyan", ratio=1)
        table.add_column(ratio=5)
        table.add_row("Best Score", str(best_score))
        table.add_row("Best Signature", str(list(best_signature)))
        table.add_row("Evaluated", str(evaluated_candidates))
        table.add_row("Trace", trace_dir or "disabled")
        self._console.print(Panel(table, title="FunSearch Run Completed", border_style="blue"))

    def _report_iteration_plain(
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


def _build_islands_table(snapshot: dict[str, Any]) -> Any:
    table = Table(title="Islands", expand=True)
    table.add_column("Island", style="cyan", justify="right")
    table.add_column("Best", style="green", justify="right")
    table.add_column("Clusters")
    if not snapshot:
        table.add_row("-", "-", "Database snapshot is not available.")
        return table

    table.caption = (
        f"evaluated_candidates={snapshot.get('evaluated_candidates')} "
        f"best_program_id={snapshot.get('best_program_id')}"
    )
    for island in snapshot.get("islands", []):
        table.add_row(
            str(island.get("index")),
            str(island.get("best_program_id")),
            str(_format_cluster_program_ids(island.get("clusters", []))),
        )
    return table


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
