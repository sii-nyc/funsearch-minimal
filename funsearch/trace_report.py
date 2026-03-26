"""把 trace 目录整理成一个可直接阅读的文本报告。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from funsearch.trace_formatting import (
    SECTION_ORDER,
    SECTION_TITLES,
    build_iteration_section_lines,
    build_run_summary_lines,
)
from funsearch.trace_viewer import load_trace_state


def build_trace_report_lines(state: dict[str, Any]) -> list[str]:
    """把完整 trace 状态展开成单个文本文件。"""

    lines = [
        "FunSearch Trace Report",
        "",
        *build_run_summary_lines(state),
    ]

    iterations = state.get("iterations", [])
    if not iterations:
        lines.extend(["", "No iterations recorded yet."])
        return lines

    for iteration_item in iterations:
        lines.extend(
            [
                "",
                "=" * 80,
                f"Iteration {iteration_item['iteration']:04d}",
                "=" * 80,
            ]
        )
        for section in SECTION_ORDER:
            lines.extend(
                [
                    "",
                    f"[{SECTION_TITLES[section]}]",
                    *build_iteration_section_lines(iteration_item, section),
                ]
            )
    return lines


def write_trace_report(trace_dir: str | Path, output_path: str | Path | None = None) -> Path:
    """读取 trace 目录并写出一个整合后的文本报告。"""

    root = Path(trace_dir).resolve()
    target_path = Path(output_path).resolve() if output_path is not None else root / "trace_report.txt"
    state = load_trace_state(root)
    target_path.write_text("\n".join(build_trace_report_lines(state)) + "\n", encoding="utf-8")
    return target_path
