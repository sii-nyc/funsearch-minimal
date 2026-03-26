"""本地 trace 可视化工具。

这个模块读取 `TraceWriter` 产出的目录，并提供两个能力：
- 把分散的 JSON / 文本文件整理成一个统一状态对象
- 在终端里实时展示搜索过程

设计目标是教学可视化，而不是通用监控系统：
- 尽量复用现有 trace 文件，不要求重写主流程
- 默认自动刷新，适合边跑边看
- 同时保留摘要、轮次导航和详细文本，便于深入理解每一轮发生了什么
"""

from __future__ import annotations

import argparse
import curses
import json
import textwrap
import time
from pathlib import Path
from typing import Any

from funsearch.trace_formatting import (
    SECTION_ORDER,
    SECTION_TITLES,
    build_iteration_list_lines,
    build_iteration_section_lines,
    build_run_summary_lines,
)


def load_trace_state(trace_dir: str | Path) -> dict[str, Any]:
    """读取 trace 目录，整理成适合前端展示的状态对象。"""

    root = Path(trace_dir)
    run_metadata = _read_json_if_exists(root / "run.json") or {}
    events = _read_jsonl(root / "events.jsonl")

    initial_snapshot = _read_json_if_exists(root / "snapshots" / "initial_database.json")
    final_snapshot = _read_json_if_exists(root / "snapshots" / "final_database.json")

    iteration_events: dict[int, dict[str, Any]] = {}
    resets: list[dict[str, Any]] = []
    run_started: dict[str, Any] | None = None
    run_completed: dict[str, Any] | None = None

    for event in events:
        event_type = event["type"]
        if event_type == "run_started":
            run_started = event
            continue
        if event_type == "run_completed":
            run_completed = event
            continue
        if event_type == "database_snapshot":
            continue

        iteration = event.get("iteration")
        if iteration is None:
            continue
        iteration_entry = iteration_events.setdefault(
            int(iteration),
            {
                "iteration": int(iteration),
                "events": [],
            },
        )
        iteration_entry["events"].append(event)
        if event_type == "islands_reset":
            resets.append(event)

    ordered_iterations = []
    for iteration in sorted(iteration_events):
        entry = iteration_events[iteration]
        ordered_iterations.append(_build_iteration_summary(root, entry))

    completed_iterations = sum(1 for item in ordered_iterations if item["status"] in {"accepted", "rejected", "completed"})
    total_iterations = run_metadata.get("search_config", {}).get("iterations")
    current_status = _describe_current_status(ordered_iterations, total_iterations, run_completed)

    final_best = None
    if run_completed is not None:
        final_best = {
            "best_program_id": run_completed.get("best_program_id"),
            "best_score": run_completed.get("best_score"),
            "best_signature": run_completed.get("best_signature"),
        }

    return {
        "trace_dir": str(root.resolve()),
        "run_metadata": run_metadata,
        "run_started": run_started,
        "run_completed": run_completed,
        "current_status": current_status,
        "completed_iterations": completed_iterations,
        "total_iterations": total_iterations,
        "initial_snapshot": initial_snapshot,
        "final_snapshot": final_snapshot,
        "final_best": final_best,
        "iterations": ordered_iterations,
        "reset_count": len(resets),
    }


def _build_iteration_summary(root: Path, iteration_entry: dict[str, Any]) -> dict[str, Any]:
    """把同一轮的多条事件合成一个更易展示的对象。"""

    iteration = iteration_entry["iteration"]
    prompt_event = _find_event(iteration_entry["events"], "prompt_sampled")
    completion_event = _find_event(iteration_entry["events"], "completion_received")
    accepted_event = _find_event(iteration_entry["events"], "candidate_accepted")
    rejected_event = _find_event(iteration_entry["events"], "candidate_rejected")
    reset_event = _find_event(iteration_entry["events"], "islands_reset")
    completed_event = _find_event(iteration_entry["events"], "iteration_completed")

    pre_snapshot_name = "initial_database.json" if iteration == 0 else f"iteration_{iteration - 1:04d}.json"
    pre_snapshot = _read_json_if_exists(root / "snapshots" / pre_snapshot_name)
    post_snapshot = _read_json_if_exists(root / "snapshots" / f"iteration_{iteration:04d}.json")

    prompt_path = _resolve_relative_path(root, prompt_event.get("prompt_path") if prompt_event else None)
    completion_path = _resolve_relative_path(root, completion_event.get("completion_path") if completion_event else None)
    candidate_path = _resolve_relative_path(
        root,
        (accepted_event or rejected_event or {}).get("candidate_program_path"),
    )

    sampled_island_index = prompt_event.get("island_index") if prompt_event else None
    sampled_island = _extract_island_summary(pre_snapshot, sampled_island_index) if pre_snapshot is not None and sampled_island_index is not None else None

    score_info = None
    status = "in_progress"
    if accepted_event is not None:
        status = "accepted"
        score_info = {
            "aggregate_score": accepted_event.get("aggregate_score"),
            "signature": accepted_event.get("signature"),
            "program_id": accepted_event.get("program_id"),
            "source_path": _stringify_path(_resolve_relative_path(root, accepted_event.get("source_path"))),
        }
    elif rejected_event is not None:
        status = "rejected"
        score_info = {
            "reason": rejected_event.get("reason"),
        }
    elif completed_event is not None:
        status = "completed"

    return {
        "iteration": iteration,
        "status": status,
        "selected_island_index": sampled_island_index,
        "sampled_programs": prompt_event.get("sampled_programs", []) if prompt_event else [],
        "selected_island": sampled_island,
        "prompt_path": str(prompt_path) if prompt_path is not None else None,
        "prompt_text": _read_text_if_exists(prompt_path),
        "completion_path": str(completion_path) if completion_path is not None else None,
        "completion_text": _read_text_if_exists(completion_path),
        "candidate_program_path": str(candidate_path) if candidate_path is not None else None,
        "candidate_program_text": _read_text_if_exists(candidate_path),
        "score_info": score_info,
        "reset_actions": reset_event.get("actions", []) if reset_event else [],
        "post_snapshot": post_snapshot,
        "best_after_iteration": {
            "best_program_id": completed_event.get("best_program_id"),
            "best_score": completed_event.get("best_score"),
            "snapshot_path": _stringify_path(_resolve_relative_path(root, completed_event.get("snapshot_path"))),
        }
        if completed_event
        else None,
    }


def _extract_island_summary(snapshot: dict[str, Any] | None, island_index: int) -> dict[str, Any] | None:
    """从快照中抽出某个 island 的程序列表，便于可视化。"""

    if snapshot is None:
        return None
    for island in snapshot.get("islands", []):
        if island.get("index") != island_index:
            continue
        programs = []
        for cluster in island.get("clusters", []):
            for program in cluster.get("programs", []):
                programs.append(
                    {
                        "program_id": program.get("program_id"),
                        "aggregate_score": program.get("aggregate_score"),
                        "signature": program.get("signature"),
                        "source_length": program.get("source_length"),
                        "source_path": program.get("source_path"),
                    }
                )
        programs.sort(key=lambda item: (item["aggregate_score"], -item["program_id"]), reverse=True)
        clusters = []
        for cluster in island.get("clusters", []):
            cluster_programs = [
                {
                    "program_id": program.get("program_id"),
                    "aggregate_score": program.get("aggregate_score"),
                    "signature": program.get("signature"),
                    "source_path": program.get("source_path"),
                }
                for program in cluster.get("programs", [])
            ]
            cluster_programs.sort(key=lambda item: (item["aggregate_score"], -item["program_id"]), reverse=True)
            if not cluster_programs:
                continue
            clusters.append(
                {
                    "signature": cluster.get("signature"),
                    "aggregate_score": cluster.get("aggregate_score"),
                    "programs": cluster_programs,
                }
            )
        clusters.sort(
            key=lambda cluster: (
                cluster.get("aggregate_score"),
                -min(program["program_id"] for program in cluster["programs"]),
            ),
            reverse=True,
        )
        return {
            "index": island.get("index"),
            "best_program_id": island.get("best_program_id"),
            "program_count": island.get("program_count"),
            "cluster_count": island.get("cluster_count"),
            "programs": programs,
            "clusters": clusters,
        }
    return None


def _describe_current_status(
    iterations: list[dict[str, Any]],
    total_iterations: int | None,
    run_completed: dict[str, Any] | None,
) -> str:
    """给当前 trace 生成一个面向人的简短状态描述。"""

    if run_completed is not None:
        return "completed"
    if not iterations:
        return "waiting_for_first_iteration"
    latest = iterations[-1]
    if latest["status"] == "in_progress":
        return f"running_iteration_{latest['iteration']}"
    if total_iterations is not None and len(iterations) < total_iterations:
        return f"waiting_for_iteration_{len(iterations)}"
    return "running"


def _find_event(events: list[dict[str, Any]], event_type: str) -> dict[str, Any] | None:
    for event in events:
        if event["type"] == event_type:
            return event
    return None


def _read_json_if_exists(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _read_text_if_exists(path: Path | None) -> str | None:
    if path is None or not path.exists():
        return None
    return path.read_text(encoding="utf-8")


def _resolve_relative_path(root: Path, relative_path: str | None) -> Path | None:
    if not relative_path:
        return None
    return root / relative_path


def _stringify_path(path: Path | None) -> str | None:
    if path is None:
        return None
    return str(path)


class TraceViewerTUI:
    """基于 curses 的交互式 trace 可视化。"""

    def __init__(self, trace_dir: str | Path, refresh_interval: float = 2.0, follow_latest: bool = True) -> None:
        self.trace_root = Path(trace_dir).resolve()
        if not self.trace_root.exists():
            raise SystemExit(f"Trace directory does not exist: {self.trace_root}")
        self.refresh_interval = refresh_interval
        self.follow_latest = follow_latest
        self.state: dict[str, Any] = {}
        self.selected_iteration: int | None = None
        self.section_index = 0
        self.detail_scroll = 0
        self.iteration_scroll = 0
        self.status_line = "Loading trace..."
        self.last_refresh_display = "never"

    def run(self) -> None:
        curses.wrapper(self._main)

    def _main(self, stdscr: Any) -> None:
        self._configure_screen(stdscr)
        self._refresh_state()
        next_refresh = time.monotonic() + self.refresh_interval

        while True:
            self._draw(stdscr)
            key = stdscr.getch()
            if key != -1 and self._handle_key(key):
                return
            if time.monotonic() >= next_refresh:
                self._refresh_state()
                next_refresh = time.monotonic() + self.refresh_interval

    def _configure_screen(self, stdscr: Any) -> None:
        stdscr.nodelay(True)
        stdscr.timeout(200)
        try:
            curses.curs_set(0)
        except curses.error:
            pass
        if curses.has_colors():
            curses.start_color()
            curses.use_default_colors()
            curses.init_pair(1, curses.COLOR_BLACK, curses.COLOR_CYAN)
            curses.init_pair(2, curses.COLOR_YELLOW, -1)
            curses.init_pair(3, curses.COLOR_GREEN, -1)
            curses.init_pair(4, curses.COLOR_RED, -1)

    def _refresh_state(self) -> None:
        try:
            self.state = load_trace_state(self.trace_root)
            self._sync_selection()
            self.last_refresh_display = time.strftime("%H:%M:%S")
            self.status_line = f"Refreshed at {self.last_refresh_display}"
        except Exception as exc:
            self.status_line = f"Refresh failed: {exc}"

    def _sync_selection(self) -> None:
        iterations = self.state.get("iterations", [])
        if not iterations:
            self.selected_iteration = None
            self.iteration_scroll = 0
            self.detail_scroll = 0
            return

        latest_iteration = iterations[-1]["iteration"]
        existing_iterations = {item["iteration"] for item in iterations}
        if self.selected_iteration is None or self.follow_latest or self.selected_iteration not in existing_iterations:
            self.selected_iteration = latest_iteration
            self.detail_scroll = 0

    def _selected_iteration_item(self) -> dict[str, Any] | None:
        for item in self.state.get("iterations", []):
            if item["iteration"] == self.selected_iteration:
                return item
        return None

    def _handle_key(self, key: int) -> bool:
        if key in (ord("q"), ord("Q")):
            return True
        if key in (curses.KEY_LEFT, ord("h")):
            self.follow_latest = False
            self._move_iteration(-1)
            return False
        if key in (curses.KEY_RIGHT, ord("l")):
            self.follow_latest = False
            self._move_iteration(1)
            return False
        if key in (ord("["), ord("p")):
            self._move_section(-1)
            return False
        if key in (ord("]"), ord("n"), 9):
            self._move_section(1)
            return False
        if key in (curses.KEY_UP, ord("k")):
            self.detail_scroll = max(0, self.detail_scroll - 1)
            return False
        if key in (curses.KEY_DOWN, ord("j")):
            self.detail_scroll += 1
            return False
        if key == curses.KEY_PPAGE:
            self.detail_scroll = max(0, self.detail_scroll - 10)
            return False
        if key == curses.KEY_NPAGE:
            self.detail_scroll += 10
            return False
        if key == ord("g"):
            self.follow_latest = False
            self._jump_to_boundary(first=True)
            return False
        if key == ord("G"):
            self._jump_to_boundary(first=False)
            return False
        if key in (ord("f"), ord("F")):
            self.follow_latest = not self.follow_latest
            if self.follow_latest:
                self._jump_to_boundary(first=False)
            return False
        if key in (ord("r"), ord("R")):
            self._refresh_state()
            return False
        return False

    def _move_iteration(self, delta: int) -> None:
        iterations = self.state.get("iterations", [])
        if not iterations:
            return
        indices = {item["iteration"]: index for index, item in enumerate(iterations)}
        current_index = indices.get(self.selected_iteration, len(iterations) - 1)
        next_index = min(max(current_index + delta, 0), len(iterations) - 1)
        self.selected_iteration = iterations[next_index]["iteration"]
        self.detail_scroll = 0

    def _jump_to_boundary(self, first: bool) -> None:
        iterations = self.state.get("iterations", [])
        if not iterations:
            return
        self.selected_iteration = iterations[0]["iteration"] if first else iterations[-1]["iteration"]
        self.detail_scroll = 0

    def _move_section(self, delta: int) -> None:
        self.section_index = (self.section_index + delta) % len(SECTION_ORDER)
        self.detail_scroll = 0

    def _draw(self, stdscr: Any) -> None:
        stdscr.erase()
        height, width = stdscr.getmaxyx()
        if height < 12 or width < 80:
            self._safe_addstr(stdscr, 0, 0, "Terminal is too small for trace viewer. Need at least 80x12.")
            self._safe_addstr(stdscr, 2, 0, "Resize the terminal or use --dump-state.")
            stdscr.refresh()
            return

        summary_lines = build_run_summary_lines(self.state) if self.state else ["Loading trace..."]
        selected_item = self._selected_iteration_item()
        section = SECTION_ORDER[self.section_index]
        detail_lines = build_iteration_section_lines(selected_item, section)
        iteration_lines = build_iteration_list_lines(self.state, self.selected_iteration)

        title = "FunSearch Trace Viewer (Terminal)"
        self._safe_addstr(stdscr, 0, 0, title, curses.A_BOLD)
        for offset, line in enumerate(summary_lines, start=1):
            self._safe_addstr(stdscr, offset, 0, line)

        divider_y = len(summary_lines) + 1
        self._safe_addstr(stdscr, divider_y, 0, "-" * width)

        body_y = divider_y + 1
        footer_y = height - 1
        body_height = footer_y - body_y
        left_width = min(34, max(26, width // 4))
        right_x = left_width + 2
        right_width = width - right_x

        left_title = f"Iterations ({len(self.state.get('iterations', []))})"
        section_title = f"Section: {SECTION_TITLES[section]}"
        follow_label = "follow latest: on" if self.follow_latest else "follow latest: off"
        self._safe_addstr(stdscr, body_y, 0, left_title, curses.A_BOLD)
        self._safe_addstr(stdscr, body_y, right_x, f"{section_title} | {follow_label}", curses.A_BOLD)

        panel_y = body_y + 1
        panel_height = body_height - 1
        self._draw_iteration_panel(stdscr, panel_y, 0, left_width, panel_height, iteration_lines)
        self._draw_detail_panel(stdscr, panel_y, right_x, right_width, panel_height, detail_lines)

        help_text = "Keys: h/l or <-/-> iteration | [/]/tab section | j/k scroll | f follow | r refresh | q quit"
        self._safe_addstr(stdscr, footer_y, 0, help_text[: width - 1], curses.color_pair(2) if curses.has_colors() else 0)
        if width > len(help_text) + 4:
            status_x = max(0, width - len(self.status_line) - 1)
            self._safe_addstr(stdscr, footer_y, status_x, self.status_line[: width - status_x - 1])
        stdscr.refresh()

    def _draw_iteration_panel(
        self,
        stdscr: Any,
        start_y: int,
        start_x: int,
        width: int,
        height: int,
        lines: list[str],
    ) -> None:
        self._safe_addstr(stdscr, start_y, start_x + width, "|")
        visible_count = max(1, height)
        selected_index = 0
        for index, line in enumerate(lines):
            if line.startswith(">"):
                selected_index = index
                break
        if selected_index < self.iteration_scroll:
            self.iteration_scroll = selected_index
        elif selected_index >= self.iteration_scroll + visible_count:
            self.iteration_scroll = selected_index - visible_count + 1

        visible_lines = lines[self.iteration_scroll : self.iteration_scroll + visible_count]
        for row, line in enumerate(visible_lines):
            attr = curses.color_pair(1) if curses.has_colors() and line.startswith(">") else 0
            self._safe_addstr(stdscr, start_y + row, start_x, line[: max(1, width - 1)], attr)

    def _draw_detail_panel(
        self,
        stdscr: Any,
        start_y: int,
        start_x: int,
        width: int,
        height: int,
        lines: list[str],
    ) -> None:
        wrapped_lines = _wrap_lines(lines, max(10, width - 1))
        if self.detail_scroll > max(0, len(wrapped_lines) - height):
            self.detail_scroll = max(0, len(wrapped_lines) - height)
        visible_lines = wrapped_lines[self.detail_scroll : self.detail_scroll + max(1, height)]
        for row, line in enumerate(visible_lines):
            self._safe_addstr(stdscr, start_y + row, start_x, line[: max(1, width - 1)])

    def _safe_addstr(self, stdscr: Any, y: int, x: int, text: str, attr: int = 0) -> None:
        height, width = stdscr.getmaxyx()
        if y < 0 or y >= height or x < 0 or x >= width:
            return
        clipped = text[: max(0, width - x - 1)]
        if not clipped:
            return
        try:
            stdscr.addstr(y, x, clipped, attr)
        except curses.error:
            return


def _wrap_lines(lines: list[str], width: int) -> list[str]:
    wrapped: list[str] = []
    for raw_line in lines:
        if raw_line == "":
            wrapped.append("")
            continue
        expanded = raw_line.expandtabs(4)
        segments = textwrap.wrap(
            expanded,
            width=width,
            replace_whitespace=False,
            drop_whitespace=False,
            break_long_words=True,
            break_on_hyphens=False,
        )
        wrapped.extend(segments or [""])
    return wrapped


def build_parser() -> argparse.ArgumentParser:
    """定义 trace viewer CLI。"""

    parser = argparse.ArgumentParser(description="Open a terminal dashboard for a FunSearch trace directory.")
    parser.add_argument("--trace-dir", required=True, help="Trace directory produced by --trace-dir during a run.")
    parser.add_argument(
        "--refresh-interval",
        type=float,
        default=2.0,
        help="Seconds between automatic refreshes in terminal mode.",
    )
    parser.add_argument(
        "--no-follow-latest",
        action="store_true",
        help="Start with follow-latest turned off.",
    )
    parser.add_argument(
        "--dump-state",
        action="store_true",
        help="Print the merged trace state as JSON and exit instead of starting the terminal viewer.",
    )
    return parser


def main() -> None:
    """trace viewer CLI 入口。"""

    args = build_parser().parse_args()
    if args.dump_state:
        print(json.dumps(load_trace_state(args.trace_dir), indent=2, ensure_ascii=False, sort_keys=True))
        return
    TraceViewerTUI(
        trace_dir=args.trace_dir,
        refresh_interval=args.refresh_interval,
        follow_latest=not args.no_follow_latest,
    ).run()


if __name__ == "__main__":
    main()
