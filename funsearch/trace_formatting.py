"""trace 展示共用的纯文本格式化函数。"""

from __future__ import annotations

import json
from typing import Any

SECTION_TITLES = {
    "summary": "Summary",
    "island": "Selected Island",
    "sampled": "Sampled Programs",
    "prompt": "Prompt",
    "completion": "Completion",
    "candidate": "Candidate Program",
    "reset": "Reset Actions",
    "snapshot": "Post Snapshot",
}
SECTION_ORDER = tuple(SECTION_TITLES)


def _format_value(value: Any) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def build_run_summary_lines(state: dict[str, Any]) -> list[str]:
    """生成顶部运行摘要。"""

    metadata = state.get("run_metadata", {})
    search_config = metadata.get("search_config", {})
    specification = metadata.get("specification", {})
    final_best = state.get("final_best") or {}

    total_iterations = state.get("total_iterations")
    if total_iterations is None:
        completed = str(state.get("completed_iterations", 0))
    else:
        completed = f"{state.get('completed_iterations', 0)}/{total_iterations}"

    return [
        f"Trace: {state.get('trace_dir', 'N/A')}",
        f"Status: {state.get('current_status', 'N/A')} | Completed: {completed} | Resets: {state.get('reset_count', 0)}",
        (
            "Problem: "
            f"target={specification.get('target_function', 'N/A')} "
            f"entrypoint={specification.get('entrypoint', 'N/A')} "
            f"inputs={len(specification.get('inputs', []))}"
        ),
        (
            "Run: "
            f"llm={metadata.get('llm_backend', 'N/A')} "
            f"iterations={search_config.get('iterations', 'N/A')} "
            f"islands={search_config.get('islands', 'N/A')} "
            f"reset_interval={search_config.get('reset_interval', 'N/A')} "
            f"prompt_versions={search_config.get('prompt_versions', 'N/A')} "
            f"seed={search_config.get('random_seed', 'N/A')}"
        ),
        (
            "Best: "
            f"score={_format_value(final_best.get('best_score'))} "
            f"program_id={_format_value(final_best.get('best_program_id'))} "
            f"signature={_format_value(final_best.get('best_signature'))}"
        ),
    ]


def build_iteration_list_lines(state: dict[str, Any], selected_iteration: int | None) -> list[str]:
    """生成左侧轮次列表。"""

    iterations = state.get("iterations", [])
    if not iterations:
        return ["No iterations yet."]

    lines = []
    for item in iterations:
        marker = ">" if item["iteration"] == selected_iteration else " "
        score_info = item.get("score_info") or {}
        score_suffix = ""
        if item["status"] == "accepted":
            score_suffix = f" score={_format_value(score_info.get('aggregate_score'))}"
        elif item["status"] == "rejected":
            score_suffix = f" reason={_format_value(score_info.get('reason'))}"
        lines.append(f"{marker} iter {item['iteration']:04d} {item['status']}{score_suffix}")
    return lines


def build_iteration_section_lines(iteration_item: dict[str, Any] | None, section: str) -> list[str]:
    """为选中轮次构建详情面板。"""

    if iteration_item is None:
        return ["No iteration selected yet."]

    if section == "summary":
        return _build_summary_section(iteration_item)
    if section == "island":
        return _build_island_section(iteration_item)
    if section == "sampled":
        return _build_sampled_programs_section(iteration_item)
    if section == "prompt":
        return _build_text_section("Prompt", iteration_item.get("prompt_path"), iteration_item.get("prompt_text"))
    if section == "completion":
        return _build_text_section("Completion", iteration_item.get("completion_path"), iteration_item.get("completion_text"))
    if section == "candidate":
        return _build_text_section(
            "Candidate Program",
            iteration_item.get("candidate_program_path"),
            iteration_item.get("candidate_program_text"),
        )
    if section == "reset":
        return _build_reset_section(iteration_item)
    if section == "snapshot":
        return _build_snapshot_section(iteration_item)
    raise ValueError(f"Unknown section: {section}")


def _build_summary_section(iteration_item: dict[str, Any]) -> list[str]:
    lines = [
        f"Iteration {iteration_item['iteration']}",
        f"Status: {iteration_item.get('status', 'N/A')}",
        f"Selected island: {_format_value(iteration_item.get('selected_island_index'))}",
    ]

    score_info = iteration_item.get("score_info") or {}
    if iteration_item.get("status") == "accepted":
        lines.extend(
            [
                f"Aggregate score: {_format_value(score_info.get('aggregate_score'))}",
                f"Signature: {_format_value(score_info.get('signature'))}",
                f"Program id: {_format_value(score_info.get('program_id'))}",
                f"Accepted source path: {_format_value(score_info.get('source_path'))}",
            ]
        )
    elif iteration_item.get("status") == "rejected":
        lines.extend(
            [
                "Candidate result: rejected",
                f"Reason: {_format_value(score_info.get('reason'))}",
            ]
        )
    else:
        lines.append("Candidate result: not available yet")

    best_after = iteration_item.get("best_after_iteration") or {}
    lines.extend(
        [
            "",
            "After iteration:",
            f"Best score: {_format_value(best_after.get('best_score'))}",
            f"Best program id: {_format_value(best_after.get('best_program_id'))}",
            f"Snapshot path: {_format_value(best_after.get('snapshot_path'))}",
            "",
            f"Prompt path: {_format_value(iteration_item.get('prompt_path'))}",
            f"Completion path: {_format_value(iteration_item.get('completion_path'))}",
            f"Candidate path: {_format_value(iteration_item.get('candidate_program_path'))}",
        ]
    )
    return lines


def _build_island_section(iteration_item: dict[str, Any]) -> list[str]:
    island = iteration_item.get("selected_island")
    if island is None:
        return ["Selected island snapshot is not available for this iteration."]

    lines = [
        f"Island index: {_format_value(island.get('index'))}",
        f"Best program id: {_format_value(island.get('best_program_id'))}",
        f"Program count: {_format_value(island.get('program_count'))}",
        f"Cluster count: {_format_value(island.get('cluster_count'))}",
        "",
        "Programs ranked by aggregate score:",
    ]
    for program in island.get("programs", []):
        lines.extend(
            [
                (
                    f"- program_id={_format_value(program.get('program_id'))} "
                    f"score={_format_value(program.get('aggregate_score'))} "
                    f"signature={_format_value(program.get('signature'))}"
                ),
                f"  source_path={_format_value(program.get('source_path'))}",
            ]
        )
    if not island.get("programs"):
        lines.append("No programs recorded in this island.")
    clusters = island.get("clusters", [])
    if clusters:
        lines.extend(["", "Clusters ranked by aggregate score:"])
        for index, cluster in enumerate(clusters):
            lines.append(
                (
                    f"- cluster={index} score={_format_value(cluster.get('aggregate_score'))} "
                    f"signature={_format_value(cluster.get('signature'))}"
                )
            )
            for program in cluster.get("programs", []):
                lines.append(
                    (
                        f"  program_id={_format_value(program.get('program_id'))} "
                        f"score={_format_value(program.get('aggregate_score'))}"
                    )
                )
    return lines


def _build_sampled_programs_section(iteration_item: dict[str, Any]) -> list[str]:
    programs = iteration_item.get("sampled_programs", [])
    if not programs:
        return ["This iteration has no prompt sampling data yet."]

    lines = ["Programs used to build the prompt:"]
    for program in programs:
        lines.extend(
            [
                (
                    f"- program_id={_format_value(program.get('program_id'))} "
                    f"score={_format_value(program.get('aggregate_score'))} "
                    f"signature={_format_value(program.get('signature'))}"
                ),
                f"  source_path={_format_value(program.get('source_path'))}",
            ]
        )
    return lines


def _build_text_section(title: str, path: str | None, content: str | None) -> list[str]:
    lines = [title, f"Path: {_format_value(path)}", ""]
    if not content:
        lines.append("File has not been generated yet.")
        return lines
    return lines + content.splitlines()


def _build_reset_section(iteration_item: dict[str, Any]) -> list[str]:
    actions = iteration_item.get("reset_actions", [])
    if not actions:
        return ["This iteration did not trigger an island reset."]

    lines = ["Island reset actions:"]
    for action in actions:
        lines.extend(
            [
                (
                    f"- island={_format_value(action.get('island_index'))} "
                    f"donor_island={_format_value(action.get('donor_island_index'))}"
                ),
                (
                    f"  donor_program_id={_format_value(action.get('donor_program_id'))} "
                    f"new_program_id={_format_value(action.get('new_program_id'))}"
                ),
            ]
        )
    return lines


def _build_snapshot_section(iteration_item: dict[str, Any]) -> list[str]:
    snapshot = iteration_item.get("post_snapshot")
    if snapshot is None:
        return ["The post-iteration database snapshot is not available yet."]

    lines = [
        f"Evaluated candidates: {_format_value(snapshot.get('evaluated_candidates'))}",
        f"Best program id: {_format_value(snapshot.get('best_program_id'))}",
        "",
        "Islands after this iteration:",
    ]
    for island in snapshot.get("islands", []):
        lines.extend(_build_snapshot_island_lines(island))
    if not snapshot.get("islands"):
        lines.append("No island data is available in this snapshot.")
    return lines


def _build_snapshot_island_lines(island: dict[str, Any]) -> list[str]:
    lines = [
        (
            f"- island={_format_value(island.get('index'))} "
            f"best_program_id={_format_value(island.get('best_program_id'))} "
            f"program_count={_format_value(island.get('program_count'))} "
            f"cluster_count={_format_value(island.get('cluster_count'))}"
        )
    ]
    clusters = sorted(
        island.get("clusters", []),
        key=lambda cluster: (
            cluster.get("aggregate_score"),
            -min(program.get("program_id") for program in cluster.get("programs", []) or [{"program_id": 0}]),
        ),
        reverse=True,
    )
    for index, cluster in enumerate(clusters):
        lines.append(
            (
                f"  cluster={index} score={_format_value(cluster.get('aggregate_score'))} "
                f"signature={_format_value(cluster.get('signature'))}"
            )
        )
        for program in sorted(
            cluster.get("programs", []),
            key=lambda item: (item.get("aggregate_score"), -item.get("program_id")),
            reverse=True,
        ):
            lines.append(
                (
                    f"    program_id={_format_value(program.get('program_id'))} "
                    f"score={_format_value(program.get('aggregate_score'))}"
                )
            )
    if not clusters:
        lines.append("  No cluster data is available for this island.")
    return lines
