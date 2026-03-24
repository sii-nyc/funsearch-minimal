"""本地 trace 可视化工具。

这个模块读取 `TraceWriter` 产出的目录，并提供两个能力：
- 把分散的 JSON / 文本文件整理成一个统一状态对象
- 启动一个本地 HTTP 服务，用网页实时展示搜索过程

设计目标是教学可视化，而不是通用监控系统：
- 尽量复用现有 trace 文件，不要求重写主流程
- 默认自动刷新，适合边跑边看
- 页面里同时保留摘要和详细文本，便于深入理解每一轮发生了什么
"""

from __future__ import annotations

import argparse
import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse


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
    """把同一轮的多条事件合成一个前端更易展示的对象。"""

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
            "source_path": _resolve_relative_path(root, accepted_event.get("source_path")),
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
            "snapshot_path": _resolve_relative_path(root, completed_event.get("snapshot_path")),
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
        return {
            "index": island.get("index"),
            "best_program_id": island.get("best_program_id"),
            "program_count": island.get("program_count"),
            "cluster_count": island.get("cluster_count"),
            "programs": programs,
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


def _build_html() -> str:
    """返回 dashboard HTML。

    页面本身不带 trace 数据，而是通过 `/api/state` 轮询。
    """

    return """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>FunSearch Trace Viewer</title>
  <style>
    :root {
      --bg: #f4f0e8;
      --paper: #fffdf8;
      --ink: #1e1b18;
      --muted: #6b6257;
      --accent: #14532d;
      --accent-soft: #dff4e6;
      --warning: #92400e;
      --warning-soft: #fff0d6;
      --danger: #7f1d1d;
      --danger-soft: #fee2e2;
      --line: #ded4c7;
      --shadow: rgba(35, 26, 15, 0.08);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: Georgia, "Noto Serif SC", "Source Han Serif SC", serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(20, 83, 45, 0.08), transparent 28%),
        linear-gradient(180deg, #f8f4ed 0%, var(--bg) 100%);
    }
    .page {
      max-width: 1400px;
      margin: 0 auto;
      padding: 24px;
    }
    .hero {
      display: grid;
      gap: 12px;
      margin-bottom: 20px;
      padding: 24px;
      background: linear-gradient(135deg, #fffdf8, #f6efe4);
      border: 1px solid var(--line);
      border-radius: 20px;
      box-shadow: 0 10px 30px var(--shadow);
    }
    .hero h1 { margin: 0; font-size: 32px; }
    .hero p { margin: 0; color: var(--muted); }
    .summary {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
      gap: 12px;
      margin-bottom: 20px;
    }
    .card {
      background: var(--paper);
      border: 1px solid var(--line);
      border-radius: 18px;
      padding: 16px;
      box-shadow: 0 10px 24px var(--shadow);
    }
    .card h2, .card h3 { margin: 0 0 10px; }
    .meta-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 8px 16px;
    }
    .metric-label { font-size: 12px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.08em; }
    .metric-value { font-size: 24px; font-weight: 700; }
    .timeline { display: grid; gap: 18px; }
    .toolbar {
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      margin-bottom: 18px;
    }
    .toolbar-group {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      align-items: center;
    }
    button {
      border: 1px solid var(--line);
      background: var(--paper);
      color: var(--ink);
      padding: 10px 14px;
      border-radius: 999px;
      font: inherit;
      cursor: pointer;
      box-shadow: 0 6px 18px var(--shadow);
    }
    button:disabled {
      cursor: not-allowed;
      opacity: 0.5;
    }
    .toggle {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      padding: 8px 12px;
      border: 1px solid var(--line);
      border-radius: 999px;
      background: var(--paper);
      box-shadow: 0 6px 18px var(--shadow);
    }
    .iteration {
      background: var(--paper);
      border: 1px solid var(--line);
      border-radius: 20px;
      box-shadow: 0 10px 24px var(--shadow);
      overflow: hidden;
    }
    .iteration-header {
      display: flex;
      flex-wrap: wrap;
      gap: 12px;
      align-items: center;
      justify-content: space-between;
      padding: 18px 20px;
      border-bottom: 1px solid var(--line);
      background: linear-gradient(180deg, rgba(20, 83, 45, 0.06), rgba(20, 83, 45, 0.01));
    }
    .status-pill {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 6px 10px;
      border-radius: 999px;
      font-size: 13px;
      font-weight: 700;
    }
    .accepted { color: var(--accent); background: var(--accent-soft); }
    .rejected { color: var(--danger); background: var(--danger-soft); }
    .in_progress, .completed { color: var(--warning); background: var(--warning-soft); }
    .iteration-body {
      display: grid;
      grid-template-columns: 1.2fr 1fr;
      gap: 16px;
      padding: 18px 20px 20px;
    }
    .section {
      display: grid;
      gap: 12px;
    }
    .mini-card {
      padding: 14px;
      border: 1px solid var(--line);
      border-radius: 14px;
      background: #fffefb;
    }
    .mini-card h4 {
      margin: 0 0 8px;
      font-size: 16px;
    }
    .program-list {
      display: grid;
      gap: 8px;
    }
    .program-row {
      padding: 10px 12px;
      background: #faf7f1;
      border-radius: 12px;
      border: 1px solid #ebe1d3;
      font-size: 14px;
    }
    details {
      border: 1px solid var(--line);
      border-radius: 14px;
      background: #fffefb;
      padding: 10px 12px;
    }
    summary {
      cursor: pointer;
      font-weight: 700;
    }
    pre {
      white-space: pre-wrap;
      word-break: break-word;
      font-size: 13px;
      line-height: 1.45;
      margin: 12px 0 0;
      padding: 14px;
      background: #1f1d1a;
      color: #f8f4ec;
      border-radius: 12px;
      overflow: auto;
    }
    .muted { color: var(--muted); }
    .kv { display: grid; gap: 6px; font-size: 14px; }
    .kv div { display: flex; gap: 10px; }
    .kv strong { min-width: 120px; }
    @media (max-width: 980px) {
      .iteration-body { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <div class="page">
    <div class="hero">
      <h1>FunSearch Trace Viewer</h1>
      <p id="subtitle">正在读取 trace...</p>
    </div>
    <div id="app"></div>
  </div>
  <script>
    function escapeHtml(value) {
      return value
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;");
    }

    function renderPrograms(programs) {
      if (!programs || programs.length === 0) {
        return '<div class="muted">无程序记录。</div>';
      }
      return '<div class="program-list">' + programs.map((program) => `
        <div class="program-row">
          <div><strong>program_id:</strong> ${program.program_id}</div>
          <div><strong>score:</strong> ${program.aggregate_score}</div>
          <div><strong>signature:</strong> ${JSON.stringify(program.signature)}</div>
          <div><strong>path:</strong> ${program.source_path}</div>
        </div>
      `).join('') + '</div>';
    }

    function renderSampledPrograms(programs) {
      if (!programs || programs.length === 0) {
        return '<div class="muted">这一轮还没有 prompt 采样信息。</div>';
      }
      return '<div class="program-list">' + programs.map((program) => `
        <div class="program-row">
          <div><strong>program_id:</strong> ${program.program_id}</div>
          <div><strong>score:</strong> ${program.aggregate_score}</div>
          <div><strong>signature:</strong> ${JSON.stringify(program.signature)}</div>
          <div><strong>path:</strong> ${program.source_path}</div>
        </div>
      `).join('') + '</div>';
    }

    function renderTextBlock(title, path, content) {
      const body = content ? `<pre>${escapeHtml(content)}</pre>` : '<div class="muted">文件尚未生成。</div>';
      const pathLine = path ? `<div class="muted">路径: ${path}</div>` : '';
      return `<details><summary>${title}</summary>${pathLine}${body}</details>`;
    }

    function renderState(state) {
      document.getElementById('subtitle').textContent =
        `${state.trace_dir} | 状态: ${state.current_status}`;

      const metadata = state.run_metadata || {};
      const searchConfig = metadata.search_config || {};
      const spec = metadata.specification || {};
      const finalBest = state.final_best || {};

      const summary = `
        <div class="summary">
          <div class="card">
            <div class="metric-label">当前状态</div>
            <div class="metric-value">${state.current_status}</div>
          </div>
          <div class="card">
            <div class="metric-label">已完成轮数</div>
            <div class="metric-value">${state.completed_iterations}${state.total_iterations != null ? ' / ' + state.total_iterations : ''}</div>
          </div>
          <div class="card">
            <div class="metric-label">Reset 次数</div>
            <div class="metric-value">${state.reset_count}</div>
          </div>
          <div class="card">
            <div class="metric-label">当前最佳分数</div>
            <div class="metric-value">${finalBest.best_score ?? 'N/A'}</div>
          </div>
        </div>
        <div class="card" style="margin-bottom: 20px;">
          <h2>Run 概览</h2>
          <div class="meta-grid">
            <div><strong>目标函数：</strong>${spec.target_function ?? 'N/A'}</div>
            <div><strong>入口函数：</strong>${spec.entrypoint ?? 'N/A'}</div>
            <div><strong>LLM 后端：</strong>${metadata.llm_backend ?? 'N/A'}</div>
            <div><strong>iterations：</strong>${searchConfig.iterations ?? 'N/A'}</div>
            <div><strong>islands：</strong>${searchConfig.islands ?? 'N/A'}</div>
            <div><strong>reset_interval：</strong>${searchConfig.reset_interval ?? 'N/A'}</div>
            <div><strong>prompt_versions：</strong>${searchConfig.prompt_versions ?? 'N/A'}</div>
            <div><strong>random_seed：</strong>${searchConfig.random_seed ?? 'N/A'}</div>
            <div><strong>最终最佳 program_id：</strong>${finalBest.best_program_id ?? 'N/A'}</div>
            <div><strong>最终最佳 signature：</strong>${finalBest.best_signature ? JSON.stringify(finalBest.best_signature) : 'N/A'}</div>
          </div>
        </div>
      `;

      const iterations = state.iterations || [];
      if (window.selectedIteration == null) {
        window.selectedIteration = iterations.length ? iterations[iterations.length - 1].iteration : null;
      }
      const availableIterations = iterations.map((item) => item.iteration);
      if (window.selectedIteration != null && !availableIterations.includes(window.selectedIteration) && iterations.length) {
        window.selectedIteration = iterations[iterations.length - 1].iteration;
      }
      if (window.followLatest && iterations.length) {
        window.selectedIteration = iterations[iterations.length - 1].iteration;
      }

      const selectedIndex = iterations.findIndex((item) => item.iteration === window.selectedIteration);
      const selectedItem = selectedIndex >= 0 ? iterations[selectedIndex] : null;
      const prevDisabled = selectedIndex <= 0;
      const nextDisabled = selectedIndex < 0 || selectedIndex >= iterations.length - 1;
      const selectedLabel = selectedItem ? `当前查看：Iteration ${selectedItem.iteration}` : '当前还没有 iteration';

      const toolbar = `
        <div class="toolbar">
          <div class="toolbar-group">
            <button id="prev-iteration" ${prevDisabled ? 'disabled' : ''}>上一轮</button>
            <button id="next-iteration" ${nextDisabled ? 'disabled' : ''}>下一轮</button>
            <button id="latest-iteration" ${!iterations.length ? 'disabled' : ''}>跳到最新</button>
            <div class="toggle">
              <input id="follow-latest" type="checkbox" ${window.followLatest ? 'checked' : ''}>
              <label for="follow-latest">自动跟随最新轮次</label>
            </div>
          </div>
          <div class="toolbar-group">
            <strong>${selectedLabel}</strong>
            <span class="muted">键盘支持：← 上一轮，→ 下一轮</span>
          </div>
        </div>
      `;

      const iterationsHtml = selectedItem ? [selectedItem].map((item) => {
        const statusClass = item.status || 'completed';
        const island = item.selected_island;
        const scoreInfo = item.score_info || {};
        const scoreHtml = item.status === 'accepted'
          ? `<div class="kv">
              <div><strong>结果：</strong><span>accepted</span></div>
              <div><strong>aggregate_score：</strong><span>${scoreInfo.aggregate_score}</span></div>
              <div><strong>signature：</strong><span>${JSON.stringify(scoreInfo.signature)}</span></div>
              <div><strong>program_id：</strong><span>${scoreInfo.program_id}</span></div>
              <div><strong>source_path：</strong><span>${scoreInfo.source_path}</span></div>
            </div>`
          : item.status === 'rejected'
            ? `<div class="kv">
                <div><strong>结果：</strong><span>rejected</span></div>
                <div><strong>原因：</strong><span>${scoreInfo.reason}</span></div>
              </div>`
            : '<div class="muted">这一轮还没有候选评估结果。</div>';

        const resetHtml = item.reset_actions && item.reset_actions.length
          ? '<div class="program-list">' + item.reset_actions.map((action) => `
              <div class="program-row">
                <div><strong>重置 island:</strong> ${action.island_index}</div>
                <div><strong>donor island:</strong> ${action.donor_island_index}</div>
                <div><strong>donor program_id:</strong> ${action.donor_program_id}</div>
                <div><strong>new program_id:</strong> ${action.new_program_id}</div>
              </div>
            `).join('') + '</div>'
          : '<div class="muted">这一轮没有 reset。</div>';

        return `
          <div class="iteration">
            <div class="iteration-header">
              <div>
                <h2>Iteration ${item.iteration}</h2>
                <div class="muted">selected island: ${item.selected_island_index ?? 'N/A'}</div>
              </div>
              <div class="status-pill ${statusClass}">${item.status}</div>
            </div>
            <div class="iteration-body">
              <div class="section">
                <div class="mini-card">
                  <h4>采样时被选中的 island</h4>
                  ${island ? `
                    <div class="kv">
                      <div><strong>island index：</strong><span>${island.index}</span></div>
                      <div><strong>best_program_id：</strong><span>${island.best_program_id}</span></div>
                      <div><strong>program_count：</strong><span>${island.program_count}</span></div>
                      <div><strong>cluster_count：</strong><span>${island.cluster_count}</span></div>
                    </div>
                    ${renderPrograms(island.programs)}
                  ` : '<div class="muted">缺少这一轮开始前的 island 快照。</div>'}
                </div>
                <div class="mini-card">
                  <h4>用于构造 prompt 的历史程序</h4>
                  ${renderSampledPrograms(item.sampled_programs)}
                </div>
                <div class="mini-card">
                  <h4>候选评估结果</h4>
                  ${scoreHtml}
                </div>
                <div class="mini-card">
                  <h4>本轮 reset</h4>
                  ${resetHtml}
                </div>
              </div>
              <div class="section">
                ${renderTextBlock('Prompt', item.prompt_path, item.prompt_text)}
                ${renderTextBlock('Completion', item.completion_path, item.completion_text)}
                ${renderTextBlock('Candidate Program', item.candidate_program_path, item.candidate_program_text)}
              </div>
            </div>
          </div>
        `;
      }).join('') : '<div class="card">还没有 iteration 事件。</div>';

      document.getElementById('app').innerHTML = summary + toolbar + `<div class="timeline">${iterationsHtml}</div>`;

      const prevButton = document.getElementById('prev-iteration');
      const nextButton = document.getElementById('next-iteration');
      const latestButton = document.getElementById('latest-iteration');
      const followLatestToggle = document.getElementById('follow-latest');

      if (prevButton) {
        prevButton.onclick = () => {
          if (selectedIndex > 0) {
            window.followLatest = false;
            window.selectedIteration = iterations[selectedIndex - 1].iteration;
            renderState(state);
          }
        };
      }
      if (nextButton) {
        nextButton.onclick = () => {
          if (selectedIndex >= 0 && selectedIndex < iterations.length - 1) {
            window.followLatest = false;
            window.selectedIteration = iterations[selectedIndex + 1].iteration;
            renderState(state);
          }
        };
      }
      if (latestButton) {
        latestButton.onclick = () => {
          if (iterations.length) {
            window.selectedIteration = iterations[iterations.length - 1].iteration;
            renderState(state);
          }
        };
      }
      if (followLatestToggle) {
        followLatestToggle.onchange = (event) => {
          window.followLatest = event.target.checked;
          if (window.followLatest && iterations.length) {
            window.selectedIteration = iterations[iterations.length - 1].iteration;
          }
          renderState(state);
        };
      }
    }

    async function refresh() {
      const response = await fetch('/api/state');
      const state = await response.json();
      renderState(state);
    }

    window.followLatest = true;
    window.selectedIteration = null;
    window.addEventListener('keydown', (event) => {
      const iterations = window.lastIterations || [];
      if (!iterations.length) {
        return;
      }
      const currentIndex = iterations.findIndex((item) => item.iteration === window.selectedIteration);
      if (event.key === 'ArrowLeft' && currentIndex > 0) {
        window.followLatest = false;
        window.selectedIteration = iterations[currentIndex - 1].iteration;
        renderState(window.lastState);
      }
      if (event.key === 'ArrowRight' && currentIndex >= 0 && currentIndex < iterations.length - 1) {
        window.followLatest = false;
        window.selectedIteration = iterations[currentIndex + 1].iteration;
        renderState(window.lastState);
      }
    });

    const originalRenderState = renderState;
    renderState = function(state) {
      window.lastState = state;
      window.lastIterations = state.iterations || [];
      originalRenderState(state);
    };

    refresh();
    setInterval(refresh, 2000);
  </script>
</body>
</html>
"""


class _TraceViewerHandler(BaseHTTPRequestHandler):
    """HTTP handler：提供 dashboard 页面和 trace 状态 API。"""

    trace_dir: Path

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self._respond_html(_build_html())
            return
        if parsed.path == "/api/state":
            self._respond_json(load_trace_state(self.trace_dir))
            return
        if parsed.path == "/api/file":
            query = parse_qs(parsed.query)
            requested = query.get("path", [""])[0]
            self._serve_text_file(requested)
            return
        self.send_error(HTTPStatus.NOT_FOUND)

    def log_message(self, format: str, *args: Any) -> None:
        del format, args
        return

    def _respond_html(self, content: str) -> None:
        payload = content.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _respond_json(self, payload_obj: dict[str, Any]) -> None:
        payload = json.dumps(payload_obj, ensure_ascii=False).encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _serve_text_file(self, requested: str) -> None:
        file_path = (self.trace_dir / requested).resolve()
        if not str(file_path).startswith(str(self.trace_dir.resolve())) or not file_path.exists():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        payload = file_path.read_text(encoding="utf-8").encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


def serve_trace(trace_dir: str | Path, host: str = "127.0.0.1", port: int = 8765) -> None:
    """启动本地 dashboard 服务。"""

    trace_root = Path(trace_dir).resolve()
    if not trace_root.exists():
        raise SystemExit(f"Trace directory does not exist: {trace_root}")

    handler = type(
        "TraceViewerHandler",
        (_TraceViewerHandler,),
        {"trace_dir": trace_root},
    )
    server = ThreadingHTTPServer((host, port), handler)
    print(f"Serving trace viewer for {trace_root}")
    print(f"Open http://{host}:{port} in your browser")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping trace viewer...")
    finally:
        server.server_close()


def build_parser() -> argparse.ArgumentParser:
    """定义 trace viewer CLI。"""

    parser = argparse.ArgumentParser(description="Serve a local dashboard for a FunSearch trace directory.")
    parser.add_argument("--trace-dir", required=True, help="Trace directory produced by --trace-dir during a run.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--dump-state",
        action="store_true",
        help="Print the merged trace state as JSON and exit instead of starting a server.",
    )
    return parser


def main() -> None:
    """trace viewer CLI 入口。"""

    args = build_parser().parse_args()
    if args.dump_state:
        print(json.dumps(load_trace_state(args.trace_dir), indent=2, ensure_ascii=False, sort_keys=True))
        return
    serve_trace(args.trace_dir, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
