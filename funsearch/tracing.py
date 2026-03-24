"""把搜索过程写成一组本地可读的 trace 文件。

trace 的目标不是追求高性能，而是便于教学和调试：
- 看每轮 prompt 长什么样
- 看模型回了什么
- 看哪些候选被接受或拒绝
- 看数据库在每轮之后变成了什么样
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from funsearch.core import ProblemSpecification, SearchConfig
from funsearch.database import ProgramDatabase, ProgramRecord


class TraceWriter:
    """把运行过程中的事件、文本产物和数据库快照落盘。"""

    def __init__(self, root_dir: str | Path) -> None:
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)
        if any(self.root_dir.iterdir()):
            raise ValueError(f"Trace directory must be empty: {self.root_dir}")

        self.events_path = self.root_dir / "events.jsonl"
        self.programs_dir = self.root_dir / "programs"
        self.prompts_dir = self.root_dir / "prompts"
        self.completions_dir = self.root_dir / "completions"
        self.candidates_dir = self.root_dir / "candidates"
        self.snapshots_dir = self.root_dir / "snapshots"
        for directory in (
            self.programs_dir,
            self.prompts_dir,
            self.completions_dir,
            self.candidates_dir,
            self.snapshots_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)

    def write_run_metadata(
        self,
        specification: ProblemSpecification,
        config: SearchConfig,
        llm_backend: str,
    ) -> None:
        """写出本次 run 的静态元信息。"""

        payload = {
            "llm_backend": llm_backend,
            "search_config": {
                "iterations": config.iterations,
                "islands": config.islands,
                "reset_interval": config.reset_interval,
                "prompt_versions": config.prompt_versions,
                "cluster_temperature": config.cluster_temperature,
                "program_temperature": config.program_temperature,
                "random_seed": config.random_seed,
            },
            "specification": {
                "target_function": specification.target_function,
                "entrypoint": specification.entrypoint,
                "inputs": list(specification.inputs),
            },
        }
        self._write_json(Path("run.json"), payload)

    def log_event(self, event_type: str, **payload: Any) -> None:
        """追加一条 JSONL 事件。"""

        event = {"type": event_type, **payload}
        with self.events_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=True, sort_keys=True) + "\n")

    def write_program(self, record: ProgramRecord) -> str:
        """按 program id 落盘完整源码；已存在则不重复写。"""

        relative_path = Path("programs") / f"program_{record.program_id:06d}.py"
        self._write_text_if_missing(relative_path, record.source)
        return relative_path.as_posix()

    def write_prompt(self, iteration: int, prompt: str) -> str:
        """写出某一轮使用的 prompt。"""

        return self._write_text(Path("prompts") / f"iteration_{iteration:04d}.txt", prompt)

    def write_completion(self, iteration: int, completion: str) -> str:
        """写出某一轮模型返回的原始 completion。"""

        return self._write_text(Path("completions") / f"iteration_{iteration:04d}.txt", completion)

    def write_candidate_program(self, iteration: int, program_source: str) -> str:
        """写出从 completion 中重建出的完整候选程序。"""

        return self._write_text(Path("candidates") / f"iteration_{iteration:04d}.py", program_source)

    def write_snapshot(self, name: str, database: ProgramDatabase) -> str:
        """把当前数据库状态序列化成一个完整快照。"""

        payload = {
            "evaluated_candidates": database.evaluated_candidates,
            "best_program_id": database.best_program().program_id,
            "islands": [],
        }
        for index, island in enumerate(database.islands):
            programs = island.all_programs()
            cluster_payloads = []
            for signature, cluster in sorted(island.clusters.items()):
                cluster_payloads.append(
                    {
                        "signature": list(signature),
                        "aggregate_score": cluster[0].aggregate_score,
                        "programs": [self._program_summary(record) for record in cluster],
                    }
                )
            payload["islands"].append(
                {
                    "index": index,
                    "best_program_id": island.best_program().program_id,
                    "program_count": len(programs),
                    "cluster_count": len(island.clusters),
                    "clusters": cluster_payloads,
                }
            )

        return self._write_json(Path("snapshots") / f"{name}.json", payload)

    def _program_summary(self, record: ProgramRecord) -> dict[str, Any]:
        """生成程序摘要，供快照和事件日志复用。"""

        return {
            "program_id": record.program_id,
            "aggregate_score": record.aggregate_score,
            "signature": list(record.signature),
            "source_length": record.source_length,
            "created_at": record.created_at,
            "source_path": self.write_program(record),
        }

    def _write_text(self, relative_path: Path, content: str) -> str:
        """写出文本文件，并返回相对路径。"""

        full_path = self.root_dir / relative_path
        full_path.write_text(content, encoding="utf-8")
        return relative_path.as_posix()

    def _write_text_if_missing(self, relative_path: Path, content: str) -> None:
        """仅在文件不存在时写入，避免重复覆盖同一程序文件。"""

        full_path = self.root_dir / relative_path
        if not full_path.exists():
            full_path.write_text(content, encoding="utf-8")

    def _write_json(self, relative_path: Path, payload: dict[str, Any]) -> str:
        """写出 JSON 文件，并返回相对路径。"""

        full_path = self.root_dir / relative_path
        full_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True, sort_keys=True) + "\n", encoding="utf-8")
        return relative_path.as_posix()
