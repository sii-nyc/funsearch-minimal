"""命令行入口。

相对论文版，这里刻意保留的是最小可运行骨架：
- 单进程、同步执行
- 没有沙箱和分布式 worker
- 岛屿重置按“评估次数”触发，而不是按墙钟时间
- cluster / program 采样温度写死在配置里
- OpenAI 兼容接口通过官方 Python SDK 调用
"""

from __future__ import annotations

import argparse
import os

from funsearch.capset import DEFAULT_INPUTS, build_capset_specification
from funsearch.console_reporter import ConsoleRunReporter
from funsearch.core import FunSearchRunner, SearchConfig
from funsearch.llm import MockLLM, OpenAICompatibleLLM
from funsearch.string_hash import (
    DEFAULT_BUCKETS,
    DEFAULT_STRINGS_PER_CASE,
    build_string_hash_specification,
)
from funsearch.tracing import TraceWriter


def build_parser() -> argparse.ArgumentParser:
    """定义 CLI 参数。

    这个项目不做复杂命令层设计，所有可调项都直接平铺成参数。
    """

    parser = argparse.ArgumentParser(description="Minimal FunSearch reproduction with small built-in search problems.")
    parser.add_argument(
        "--problem",
        choices=("capset", "string-hash"),
        default="capset",
        help="Which built-in problem to run.",
    )
    parser.add_argument(
        "--llm",
        choices=("openai-compatible", "mock"),
        default="openai-compatible",
        help="Which LLM backend to use.",
    )
    parser.add_argument("--base-url", default=os.environ.get("FUNSEARCH_BASE_URL"))
    parser.add_argument("--api-key", default=os.environ.get("FUNSEARCH_API_KEY", ""))
    parser.add_argument("--model", default=os.environ.get("FUNSEARCH_MODEL"))
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--islands", type=int, default=4)
    parser.add_argument("--reset-interval", type=int, default=10)
    parser.add_argument(
        "--prompt-versions",
        type=int,
        default=2,
        help="Number of historical versions sampled into each best-shot prompt.",
    )
    parser.add_argument(
        "--inputs",
        default=",".join(str(value) for value in DEFAULT_INPUTS),
        help="Comma-separated cap set dimensions to evaluate when --problem=capset.",
    )
    parser.add_argument(
        "--string-hash-buckets",
        type=int,
        default=DEFAULT_BUCKETS,
        help="Bucket count for --problem=string-hash.",
    )
    parser.add_argument(
        "--string-hash-strings-per-case",
        type=int,
        default=DEFAULT_STRINGS_PER_CASE,
        help="Number of strings in each string-hash evaluation case.",
    )
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--trace-dir",
        help="Optional empty directory where prompts, completions, events, and database snapshots are written.",
    )
    parser.add_argument(
        "--no-live-report",
        action="store_true",
        help="Do not stream per-iteration records to stdout during the run.",
    )
    return parser


def main() -> None:
    """解析参数、构建问题、选择 LLM，然后运行搜索。"""

    args = build_parser().parse_args()
    if args.problem == "capset":
        # cap set 问题的输入是一组维度，例如 1,2,3,4。
        inputs = tuple(int(part.strip()) for part in args.inputs.split(",") if part.strip())
        specification = build_capset_specification(inputs)
    else:
        if args.string_hash_buckets <= 0:
            raise SystemExit("--string-hash-buckets must be a positive integer")
        if args.string_hash_strings_per_case <= 0:
            raise SystemExit("--string-hash-strings-per-case must be a positive integer")
        specification = build_string_hash_specification(
            num_buckets=args.string_hash_buckets,
            strings_per_case=args.string_hash_strings_per_case,
        )
    if args.prompt_versions <= 0:
        raise SystemExit("--prompt-versions must be a positive integer")
    config = SearchConfig(
        iterations=args.iterations,
        islands=args.islands,
        reset_interval=args.reset_interval,
        prompt_versions=args.prompt_versions,
        random_seed=args.seed,
    )

    if args.llm == "mock":
        # mock LLM 让整个项目可以完全离线演示。
        llm = MockLLM()
    else:
        if not args.base_url:
            raise SystemExit("--base-url is required when --llm=openai-compatible")
        if not args.model:
            raise SystemExit("--model is required when --llm=openai-compatible")
        llm = OpenAICompatibleLLM(
            base_url=args.base_url,
            api_key=args.api_key,
            model=args.model,
            temperature=args.temperature,
        )

    trace_writer = TraceWriter(args.trace_dir) if args.trace_dir else None
    progress_reporter = None if args.no_live_report else ConsoleRunReporter()
    result = FunSearchRunner(
        specification=specification,
        llm=llm,
        config=config,
        trace_writer=trace_writer,
        progress_reporter=progress_reporter,
    ).run()
    print(f"Best aggregate score: {result.best_score}")
    print(f"Best signature: {result.best_signature}")
    if result.trace_dir is not None:
        print(f"Trace directory: {result.trace_dir}")
    print("Best program:")
    print(result.best_program)


if __name__ == "__main__":
    main()
