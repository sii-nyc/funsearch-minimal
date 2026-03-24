"""Command-line interface for the minimal FunSearch reproduction.

Simplifications relative to the paper:
- Single-process, synchronous search.
- No sandboxing, timeouts, or distributed workers.
- Island resets happen every fixed number of evaluated candidates.
- Fixed softmax temperatures for cluster and program sampling.
- OpenAI-compatible API access uses the Python standard library only.
"""

from __future__ import annotations

import argparse
import os

from funsearch.capset import DEFAULT_INPUTS, build_capset_specification
from funsearch.core import FunSearchRunner, SearchConfig
from funsearch.llm import MockLLM, OpenAICompatibleLLM
from funsearch.tracing import TraceWriter


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Minimal FunSearch reproduction for the cap set problem.")
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
        "--inputs",
        default=",".join(str(value) for value in DEFAULT_INPUTS),
        help="Comma-separated cap set dimensions to evaluate, for example 1,2,3,4.",
    )
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--trace-dir",
        help="Optional empty directory where prompts, completions, events, and database snapshots are written.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    inputs = tuple(int(part.strip()) for part in args.inputs.split(",") if part.strip())
    specification = build_capset_specification(inputs)
    config = SearchConfig(
        iterations=args.iterations,
        islands=args.islands,
        reset_interval=args.reset_interval,
        random_seed=args.seed,
    )

    if args.llm == "mock":
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
    result = FunSearchRunner(
        specification=specification,
        llm=llm,
        config=config,
        trace_writer=trace_writer,
    ).run()
    print(f"Best aggregate score: {result.best_score}")
    print(f"Best signature: {result.best_signature}")
    if result.trace_dir is not None:
        print(f"Trace directory: {result.trace_dir}")
    print("Best program:")
    print(result.best_program)


if __name__ == "__main__":
    main()
