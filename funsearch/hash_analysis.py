"""分析一个 string-hash 程序在固定输入上的桶分布。"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from funsearch.core import ProblemSpecification, evaluate_program_detailed
from funsearch.string_hash import (
    DEFAULT_BUCKETS,
    DEFAULT_RANDOM_SEED,
    DEFAULT_STRINGS_PER_CASE,
    build_bucket_assignments,
    build_bucket_histogram,
    build_string_hash_specification,
    compute_bucket_variance,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect bucket usage for one string-hash program.")
    parser.add_argument(
        "--program",
        required=True,
        help="Path to a full Python program containing main(problem) and hash_string(s).",
    )
    parser.add_argument(
        "--string-hash-buckets",
        type=int,
        default=DEFAULT_BUCKETS,
        help="Bucket count for the fixed string-hash corpus.",
    )
    parser.add_argument(
        "--string-hash-strings-per-case",
        type=int,
        default=DEFAULT_STRINGS_PER_CASE,
        help="Number of strings in the fixed string-hash corpus.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_RANDOM_SEED,
        help="Random seed used to build the fixed mixed corpus.",
    )
    return parser


def analyze_program_source(
    program_source: str,
    specification: ProblemSpecification,
) -> dict[str, Any]:
    """执行并分析一个完整程序在当前 string-hash 输入上的表现。"""

    result, error = evaluate_program_detailed(program_source, specification)
    if result is None:
        raise ValueError(f"Program rejected by evaluator: {error}")

    namespace: dict[str, Any] = {}
    exec(program_source, namespace)
    hash_string = namespace.get("hash_string")
    if not callable(hash_string):
        raise ValueError("Program does not define a callable hash_string(s).")

    if len(specification.inputs) != 1:
        raise ValueError(
            "This analysis tool expects the current string-hash problem to have exactly one input."
        )

    problem = specification.inputs[0]
    strings = list(problem["strings"])
    num_buckets = int(problem["num_buckets"])
    bucket_strings = build_bucket_assignments(strings, num_buckets, hash_string)
    loads = [len(bucket) for bucket in bucket_strings]
    used_buckets = sum(1 for load in loads if load > 0)

    return {
        "label": problem.get("label"),
        "aggregate_score": result.aggregate_score,
        "signature": result.signature,
        "num_strings": len(strings),
        "num_buckets": num_buckets,
        "used_buckets": used_buckets,
        "empty_buckets": num_buckets - used_buckets,
        "collision_count": len(strings) - used_buckets,
        "buckets_with_collisions": sum(1 for load in loads if load > 1),
        "max_bucket_load": max(loads, default=0),
        "variance": compute_bucket_variance(bucket_strings),
        "bucket_histogram": build_bucket_histogram(bucket_strings),
        "bucket_loads": loads,
        "bucket_strings": bucket_strings,
    }


def build_bucket_report(analysis: dict[str, Any]) -> str:
    """把桶分析结果格式化成一份可直接阅读的文本报告。"""

    lines = [
        "String Hash Analysis",
        f"Label: {analysis['label']}",
        f"Aggregate score: {analysis['aggregate_score']}",
        f"Signature: {analysis['signature']}",
        f"Strings: {analysis['num_strings']} | Buckets: {analysis['num_buckets']}",
        (
            "Usage: "
            f"used={analysis['used_buckets']} empty={analysis['empty_buckets']} "
            f"collisions={analysis['collision_count']} collision_buckets={analysis['buckets_with_collisions']} "
            f"max_load={analysis['max_bucket_load']}"
        ),
        f"Variance: {analysis['variance']}",
        f"Load histogram: {analysis['bucket_histogram']}",
        "",
        "Bucket loads:",
    ]

    bucket_loads = analysis["bucket_loads"]
    for start in range(0, len(bucket_loads), 8):
        chunk = bucket_loads[start : start + 8]
        lines.append("  " + " ".join(f"{start + offset:03d}:{load}" for offset, load in enumerate(chunk)))

    lines.extend(["", "Bucket contents:"])
    for bucket_index, bucket in enumerate(analysis["bucket_strings"]):
        lines.append(f"- bucket {bucket_index:03d} load={len(bucket)}")
        for text in bucket:
            lines.append(f"  {text}")
    return "\n".join(lines)


def main() -> None:
    args = build_parser().parse_args()
    if args.string_hash_buckets <= 0:
        raise SystemExit("--string-hash-buckets must be a positive integer")
    if args.string_hash_strings_per_case <= 0:
        raise SystemExit("--string-hash-strings-per-case must be a positive integer")

    program_path = Path(args.program)
    program_source = program_path.read_text(encoding="utf-8")
    specification = build_string_hash_specification(
        random_seed=args.seed,
        num_buckets=args.string_hash_buckets,
        strings_per_case=args.string_hash_strings_per_case,
    )
    analysis = analyze_program_source(program_source, specification)
    print(build_bucket_report(analysis))


if __name__ == "__main__":
    main()
