"""字符串哈希 demo 问题。

这个例子比 cap set 更小、更直观：
- 固定 `hash_string(s)` 主流程
- 只进化 `mix_char(h, i, c)`
- evaluator 看字符串映射到桶后是否足够均匀

它的教学重点是：让读者清楚看到“只搜索一个很小的函数”也能形成完整的
FunSearch 闭环。
"""

from __future__ import annotations

import random
from collections import Counter
from statistics import mean
from textwrap import dedent
from typing import Callable, Iterable

from funsearch.core import ProblemSpecification

DEFAULT_BUCKETS = 17
DEFAULT_RANDOM_SEED = 0
DEFAULT_STRINGS_PER_CASE = 24
_TENANTS = ("acme", "globex", "initech", "umbrella", "stark", "wayne")
_REGIONS = ("us-east-1", "us-west-2", "eu-west-1", "ap-southeast-1")
_SERVICES = ("auth", "billing", "catalog", "notifications", "search", "storage")
_OPERATIONS = ("list-users", "sync-profile", "refresh-cache", "checkout", "upload-part", "rebuild-index")
_FIRST_NAMES = ("alex", "sam", "jordan", "taylor", "morgan", "casey")
_LAST_NAMES = ("chen", "patel", "garcia", "kim", "brown", "ivanov")
_METRICS = ("latency_ms", "request_count", "cache_hit_ratio", "queue_depth")
_ENVIRONMENTS = ("prod", "staging")

STRING_HASH_SEED_PROGRAM = dedent(
    '''\
    """Searches for a character-mixing function inside a tiny string hash."""


    def main(problem):
        """Scores one batch of strings by how evenly it fills the buckets."""
        strings = problem["strings"]
        num_buckets = problem["num_buckets"]
        buckets = [0] * num_buckets
        for text in strings:
            buckets[hash_string(text) % num_buckets] += 1

        expected_load = len(strings) / num_buckets
        variance = sum((load - expected_load) ** 2 for load in buckets) / num_buckets
        return -variance


    def hash_string(s):
        """Hashes a string by repeatedly applying `mix_char`."""
        h = 0
        for i, ch in enumerate(s):
            h = mix_char(h, i, ord(ch))
        return h ^ len(s)


    def mix_char(h, i, c):
        """Mixes one character into the running hash state."""
        return (h + c) & 0xFFFFFFFF
    '''
)


def _make_api_path(index: int) -> str:
    tenant = _TENANTS[index % len(_TENANTS)]
    region = _REGIONS[(index // 2) % len(_REGIONS)]
    operation = _OPERATIONS[index % len(_OPERATIONS)]
    method = ("GET", "POST", "PUT")[index % 3]
    user_id = 1000 + index * 7
    return (
        f"{method} /api/v{1 + (index % 3)}/tenants/{tenant}/users/{user_id}"
        f"/{operation}?region={region}&include=profile,roles"
    )


def _make_storage_key(index: int) -> str:
    tenant = _TENANTS[(index + 1) % len(_TENANTS)]
    region = _REGIONS[index % len(_REGIONS)]
    service = _SERVICES[(index // 2) % len(_SERVICES)]
    day = 1 + (index % 28)
    hour = (index * 3) % 24
    return (
        f"s3://logs-{region}/{tenant}/{service}/2026/03/{day:02d}/{hour:02d}/"
        f"request-{index + 1:05d}.json.gz"
    )


def _make_email_address(index: int) -> str:
    first_name = _FIRST_NAMES[index % len(_FIRST_NAMES)]
    last_name = _LAST_NAMES[(index // 2) % len(_LAST_NAMES)]
    service = _SERVICES[index % len(_SERVICES)]
    tenant = _TENANTS[(index // 3) % len(_TENANTS)]
    return f"{first_name}.{last_name}+{service}{index % 17}@{tenant}.example.com"


def _make_metric_name(index: int) -> str:
    environment = _ENVIRONMENTS[index % len(_ENVIRONMENTS)]
    region = _REGIONS[(index // 2) % len(_REGIONS)]
    service = _SERVICES[index % len(_SERVICES)]
    operation = _OPERATIONS[(index // 3) % len(_OPERATIONS)].replace("-", "_")
    metric = _METRICS[index % len(_METRICS)]
    return f"metrics.{environment}.{region}.{service}.{operation}.{metric}"


def _make_job_name(index: int) -> str:
    environment = _ENVIRONMENTS[index % len(_ENVIRONMENTS)]
    region = _REGIONS[index % len(_REGIONS)]
    service = _SERVICES[(index // 2) % len(_SERVICES)]
    worker = (index % 12) + 1
    attempt = (index % 4) + 1
    return (
        f"job/{environment}/{region}/{service}/worker-{worker:02d}/"
        f"attempt-{attempt:02d}/run-{index + 1:05d}"
    )


def _make_file_path(index: int) -> str:
    service = _SERVICES[index % len(_SERVICES)]
    tenant = _TENANTS[(index // 2) % len(_TENANTS)]
    day = 1 + ((index * 5) % 28)
    return f"/srv/{service}/{tenant}/releases/2026-03-{day:02d}/config/shard_{index % 16:02d}.yaml"


def make_realistic_strings(
    rng: random.Random,
    count: int = DEFAULT_STRINGS_PER_CASE,
) -> list[str]:
    """构造一组更接近真实系统数据的混合字符串。"""

    builders = (
        _make_api_path,
        _make_storage_key,
        _make_email_address,
        _make_metric_name,
        _make_job_name,
        _make_file_path,
    )
    offsets = list(range(len(builders)))
    rng.shuffle(offsets)

    strings_out = []
    for index in range(count):
        family_index = offsets[index % len(builders)]
        variant_index = index // len(builders)
        strings_out.append(builders[family_index](variant_index))
    return strings_out


def build_bucket_assignments(
    strings: Iterable[str],
    num_buckets: int,
    hash_string: Callable[[str], int],
) -> list[list[str]]:
    """把字符串按 hash 结果分配到桶里。"""

    bucket_strings = [[] for _ in range(num_buckets)]
    for text in strings:
        bucket_index = hash_string(text) % num_buckets
        bucket_strings[bucket_index].append(text)
    return bucket_strings


def compute_bucket_variance(bucket_strings: list[list[str]]) -> float:
    """按 evaluator 同样的公式计算 bucket load variance。"""

    num_buckets = len(bucket_strings)
    total_strings = sum(len(bucket) for bucket in bucket_strings)
    expected_load = total_strings / num_buckets
    return sum((len(bucket) - expected_load) ** 2 for bucket in bucket_strings) / num_buckets


def build_bucket_histogram(bucket_strings: list[list[str]]) -> dict[int, int]:
    """统计“load -> 有多少个桶”这样的直方图。"""

    return dict(sorted(Counter(len(bucket) for bucket in bucket_strings).items()))


def build_string_hash_inputs(
    *,
    random_seed: int = DEFAULT_RANDOM_SEED,
    num_buckets: int = DEFAULT_BUCKETS,
    strings_per_case: int = DEFAULT_STRINGS_PER_CASE,
) -> tuple[dict[str, object], ...]:
    """构造一个固定的混合真实语料集。

    当前版本只保留一个输入项，让搜索直接优化一个更复杂的哈希任务。
    """

    rng = random.Random(random_seed)
    return (
        {
            "label": "mixed_realistic_strings",
            "strings": make_realistic_strings(rng, count=strings_per_case),
            "num_buckets": num_buckets,
        },
    )


def build_string_hash_specification(
    inputs: Iterable[dict[str, object]] | None = None,
    *,
    random_seed: int = DEFAULT_RANDOM_SEED,
    num_buckets: int = DEFAULT_BUCKETS,
    strings_per_case: int = DEFAULT_STRINGS_PER_CASE,
) -> ProblemSpecification:
    """构造字符串哈希问题的 `ProblemSpecification`。"""

    # 同一次运行里，这些输入会被所有候选程序共享，用作固定评测集。
    normalized_inputs = tuple(inputs) if inputs is not None else build_string_hash_inputs(
        random_seed=random_seed,
        num_buckets=num_buckets,
        strings_per_case=strings_per_case,
    )
    return ProblemSpecification(
        seed_program=STRING_HASH_SEED_PROGRAM,
        target_function="mix_char",
        entrypoint="main",
        inputs=normalized_inputs,
        aggregate_scores=lambda scores: float(mean(scores)),
    )
