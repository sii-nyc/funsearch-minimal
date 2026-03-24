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
import string
from statistics import mean
from textwrap import dedent
from typing import Iterable

from funsearch.core import ProblemSpecification

DEFAULT_BUCKETS = 17
DEFAULT_RANDOM_SEED = 0
DEFAULT_STRINGS_PER_CASE = 24
_IDENTIFIER_BASE_STRINGS = (
    "get_user",
    "get_users",
    "get_user_by_id",
    "get_user_name",
    "set_user",
    "set_user_name",
    "set_user_email",
    "load_user",
    "load_user_profile",
    "save_user",
    "save_user_profile",
    "delete_user",
    "delete_user_cache",
    "user_to_json",
    "user_from_json",
    "parse_user_id",
    "format_user_name",
    "update_user",
    "update_user_name",
    "update_user_email",
    "list_users",
    "list_user_groups",
    "find_user",
    "find_user_by_name",
)

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
        return (h * 33 + c + i) & 0xFFFFFFFF
    '''
)


def make_random_strings(
    rng: random.Random,
    count: int = DEFAULT_STRINGS_PER_CASE,
    min_length: int = 4,
    max_length: int = 10,
) -> list[str]:
    """生成随机小写字符串。"""

    alphabet = string.ascii_lowercase
    strings_out = []
    for _ in range(count):
        length = rng.randint(min_length, max_length)
        strings_out.append("".join(rng.choice(alphabet) for _ in range(length)))
    return strings_out


def make_prefixed_strings(count: int = DEFAULT_STRINGS_PER_CASE) -> list[str]:
    """生成共享前缀的字符串，测试后缀差异是否能被哈希识别。"""

    return [f"user_{index:04d}" for index in range(1, count + 1)]


def make_suffixed_strings(count: int = DEFAULT_STRINGS_PER_CASE) -> list[str]:
    """生成共享后缀的字符串，测试前缀差异是否能被哈希识别。"""

    return [f"file_{index:03d}.txt" for index in range(1, count + 1)]


def make_identifier_strings(count: int = DEFAULT_STRINGS_PER_CASE) -> list[str]:
    """生成一批相互相似的标识符风格字符串。"""

    if count <= len(_IDENTIFIER_BASE_STRINGS):
        return list(_IDENTIFIER_BASE_STRINGS[:count])

    identifiers = list(_IDENTIFIER_BASE_STRINGS)
    for index in range(count - len(_IDENTIFIER_BASE_STRINGS)):
        identifiers.append(f"user_helper_{index + 1:03d}")
    return identifiers


def build_string_hash_inputs(
    *,
    random_seed: int = DEFAULT_RANDOM_SEED,
    num_buckets: int = DEFAULT_BUCKETS,
    strings_per_case: int = DEFAULT_STRINGS_PER_CASE,
) -> tuple[dict[str, object], ...]:
    """构造一个小而固定的测试集。

    这里每个输入项都是一个小字典，里面放：
    - `strings`：这组待哈希的字符串
    - `num_buckets`：桶数量
    - `label`：给人看的标签，方便理解这组数据在测什么
    """

    rng = random.Random(random_seed)
    return (
        {
            "label": "random_lowercase",
            "strings": make_random_strings(rng, count=strings_per_case),
            "num_buckets": num_buckets,
        },
        {
            "label": "shared_prefix",
            "strings": make_prefixed_strings(strings_per_case),
            "num_buckets": num_buckets,
        },
        {
            "label": "shared_suffix",
            "strings": make_suffixed_strings(strings_per_case),
            "num_buckets": num_buckets,
        },
        {
            "label": "identifiers",
            "strings": make_identifier_strings(strings_per_case),
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
