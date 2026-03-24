"""String-hash demo problem for the minimal FunSearch reproduction."""

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

STRING_HASH_SEED_PROGRAM = dedent(
    '''\
    """Searches for a character-mixing function inside a tiny string hash."""


    def main(problem):
        """Scores one batch of strings by bucket collisions and load variance."""
        strings = problem["strings"]
        num_buckets = problem["num_buckets"]
        buckets = [0] * num_buckets
        for text in strings:
            buckets[hash_string(text) % num_buckets] += 1

        collisions = len(strings) - sum(1 for load in buckets if load)
        expected_load = len(strings) / num_buckets
        variance = sum((load - expected_load) ** 2 for load in buckets) / num_buckets
        return -(collisions + variance)


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
    """Builds lowercase strings with varied lengths."""

    alphabet = string.ascii_lowercase
    strings_out = []
    for _ in range(count):
        length = rng.randint(min_length, max_length)
        strings_out.append("".join(rng.choice(alphabet) for _ in range(length)))
    return strings_out


def make_prefixed_strings(count: int = DEFAULT_STRINGS_PER_CASE) -> list[str]:
    """Builds strings that differ mostly near the end."""

    return [f"user_{index:04d}" for index in range(1, count + 1)]


def make_suffixed_strings(count: int = DEFAULT_STRINGS_PER_CASE) -> list[str]:
    """Builds strings that share the same suffix."""

    return [f"file_{index:03d}.txt" for index in range(1, count + 1)]


def make_identifier_strings() -> list[str]:
    """Builds short identifier-like names with overlapping fragments."""

    return [
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
    ]


def build_string_hash_inputs(
    *,
    random_seed: int = DEFAULT_RANDOM_SEED,
    num_buckets: int = DEFAULT_BUCKETS,
) -> tuple[dict[str, object], ...]:
    """Builds a small deterministic dataset with several string families."""

    rng = random.Random(random_seed)
    return (
        {
            "label": "random_lowercase",
            "strings": make_random_strings(rng),
            "num_buckets": num_buckets,
        },
        {
            "label": "shared_prefix",
            "strings": make_prefixed_strings(),
            "num_buckets": num_buckets,
        },
        {
            "label": "shared_suffix",
            "strings": make_suffixed_strings(),
            "num_buckets": num_buckets,
        },
        {
            "label": "identifiers",
            "strings": make_identifier_strings(),
            "num_buckets": num_buckets,
        },
    )


def build_string_hash_specification(
    inputs: Iterable[dict[str, object]] | None = None,
) -> ProblemSpecification:
    """Builds a tiny FunSearch problem around a fixed string hash skeleton."""

    normalized_inputs = tuple(inputs) if inputs is not None else build_string_hash_inputs()
    return ProblemSpecification(
        seed_program=STRING_HASH_SEED_PROGRAM,
        target_function="mix_char",
        entrypoint="main",
        inputs=normalized_inputs,
        aggregate_scores=lambda scores: float(mean(scores)),
    )
