"""cap set 问题的辅助函数和问题定义。

这是仓库最接近论文原始问题的示例：
- 固定 `solve(n)` 框架
- 只进化 `priority(element, n)`
- evaluator 检查输出是不是合法 cap set，并返回其大小
"""

from __future__ import annotations

from itertools import product
from statistics import mean
from textwrap import dedent
from typing import Iterable

from funsearch.core import ProblemSpecification

Element = tuple[int, ...]

DEFAULT_INPUTS = (1, 2, 3, 4)

CAP_SET_SEED_PROGRAM = dedent(
    '''\
    """Finds large cap sets."""
    from funsearch import capset


    def main(n):
        """Runs `solve` on an n-dimensional cap set and evaluates the output."""
        solution = solve(n)
        return evaluate(solution, n)


    def evaluate(candidate_set, n):
        """Returns the size of candidate_set if it is a cap set, None otherwise."""
        if capset.is_cap_set(candidate_set, n):
            return len(candidate_set)
        return None


    def solve(n):
        """Builds a cap set of dimension `n` using `priority`."""
        elements = capset.all_vectors(n)
        elements = sorted(elements, key=lambda element: priority(element, n), reverse=True)
        chosen = []
        for element in elements:
            if capset.can_add_to_cap_set(element, chosen):
                chosen.append(element)
        return chosen


    def priority(element, n):
        """Returns the priority with which we want to add `element` to the cap set."""
        return 0.0
    '''
)


def all_vectors(n: int) -> list[Element]:
    """返回 Z_3^n 中的所有向量。"""

    return [tuple(vector) for vector in product(range(3), repeat=n)]


def third_on_line(left: Element, right: Element) -> Element:
    """给定两个点，返回与它们共线的第三个点。"""

    return tuple((-x - y) % 3 for x, y in zip(left, right))


def can_add_to_cap_set(element: Element, candidate_set: Iterable[Element]) -> bool:
    """判断把 `element` 加进去后，是否仍然保持 cap set 性质。"""

    existing = list(candidate_set)
    if element in existing:
        return False

    # 如果 `element` 与已有某个点形成的第三点也已经在集合里，就会产生一条线。
    existing_set = set(existing)
    for other in existing:
        if third_on_line(element, other) in existing_set:
            return False
    return True


def is_cap_set(candidate_set: Iterable[Element], n: int) -> bool:
    """当且仅当集合中不存在三点共线时返回 True。"""

    elements = [tuple(element) for element in candidate_set]
    if any(len(element) != n for element in elements):
        return False
    if len(set(elements)) != len(elements):
        return False

    element_set = set(elements)
    for index, left in enumerate(elements):
        for right in elements[index + 1 :]:
            if third_on_line(left, right) in element_set:
                return False
    return True


def build_capset_specification(inputs: Iterable[int] = DEFAULT_INPUTS) -> ProblemSpecification:
    """构造 cap set 问题的 `ProblemSpecification`。"""

    normalized_inputs = tuple(int(value) for value in inputs)
    return ProblemSpecification(
        seed_program=CAP_SET_SEED_PROGRAM,
        target_function="priority",
        entrypoint="main",
        inputs=normalized_inputs,
        aggregate_scores=lambda scores: float(mean(scores)),
    )
