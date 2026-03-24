"""Program records, diversity-preserving islands, and database resets."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class ProgramRecord:
    """A correct candidate program stored in the database."""

    source: str
    function_source: str
    signature: tuple[float, ...]
    aggregate_score: float
    source_length: int
    created_at: int

    @property
    def program_id(self) -> int:
        return self.created_at

    def clone(self, created_at: int) -> "ProgramRecord":
        return ProgramRecord(
            source=self.source,
            function_source=self.function_source,
            signature=self.signature,
            aggregate_score=self.aggregate_score,
            source_length=self.source_length,
            created_at=created_at,
        )


@dataclass(frozen=True)
class ResetAction:
    """Describes how one island was reseeded during a reset."""

    island_index: int
    donor_island_index: int
    donor_program_id: int
    new_program_id: int


def _softmax_weights(values: Iterable[float], temperature: float) -> list[float]:
    normalized_temperature = max(temperature, 1e-6)
    numbers = list(values)
    if not numbers:
        return []
    maximum = max(numbers)
    exps = [math.exp((value - maximum) / normalized_temperature) for value in numbers]
    total = sum(exps)
    return [value / total for value in exps]


class Island:
    """A single subpopulation of programs grouped by score signature."""

    def __init__(self) -> None:
        self.clusters: dict[tuple[float, ...], list[ProgramRecord]] = {}

    def add(self, record: ProgramRecord) -> None:
        self.clusters.setdefault(record.signature, []).append(record)

    def all_programs(self) -> list[ProgramRecord]:
        records: list[ProgramRecord] = []
        for cluster in self.clusters.values():
            records.extend(cluster)
        return records

    def best_program(self) -> ProgramRecord:
        return max(self.all_programs(), key=lambda record: (record.aggregate_score, -record.created_at))

    def replace_with(self, record: ProgramRecord) -> None:
        self.clusters = {record.signature: [record]}

    def sample_programs(
        self,
        sample_size: int,
        rng: random.Random,
        cluster_temperature: float,
        program_temperature: float,
    ) -> list[ProgramRecord]:
        all_records = self.all_programs()
        if len(all_records) <= sample_size:
            padded = list(all_records)
            while len(padded) < sample_size:
                padded.append(self.best_program())
            return padded

        selected: list[ProgramRecord] = []
        seen_ids: set[int] = set()
        attempts = 0
        while len(selected) < sample_size and attempts < sample_size * 10:
            candidate = self._sample_one(rng, cluster_temperature, program_temperature)
            if id(candidate) not in seen_ids:
                selected.append(candidate)
                seen_ids.add(id(candidate))
            attempts += 1
        while len(selected) < sample_size:
            selected.append(self.best_program())
        return selected

    def _sample_one(
        self,
        rng: random.Random,
        cluster_temperature: float,
        program_temperature: float,
    ) -> ProgramRecord:
        clusters = list(self.clusters.items())
        cluster_scores = [records[0].aggregate_score for _, records in clusters]
        cluster_weights = _softmax_weights(cluster_scores, cluster_temperature)
        chosen_signature, chosen_cluster = rng.choices(clusters, weights=cluster_weights, k=1)[0]
        del chosen_signature

        shorter_is_better = [-record.source_length for record in chosen_cluster]
        program_weights = _softmax_weights(shorter_is_better, program_temperature)
        return rng.choices(chosen_cluster, weights=program_weights, k=1)[0]


class ProgramDatabase:
    """A simplified island-based program database."""

    def __init__(
        self,
        islands: list[Island],
        rng: random.Random,
        cluster_temperature: float,
        program_temperature: float,
        next_timestamp: int,
    ) -> None:
        self.islands = islands
        self.rng = rng
        self.cluster_temperature = cluster_temperature
        self.program_temperature = program_temperature
        self._next_timestamp = next_timestamp
        self.evaluated_candidates = 0

    @classmethod
    def from_seed(
        cls,
        seed_record: ProgramRecord,
        island_count: int,
        rng: random.Random,
        cluster_temperature: float,
        program_temperature: float,
    ) -> "ProgramDatabase":
        islands = []
        next_timestamp = seed_record.created_at + 1
        for _ in range(island_count):
            island = Island()
            island.add(seed_record.clone(next_timestamp))
            next_timestamp += 1
            islands.append(island)
        return cls(islands, rng, cluster_temperature, program_temperature, next_timestamp)

    def sample_prompt_records(self, sample_size: int) -> tuple[int, list[ProgramRecord]]:
        island_index = self.rng.randrange(len(self.islands))
        island = self.islands[island_index]
        return island_index, island.sample_programs(
            sample_size,
            self.rng,
            self.cluster_temperature,
            self.program_temperature,
        )

    def add_program(
        self,
        island_index: int,
        source: str,
        function_source: str,
        signature: tuple[float, ...],
        aggregate_score: float,
    ) -> ProgramRecord:
        record = ProgramRecord(
            source=source,
            function_source=function_source,
            signature=signature,
            aggregate_score=aggregate_score,
            source_length=len(source),
            created_at=self._claim_timestamp(),
        )
        self.islands[island_index].add(record)
        return record

    def best_program(self) -> ProgramRecord:
        return max(
            (island.best_program() for island in self.islands),
            key=lambda record: (record.aggregate_score, -record.created_at),
        )

    def record_evaluation(self, reset_interval: int) -> list[ResetAction]:
        self.evaluated_candidates += 1
        if reset_interval > 0 and self.evaluated_candidates % reset_interval == 0:
            return self._reset_worst_half()
        return []

    def _reset_worst_half(self) -> list[ResetAction]:
        if len(self.islands) < 2:
            return []

        island_ranking = sorted(
            enumerate(self.islands),
            key=lambda item: item[1].best_program().aggregate_score,
        )
        reset_count = len(self.islands) // 2
        to_reset = island_ranking[:reset_count]
        survivors = island_ranking[reset_count:]
        survivor_indices = [index for index, _ in survivors]
        if not survivor_indices:
            return []

        actions = []
        for index, _ in to_reset:
            donor_island_index = self.rng.choice(survivor_indices)
            donor_island = self.islands[donor_island_index]
            donor_program = donor_island.best_program()
            cloned_program = donor_program.clone(self._claim_timestamp())
            self.islands[index].replace_with(cloned_program)
            actions.append(
                ResetAction(
                    island_index=index,
                    donor_island_index=donor_island_index,
                    donor_program_id=donor_program.program_id,
                    new_program_id=cloned_program.program_id,
                )
            )
        return actions

    def _claim_timestamp(self) -> int:
        value = self._next_timestamp
        self._next_timestamp += 1
        return value
