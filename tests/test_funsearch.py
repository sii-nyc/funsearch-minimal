from __future__ import annotations

import json
import random
import tempfile
import unittest

from funsearch.capset import build_capset_specification, can_add_to_cap_set, is_cap_set
from funsearch.core import FunSearchRunner, SearchConfig, evaluate_program
from funsearch.database import Island, ProgramDatabase, ProgramRecord
from funsearch.llm import MockLLM
from funsearch.tracing import TraceWriter
from funsearch.prompting import build_prompt, extract_function_source, extract_generated_function, replace_function


class CapSetTests(unittest.TestCase):
    def test_capset_validity_helpers(self) -> None:
        self.assertTrue(is_cap_set([(0,), (1, 1)], 1) is False)
        self.assertTrue(is_cap_set([(0, 0), (1, 1)], 2))
        self.assertTrue(can_add_to_cap_set((2, 2), [(0, 0), (1, 1)]) is False)
        self.assertTrue(can_add_to_cap_set((0, 1), [(0, 0), (1, 1)]))

    def test_seed_program_is_valid(self) -> None:
        specification = build_capset_specification((1, 2))
        result = evaluate_program(specification.seed_program, specification)
        self.assertIsNotNone(result)
        self.assertEqual(result.signature, (2.0, 4.0))


class PromptingTests(unittest.TestCase):
    def test_prompt_contains_versions_and_empty_target(self) -> None:
        specification = build_capset_specification((1, 2))
        seed_function = extract_function_source(specification.seed_program, specification.target_function)
        variant_function = "def priority(element, n):\n    return float(sum(element))\n"
        variant_program = replace_function(specification.seed_program, specification.target_function, variant_function)
        records = [
            ProgramRecord(
                source=specification.seed_program,
                function_source=seed_function,
                signature=(2.0, 4.0),
                aggregate_score=3.0,
                source_length=len(specification.seed_program),
                created_at=0,
            ),
            ProgramRecord(
                source=variant_program,
                function_source=variant_function,
                signature=(2.0, 4.0),
                aggregate_score=3.5,
                source_length=len(variant_program),
                created_at=1,
            ),
        ]

        prompt = build_prompt(specification.seed_program, specification.target_function, records)

        self.assertIn("def priority_v0(", prompt)
        self.assertIn("def priority_v1(", prompt)
        self.assertIn("def priority_v2(", prompt)
        self.assertNotIn("def solve(", prompt)

    def test_extract_generated_function_from_fenced_code(self) -> None:
        completion = """Here is a candidate.

```python
def priority_v2(element, n):
    \"\"\"Returns the priority with which we want to add `element` to the cap set.\"\"\"
    return float(sum(element))
```
"""
        extracted = extract_generated_function(completion, "priority_v2", "priority")
        self.assertIsNotNone(extracted)
        self.assertIn("def priority(", extracted)
        self.assertNotIn("priority_v2", extracted)


class EvaluatorAndDatabaseTests(unittest.TestCase):
    def test_invalid_program_is_rejected(self) -> None:
        specification = build_capset_specification((1,))
        invalid_function = "def priority(element, n):\n    return complex(1, 1)\n"
        invalid_program = replace_function(specification.seed_program, specification.target_function, invalid_function)
        self.assertIsNone(evaluate_program(invalid_program, specification))

    def test_clusters_preserve_same_signature_together(self) -> None:
        island = Island()
        record_a = ProgramRecord("a", "def f():\n    return 1\n", (1.0, 2.0), 1.5, 10, 0)
        record_b = ProgramRecord("b", "def f():\n    return 2\n", (1.0, 2.0), 1.5, 12, 1)
        island.add(record_a)
        island.add(record_b)

        self.assertEqual(len(island.clusters), 1)
        self.assertEqual(len(island.clusters[(1.0, 2.0)]), 2)

    def test_reset_reseeds_worst_islands_from_survivors(self) -> None:
        seed = ProgramRecord("seed", "def priority():\n    return 0\n", (1.0,), 1.0, 10, 0)
        database = ProgramDatabase.from_seed(
            seed_record=seed,
            island_count=4,
            rng=random.Random(0),
            cluster_temperature=1.0,
            program_temperature=0.2,
        )
        database.add_program(2, "high-a", "def priority():\n    return 10\n", (10.0,), 10.0)
        database.add_program(3, "high-b", "def priority():\n    return 11\n", (11.0,), 11.0)

        database.record_evaluation(reset_interval=1)

        best_scores = [island.best_program().aggregate_score for island in database.islands]
        self.assertTrue(all(score >= 10.0 for score in best_scores))


class IntegrationTests(unittest.TestCase):
    def test_mock_search_returns_reproducible_best_program(self) -> None:
        specification = build_capset_specification((1, 2))
        seed_result = evaluate_program(specification.seed_program, specification)
        self.assertIsNotNone(seed_result)

        runner = FunSearchRunner(
            specification=specification,
            llm=MockLLM(),
            config=SearchConfig(iterations=4, islands=2, reset_interval=2, random_seed=0),
        )
        result = runner.run()
        rerun_result = evaluate_program(result.best_program, specification)

        self.assertIsNotNone(rerun_result)
        self.assertGreaterEqual(result.best_score, seed_result.aggregate_score)
        self.assertEqual(result.best_signature, rerun_result.signature)
        self.assertEqual(result.best_score, rerun_result.aggregate_score)

    def test_trace_writer_persists_events_and_snapshots(self) -> None:
        specification = build_capset_specification((1, 2))
        with tempfile.TemporaryDirectory() as temp_dir:
            trace_dir = f"{temp_dir}/trace"
            runner = FunSearchRunner(
                specification=specification,
                llm=MockLLM(),
                config=SearchConfig(iterations=2, islands=2, reset_interval=2, random_seed=0),
                trace_writer=TraceWriter(trace_dir),
            )
            result = runner.run()

            self.assertEqual(result.trace_dir, trace_dir)

            with open(f"{trace_dir}/run.json", encoding="utf-8") as handle:
                run_metadata = json.load(handle)
            self.assertEqual(run_metadata["specification"]["inputs"], [1, 2])

            with open(f"{trace_dir}/events.jsonl", encoding="utf-8") as handle:
                events = [json.loads(line) for line in handle]
            event_types = [event["type"] for event in events]
            self.assertIn("prompt_sampled", event_types)
            self.assertIn("completion_received", event_types)
            self.assertIn("iteration_completed", event_types)
            self.assertIn("run_completed", event_types)

            with open(f"{trace_dir}/snapshots/final_database.json", encoding="utf-8") as handle:
                final_snapshot = json.load(handle)
            self.assertIn("islands", final_snapshot)
            self.assertTrue(final_snapshot["islands"])


if __name__ == "__main__":
    unittest.main()
