from __future__ import annotations

import json
import math
import random
import tempfile
import unittest
from pathlib import Path
from textwrap import dedent

from funsearch.capset import build_capset_specification, can_add_to_cap_set, is_cap_set
from funsearch.cli import build_parser
from funsearch.core import FunSearchRunner, SearchConfig, evaluate_program
from funsearch.database import Island, ProgramDatabase, ProgramRecord
from funsearch.llm import MockLLM
from funsearch.tracing import TraceWriter
from funsearch.prompting import build_prompt, extract_function_source, extract_generated_function, replace_function
from funsearch.string_hash import (
    DEFAULT_BUCKETS,
    DEFAULT_STRINGS_PER_CASE,
    build_string_hash_inputs,
    build_string_hash_specification,
    make_identifier_strings,
    make_prefixed_strings,
    make_random_strings,
    make_suffixed_strings,
)
from funsearch.trace_viewer import load_trace_state


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


class StringHashTests(unittest.TestCase):
    def test_dataset_builders_are_deterministic_and_representative(self) -> None:
        first = build_string_hash_inputs(random_seed=7)
        second = build_string_hash_inputs(random_seed=7)

        self.assertEqual(first, second)
        self.assertEqual([case["label"] for case in first], ["random_lowercase", "shared_prefix", "shared_suffix", "identifiers"])
        self.assertEqual(make_prefixed_strings()[:2], ["user_0001", "user_0002"])
        self.assertEqual(make_suffixed_strings()[:2], ["file_001.txt", "file_002.txt"])
        self.assertIn("get_user_by_id", make_identifier_strings())
        self.assertEqual(len(make_random_strings(random.Random(3))), 24)

    def test_string_hash_inputs_respect_bucket_and_case_sizes(self) -> None:
        inputs = build_string_hash_inputs(num_buckets=23, strings_per_case=7)

        self.assertEqual(len(inputs), 4)
        self.assertTrue(all(case["num_buckets"] == 23 for case in inputs))
        self.assertTrue(all(len(case["strings"]) == 7 for case in inputs))

    def test_string_hash_specification_uses_custom_dataset_size(self) -> None:
        specification = build_string_hash_specification(num_buckets=19, strings_per_case=5)

        self.assertTrue(all(case["num_buckets"] == 19 for case in specification.inputs))
        self.assertTrue(all(len(case["strings"]) == 5 for case in specification.inputs))

    def test_seed_program_is_valid(self) -> None:
        specification = build_string_hash_specification()
        result = evaluate_program(specification.seed_program, specification)

        self.assertIsNotNone(result)
        self.assertEqual(len(result.signature), 4)
        self.assertTrue(all(math.isfinite(score) for score in result.signature))
        self.assertTrue(math.isfinite(result.aggregate_score))

    def test_better_mixer_beats_a_weak_one(self) -> None:
        specification = build_string_hash_specification()
        weak_function = "def mix_char(h, i, c):\n    return h\n"
        strong_function = dedent(
            """\
            def mix_char(h, i, c):
                h ^= c + i * 17
                h = (h * 131) & 0xFFFFFFFF
                return h
            """
        )

        weak_program = replace_function(specification.seed_program, specification.target_function, weak_function)
        strong_program = replace_function(specification.seed_program, specification.target_function, strong_function)
        weak_result = evaluate_program(weak_program, specification)
        strong_result = evaluate_program(strong_program, specification)

        self.assertIsNotNone(weak_result)
        self.assertIsNotNone(strong_result)
        assert weak_result is not None
        assert strong_result is not None
        self.assertGreater(strong_result.aggregate_score, weak_result.aggregate_score)


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
        self.assertIn("def solve(", prompt)
        self.assertNotIn("def priority(element, n):", prompt)
        self.assertIn("Return Python code only.", prompt)
        self.assertIn("Output exactly one function definition", prompt)
        self.assertIn("aggregate_score=3.5", prompt)
        self.assertIn("Goal: maximize the aggregate score", prompt)
        self.assertIn("Best aggregate_score among the shown versions: 3.5", prompt)
        self.assertIn("`main`:", prompt)

    def test_string_hash_prompt_includes_fixed_helper_source(self) -> None:
        specification = build_string_hash_specification()
        seed_function = extract_function_source(specification.seed_program, specification.target_function)
        records = [
            ProgramRecord(
                source=specification.seed_program,
                function_source=seed_function,
                signature=(-1.0, -2.0, -3.0, -4.0),
                aggregate_score=-2.5,
                source_length=len(specification.seed_program),
                created_at=0,
            )
        ]

        prompt = build_prompt(specification.seed_program, specification.target_function, records)

        self.assertIn("def main(problem):", prompt)
        self.assertIn("def hash_string(s):", prompt)
        self.assertNotIn("def mix_char(h, i, c):", prompt)
        self.assertIn("def mix_char_v1(h, i, c):", prompt)

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

    def test_dead_loop_program_times_out(self) -> None:
        specification = build_capset_specification((1,))
        hanging_function = "def priority(element, n):\n    while True:\n        pass\n"
        hanging_program = replace_function(specification.seed_program, specification.target_function, hanging_function)
        self.assertIsNone(evaluate_program(hanging_program, specification))

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

    def test_mock_search_runs_on_string_hash_problem(self) -> None:
        specification = build_string_hash_specification()
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
        self.assertEqual(len(result.best_signature), 4)
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

    def test_trace_viewer_loads_iteration_timeline(self) -> None:
        specification = build_string_hash_specification()
        with tempfile.TemporaryDirectory() as temp_dir:
            trace_dir = f"{temp_dir}/trace"
            FunSearchRunner(
                specification=specification,
                llm=MockLLM(),
                config=SearchConfig(iterations=2, islands=2, reset_interval=2, random_seed=0),
                trace_writer=TraceWriter(trace_dir),
            ).run()

            state = load_trace_state(trace_dir)

            self.assertEqual(Path(state["trace_dir"]), Path(trace_dir).resolve())
            self.assertEqual(state["completed_iterations"], 2)
            self.assertEqual(len(state["iterations"]), 2)
            self.assertEqual(state["run_metadata"]["specification"]["target_function"], "mix_char")
            first_iteration = state["iterations"][0]
            self.assertEqual(first_iteration["iteration"], 0)
            self.assertIsNotNone(first_iteration["selected_island"])
            self.assertTrue(first_iteration["prompt_text"])
            self.assertTrue(first_iteration["completion_text"])
            self.assertIn(first_iteration["status"], {"accepted", "rejected"})
            json.dumps(state)


class CliTests(unittest.TestCase):
    def test_problem_flag_defaults_to_capset(self) -> None:
        args = build_parser().parse_args(["--llm", "mock"])
        self.assertEqual(args.problem, "capset")

    def test_problem_flag_accepts_string_hash(self) -> None:
        args = build_parser().parse_args(["--problem", "string-hash", "--llm", "mock"])
        self.assertEqual(args.problem, "string-hash")

    def test_string_hash_cli_flags_have_expected_defaults(self) -> None:
        args = build_parser().parse_args(["--problem", "string-hash", "--llm", "mock"])
        self.assertEqual(args.string_hash_buckets, DEFAULT_BUCKETS)
        self.assertEqual(args.string_hash_strings_per_case, DEFAULT_STRINGS_PER_CASE)

    def test_string_hash_cli_flags_can_be_overridden(self) -> None:
        args = build_parser().parse_args(
            [
                "--problem",
                "string-hash",
                "--llm",
                "mock",
                "--string-hash-buckets",
                "23",
                "--string-hash-strings-per-case",
                "7",
            ]
        )
        self.assertEqual(args.string_hash_buckets, 23)
        self.assertEqual(args.string_hash_strings_per_case, 7)


if __name__ == "__main__":
    unittest.main()
