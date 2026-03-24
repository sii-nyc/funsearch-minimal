# AGENTS.md

## Project Intent

This repository is a **minimal, readability-first educational reproduction** of the FunSearch method from the paper *Mathematical discoveries from program search with large language models*.

The goal is to make the core algorithm easy to inspect and modify, not to build a production system. Prefer the simplest correct implementation that preserves the paper's main loop.

## Core Constraints

- Keep the implementation focused on the FunSearch core loop:
  - define a problem with `evaluate(...)` plus a program skeleton
  - evolve exactly one target function inside that skeleton
  - execute candidates and score them
  - keep only valid programs in the database
  - build prompts from historical programs
  - use simplified island-based evolutionary search
- Do **not** over-engineer. Avoid adding:
  - distributed workers
  - concurrency
  - sandboxing or resource isolation
  - plugin systems or large abstractions
  - extra example problems unless explicitly requested
- Prefer the Python standard library where practical.
- Do not add the `openai` SDK or other client libraries without explicit approval. The current OpenAI-compatible client intentionally uses standard-library HTTP only.

## Current Architecture

- `main.py`: CLI entrypoint.
- `funsearch/cli.py`: argument parsing and backend selection.
- `funsearch/core.py`: `ProblemSpecification`, evaluator, search loop, and result types.
- `funsearch/capset.py`: cap set utilities and the seed specification used by the search.
- `funsearch/prompting.py`: best-shot prompt construction and AST-based extraction/replacement of the evolved function.
- `funsearch/database.py`: simplified island model, signature clustering, sampling, and resets.
- `funsearch/llm.py`: `LLMClient`, `OpenAICompatibleLLM`, and `MockLLM`.
- `tests/test_funsearch.py`: unit and integration tests.

## Important Invariants

- The skeleton is immutable except for the single target function being evolved.
- Prompt construction must remain “best-shot” style:
  - sample historical programs from one island
  - rename them to versioned functions like `priority_v0`, `priority_v1`
  - append an empty next version such as `priority_v2`
- Candidate extraction should only accept the generated versioned target function and ignore unrelated code.
- The evaluator must reject candidates that are not executable or whose scores are invalid:
  - syntax/runtime failure
  - missing entrypoint
  - non-numeric score
  - non-finite score
  - invalid cap set output
- The program database should preserve diversity via signature-based clustering inside each island.
- Island resets are intentionally simplified to happen every fixed number of evaluated candidates, not by wall-clock time.

## Python Environment

This project uses **uv** for Python environment and dependency management.

- Use `uv` for all Python-related commands in this repository.
- Run Python code and Python-based tools with `uv run`.
- Do NOT use `python` or `pip` directly, including `python`, `python3`, `python -m`, `pip`, `pip3`, or `python -m pip`.
- Do NOT change dependencies or dependency files without explicit approval. This includes `uv add`, `uv remove`, `uv pip install`, and manual edits to `pyproject.toml` or `uv.lock`.
- If a new dependency is needed, stop and ask for approval first. State the exact package and why it is needed.

## Recommended Commands

Run tests:

```bash
uv run python -m unittest -v
```

Run the offline mock demo:

```bash
uv run python main.py --llm mock --iterations 8 --islands 4 --reset-interval 4 --inputs 1,2,3,4
```

Run against an OpenAI-compatible API:

```bash
uv run python main.py \
  --llm openai-compatible \
  --base-url http://localhost:8000/v1 \
  --api-key dummy \
  --model your-model-name \
  --iterations 8 \
  --islands 4 \
  --reset-interval 4 \
  --inputs 1,2,3,4
```

## Editing Guidance

- Preserve readability over cleverness.
- Keep comments short and explanatory, especially around prompt construction, AST rewriting, evaluator behavior, and island resets.
- When changing algorithm behavior, update both `README.md` and tests if the user-visible behavior or documented simplifications change.
- If you touch prompt/evaluator/database logic, add or update tests for the affected invariant.
