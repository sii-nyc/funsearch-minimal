# Minimal FunSearch Reproduction

This repository contains a minimal, readability-first Python reproduction of the core FunSearch loop from the paper _Mathematical discoveries from program search with large language models_.

It intentionally keeps only the central ideas:

- define a problem with an `evaluate(...)` function plus a program skeleton
- evolve only one target function inside that skeleton
- score candidates by executing them
- keep valid programs in a diverse database
- build best-shot prompts from historical high-scoring programs
- use a simplified island-based evolutionary search

## Simplifications

Compared with the paper, this version is intentionally small:

- single-process and synchronous
- no distributed sampler/evaluator/database workers
- no sandboxing, timeout, or resource isolation
- island resets happen every fixed number of evaluated candidates
- fixed-temperature sampling for clusters and programs
- OpenAI-compatible API calls use the official OpenAI Python SDK

## Layout

- `main.py`: CLI entrypoint
- `funsearch/capset.py`: cap set utilities and the paper-style problem specification
- `funsearch/core.py`: specification, evaluator, and search loop
- `funsearch/database.py`: islands, clustering, and resets
- `funsearch/prompting.py`: best-shot prompt building and AST-based candidate extraction
- `funsearch/llm.py`: OpenAI-compatible client via the OpenAI SDK plus offline `MockLLM`

## Run

Mock LLM, fully offline:

```bash
uv run python main.py --llm mock --iterations 8 --islands 4 --reset-interval 4 --inputs 1,2,3,4
```

OpenAI-compatible API:

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

The CLI prints:

- best aggregate score
- per-input score signature
- optional trace directory
- the full best program currently stored in the database

## Saving The Evolution Process

If you want to inspect the search history, pass `--trace-dir` and point it at an empty directory:

```bash
uv run python main.py \
  --llm mock \
  --iterations 8 \
  --islands 4 \
  --reset-interval 4 \
  --inputs 1,2,3,4 \
  --trace-dir runs/mock-trace
```

The trace directory contains:

- `run.json`: run configuration and problem metadata
- `events.jsonl`: append-only event log for sampling, completion, acceptance, rejection, reset, and iteration end
- `prompts/`: one prompt per iteration
- `completions/`: raw model output per iteration
- `candidates/`: reconstructed full candidate programs per iteration when extraction succeeds
- `programs/`: all accepted programs currently referenced by the database
- `snapshots/`: full database state after initialization, after every iteration, and at the end

This layout is meant for two inspection styles:

- sequential replay from `events.jsonl`
- direct inspection of “what was in the pool at iteration N” via `snapshots/iteration_XXXX.json`

## Tests

```bash
uv run python -m unittest -v
```
