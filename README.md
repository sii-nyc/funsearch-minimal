# Minimal FunSearch Reproduction

Chinese documentation: [README.zh-CN.md](/Users/hariseldon/Desktop/codes/funsearch/README.zh-CN.md)

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
- no sandboxing or resource isolation
- a small fixed execution timeout rejects candidates that hang
- island resets happen every fixed number of evaluated candidates
- fixed-temperature sampling for clusters and programs
- OpenAI-compatible API calls use the official OpenAI Python SDK
- two small built-in example problems: cap set and string hashing

## Layout

- `main.py`: CLI entrypoint
- `funsearch/capset.py`: cap set utilities and the paper-style problem specification
- `funsearch/string_hash.py`: a tiny string-hash demo that evolves `mix_char(h, i, c)`
- `funsearch/core.py`: specification, evaluator, and search loop
- `funsearch/database.py`: islands, clustering, and resets
- `funsearch/prompting.py`: best-shot prompt building and AST-based candidate extraction
- `funsearch/llm.py`: OpenAI-compatible client via the OpenAI SDK plus offline `MockLLM`

Prompt construction keeps the skeleton read-only, but it now includes the fixed helper/source context from the seed program so the model can see how the evolved function is actually used.

## Run

Mock LLM, fully offline:

```bash
uv run python main.py --llm mock --iterations 8 --islands 4 --reset-interval 4 --inputs 1,2,3,4
```

Mock LLM on the string-hash demo:

```bash
uv run python main.py --problem string-hash --llm mock --iterations 8 --islands 4 --reset-interval 4
```

String-hash demo with custom bucket count and dataset size:

```bash
uv run python main.py \
  --problem string-hash \
  --llm mock \
  --iterations 8 \
  --islands 4 \
  --reset-interval 4 \
  --string-hash-buckets 23 \
  --string-hash-strings-per-case 12
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

During the run, the CLI now also streams per-iteration records directly to stdout by default. That inline report includes:

- the selected island and its current programs
- the sampled historical programs used to build the prompt
- the full prompt text
- the raw completion
- the reconstructed candidate program when extraction succeeds
- acceptance / rejection details
- reset actions
- a short post-iteration database snapshot summary

The string-hash problem is meant as a compact teaching example: `hash_string(s)` stays fixed, FunSearch only mutates `mix_char(h, i, c)`, and the evaluator scores how evenly the resulting hash spreads several small string families across buckets.

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

If you only want the final result without the inline per-iteration report, add:

```bash
--no-live-report
```

If you save a trace and later want to inspect it again after the run, you can still point `funsearch.trace_viewer` at the trace directory:

```bash
uv run python -m funsearch.trace_viewer --trace-dir runs/mock-trace
```

The terminal dashboard auto-refreshes and keeps the same core information that used to be shown in the browser view:

- run status and best score
- iteration timeline with previous/next navigation
- selected island contents
- sampled historical programs used to build the prompt
- prompt text, raw completion, and reconstructed candidate program
- acceptance / rejection result and reset actions
- post-iteration database snapshot summary

Useful keys:

- `h` / `l` or left / right arrow: previous / next iteration
- `[` / `]` or `tab`: previous / next detail section
- `j` / `k`: scroll the current section
- `f`: toggle follow-latest
- `r`: refresh immediately
- `q`: quit

## Tests

```bash
uv run python -m unittest -v
```
