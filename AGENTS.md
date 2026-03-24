# AGENTS.md 

## Python Environment

This project uses **uv** for Python environment and dependency management.

- Use `uv` for all Python-related commands in this repository.
- Run Python code and Python-based tools with `uv run`.
- Do NOT use `python` or `pip` directly, including `python`, `python3`, `python -m`, `pip`, `pip3`, or `python -m pip`.
- Do NOT change dependencies or dependency files without explicit approval. This includes `uv add`, `uv remove`, `uv pip install`, and manual edits to `pyproject.toml` or `uv.lock`.
- If a new dependency is needed, stop and ask for approval first. State the exact package and why it is needed.

### Examples

```bash
uv sync
uv run python script.py
uv run python -m package.module
uv run pytest tests/unit -x -q
uv run ruff check .
```