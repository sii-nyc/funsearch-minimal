"""Prompt construction and AST-based function extraction."""

from __future__ import annotations

import ast
import re
from typing import Iterable

from funsearch.database import ProgramRecord


def build_prompt(seed_program: str, target_function: str, sampled_programs: Iterable[ProgramRecord]) -> str:
    """Builds a best-shot prompt from the program prelude and prior versions."""

    sampled_list = sorted(sampled_programs, key=lambda program: program.aggregate_score)
    versioned_blocks = [
        rename_function_source(program.function_source, f"{target_function}_v{index}")
        for index, program in enumerate(sampled_list)
    ]
    next_index = len(versioned_blocks)
    prompt_lines = [
        _extract_module_prelude(seed_program).rstrip(),
        "",
        *[block.rstrip() + "\n" for block in versioned_blocks],
        _build_empty_target(seed_program, target_function, f"{target_function}_v{next_index}"),
        "",
        f"# Complete `{target_function}_v{next_index}` only. Return Python code for that function.",
    ]
    return "\n".join(prompt_lines).strip() + "\n"


def extract_function_source(program_source: str, function_name: str) -> str:
    """Extracts a top-level function from program_source."""

    module = ast.parse(program_source)
    for node in module.body:
        if isinstance(node, ast.FunctionDef) and node.name == function_name:
            return ast.unparse(node)
    raise ValueError(f"Function {function_name!r} not found in program.")


def rename_function_source(function_source: str, new_name: str) -> str:
    """Renames the function in function_source and returns normalized source."""

    function_module = ast.parse(function_source)
    function_node = function_module.body[0]
    if not isinstance(function_node, ast.FunctionDef):
        raise ValueError("Expected a single function definition.")
    renamed = ast.FunctionDef(
        name=new_name,
        args=function_node.args,
        body=function_node.body,
        decorator_list=function_node.decorator_list,
        returns=function_node.returns,
        type_comment=function_node.type_comment,
        type_params=function_node.type_params,
    )
    ast.fix_missing_locations(renamed)
    return ast.unparse(renamed)


def extract_generated_function(completion: str, function_name: str, renamed_to: str) -> str | None:
    """Extracts function_name from a model completion and renames it to renamed_to."""

    for block in _candidate_function_blocks(completion, function_name):
        try:
            function_module = ast.parse(block)
        except SyntaxError:
            continue
        if not function_module.body:
            continue
        function_node = function_module.body[0]
        if isinstance(function_node, ast.FunctionDef) and function_node.name == function_name:
            return rename_function_source(ast.unparse(function_node), renamed_to)
    return None


def replace_function(program_source: str, function_name: str, new_function_source: str) -> str:
    """Replaces function_name inside program_source with new_function_source."""

    module = ast.parse(program_source)
    replacement_module = ast.parse(new_function_source)
    replacement_node = replacement_module.body[0]
    if not isinstance(replacement_node, ast.FunctionDef):
        raise ValueError("Replacement source must define a function.")

    updated_body = []
    replaced = False
    for node in module.body:
        if isinstance(node, ast.FunctionDef) and node.name == function_name:
            updated_body.append(replacement_node)
            replaced = True
        else:
            updated_body.append(node)
    if not replaced:
        raise ValueError(f"Function {function_name!r} not found in program.")

    updated_module = ast.Module(body=updated_body, type_ignores=[])
    ast.fix_missing_locations(updated_module)
    return ast.unparse(updated_module) + "\n"


def _extract_module_prelude(program_source: str) -> str:
    module = ast.parse(program_source)
    prelude = []
    for node in module.body:
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            prelude.append(ast.unparse(node))
            continue
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            prelude.append(ast.unparse(node))
            continue
        break
    return "\n\n".join(prelude)


def _build_empty_target(program_source: str, function_name: str, new_name: str) -> str:
    target_source = extract_function_source(program_source, function_name)
    target_module = ast.parse(target_source)
    target_node = target_module.body[0]
    if not isinstance(target_node, ast.FunctionDef):
        raise ValueError("Target source must be a function.")

    header_lines = [f"def {new_name}({ast.unparse(target_node.args)}):"]
    docstring = ast.get_docstring(target_node)
    if docstring:
        header_lines.append(f'    """{docstring}"""')
    return "\n".join(header_lines)


def _candidate_function_blocks(completion: str, function_name: str) -> list[str]:
    blocks = []
    blocks.extend(_extract_blocks_from_text(completion, function_name))
    for fenced in re.findall(r"```(?:python)?\n(.*?)```", completion, flags=re.DOTALL):
        blocks.extend(_extract_blocks_from_text(fenced, function_name))
    return blocks


def _extract_blocks_from_text(text: str, function_name: str) -> list[str]:
    lines = text.splitlines()
    blocks = []
    target_prefix = f"def {function_name}("
    for index, line in enumerate(lines):
        if line.lstrip().startswith(target_prefix):
            indent = len(line) - len(line.lstrip())
            block = [line[indent:]]
            cursor = index + 1
            while cursor < len(lines):
                next_line = lines[cursor]
                stripped = next_line.strip()
                current_indent = len(next_line) - len(next_line.lstrip())
                if stripped and current_indent <= indent:
                    break
                block.append(next_line[indent:])
                cursor += 1
            blocks.append("\n".join(block).strip() + "\n")
    return blocks
