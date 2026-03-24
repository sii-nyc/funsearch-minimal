"""Prompt 构造和基于 AST 的函数抽取/替换。

这个模块的职责非常关键：保证 FunSearch 只改“目标函数”本身，而不会把
 整个程序骨架都交给模型自由发挥。
"""

from __future__ import annotations

import ast
import re
from typing import Iterable

from funsearch.database import ProgramRecord


def build_prompt(seed_program: str, target_function: str, sampled_programs: Iterable[ProgramRecord]) -> str:
    """根据 seed program 和历史程序构造 best-shot prompt。

    prompt 的结构是：
    - 模块前导部分（docstring / import）
    - 若干历史版本：`target_v0`, `target_v1`, ...
    - 一个空白的新版本：`target_vN`

    模型只能补全最后这个新版本。
    """

    # 历史程序按分数排序后重命名，形成清晰的“版本演化”上下文。
    sampled_list = sorted(sampled_programs, key=lambda program: program.aggregate_score)
    versioned_blocks = [
        rename_function_source(program.function_source, f"{target_function}_v{index}")
        for index, program in enumerate(sampled_list)
    ]
    next_index = len(versioned_blocks)
    best_program = max(sampled_list, key=lambda program: program.aggregate_score)
    score_lines = [
        f"# {target_function}_v{index}: signature={program.signature}, aggregate_score={program.aggregate_score}"
        for index, program in enumerate(sampled_list)
    ]
    prompt_lines = [
        f"# You are improving `{target_function}` inside a fixed program skeleton.",
        "# Task: propose a new candidate implementation that may score better than the earlier versions.",
        "# Goal: maximize the aggregate score produced by the evaluator across the fixed input set.",
        "# The surrounding program is fixed. Do not rewrite other functions, imports, or module-level code.",
        "# Return Python code only.",
        f"# Output exactly one function definition: `{target_function}_v{next_index}`.",
        "# Do not include markdown fences, explanations, tests, or any extra text outside the function.",
        "",
        "# Problem summary extracted from the fixed program:",
        *[f"# - {line}" for line in _summarize_program_for_prompt(seed_program, target_function)],
        "",
        "# Previous versions are shown below together with the scores they achieved.",
        *score_lines,
        f"# Best aggregate_score among the shown versions: {best_program.aggregate_score}",
        f"# Best signature among the shown versions: {best_program.signature}",
        "",
        _extract_module_prelude(seed_program).rstrip(),
        "",
        *[block.rstrip() + "\n" for block in versioned_blocks],
        _build_empty_target(seed_program, target_function, f"{target_function}_v{next_index}"),
        "",
        f"# Complete `{target_function}_v{next_index}` only.",
        f"# The function should be a plausible improvement over the previous `{target_function}` versions.",
    ]
    return "\n".join(prompt_lines).strip() + "\n"


def extract_function_source(program_source: str, function_name: str) -> str:
    """从完整程序源码里抽出某个顶层函数。"""

    module = ast.parse(program_source)
    for node in module.body:
        if isinstance(node, ast.FunctionDef) and node.name == function_name:
            return ast.unparse(node)
    raise ValueError(f"Function {function_name!r} not found in program.")


def rename_function_source(function_source: str, new_name: str) -> str:
    """给一段函数源码改名，并返回格式化后的源码。"""

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
    """从模型输出中提取目标函数，并改回真实函数名。

    例如 prompt 里要求补 `priority_v2`，但真正写回程序时需要把它改回
    `priority`。
    """

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
    """在完整程序骨架中替换目标函数。"""

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
    """提取模块开头的说明和 import，作为 prompt 前导。"""

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


def _summarize_program_for_prompt(program_source: str, target_function: str) -> list[str]:
    """提取固定程序骨架中的问题摘要，帮助模型理解评分目标。"""

    module = ast.parse(program_source)
    summary_lines = []

    module_docstring = ast.get_docstring(module)
    if module_docstring:
        summary_lines.append(module_docstring)

    for node in module.body:
        if not isinstance(node, ast.FunctionDef) or node.name == target_function:
            continue
        docstring = ast.get_docstring(node)
        if docstring:
            summary_lines.append(f"`{node.name}`: {docstring}")
    return summary_lines


def _build_empty_target(program_source: str, function_name: str, new_name: str) -> str:
    """构造一个只有函数头和 docstring 的“待补全版本”。"""

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
    """从普通文本和 fenced code block 中都尝试提取候选函数块。"""

    blocks = []
    blocks.extend(_extract_blocks_from_text(completion, function_name))
    for fenced in re.findall(r"```(?:python)?\n(.*?)```", completion, flags=re.DOTALL):
        blocks.extend(_extract_blocks_from_text(fenced, function_name))
    return blocks


def _extract_blocks_from_text(text: str, function_name: str) -> list[str]:
    """按缩进规则从文本里切出一个或多个同名函数定义。"""

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
