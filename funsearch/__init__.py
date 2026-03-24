"""对外暴露这个教学版 FunSearch 最常用的入口。

这里集中导出：
- 运行搜索所需的核心类型
- LLM 接口与 mock 实现
- 两个内置问题的 builder

这样外部使用者通常只需要 `import funsearch` 即可开始实验。
"""

from funsearch.capset import build_capset_specification
from funsearch.core import FunSearchRunner, SearchConfig, SearchResult
from funsearch.llm import LLMClient, MockLLM, OpenAICompatibleLLM
from funsearch.string_hash import build_string_hash_specification

__all__ = [
    "FunSearchRunner",
    "LLMClient",
    "MockLLM",
    "OpenAICompatibleLLM",
    "SearchConfig",
    "SearchResult",
    "build_capset_specification",
    "build_string_hash_specification",
]
