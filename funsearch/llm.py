"""LLM 抽象层。

这个项目刻意把 LLM 接口压缩得很小：
- 只需要 `generate(prompt) -> str`
- 一个真实实现：走 OpenAI 兼容 API
- 一个假实现：返回手写候选，方便离线演示
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from openai import OpenAI


class LLMClient(ABC):
    """搜索循环真正依赖的唯一接口。"""

    @abstractmethod
    def generate(self, prompt: str) -> str:
        raise NotImplementedError


class OpenAICompatibleLLM(LLMClient):
    """通过官方 OpenAI SDK 调用兼容的 chat completions 接口。"""

    def __init__(self, base_url: str, api_key: str, model: str, temperature: float = 0.8) -> None:
        self.model = model
        self.temperature = temperature
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url.rstrip("/"),
        )

    def generate(self, prompt: str) -> str:
        """把 prompt 作为单轮用户消息发给模型，并取回文本内容。"""

        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
        )
        content = completion.choices[0].message.content
        if content is None:
            raise ValueError("OpenAI SDK returned an empty completion.")
        return content


class MockLLM(LLMClient):
    """返回手写候选函数，让搜索循环在离线状态下也能跑通。"""

    def __init__(self) -> None:
        self._index = 0
        # cap set 问题用的一组手写候选。
        self._priority_responses = [
            """def priority_v2(element, n):
    \"\"\"Returns the priority with which we want to add `element` to the cap set.\"\"\"
    return sum(element)
""",
            """def priority_v2(element, n):
    \"\"\"Returns the priority with which we want to add `element` to the cap set.\"\"\"
    zeros = sum(coordinate == 0 for coordinate in element)
    return float(zeros * (n + 1) - sum(element))
""",
            """```python
def priority_v2(element, n):
    \"\"\"Returns the priority with which we want to add `element` to the cap set.\"\"\"
    zeros = sum(coordinate == 0 for coordinate in element)
    twos = sum(coordinate == 2 for coordinate in element)
    return float(zeros * 3 - twos)
```
""",
            """def priority_v2(element, n):
    \"\"\"Returns the priority with which we want to add `element` to the cap set.\"\"\"
    balance = sum(1 if coordinate == 0 else -1 for coordinate in element)
    return float(balance)
""",
        ]
        # string-hash 问题用的一组手写候选。
        self._mix_char_responses = [
            """def mix_char_v2(h, i, c):
    \"\"\"Mixes one character into the running hash state.\"\"\"
    return ((h << 5) - h + c + i) & 0xFFFFFFFF
""",
            """def mix_char_v2(h, i, c):
    \"\"\"Mixes one character into the running hash state.\"\"\"
    h ^= c + i * 17
    return (h * 131) & 0xFFFFFFFF
""",
            """```python
def mix_char_v2(h, i, c):
    \"\"\"Mixes one character into the running hash state.\"\"\"
    h += c ^ (i * 29)
    h ^= h >> 13
    return (h * 257) & 0xFFFFFFFF
```
""",
            """def mix_char_v2(h, i, c):
    \"\"\"Mixes one character into the running hash state.\"\"\"
    return (h * 65599 + (c ^ i)) & 0xFFFFFFFF
""",
        ]

    def generate(self, prompt: str) -> str:
        """根据 prompt 中的目标函数名，选择对应问题的 mock 响应。"""

        if "def mix_char_v" in prompt:
            responses = self._mix_char_responses
        else:
            responses = self._priority_responses
        response = responses[self._index % len(responses)]
        self._index += 1
        return response
