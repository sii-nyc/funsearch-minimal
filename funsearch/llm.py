"""Minimal LLM interfaces for OpenAI-compatible APIs and offline mocks."""

from __future__ import annotations

from abc import ABC, abstractmethod

from openai import OpenAI


class LLMClient(ABC):
    """The only interface the search loop needs from an LLM."""

    @abstractmethod
    def generate(self, prompt: str) -> str:
        raise NotImplementedError


class OpenAICompatibleLLM(LLMClient):
    """Calls a chat-completions endpoint through the official OpenAI Python SDK."""

    def __init__(self, base_url: str, api_key: str, model: str, temperature: float = 0.8) -> None:
        self.model = model
        self.temperature = temperature
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url.rstrip("/"),
        )

    def generate(self, prompt: str) -> str:
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
    """Returns hand-written mutations so the search loop is runnable offline."""

    def __init__(self) -> None:
        self._index = 0
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
        if "def mix_char_v" in prompt:
            responses = self._mix_char_responses
        else:
            responses = self._priority_responses
        response = responses[self._index % len(responses)]
        self._index += 1
        return response
