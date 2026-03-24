"""Minimal LLM interfaces for OpenAI-compatible APIs and offline mocks."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from urllib import request


class LLMClient(ABC):
    """The only interface the search loop needs from an LLM."""

    @abstractmethod
    def generate(self, prompt: str) -> str:
        raise NotImplementedError


class OpenAICompatibleLLM(LLMClient):
    """Calls a chat-completions endpoint that follows the OpenAI wire format."""

    def __init__(self, base_url: str, api_key: str, model: str, temperature: float = 0.8) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.temperature = temperature

    def generate(self, prompt: str) -> str:
        payload = json.dumps(
            {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": self.temperature,
            }
        ).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        endpoint = f"{self.base_url}/chat/completions"
        http_request = request.Request(endpoint, data=payload, headers=headers, method="POST")
        with request.urlopen(http_request) as response:
            data = json.loads(response.read().decode("utf-8"))
        return data["choices"][0]["message"]["content"]


class MockLLM(LLMClient):
    """Returns hand-written mutations so the search loop is runnable offline."""

    def __init__(self) -> None:
        self._index = 0
        self._responses = [
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

    def generate(self, prompt: str) -> str:
        del prompt
        response = self._responses[self._index % len(self._responses)]
        self._index += 1
        return response
