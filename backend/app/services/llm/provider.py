import json
from abc import ABC, abstractmethod
from typing import AsyncGenerator, Optional
import httpx
from backend.app.core.config import settings
from backend.app.services.llm.prompt import SYSTEM_PROMPT

class BaseLLMProvider(ABC):
    @abstractmethod
    async def generate(self, prompt: str, system_prompt: str = SYSTEM_PROMPT) -> str:
        pass

    @abstractmethod
    async def generate_stream(
        self, prompt: str, system_prompt: str = SYSTEM_PROMPT
    ) -> AsyncGenerator[str, None]:
        pass

class OllamaProvider(BaseLLMProvider):
    def __init__(self, base_url: str = settings.OLLAMA_BASE_URL, model: str = settings.OLLAMA_MODEL):
        self.base_url = base_url.rstrip("/")
        self.model = model

    async def generate(self, prompt: str, system_prompt: str = SYSTEM_PROMPT) -> str:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "system": system_prompt,
                    "stream": False
                }
            )
            resp.raise_for_status()
            data = resp.json()
            return data.get("response", "")

    async def generate_stream(
        self, prompt: str, system_prompt: str = SYSTEM_PROMPT
    ) -> AsyncGenerator[str, None]:
        async with httpx.AsyncClient(timeout=120.0) as client:
            async with client.stream(
                "POST",
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "system": system_prompt,
                    "stream": True
                }
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if line:
                        chunk = json.loads(line)
                        token = chunk.get("response", "")
                        if token:
                            yield token
                        if chunk.get("done", False):
                            break

class OpenAIProvider(BaseLLMProvider):
    def __init__(self, api_key: str = settings.OPENAI_API_KEY, model: str = settings.OPENAI_MODEL):
        self.api_key = api_key
        self.model = model
        self.base_url = "https://api.openai.com/v1"

    async def generate(self, prompt: str, system_prompt: str = SYSTEM_PROMPT) -> str:
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            "stream": False
        }
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(f"{self.base_url}/chat/completions", headers=headers, json=payload)
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]

    async def generate_stream(
        self, prompt: str, system_prompt: str = SYSTEM_PROMPT
    ) -> AsyncGenerator[str, None]:
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            "stream": True
        }
        async with httpx.AsyncClient(timeout=120.0) as client:
            async with client.stream(
                "POST", f"{self.base_url}/chat/completions", headers=headers, json=payload
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if line.startswith("data: ") and line != "data: [DONE]":
                        chunk = json.loads(line[6:])
                        delta = chunk["choices"][0]["delta"].get("content", "")
                        if delta:
                            yield delta

class MockProvider(BaseLLMProvider):
    """Deterministic fallback provider for testing and offline environments."""
    async def generate(self, prompt: str, system_prompt: str = SYSTEM_PROMPT) -> str:
        return "Based on your Second Brain notes [1], the system architecture relies on hybrid retrieval and parent-child chunking [2]."

    async def generate_stream(
        self, prompt: str, system_prompt: str = SYSTEM_PROMPT
    ) -> AsyncGenerator[str, None]:
        tokens = [
            "Based on ", "your Second ", "Brain notes ", "[1], ",
            "the architecture ", "implements hybrid ", "retrieval ", "with ",
            "reciprocal rank ", "fusion ", "and cross-encoder ", "reranking ", "[2]."
        ]
        for t in tokens:
            yield t

def get_llm_provider() -> BaseLLMProvider:
    if settings.LLM_PROVIDER == "openai" and settings.OPENAI_API_KEY:
        return OpenAIProvider()
    elif settings.LLM_PROVIDER == "ollama":
        return OllamaProvider()
    return MockProvider()
