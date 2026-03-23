import asyncio
from dataclasses import dataclass

import httpx
from openai import AsyncOpenAI


@dataclass
class GenerateResult:
    content: str
    prompt_tokens: int = 0
    completion_tokens: int = 0


class BenchmarkClient:
    """OpenAI-compatible client for vLLM models."""

    def __init__(self, base_url: str, api_key: str = "", timeout: int = 120, max_concurrent: int = 4):
        self.client = AsyncOpenAI(
            base_url=base_url,
            api_key=api_key or "no-key",
            timeout=timeout,
        )
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.max_retries = 3

    async def generate(
        self, model: str, messages: list, temperature: float, max_tokens: int,
        enable_thinking: bool = True,
    ) -> GenerateResult:
        async with self.semaphore:
            last_error = None
            for attempt in range(self.max_retries):
                try:
                    kwargs = dict(
                        model=model,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                    if not enable_thinking:
                        kwargs["extra_body"] = {
                            "chat_template_kwargs": {"enable_thinking": False}
                        }
                    response = await self.client.chat.completions.create(**kwargs)
                    usage = response.usage
                    return GenerateResult(
                        content=response.choices[0].message.content or "",
                        prompt_tokens=usage.prompt_tokens if usage else 0,
                        completion_tokens=usage.completion_tokens if usage else 0,
                    )
                except Exception as e:
                    last_error = e
                    if attempt < self.max_retries - 1:
                        await asyncio.sleep(2 ** attempt)
            raise last_error


class ClaudeNativeClient:
    """Claude native API client (/v1/messages) with thinking support."""

    def __init__(self, base_url: str, api_key: str = "", timeout: int = 120, max_concurrent: int = 4):
        # base_url should be like http://host:8317/v1 — strip /v1 for native API
        self.base_url = base_url.rstrip("/")
        if self.base_url.endswith("/v1"):
            self.base_url = self.base_url[:-3]
        self.api_key = api_key
        self.timeout = timeout
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.max_retries = 3

    async def generate(
        self, model: str, messages: list, temperature: float, max_tokens: int,
        enable_thinking: bool = True, thinking_budget: int = 8192,
    ) -> GenerateResult:
        async with self.semaphore:
            last_error = None
            for attempt in range(self.max_retries):
                try:
                    body = {
                        "model": model,
                        "messages": messages,
                        "max_tokens": max_tokens,
                    }

                    if enable_thinking:
                        body["thinking"] = {
                            "type": "enabled",
                            "budget_tokens": thinking_budget,
                        }
                        # temperature must be 1 when thinking is enabled for Claude
                        body["temperature"] = 1.0
                    else:
                        body["thinking"] = {"type": "disabled"}
                        body["temperature"] = temperature

                    headers = {
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                        "anthropic-version": "2023-06-01",
                    }

                    async with httpx.AsyncClient(timeout=self.timeout) as client:
                        response = await client.post(
                            f"{self.base_url}/v1/messages",
                            json=body,
                            headers=headers,
                        )
                        response.raise_for_status()
                        data = response.json()

                    # Extract text content (skip thinking blocks)
                    text_parts = []
                    for block in data.get("content", []):
                        if block["type"] == "text":
                            text_parts.append(block["text"])

                    usage = data.get("usage", {})
                    return GenerateResult(
                        content="\n".join(text_parts),
                        prompt_tokens=usage.get("input_tokens", 0),
                        completion_tokens=usage.get("output_tokens", 0),
                    )
                except Exception as e:
                    last_error = e
                    if attempt < self.max_retries - 1:
                        await asyncio.sleep(2 ** attempt)
            raise last_error


def create_client(model_config: dict, settings: dict) -> BenchmarkClient | ClaudeNativeClient:
    """Create appropriate client based on model's api_type."""
    api_type = model_config.get("api_type", "openai")
    if api_type == "claude":
        return ClaudeNativeClient(
            base_url=model_config["base_url"],
            api_key=model_config.get("api_key", ""),
            timeout=settings.get("timeout", 120),
            max_concurrent=settings.get("concurrent_requests", 4),
        )
    else:
        return BenchmarkClient(
            base_url=model_config["base_url"],
            api_key=model_config.get("api_key", ""),
            timeout=settings.get("timeout", 120),
            max_concurrent=settings.get("concurrent_requests", 4),
        )
