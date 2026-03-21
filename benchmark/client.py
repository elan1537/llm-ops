import asyncio
from dataclasses import dataclass

from openai import AsyncOpenAI


@dataclass
class GenerateResult:
    content: str
    prompt_tokens: int = 0
    completion_tokens: int = 0


class BenchmarkClient:
    def __init__(self, base_url: str, api_key: str = "", timeout: int = 120, max_concurrent: int = 4):
        self.client = AsyncOpenAI(
            base_url=base_url,
            api_key=api_key or "no-key",
            timeout=timeout,
        )
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.max_retries = 3

    async def generate(
        self, model: str, messages: list, temperature: float, max_tokens: int
    ) -> GenerateResult:
        async with self.semaphore:
            last_error = None
            for attempt in range(self.max_retries):
                try:
                    response = await self.client.chat.completions.create(
                        model=model,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
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
