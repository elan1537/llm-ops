import asyncio

from openai import AsyncOpenAI


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
    ) -> str:
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
                    return response.choices[0].message.content or ""
                except Exception as e:
                    last_error = e
                    if attempt < self.max_retries - 1:
                        await asyncio.sleep(2 ** attempt)
            raise last_error
