import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from benchmark.client import BenchmarkClient


@pytest.fixture
def client():
    return BenchmarkClient(
        base_url="http://localhost:8000/v1",
        api_key="test",
        timeout=10,
        max_concurrent=2,
    )


class TestGenerate:
    def test_returns_response_content(self, client):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Hello!"))]

        with patch.object(
            client.client.chat.completions, "create", new_callable=AsyncMock, return_value=mock_response
        ):
            result = asyncio.run(client.generate(
                model="test-model",
                messages=[{"role": "user", "content": "Hi"}],
                temperature=0.0,
                max_tokens=100,
            ))
            assert result == "Hello!"

    def test_retries_on_failure(self, client):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="OK"))]

        call_count = 0
        async def flaky_create(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise Exception("Server error")
            return mock_response

        with patch.object(
            client.client.chat.completions, "create", side_effect=flaky_create
        ):
            result = asyncio.run(client.generate(
                model="test-model",
                messages=[{"role": "user", "content": "Hi"}],
                temperature=0.0,
                max_tokens=100,
            ))
            assert result == "OK"
            assert call_count == 3

    def test_raises_after_max_retries(self, client):
        async def always_fail(**kwargs):
            raise Exception("Server error")

        with patch.object(
            client.client.chat.completions, "create", side_effect=always_fail
        ):
            with pytest.raises(Exception, match="Server error"):
                asyncio.run(client.generate(
                    model="test-model",
                    messages=[{"role": "user", "content": "Hi"}],
                    temperature=0.0,
                    max_tokens=100,
                ))


class TestConcurrency:
    def test_semaphore_limits_concurrent_calls(self, client):
        max_concurrent_seen = 0
        current_concurrent = 0

        async def slow_create(**kwargs):
            nonlocal max_concurrent_seen, current_concurrent
            current_concurrent += 1
            max_concurrent_seen = max(max_concurrent_seen, current_concurrent)
            await asyncio.sleep(0.05)
            current_concurrent -= 1
            mock = MagicMock()
            mock.choices = [MagicMock(message=MagicMock(content="ok"))]
            return mock

        with patch.object(
            client.client.chat.completions, "create", side_effect=slow_create
        ):
            async def run_batch():
                tasks = [
                    client.generate("m", [{"role": "user", "content": "hi"}], 0.0, 10)
                    for _ in range(6)
                ]
                await asyncio.gather(*tasks)

            asyncio.run(run_batch())
            assert max_concurrent_seen <= 2
