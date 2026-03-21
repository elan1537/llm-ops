# LLM Benchmark Framework Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a benchmark framework that evaluates 3 LLM models across 5 datasets (Korean QA, reading comprehension, summarization, math reasoning, vision/OCR) with auto-scoring and LLM-as-Judge.

**Architecture:** Python async pipeline — config.yaml drives which models/datasets to run, OpenAI-compatible client calls the gateway (localhost:8000) and Judge (CLIProxyAPI), evaluators score responses, results save as JSON with CLI summary.

**Tech Stack:** Python 3.11, openai (async client), HuggingFace datasets, tqdm, Pillow, pytest

**Spec:** `docs/superpowers/specs/2026-03-21-llm-benchmark-design.md`

---

## File Map

```
benchmark/
├── __init__.py              # Package marker
├── config.yaml              # Benchmark configuration
├── config.py                # Config loader + validation
├── client.py                # OpenAI-compatible async API client with retry
├── run.py                   # CLI entry point + orchestration
├── datasets/
│   ├── __init__.py          # Registry: DATASET_REGISTRY dict
│   ├── base.py              # Sample dataclass + BaseDataset ABC
│   ├── kmmlu.py             # KMMLU loader (Korean multiple-choice QA)
│   ├── korquad.py           # KorQuAD 1.0 loader (Korean reading comp)
│   ├── summarization.py     # XL-Sum Korean loader
│   ├── gsm8k.py             # GSM8K math reasoning loader
│   └── docvqa.py            # DocVQA vision/OCR loader
├── evaluators/
│   ├── __init__.py          # Registry: EVALUATOR_REGISTRY dict
│   ├── exact_match.py       # Multiple-choice / numeric exact match
│   ├── f1_em.py             # Token-level F1 + Exact Match
│   ├── anls.py              # Average Normalized Levenshtein Similarity
│   └── llm_judge.py         # Claude Opus judge via CLIProxyAPI
├── results/
│   └── .gitkeep
└── requirements.txt         # Benchmark-specific dependencies

tests/benchmark/
├── __init__.py
├── test_config.py           # Config loading + validation tests
├── test_client.py           # API client tests (mocked)
├── test_evaluators.py       # All evaluator unit tests
├── test_datasets.py         # Dataset loading + prompt formatting tests
└── test_run.py              # CLI runner integration test
```

**Key routing note:** The gateway routes by model `name` (the key in models.yaml, e.g. `"qwen3.5-122b"`), NOT by `model_id`. The benchmark config's `name` field is what gets sent as the `model` parameter in API calls. `model_id` is metadata for result display only.

---

### Task 1: Project Scaffolding + Config

**Files:**
- Create: `benchmark/__init__.py`
- Create: `benchmark/config.yaml`
- Create: `benchmark/config.py`
- Create: `benchmark/requirements.txt`
- Create: `benchmark/results/.gitkeep`
- Create: `benchmark/datasets/__init__.py`
- Create: `benchmark/evaluators/__init__.py`
- Create: `tests/benchmark/__init__.py`
- Create: `tests/benchmark/test_config.py`

- [ ] **Step 1: Create directory structure and package files**

```bash
mkdir -p benchmark/datasets benchmark/evaluators benchmark/results tests/benchmark
```

`benchmark/__init__.py`:
```python
```

`benchmark/datasets/__init__.py`:
```python
DATASET_REGISTRY: dict[str, type] = {}


def register_dataset(name: str):
    def decorator(cls):
        DATASET_REGISTRY[name] = cls
        return cls
    return decorator
```

`benchmark/evaluators/__init__.py`:
```python
EVALUATOR_REGISTRY: dict[str, type] = {}


def register_evaluator(name: str):
    def decorator(cls):
        EVALUATOR_REGISTRY[name] = cls
        return cls
    return decorator
```

`benchmark/results/.gitkeep`:
```
```

`tests/benchmark/__init__.py`:
```python
```

- [ ] **Step 2: Create benchmark/requirements.txt**

```
openai>=1.0.0
datasets>=2.14.0
tqdm>=4.60.0
Pillow>=10.0.0
```

- [ ] **Step 3: Create benchmark/config.yaml**

```yaml
models:
  - name: qwen3.5-122b
    base_url: http://localhost:8000/v1
    model_id: Qwen/Qwen3.5-122B-A10B-GPTQ-Int4
    vision: true
  - name: qwen3.5-35b
    base_url: http://localhost:8000/v1
    model_id: Qwen/Qwen3.5-35B-A3B-GPTQ-Int4
    vision: true
  - name: nemotron-3-nano
    base_url: http://localhost:8000/v1
    model_id: nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16
    vision: false

judge:
  base_url: http://CHANGE_ME:8080/v1
  model_id: claude-opus-4-20250514
  api_key: ""

datasets:
  kmmlu:
    enabled: true
    samples: 200
    evaluator: exact_match
  korquad:
    enabled: true
    samples: 200
    evaluator: f1_em
  xlsum_ko:
    enabled: true
    samples: 100
    evaluator: llm_judge
  gsm8k:
    enabled: true
    samples: 200
    evaluator: exact_match
  docvqa:
    enabled: true
    samples: 100
    evaluator: anls
    vision_only: true
    anls_judge_threshold: 0.5

settings:
  temperatures: [0.0, 0.5]
  stochastic_runs: 3
  concurrent_requests: 4
  timeout: 120
  max_tokens: 2048
  results_dir: benchmark/results
```

- [ ] **Step 4: Write the failing test for config loading**

`tests/benchmark/test_config.py`:
```python
import pytest
import yaml

from benchmark.config import load_config, validate_config, ConfigError


@pytest.fixture
def write_yaml(tmp_path):
    def _write(data):
        path = tmp_path / "config.yaml"
        path.write_text(yaml.dump(data, default_flow_style=False))
        return str(path)
    return _write


VALID_CONFIG = {
    "models": [
        {"name": "test-model", "base_url": "http://localhost:8000/v1",
         "model_id": "test/model", "vision": False}
    ],
    "judge": {"base_url": "http://judge:8080/v1", "model_id": "claude-opus-4-20250514", "api_key": ""},
    "datasets": {
        "kmmlu": {"enabled": True, "samples": 10, "evaluator": "exact_match"}
    },
    "settings": {
        "temperatures": [0.0],
        "stochastic_runs": 3,
        "concurrent_requests": 4,
        "timeout": 120,
        "max_tokens": 2048,
        "results_dir": "benchmark/results",
    },
}


class TestLoadConfig:
    def test_loads_valid_config(self, write_yaml):
        path = write_yaml(VALID_CONFIG)
        config = load_config(path)
        assert len(config["models"]) == 1
        assert config["models"][0]["name"] == "test-model"

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/config.yaml")

    def test_missing_models_section_raises(self, write_yaml):
        path = write_yaml({"datasets": {}, "settings": {}})
        with pytest.raises(ConfigError, match="models"):
            load_config(path)

    def test_missing_settings_section_raises(self, write_yaml):
        data = {**VALID_CONFIG}
        del data["settings"]
        path = write_yaml(data)
        with pytest.raises(ConfigError, match="settings"):
            load_config(path)


class TestValidateConfig:
    def test_valid_config_passes(self, write_yaml):
        path = write_yaml(VALID_CONFIG)
        config = load_config(path)
        validate_config(config)  # should not raise

    def test_duplicate_model_names_raises(self, write_yaml):
        data = {**VALID_CONFIG, "models": [
            {"name": "dup", "base_url": "http://a/v1", "model_id": "a", "vision": False},
            {"name": "dup", "base_url": "http://b/v1", "model_id": "b", "vision": False},
        ]}
        path = write_yaml(data)
        config = load_config(path)
        with pytest.raises(ConfigError, match="Duplicate model name"):
            validate_config(config)

    def test_unknown_evaluator_raises(self, write_yaml):
        data = {**VALID_CONFIG, "datasets": {
            "test": {"enabled": True, "samples": 10, "evaluator": "nonexistent"}
        }}
        path = write_yaml(data)
        config = load_config(path)
        with pytest.raises(ConfigError, match="Unknown evaluator"):
            validate_config(config)
```

- [ ] **Step 5: Run test to verify it fails**

Run: `cd /home/separk/workspace/infra && python -m pytest tests/benchmark/test_config.py -v`
Expected: FAIL (ImportError — benchmark.config does not exist)

- [ ] **Step 6: Implement config.py**

`benchmark/config.py`:
```python
import yaml

VALID_EVALUATORS = {"exact_match", "f1_em", "anls", "llm_judge"}


class ConfigError(Exception):
    pass


def load_config(path: str) -> dict:
    with open(path) as f:
        config = yaml.safe_load(f)

    for section in ("models", "settings"):
        if section not in config:
            raise ConfigError(f"Missing required section: {section}")

    return config


def validate_config(config: dict) -> None:
    names = [m["name"] for m in config["models"]]
    if len(names) != len(set(names)):
        dupes = [n for n in names if names.count(n) > 1]
        raise ConfigError(f"Duplicate model name: {dupes[0]}")

    for ds_name, ds_cfg in config.get("datasets", {}).items():
        if not ds_cfg.get("enabled", False):
            continue
        evaluator = ds_cfg.get("evaluator")
        if evaluator not in VALID_EVALUATORS:
            raise ConfigError(f"Unknown evaluator '{evaluator}' in dataset '{ds_name}'")
```

- [ ] **Step 7: Run test to verify it passes**

Run: `cd /home/separk/workspace/infra && python -m pytest tests/benchmark/test_config.py -v`
Expected: All PASS

- [ ] **Step 8: Commit**

```bash
git add benchmark/ tests/benchmark/
git commit -m "feat(benchmark): add project scaffolding and config loader"
```

---

### Task 2: API Client with Retry

**Files:**
- Create: `benchmark/client.py`
- Create: `tests/benchmark/test_client.py`

- [ ] **Step 1: Write the failing test**

`tests/benchmark/test_client.py`:
```python
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
            assert max_concurrent_seen <= 2  # max_concurrent=2 in fixture
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/separk/workspace/infra && python -m pytest tests/benchmark/test_client.py -v`
Expected: FAIL (ImportError)

- [ ] **Step 3: Implement client.py**

`benchmark/client.py`:
```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/separk/workspace/infra && python -m pytest tests/benchmark/test_client.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add benchmark/client.py tests/benchmark/test_client.py
git commit -m "feat(benchmark): add async API client with retry and concurrency"
```

---

### Task 3: Auto Evaluators (exact_match, f1_em, anls)

**Files:**
- Create: `benchmark/evaluators/exact_match.py`
- Create: `benchmark/evaluators/f1_em.py`
- Create: `benchmark/evaluators/anls.py`
- Create: `tests/benchmark/test_evaluators.py`

- [ ] **Step 1: Write the failing tests**

`tests/benchmark/test_evaluators.py`:
```python
import pytest

from benchmark.evaluators.exact_match import ExactMatchEvaluator
from benchmark.evaluators.f1_em import F1EMEvaluator
from benchmark.evaluators.anls import ANLSEvaluator


class TestExactMatch:
    def test_perfect_score(self):
        ev = ExactMatchEvaluator()
        result = ev.evaluate(
            predictions=["A", "B", "C"],
            references=["A", "B", "C"],
        )
        assert result["score"] == 1.0
        assert result["correct"] == 3
        assert result["total"] == 3

    def test_partial_score(self):
        ev = ExactMatchEvaluator()
        result = ev.evaluate(
            predictions=["A", "B", "D"],
            references=["A", "B", "C"],
        )
        assert result["score"] == pytest.approx(2 / 3)

    def test_case_insensitive(self):
        ev = ExactMatchEvaluator()
        result = ev.evaluate(predictions=["a"], references=["A"])
        assert result["score"] == 1.0

    def test_strips_whitespace(self):
        ev = ExactMatchEvaluator()
        result = ev.evaluate(predictions=["  A  "], references=["A"])
        assert result["score"] == 1.0

    def test_extracts_answer_letter_from_text(self):
        ev = ExactMatchEvaluator()
        result = ev.evaluate(
            predictions=["The answer is B.", "정답은 A입니다"],
            references=["B", "A"],
        )
        assert result["score"] == 1.0


class TestF1EM:
    def test_exact_match(self):
        ev = F1EMEvaluator()
        result = ev.evaluate(
            predictions=["서울은 한국의 수도입니다"],
            references=["서울은 한국의 수도입니다"],
        )
        assert result["em"] == 1.0
        assert result["f1"] == 1.0

    def test_partial_overlap(self):
        ev = F1EMEvaluator()
        result = ev.evaluate(
            predictions=["서울"],
            references=["서울은 한국의 수도"],
        )
        assert result["em"] == 0.0
        assert result["f1"] > 0.0  # partial token overlap

    def test_no_overlap(self):
        ev = F1EMEvaluator()
        result = ev.evaluate(
            predictions=["부산"],
            references=["서울"],
        )
        assert result["f1"] == 0.0


class TestANLS:
    def test_perfect_match(self):
        ev = ANLSEvaluator()
        result = ev.evaluate(predictions=["hello"], references=["hello"])
        assert result["anls"] == 1.0

    def test_partial_match(self):
        ev = ANLSEvaluator()
        result = ev.evaluate(predictions=["helo"], references=["hello"])
        assert result["anls"] > 0.5

    def test_completely_wrong(self):
        ev = ANLSEvaluator()
        result = ev.evaluate(predictions=["xyz"], references=["abcdef"])
        assert result["anls"] < 0.5

    def test_empty_prediction(self):
        ev = ANLSEvaluator()
        result = ev.evaluate(predictions=[""], references=["hello"])
        assert result["anls"] == 0.0

    def test_threshold_applied(self):
        """ANLS score < threshold (0.0 by default in scoring) should be 0."""
        ev = ANLSEvaluator()
        result = ev.evaluate(predictions=["completely wrong answer here"], references=["x"])
        assert result["anls"] == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/separk/workspace/infra && python -m pytest tests/benchmark/test_evaluators.py -v`
Expected: FAIL (ImportError)

- [ ] **Step 3: Implement exact_match.py**

`benchmark/evaluators/exact_match.py`:
```python
import re

from benchmark.evaluators import register_evaluator


def _normalize(text: str) -> str:
    text = text.strip().upper()
    # Try to extract a single answer letter (A/B/C/D) from verbose responses
    patterns = [
        r"(?:answer|정답)[은는\s:]*([A-D])",
        r"^([A-D])[\.\)\s]",
        r"\b([A-D])\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).upper()
    return text


@register_evaluator("exact_match")
class ExactMatchEvaluator:
    def evaluate(self, predictions: list[str], references: list[str]) -> dict:
        correct = 0
        details = []
        for pred, ref in zip(predictions, references):
            match = _normalize(pred) == _normalize(ref)
            if match:
                correct += 1
            details.append({"prediction": pred, "reference": ref, "correct": match})

        total = len(references)
        return {
            "score": correct / total if total else 0.0,
            "correct": correct,
            "total": total,
            "details": details,
        }
```

- [ ] **Step 4: Implement f1_em.py**

`benchmark/evaluators/f1_em.py`:
```python
import re
from collections import Counter

from benchmark.evaluators import register_evaluator


def _tokenize(text: str) -> list[str]:
    """Simple whitespace + character tokenizer for Korean/English."""
    text = text.strip().lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return text.split()


def _compute_f1(prediction: str, reference: str) -> float:
    pred_tokens = _tokenize(prediction)
    ref_tokens = _tokenize(reference)
    if not pred_tokens or not ref_tokens:
        return float(pred_tokens == ref_tokens)

    common = Counter(pred_tokens) & Counter(ref_tokens)
    num_common = sum(common.values())
    if num_common == 0:
        return 0.0

    precision = num_common / len(pred_tokens)
    recall = num_common / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def _compute_em(prediction: str, reference: str) -> float:
    return float(_tokenize(prediction) == _tokenize(reference))


@register_evaluator("f1_em")
class F1EMEvaluator:
    def evaluate(self, predictions: list[str], references: list[str]) -> dict:
        f1_scores = []
        em_scores = []
        details = []

        for pred, ref in zip(predictions, references):
            f1 = _compute_f1(pred, ref)
            em = _compute_em(pred, ref)
            f1_scores.append(f1)
            em_scores.append(em)
            details.append({"prediction": pred, "reference": ref, "f1": f1, "em": em})

        n = len(references)
        return {
            "f1": sum(f1_scores) / n if n else 0.0,
            "em": sum(em_scores) / n if n else 0.0,
            "total": n,
            "details": details,
        }
```

- [ ] **Step 5: Implement anls.py**

`benchmark/evaluators/anls.py`:
```python
from benchmark.evaluators import register_evaluator


def _edit_distance(s1: str, s2: str) -> int:
    """Compute Levenshtein edit distance."""
    if len(s1) < len(s2):
        return _edit_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)

    prev_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr_row = [i + 1]
        for j, c2 in enumerate(s2):
            cost = 0 if c1 == c2 else 1
            curr_row.append(min(
                curr_row[j] + 1,
                prev_row[j + 1] + 1,
                prev_row[j] + cost,
            ))
        prev_row = curr_row
    return prev_row[-1]


def _anls_score(prediction: str, reference: str, threshold: float = 0.5) -> float:
    """Compute ANLS (Average Normalized Levenshtein Similarity) for one pair."""
    pred = prediction.strip().lower()
    ref = reference.strip().lower()
    if not ref and not pred:
        return 1.0
    if not ref or not pred:
        return 0.0

    dist = _edit_distance(pred, ref)
    max_len = max(len(pred), len(ref))
    nls = 1.0 - dist / max_len
    return nls if nls >= threshold else 0.0


@register_evaluator("anls")
class ANLSEvaluator:
    def evaluate(
        self, predictions: list[str], references: list[str], threshold: float = 0.5
    ) -> dict:
        scores = []
        details = []
        for pred, ref in zip(predictions, references):
            score = _anls_score(pred, ref, threshold)
            scores.append(score)
            details.append({"prediction": pred, "reference": ref, "anls": score})

        n = len(references)
        return {
            "anls": sum(scores) / n if n else 0.0,
            "total": n,
            "details": details,
        }
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `cd /home/separk/workspace/infra && python -m pytest tests/benchmark/test_evaluators.py -v`
Expected: All PASS

- [ ] **Step 7: Commit**

```bash
git add benchmark/evaluators/ tests/benchmark/test_evaluators.py
git commit -m "feat(benchmark): add exact_match, f1_em, and anls evaluators"
```

---

### Task 4: LLM Judge Evaluator

**Files:**
- Create: `benchmark/evaluators/llm_judge.py`
- Modify: `tests/benchmark/test_evaluators.py` (append)

- [ ] **Step 1: Write the failing test**

Append to `tests/benchmark/test_evaluators.py`:
```python
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from benchmark.evaluators.llm_judge import LLMJudgeEvaluator


class TestLLMJudge:
    def test_parses_numeric_score(self):
        mock_client = MagicMock()
        mock_client.generate = AsyncMock(return_value="4")

        ev = LLMJudgeEvaluator()
        result = asyncio.run(ev.evaluate_async(
            client=mock_client,
            judge_model="claude-opus",
            predictions=["모델의 요약"],
            references=["원문 요약"],
            sources=["원문 텍스트"],
        ))
        assert result["mean_score"] == 4.0
        assert len(result["scores"]) == 1

    def test_parses_score_from_verbose_response(self):
        mock_client = MagicMock()
        mock_client.generate = AsyncMock(
            return_value="충실도: 4\n간결성: 3\n유창성: 5\n\n종합 점수: 4"
        )

        ev = LLMJudgeEvaluator()
        result = asyncio.run(ev.evaluate_async(
            client=mock_client,
            judge_model="claude-opus",
            predictions=["요약"],
            references=["참조"],
            sources=["원문"],
        ))
        assert result["mean_score"] == 4.0

    def test_handles_unparseable_score(self):
        mock_client = MagicMock()
        mock_client.generate = AsyncMock(return_value="I cannot evaluate this")

        ev = LLMJudgeEvaluator()
        result = asyncio.run(ev.evaluate_async(
            client=mock_client,
            judge_model="claude-opus",
            predictions=["요약"],
            references=["참조"],
            sources=["원문"],
        ))
        assert result["scores"][0] is None
        assert result["errors"] == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/separk/workspace/infra && python -m pytest tests/benchmark/test_evaluators.py::TestLLMJudge -v`
Expected: FAIL (ImportError)

- [ ] **Step 3: Implement llm_judge.py**

`benchmark/evaluators/llm_judge.py`:
```python
import asyncio
import re

from benchmark.evaluators import register_evaluator

JUDGE_PROMPT_TEMPLATE = """당신은 텍스트 요약 품질을 평가하는 전문가입니다.

아래 원문과 모델이 생성한 요약을 비교하여 평가해주세요.

## 원문
{source}

## 참조 요약
{reference}

## 모델 생성 요약
{prediction}

## 평가 기준 (각 1~5점)
- 충실도: 원문의 핵심 내용을 정확히 반영하는가
- 간결성: 불필요한 내용 없이 핵심만 담았는가
- 유창성: 문장이 자연스럽고 읽기 쉬운가

## 응답 형식
충실도: [점수]
간결성: [점수]
유창성: [점수]

종합 점수: [1~5 정수]"""


def _parse_score(text: str) -> int | None:
    """Extract the final overall score (종합 점수) from judge response."""
    match = re.search(r"종합\s*점수[:\s]*(\d)", text)
    if match:
        score = int(match.group(1))
        if 1 <= score <= 5:
            return score

    # Fallback: try to find a single digit 1-5
    numbers = re.findall(r"\b([1-5])\b", text)
    if numbers:
        return int(numbers[-1])

    return None


@register_evaluator("llm_judge")
class LLMJudgeEvaluator:
    async def evaluate_async(
        self,
        client,
        judge_model: str,
        predictions: list[str],
        references: list[str],
        sources: list[str],
    ) -> dict:
        scores = []
        errors = 0
        details = []

        for pred, ref, src in zip(predictions, references, sources):
            prompt = JUDGE_PROMPT_TEMPLATE.format(
                source=src, reference=ref, prediction=pred,
            )
            try:
                response = await client.generate(
                    model=judge_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=512,
                )
                score = _parse_score(response)
                if score is None:
                    errors += 1
                scores.append(score)
                details.append({
                    "prediction": pred, "reference": ref,
                    "judge_response": response, "score": score,
                })
            except Exception as e:
                scores.append(None)
                errors += 1
                details.append({
                    "prediction": pred, "reference": ref,
                    "error": str(e), "score": None,
                })

        valid_scores = [s for s in scores if s is not None]
        mean = sum(valid_scores) / len(valid_scores) if valid_scores else 0.0

        return {
            "mean_score": mean,
            "scores": scores,
            "errors": errors,
            "total": len(predictions),
            "details": details,
        }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/separk/workspace/infra && python -m pytest tests/benchmark/test_evaluators.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add benchmark/evaluators/llm_judge.py tests/benchmark/test_evaluators.py
git commit -m "feat(benchmark): add LLM judge evaluator with rubric scoring"
```

---

### Task 5: Dataset Base Class + KMMLU

**Files:**
- Create: `benchmark/datasets/base.py`
- Create: `benchmark/datasets/kmmlu.py`
- Create: `tests/benchmark/test_datasets.py`

- [ ] **Step 1: Write the failing test**

`tests/benchmark/test_datasets.py`:
```python
import pytest
from unittest.mock import patch, MagicMock

from benchmark.datasets.base import Sample, BaseDataset
from benchmark.datasets.kmmlu import KMMLUDataset


class TestSample:
    def test_text_sample(self):
        s = Sample(id="1", prompt="Hello", reference="A", metadata={})
        assert s.prompt == "Hello"
        assert s.reference == "A"

    def test_vision_sample(self):
        content = [
            {"type": "text", "text": "What is this?"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
        ]
        s = Sample(id="1", prompt=content, reference="cat", metadata={})
        assert isinstance(s.prompt, list)


class TestKMMLU:
    @pytest.fixture
    def mock_hf_dataset(self):
        """Mock HuggingFace datasets.load_dataset for KMMLU."""
        rows = [
            {"input": "한국의 수도는?", "A": "서울", "B": "부산", "C": "대전", "D": "인천", "output": "1"},
            {"input": "1+1=?", "A": "1", "B": "2", "C": "3", "D": "4", "output": "2"},
            {"input": "물의 화학식은?", "A": "CO2", "B": "H2O", "C": "NaCl", "D": "O2", "output": "2"},
        ]
        mock_ds = MagicMock()
        mock_ds.__len__ = lambda self: len(rows)
        mock_ds.__getitem__ = lambda self, idx: rows[idx] if isinstance(idx, int) else rows
        mock_ds.select = lambda indices: MagicMock(
            __iter__=lambda self: iter([rows[i] for i in indices]),
            __len__=lambda self: len(indices),
        )
        mock_ds.shuffle = lambda seed: mock_ds
        return mock_ds

    def test_load_samples(self, mock_hf_dataset):
        with patch("benchmark.datasets.kmmlu.load_dataset", return_value={"test": mock_hf_dataset}):
            ds = KMMLUDataset({"samples": 2})
            samples = ds.load_samples(n=2, seed=42)
            assert len(samples) == 2
            assert all(isinstance(s, Sample) for s in samples)

    def test_prompt_format(self, mock_hf_dataset):
        with patch("benchmark.datasets.kmmlu.load_dataset", return_value={"test": mock_hf_dataset}):
            ds = KMMLUDataset({"samples": 1})
            samples = ds.load_samples(n=1, seed=42)
            prompt = samples[0].prompt
            assert "A." in prompt
            assert "B." in prompt
            assert "C." in prompt
            assert "D." in prompt

    def test_reference_is_letter(self, mock_hf_dataset):
        with patch("benchmark.datasets.kmmlu.load_dataset", return_value={"test": mock_hf_dataset}):
            ds = KMMLUDataset({"samples": 1})
            samples = ds.load_samples(n=1, seed=42)
            assert samples[0].reference in ("A", "B", "C", "D")

    def test_name(self):
        ds = KMMLUDataset({})
        assert ds.name == "kmmlu"

    def test_requires_vision_false(self):
        ds = KMMLUDataset({})
        assert ds.requires_vision is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/separk/workspace/infra && python -m pytest tests/benchmark/test_datasets.py -v`
Expected: FAIL (ImportError)

- [ ] **Step 3: Implement base.py**

`benchmark/datasets/base.py`:
```python
from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class Sample:
    id: str
    prompt: str | list  # str for text, list[dict] for vision (OpenAI content format)
    reference: str
    metadata: dict = field(default_factory=dict)


class BaseDataset(ABC):
    def __init__(self, config: dict):
        self.config = config

    @abstractmethod
    def load_samples(self, n: int, seed: int = 42) -> list[Sample]:
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @property
    def requires_vision(self) -> bool:
        return False
```

- [ ] **Step 4: Implement kmmlu.py**

`benchmark/datasets/kmmlu.py`:
```python
import random

from datasets import load_dataset

from benchmark.datasets import register_dataset
from benchmark.datasets.base import BaseDataset, Sample

INDEX_TO_LETTER = {1: "A", 2: "B", 3: "C", 4: "D", "1": "A", "2": "B", "3": "C", "4": "D"}

PROMPT_TEMPLATE = """다음 질문에 대해 A, B, C, D 중 하나만 답하세요.

질문: {question}
A. {a}
B. {b}
C. {c}
D. {d}

정답:"""


@register_dataset("kmmlu")
class KMMLUDataset(BaseDataset):
    @property
    def name(self) -> str:
        return "kmmlu"

    def load_samples(self, n: int, seed: int = 42) -> list[Sample]:
        ds = load_dataset("HAERAE-HUB/KMMLU", "all")
        data = ds["test"]

        random.seed(seed)
        total = len(data)
        indices = random.sample(range(total), min(n, total))

        samples = []
        for idx in indices:
            row = data[idx]
            prompt = PROMPT_TEMPLATE.format(
                question=row["input"],
                a=row["A"],
                b=row["B"],
                c=row["C"],
                d=row["D"],
            )
            ref_raw = row["output"]
            reference = INDEX_TO_LETTER.get(ref_raw, str(ref_raw).strip().upper())

            samples.append(Sample(
                id=f"kmmlu_{idx}",
                prompt=prompt,
                reference=reference,
                metadata={"index": idx},
            ))
        return samples
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd /home/separk/workspace/infra && python -m pytest tests/benchmark/test_datasets.py -v`
Expected: All PASS

- [ ] **Step 6: Commit**

```bash
git add benchmark/datasets/base.py benchmark/datasets/kmmlu.py tests/benchmark/test_datasets.py
git commit -m "feat(benchmark): add dataset base class and KMMLU loader"
```

---

### Task 6: KorQuAD + GSM8K Datasets

**Files:**
- Create: `benchmark/datasets/korquad.py`
- Create: `benchmark/datasets/gsm8k.py`
- Modify: `tests/benchmark/test_datasets.py` (append)

- [ ] **Step 1: Write the failing tests**

Append to `tests/benchmark/test_datasets.py`:
```python
from benchmark.datasets.korquad import KorQuADDataset
from benchmark.datasets.gsm8k import GSM8KDataset


class TestKorQuAD:
    @pytest.fixture
    def mock_hf_dataset(self):
        rows = [
            {
                "context": "서울은 대한민국의 수도이며 가장 큰 도시이다.",
                "question": "대한민국의 수도는 어디인가?",
                "answers": {"text": ["서울"], "answer_start": [0]},
            },
            {
                "context": "파이썬은 1991년에 만들어진 프로그래밍 언어이다.",
                "question": "파이썬은 언제 만들어졌는가?",
                "answers": {"text": ["1991년"], "answer_start": [4]},
            },
        ]
        mock_ds = MagicMock()
        mock_ds.__len__ = lambda self: len(rows)
        mock_ds.__getitem__ = lambda self, idx: rows[idx]
        mock_ds.shuffle = lambda seed: mock_ds
        return mock_ds

    def test_load_samples(self, mock_hf_dataset):
        with patch("benchmark.datasets.korquad.load_dataset", return_value={"validation": mock_hf_dataset}):
            ds = KorQuADDataset({"samples": 2})
            samples = ds.load_samples(n=2, seed=42)
            assert len(samples) == 2

    def test_prompt_contains_context_and_question(self, mock_hf_dataset):
        with patch("benchmark.datasets.korquad.load_dataset", return_value={"validation": mock_hf_dataset}):
            ds = KorQuADDataset({"samples": 1})
            samples = ds.load_samples(n=1, seed=42)
            assert "서울은 대한민국의 수도" in samples[0].prompt
            assert "수도는 어디" in samples[0].prompt

    def test_reference_is_answer_text(self, mock_hf_dataset):
        with patch("benchmark.datasets.korquad.load_dataset", return_value={"validation": mock_hf_dataset}):
            ds = KorQuADDataset({"samples": 1})
            samples = ds.load_samples(n=1, seed=42)
            assert samples[0].reference == "서울"


class TestGSM8K:
    @pytest.fixture
    def mock_hf_dataset(self):
        rows = [
            {"question": "3+5=?", "answer": "3+5=8\n#### 8"},
            {"question": "10*2=?", "answer": "10*2=20\n#### 20"},
        ]
        mock_ds = MagicMock()
        mock_ds.__len__ = lambda self: len(rows)
        mock_ds.__getitem__ = lambda self, idx: rows[idx]
        mock_ds.shuffle = lambda seed: mock_ds
        return mock_ds

    def test_load_samples(self, mock_hf_dataset):
        with patch("benchmark.datasets.gsm8k.load_dataset", return_value={"test": mock_hf_dataset}):
            ds = GSM8KDataset({"samples": 2})
            samples = ds.load_samples(n=2, seed=42)
            assert len(samples) == 2

    def test_reference_is_final_number(self, mock_hf_dataset):
        with patch("benchmark.datasets.gsm8k.load_dataset", return_value={"test": mock_hf_dataset}):
            ds = GSM8KDataset({"samples": 1})
            samples = ds.load_samples(n=1, seed=42)
            assert samples[0].reference in ("8", "20")

    def test_prompt_asks_for_step_by_step(self, mock_hf_dataset):
        with patch("benchmark.datasets.gsm8k.load_dataset", return_value={"test": mock_hf_dataset}):
            ds = GSM8KDataset({"samples": 1})
            samples = ds.load_samples(n=1, seed=42)
            assert "단계별" in samples[0].prompt or "step" in samples[0].prompt.lower()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/separk/workspace/infra && python -m pytest tests/benchmark/test_datasets.py::TestKorQuAD tests/benchmark/test_datasets.py::TestGSM8K -v`
Expected: FAIL (ImportError)

- [ ] **Step 3: Implement korquad.py**

`benchmark/datasets/korquad.py`:
```python
import random

from datasets import load_dataset

from benchmark.datasets import register_dataset
from benchmark.datasets.base import BaseDataset, Sample

PROMPT_TEMPLATE = """다음 문맥을 읽고 질문에 답하세요. 답은 문맥에서 직접 찾아 간결하게 작성하세요.

문맥: {context}

질문: {question}

답:"""


@register_dataset("korquad")
class KorQuADDataset(BaseDataset):
    @property
    def name(self) -> str:
        return "korquad"

    def load_samples(self, n: int, seed: int = 42) -> list[Sample]:
        ds = load_dataset("squad_kor_v1")
        data = ds["validation"]

        random.seed(seed)
        total = len(data)
        indices = random.sample(range(total), min(n, total))

        samples = []
        for idx in indices:
            row = data[idx]
            prompt = PROMPT_TEMPLATE.format(
                context=row["context"],
                question=row["question"],
            )
            reference = row["answers"]["text"][0]

            samples.append(Sample(
                id=f"korquad_{idx}",
                prompt=prompt,
                reference=reference,
                metadata={"index": idx},
            ))
        return samples
```

- [ ] **Step 4: Implement gsm8k.py**

`benchmark/datasets/gsm8k.py`:
```python
import re
import random

from datasets import load_dataset

from benchmark.datasets import register_dataset
from benchmark.datasets.base import BaseDataset, Sample

PROMPT_TEMPLATE = """다음 수학 문제를 단계별로 풀어주세요. 최종 답은 마지막 줄에 숫자만 작성하세요.

문제: {question}

풀이:"""


def _extract_answer(answer_text: str) -> str:
    """Extract the final numeric answer after ####."""
    match = re.search(r"####\s*(.+)", answer_text)
    if match:
        return match.group(1).strip().replace(",", "")
    return answer_text.strip()


@register_dataset("gsm8k")
class GSM8KDataset(BaseDataset):
    @property
    def name(self) -> str:
        return "gsm8k"

    def load_samples(self, n: int, seed: int = 42) -> list[Sample]:
        ds = load_dataset("openai/gsm8k", "main")
        data = ds["test"]

        random.seed(seed)
        total = len(data)
        indices = random.sample(range(total), min(n, total))

        samples = []
        for idx in indices:
            row = data[idx]
            prompt = PROMPT_TEMPLATE.format(question=row["question"])
            reference = _extract_answer(row["answer"])

            samples.append(Sample(
                id=f"gsm8k_{idx}",
                prompt=prompt,
                reference=reference,
                metadata={"index": idx, "full_answer": row["answer"]},
            ))
        return samples
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd /home/separk/workspace/infra && python -m pytest tests/benchmark/test_datasets.py -v`
Expected: All PASS

- [ ] **Step 6: Commit**

```bash
git add benchmark/datasets/korquad.py benchmark/datasets/gsm8k.py tests/benchmark/test_datasets.py
git commit -m "feat(benchmark): add KorQuAD and GSM8K dataset loaders"
```

---

### Task 7: Summarization Dataset (XL-Sum Korean)

**Files:**
- Create: `benchmark/datasets/summarization.py`
- Modify: `tests/benchmark/test_datasets.py` (append)

- [ ] **Step 1: Write the failing test**

Append to `tests/benchmark/test_datasets.py`:
```python
from benchmark.datasets.summarization import XLSumKoDataset


class TestXLSumKo:
    @pytest.fixture
    def mock_hf_dataset(self):
        rows = [
            {"text": "서울에서 대규모 축제가 열렸다. 수만 명의 시민이 참여했다.", "summary": "서울 대규모 축제에 수만 명 참여"},
            {"text": "새로운 AI 기술이 발표되었다.", "summary": "새 AI 기술 발표"},
        ]
        mock_ds = MagicMock()
        mock_ds.__len__ = lambda self: len(rows)
        mock_ds.__getitem__ = lambda self, idx: rows[idx]
        mock_ds.shuffle = lambda seed: mock_ds
        return mock_ds

    def test_load_samples(self, mock_hf_dataset):
        with patch("benchmark.datasets.summarization.load_dataset", return_value={"test": mock_hf_dataset}):
            ds = XLSumKoDataset({"samples": 2})
            samples = ds.load_samples(n=2, seed=42)
            assert len(samples) == 2

    def test_prompt_contains_source_text(self, mock_hf_dataset):
        with patch("benchmark.datasets.summarization.load_dataset", return_value={"test": mock_hf_dataset}):
            ds = XLSumKoDataset({"samples": 1})
            samples = ds.load_samples(n=1, seed=42)
            assert "서울에서 대규모 축제" in samples[0].prompt

    def test_reference_is_summary(self, mock_hf_dataset):
        with patch("benchmark.datasets.summarization.load_dataset", return_value={"test": mock_hf_dataset}):
            ds = XLSumKoDataset({"samples": 1})
            samples = ds.load_samples(n=1, seed=42)
            assert "축제" in samples[0].reference or "AI" in samples[0].reference

    def test_metadata_has_source(self, mock_hf_dataset):
        with patch("benchmark.datasets.summarization.load_dataset", return_value={"test": mock_hf_dataset}):
            ds = XLSumKoDataset({"samples": 1})
            samples = ds.load_samples(n=1, seed=42)
            assert "source" in samples[0].metadata
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/separk/workspace/infra && python -m pytest tests/benchmark/test_datasets.py::TestXLSumKo -v`
Expected: FAIL (ImportError)

- [ ] **Step 3: Implement summarization.py**

`benchmark/datasets/summarization.py`:
```python
import random

from datasets import load_dataset

from benchmark.datasets import register_dataset
from benchmark.datasets.base import BaseDataset, Sample

PROMPT_TEMPLATE = """다음 텍스트를 핵심 내용 위주로 간결하게 요약하세요.

텍스트:
{text}

요약:"""


@register_dataset("xlsum_ko")
class XLSumKoDataset(BaseDataset):
    @property
    def name(self) -> str:
        return "xlsum_ko"

    def load_samples(self, n: int, seed: int = 42) -> list[Sample]:
        ds = load_dataset("csebuetnlp/xlsum", "korean")
        data = ds["test"]

        random.seed(seed)
        total = len(data)
        indices = random.sample(range(total), min(n, total))

        samples = []
        for idx in indices:
            row = data[idx]
            prompt = PROMPT_TEMPLATE.format(text=row["text"])

            samples.append(Sample(
                id=f"xlsum_ko_{idx}",
                prompt=prompt,
                reference=row["summary"],
                metadata={"index": idx, "source": row["text"]},
            ))
        return samples
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/separk/workspace/infra && python -m pytest tests/benchmark/test_datasets.py::TestXLSumKo -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add benchmark/datasets/summarization.py tests/benchmark/test_datasets.py
git commit -m "feat(benchmark): add XL-Sum Korean summarization dataset loader"
```

---

### Task 8: DocVQA Dataset (Vision)

**Files:**
- Create: `benchmark/datasets/docvqa.py`
- Modify: `tests/benchmark/test_datasets.py` (append)

- [ ] **Step 1: Write the failing test**

Append to `tests/benchmark/test_datasets.py`:
```python
from benchmark.datasets.docvqa import DocVQADataset
from PIL import Image
import io
import base64


class TestDocVQA:
    @pytest.fixture
    def mock_hf_dataset(self):
        """Create mock dataset with PIL Image objects."""
        img = Image.new("RGB", (100, 100), color="white")

        rows = [
            {"image": img, "question": "What is the title?", "answers": ["Annual Report"]},
            {"image": img, "question": "What is the date?", "answers": ["2024-01-01"]},
        ]
        mock_ds = MagicMock()
        mock_ds.__len__ = lambda self: len(rows)
        mock_ds.__getitem__ = lambda self, idx: rows[idx]
        mock_ds.shuffle = lambda seed: mock_ds
        return mock_ds

    def test_load_samples(self, mock_hf_dataset):
        with patch("benchmark.datasets.docvqa.load_dataset", return_value={"test": mock_hf_dataset}):
            ds = DocVQADataset({"samples": 2})
            samples = ds.load_samples(n=2, seed=42)
            assert len(samples) == 2

    def test_prompt_is_vision_format(self, mock_hf_dataset):
        with patch("benchmark.datasets.docvqa.load_dataset", return_value={"test": mock_hf_dataset}):
            ds = DocVQADataset({"samples": 1})
            samples = ds.load_samples(n=1, seed=42)
            prompt = samples[0].prompt
            assert isinstance(prompt, list)
            types = [item["type"] for item in prompt]
            assert "text" in types
            assert "image_url" in types

    def test_image_is_base64_encoded(self, mock_hf_dataset):
        with patch("benchmark.datasets.docvqa.load_dataset", return_value={"test": mock_hf_dataset}):
            ds = DocVQADataset({"samples": 1})
            samples = ds.load_samples(n=1, seed=42)
            prompt = samples[0].prompt
            img_item = [i for i in prompt if i["type"] == "image_url"][0]
            assert img_item["image_url"]["url"].startswith("data:image/png;base64,")

    def test_requires_vision_true(self):
        ds = DocVQADataset({})
        assert ds.requires_vision is True

    def test_reference_is_first_answer(self, mock_hf_dataset):
        with patch("benchmark.datasets.docvqa.load_dataset", return_value={"test": mock_hf_dataset}):
            ds = DocVQADataset({"samples": 1})
            samples = ds.load_samples(n=1, seed=42)
            assert samples[0].reference in ("Annual Report", "2024-01-01")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/separk/workspace/infra && python -m pytest tests/benchmark/test_datasets.py::TestDocVQA -v`
Expected: FAIL (ImportError)

- [ ] **Step 3: Implement docvqa.py**

`benchmark/datasets/docvqa.py`:
```python
import base64
import io
import random

from datasets import load_dataset

from benchmark.datasets import register_dataset
from benchmark.datasets.base import BaseDataset, Sample

QUESTION_PREFIX = "다음 문서 이미지를 보고 질문에 답하세요.\n\n질문: "


def _image_to_base64(image) -> str:
    """Convert PIL Image to base64 data URI."""
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


@register_dataset("docvqa")
class DocVQADataset(BaseDataset):
    @property
    def name(self) -> str:
        return "docvqa"

    @property
    def requires_vision(self) -> bool:
        return True

    def load_samples(self, n: int, seed: int = 42) -> list[Sample]:
        ds = load_dataset("lmms-lab/DocVQA")
        data = ds["test"]

        random.seed(seed)
        total = len(data)
        indices = random.sample(range(total), min(n, total))

        samples = []
        for idx in indices:
            row = data[idx]
            image_uri = _image_to_base64(row["image"])

            prompt = [
                {"type": "text", "text": f"{QUESTION_PREFIX}{row['question']}"},
                {"type": "image_url", "image_url": {"url": image_uri}},
            ]

            answers = row["answers"]
            reference = answers[0] if answers else ""

            samples.append(Sample(
                id=f"docvqa_{idx}",
                prompt=prompt,
                reference=reference,
                metadata={"index": idx, "all_answers": answers},
            ))
        return samples
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/separk/workspace/infra && python -m pytest tests/benchmark/test_datasets.py::TestDocVQA -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add benchmark/datasets/docvqa.py tests/benchmark/test_datasets.py
git commit -m "feat(benchmark): add DocVQA vision dataset loader"
```

---

### Task 9: CLI Runner (run.py)

**Files:**
- Create: `benchmark/run.py`
- Create: `tests/benchmark/test_run.py`

This is the orchestration layer that ties everything together.

- [ ] **Step 1: Write the failing test**

`tests/benchmark/test_run.py`:
```python
import asyncio
import json
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml

from benchmark.run import BenchmarkRunner


@pytest.fixture
def minimal_config(tmp_path):
    config = {
        "models": [
            {"name": "test-model", "base_url": "http://localhost:8000/v1",
             "model_id": "test/model", "vision": False}
        ],
        "judge": {"base_url": "http://judge:8080/v1", "model_id": "claude-opus", "api_key": ""},
        "datasets": {
            "kmmlu": {"enabled": True, "samples": 2, "evaluator": "exact_match"},
        },
        "settings": {
            "temperatures": [0.0],
            "stochastic_runs": 1,
            "concurrent_requests": 2,
            "timeout": 10,
            "max_tokens": 100,
            "results_dir": str(tmp_path / "results"),
        },
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.dump(config))
    return str(path), config


class TestBenchmarkRunner:
    def test_init_creates_results_dir(self, minimal_config):
        config_path, config = minimal_config
        runner = BenchmarkRunner(config_path)
        assert os.path.isdir(config["settings"]["results_dir"])

    def test_filter_models_for_vision_dataset(self, minimal_config):
        config_path, _ = minimal_config
        runner = BenchmarkRunner(config_path)
        # vision_only dataset should skip non-vision model
        models = runner._filter_models(vision_only=True)
        assert len(models) == 0  # test-model has vision=False

    def test_filter_models_for_text_dataset(self, minimal_config):
        config_path, _ = minimal_config
        runner = BenchmarkRunner(config_path)
        models = runner._filter_models(vision_only=False)
        assert len(models) == 1

    def test_run_stores_results_json(self, minimal_config):
        config_path, config = minimal_config

        from benchmark.datasets.base import Sample
        mock_samples = [
            Sample(id="1", prompt="Q1", reference="A", metadata={}),
            Sample(id="2", prompt="Q2", reference="B", metadata={}),
        ]

        with patch("benchmark.run.DATASET_REGISTRY", {
            "kmmlu": MagicMock(return_value=MagicMock(
                load_samples=MagicMock(return_value=mock_samples),
                name="kmmlu",
                requires_vision=False,
            ))
        }), patch("benchmark.run.EVALUATOR_REGISTRY", {
            "exact_match": MagicMock(return_value=MagicMock(
                evaluate=MagicMock(return_value={"score": 0.5, "correct": 1, "total": 2, "details": []})
            ))
        }), patch("benchmark.run.BenchmarkClient") as MockClient:
            mock_client_instance = MagicMock()
            mock_client_instance.generate = AsyncMock(return_value="A")
            MockClient.return_value = mock_client_instance

            runner = BenchmarkRunner(config_path)
            asyncio.run(runner.run())

            results_dir = config["settings"]["results_dir"]
            result_files = os.listdir(results_dir)
            json_files = [f for f in result_files if f.endswith(".json")]
            assert len(json_files) == 1

            with open(os.path.join(results_dir, json_files[0])) as f:
                data = json.load(f)
            assert "kmmlu" in data["results"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/separk/workspace/infra && python -m pytest tests/benchmark/test_run.py -v`
Expected: FAIL (ImportError)

- [ ] **Step 3: Implement run.py**

`benchmark/run.py`:
```python
import argparse
import asyncio
import json
import os
import statistics
from datetime import datetime

from tqdm import tqdm

from benchmark.client import BenchmarkClient
from benchmark.config import load_config, validate_config
from benchmark.datasets import DATASET_REGISTRY
from benchmark.datasets.base import Sample
from benchmark.evaluators import EVALUATOR_REGISTRY

# Force dataset/evaluator registration by importing all modules
import benchmark.datasets.kmmlu
import benchmark.datasets.korquad
import benchmark.datasets.gsm8k
import benchmark.datasets.summarization
import benchmark.datasets.docvqa
import benchmark.evaluators.exact_match
import benchmark.evaluators.f1_em
import benchmark.evaluators.anls
import benchmark.evaluators.llm_judge


class BenchmarkRunner:
    def __init__(self, config_path: str):
        self.config = load_config(config_path)
        validate_config(self.config)
        self.settings = self.config["settings"]
        os.makedirs(self.settings["results_dir"], exist_ok=True)

    def _filter_models(self, vision_only: bool) -> list[dict]:
        if vision_only:
            return [m for m in self.config["models"] if m.get("vision", False)]
        return list(self.config["models"])

    async def _run_model_on_samples(
        self, client: BenchmarkClient, model_name: str, samples: list[Sample],
        temperature: float,
    ) -> list[str]:
        """Run model on all samples, return list of predictions."""
        tasks = []
        for sample in samples:
            if isinstance(sample.prompt, list):
                messages = [{"role": "user", "content": sample.prompt}]
            else:
                messages = [{"role": "user", "content": sample.prompt}]
            tasks.append(client.generate(
                model=model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=self.settings["max_tokens"],
            ))

        predictions = []
        for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc=f"  {model_name}", leave=False):
            try:
                result = await coro
                predictions.append(result)
            except Exception as e:
                predictions.append(f"ERROR: {e}")

        return predictions

    async def _evaluate(
        self, ds_name: str, ds_config: dict, evaluator_name: str,
        predictions: list[str], samples: list[Sample],
        judge_client: BenchmarkClient | None = None,
    ) -> dict:
        evaluator_cls = EVALUATOR_REGISTRY[evaluator_name]
        evaluator = evaluator_cls()

        references = [s.reference for s in samples]

        if evaluator_name == "llm_judge":
            sources = [s.metadata.get("source", "") for s in samples]
            return await evaluator.evaluate_async(
                client=judge_client,
                judge_model=self.config["judge"]["model_id"],
                predictions=predictions,
                references=references,
                sources=sources,
            )
        elif evaluator_name == "anls":
            threshold = ds_config.get("anls_judge_threshold", 0.5)
            result = evaluator.evaluate(predictions, references, threshold=threshold)

            # Re-evaluate low-ANLS samples with Judge if available
            if judge_client and threshold > 0:
                low_indices = [
                    i for i, d in enumerate(result["details"])
                    if d["anls"] < threshold
                ]
                if low_indices:
                    judge_eval = EVALUATOR_REGISTRY["llm_judge"]()
                    low_preds = [predictions[i] for i in low_indices]
                    low_refs = [references[i] for i in low_indices]
                    low_sources = ["(document image)" for _ in low_indices]
                    judge_result = await judge_eval.evaluate_async(
                        client=judge_client,
                        judge_model=self.config["judge"]["model_id"],
                        predictions=low_preds,
                        references=low_refs,
                        sources=low_sources,
                    )
                    result["judge_fallback"] = {
                        "count": len(low_indices),
                        "mean_score": judge_result["mean_score"],
                    }
            return result
        else:
            return evaluator.evaluate(predictions, references)

    async def run(
        self, dataset_filter: list[str] | None = None,
        model_filter: list[str] | None = None,
        sample_override: int | None = None,
    ):
        results = {}
        timestamp = datetime.now().isoformat(timespec="seconds")
        temperatures = self.settings["temperatures"]
        stochastic_runs = self.settings["stochastic_runs"]

        judge_client = None
        judge_cfg = self.config.get("judge", {})
        if judge_cfg.get("base_url"):
            judge_client = BenchmarkClient(
                base_url=judge_cfg["base_url"],
                api_key=judge_cfg.get("api_key", ""),
                timeout=self.settings["timeout"],
                max_concurrent=2,
            )

        for ds_name, ds_config in self.config["datasets"].items():
            if not ds_config.get("enabled", False):
                continue
            if dataset_filter and ds_name not in dataset_filter:
                continue

            print(f"\n[{ds_name.upper()}]")

            ds_cls = DATASET_REGISTRY.get(ds_name)
            if not ds_cls:
                print(f"  WARNING: dataset '{ds_name}' not registered, skipping")
                continue

            dataset = ds_cls(ds_config)
            vision_only = ds_config.get("vision_only", False)
            models = self._filter_models(vision_only)

            if model_filter:
                models = [m for m in models if m["name"] in model_filter]

            n_samples = sample_override or ds_config.get("samples", 100)
            samples = dataset.load_samples(n=n_samples, seed=42)

            results[ds_name] = {}

            for model in models:
                model_name = model["name"]
                client = BenchmarkClient(
                    base_url=model["base_url"],
                    api_key="",
                    timeout=self.settings["timeout"],
                    max_concurrent=self.settings["concurrent_requests"],
                )

                results[ds_name][model_name] = {}

                for temp in temperatures:
                    runs_needed = 1 if temp == 0.0 else stochastic_runs

                    if runs_needed == 1:
                        predictions = await self._run_model_on_samples(
                            client, model_name, samples, temp,
                        )
                        eval_result = await self._evaluate(
                            ds_name, ds_config, ds_config["evaluator"],
                            predictions, samples, judge_client,
                        )
                        results[ds_name][model_name][f"temperature_{temp}"] = eval_result
                        self._print_result(ds_name, model_name, temp, eval_result)
                    else:
                        run_results = []
                        for run_i in range(runs_needed):
                            predictions = await self._run_model_on_samples(
                                client, model_name, samples, temp,
                            )
                            eval_result = await self._evaluate(
                                ds_name, ds_config, ds_config["evaluator"],
                                predictions, samples, judge_client,
                            )
                            run_results.append(eval_result)

                        agg = self._aggregate_runs(run_results)
                        results[ds_name][model_name][f"temperature_{temp}"] = agg
                        self._print_result(ds_name, model_name, temp, agg)

        # Save results
        output = {
            "timestamp": timestamp,
            "config_snapshot": self.config,
            "results": results,
        }
        filename = f"{timestamp.replace(':', '-')}_benchmark.json"
        filepath = os.path.join(self.settings["results_dir"], filename)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)

        print(f"\nResults saved: {filepath}")

    def _aggregate_runs(self, run_results: list[dict]) -> dict:
        """Aggregate multiple stochastic runs into mean ± std."""
        score_key = self._find_score_key(run_results[0])
        scores = [r[score_key] for r in run_results]
        return {
            "mean_score": statistics.mean(scores),
            "std": statistics.stdev(scores) if len(scores) > 1 else 0.0,
            "runs": scores,
            "total": run_results[0].get("total", 0),
            "errors": sum(r.get("errors", 0) for r in run_results),
        }

    def _find_score_key(self, result: dict) -> str:
        for key in ("score", "f1", "anls", "mean_score"):
            if key in result:
                return key
        return "score"

    def _print_result(self, ds_name: str, model_name: str, temp: float, result: dict):
        score_key = self._find_score_key(result)
        score = result.get("mean_score", result.get(score_key, 0))
        errors = result.get("errors", 0)
        total = result.get("total", 0)

        if "std" in result:
            std = result["std"]
            print(f"  {model_name:<25} t={temp}: {score:.1%} (+/-{std:.1%})")
        elif score_key in ("f1", "em"):
            f1 = result.get("f1", 0)
            em = result.get("em", 0)
            print(f"  {model_name:<25} t={temp}: F1={f1:.1%} EM={em:.1%}")
        elif score_key == "mean_score":
            print(f"  {model_name:<25} t={temp}: {score:.1f}/5.0 (judge)")
        elif score_key == "anls":
            anls = result.get("anls", 0)
            print(f"  {model_name:<25} t={temp}: {anls:.1%} ANLS")
        else:
            correct = result.get("correct", 0)
            error_str = f", {errors} errors" if errors else ""
            print(f"  {model_name:<25} t={temp}: {score:.1%} ({correct}/{total}{error_str})")


def main():
    parser = argparse.ArgumentParser(description="LLM Benchmark Runner")
    parser.add_argument(
        "--config", default="benchmark/config.yaml",
        help="Path to benchmark config (default: benchmark/config.yaml)",
    )
    parser.add_argument("--dataset", nargs="+", help="Run only these datasets")
    parser.add_argument("--model", nargs="+", help="Run only these models")
    parser.add_argument("--samples", type=int, help="Override sample count")
    args = parser.parse_args()

    runner = BenchmarkRunner(args.config)
    asyncio.run(runner.run(
        dataset_filter=args.dataset,
        model_filter=args.model,
        sample_override=args.samples,
    ))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/separk/workspace/infra && python -m pytest tests/benchmark/test_run.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add benchmark/run.py tests/benchmark/test_run.py
git commit -m "feat(benchmark): add CLI runner with multi-temp and stochastic support"
```

---

### Task 10: Integration Smoke Test

**Files:**
- Modify: `tests/benchmark/test_run.py` (append integration test)

- [ ] **Step 1: Write the integration test**

Append to `tests/benchmark/test_run.py`:
```python
class TestIntegration:
    """Smoke test: loads real config, mocks only API calls."""

    def test_full_pipeline_with_mocked_api(self, tmp_path):
        config = {
            "models": [
                {"name": "model-a", "base_url": "http://localhost:8000/v1",
                 "model_id": "test/a", "vision": True},
                {"name": "model-b", "base_url": "http://localhost:8000/v1",
                 "model_id": "test/b", "vision": False},
            ],
            "judge": {"base_url": "http://judge:8080/v1", "model_id": "claude-opus", "api_key": ""},
            "datasets": {
                "kmmlu": {"enabled": True, "samples": 3, "evaluator": "exact_match"},
                "gsm8k": {"enabled": True, "samples": 3, "evaluator": "exact_match"},
                "docvqa": {"enabled": True, "samples": 2, "evaluator": "anls",
                           "vision_only": True, "anls_judge_threshold": 0.5},
            },
            "settings": {
                "temperatures": [0.0],
                "stochastic_runs": 1,
                "concurrent_requests": 2,
                "timeout": 10,
                "max_tokens": 100,
                "results_dir": str(tmp_path / "results"),
            },
        }
        config_path = str(tmp_path / "config.yaml")
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        from benchmark.datasets.base import Sample

        kmmlu_samples = [
            Sample(id="k1", prompt="Q?", reference="A", metadata={}),
            Sample(id="k2", prompt="Q?", reference="B", metadata={}),
            Sample(id="k3", prompt="Q?", reference="C", metadata={}),
        ]
        gsm8k_samples = [
            Sample(id="g1", prompt="Math?", reference="42", metadata={}),
            Sample(id="g2", prompt="Math?", reference="7", metadata={}),
            Sample(id="g3", prompt="Math?", reference="100", metadata={}),
        ]
        docvqa_samples = [
            Sample(id="d1", prompt=[{"type": "text", "text": "Q?"}], reference="yes", metadata={}),
            Sample(id="d2", prompt=[{"type": "text", "text": "Q?"}], reference="no", metadata={}),
        ]

        mock_datasets = {
            "kmmlu": MagicMock(return_value=MagicMock(
                load_samples=MagicMock(return_value=kmmlu_samples),
                name="kmmlu", requires_vision=False,
            )),
            "gsm8k": MagicMock(return_value=MagicMock(
                load_samples=MagicMock(return_value=gsm8k_samples),
                name="gsm8k", requires_vision=False,
            )),
            "docvqa": MagicMock(return_value=MagicMock(
                load_samples=MagicMock(return_value=docvqa_samples),
                name="docvqa", requires_vision=True,
            )),
        }

        with patch("benchmark.run.DATASET_REGISTRY", mock_datasets), \
             patch("benchmark.run.BenchmarkClient") as MockClient:

            mock_instance = MagicMock()
            responses = iter(["A", "B", "C", "42", "7", "100", "A", "B", "C",
                              "42", "7", "100", "yes", "no"])
            mock_instance.generate = AsyncMock(side_effect=lambda **kw: next(responses, "X"))
            MockClient.return_value = mock_instance

            runner = BenchmarkRunner(config_path)
            asyncio.run(runner.run())

            result_files = [f for f in os.listdir(str(tmp_path / "results")) if f.endswith(".json")]
            assert len(result_files) == 1

            with open(str(tmp_path / "results" / result_files[0])) as f:
                data = json.load(f)

            # kmmlu: both models ran
            assert "model-a" in data["results"]["kmmlu"]
            assert "model-b" in data["results"]["kmmlu"]
            # docvqa: only model-a (vision=True), model-b skipped
            assert "model-a" in data["results"]["docvqa"]
            assert "model-b" not in data["results"]["docvqa"]
```

- [ ] **Step 2: Run test to verify it passes**

Run: `cd /home/separk/workspace/infra && python -m pytest tests/benchmark/test_run.py::TestIntegration -v`
Expected: PASS

- [ ] **Step 3: Run all benchmark tests together**

Run: `cd /home/separk/workspace/infra && python -m pytest tests/benchmark/ -v`
Expected: All PASS

- [ ] **Step 4: Commit**

```bash
git add tests/benchmark/test_run.py
git commit -m "test(benchmark): add integration smoke test for full pipeline"
```

---

### Task 11: Final Wiring + .gitignore + README Update

**Files:**
- Modify: `.gitignore` (add benchmark results)
- Verify: `python -m benchmark.run --help` works

- [ ] **Step 1: Update .gitignore**

Append to `.gitignore`:
```
benchmark/results/*.json
```

- [ ] **Step 2: Install dependencies**

Run: `pip install -r benchmark/requirements.txt`

- [ ] **Step 3: Verify CLI works**

Run: `cd /home/separk/workspace/infra && python -m benchmark.run --help`
Expected:
```
usage: run.py [-h] [--config CONFIG] [--dataset DATASET [DATASET ...]] [--model MODEL [MODEL ...]] [--samples SAMPLES]
```

- [ ] **Step 4: Run full test suite (existing + benchmark)**

Run: `cd /home/separk/workspace/infra && python -m pytest tests/ -v`
Expected: All PASS (existing tests + new benchmark tests)

- [ ] **Step 5: Commit**

```bash
git add .gitignore benchmark/
git commit -m "feat(benchmark): complete benchmark framework with all datasets and evaluators"
```
