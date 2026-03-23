# Benchmark Reliability Improvements Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix benchmark measurement reliability — standardize prompts, harden evaluators, add result analysis CLI, and unify API client structure.

**Architecture:** Four independent improvements: (1) common thinking-strip utility shared by all evaluators, (2) stronger answer extraction in evaluators, (3) stricter output-format prompts in all datasets, (4) `create_client()` factory + `analyze.py` CLI.

**Tech Stack:** Python 3.11, pytest, existing benchmark framework

**Spec:** `docs/superpowers/specs/2026-03-23-benchmark-reliability-design.md`

---

## File Map

```
benchmark/
├── evaluators/
│   ├── common.py           # NEW — strip_thinking(), extract_answer_tag()
│   ├── exact_match.py      # MODIFY — rewrite _normalize with ordered patterns
│   ├── f1_em.py            # MODIFY — rewrite _extract_answer
│   ├── cer.py              # MODIFY — use common.strip_thinking
│   ├── anls.py             # MODIFY — use common.strip_thinking
│   └── llm_judge.py        # (no change)
├── datasets/
│   ├── kmmlu.py            # MODIFY — prompt
│   ├── mmlu.py             # MODIFY — prompt
│   ├── korquad.py          # MODIFY — prompt
│   ├── gsm8k.py            # MODIFY — prompt
│   ├── longbench.py        # MODIFY — prompt
│   └── ruler.py            # MODIFY — prompt
├── client.py               # MODIFY — add create_client() factory
├── run.py                  # MODIFY — use create_client()
└── analyze.py              # NEW — result analysis CLI

tests/benchmark/
├── test_evaluators.py      # MODIFY — add robustness tests
└── test_analyze.py         # NEW
```

---

### Task 1: Common Evaluator Utility

**Files:**
- Create: `benchmark/evaluators/common.py`
- Modify: `benchmark/evaluators/cer.py`
- Modify: `benchmark/evaluators/anls.py`

- [ ] **Step 1: Create common.py**

`benchmark/evaluators/common.py`:
```python
"""Common utilities shared by all evaluators."""

import re


def strip_thinking(text: str) -> str:
    """Remove <think>...</think> blocks and </think> tags from model output."""
    # Remove full <think>...</think> blocks
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    # Handle unclosed <think> (take everything after </think>)
    if "</think>" in text:
        text = text.split("</think>")[-1]
    return text.strip()


def extract_answer_tag(text: str) -> str | None:
    """Extract answer from 'Answer: X' format. Returns None if not found."""
    match = re.search(r"(?:Answer|답)\s*:\s*(.+?)(?:\n|$)", text, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return None
```

- [ ] **Step 2: Update cer.py to use common utility**

In `benchmark/evaluators/cer.py`, replace the inline `</think>` stripping:

```python
# Old:
def _normalize_for_cer(text: str) -> str:
    if "</think>" in text:
        text = text.split("</think>")[-1]
    ...

# New:
from benchmark.evaluators.common import strip_thinking

def _normalize_for_cer(text: str) -> str:
    text = strip_thinking(text)
    text = re.sub(r"[#*_`~\[\]()>|]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()
```

- [ ] **Step 3: Update anls.py to use common utility**

In `benchmark/evaluators/anls.py`, add `strip_thinking` to `_anls_score`:

```python
from benchmark.evaluators.common import strip_thinking

def _anls_score(prediction: str, reference: str, threshold: float = 0.5) -> float:
    pred = strip_thinking(prediction).strip().lower()
    ref = reference.strip().lower()
    ...
```

- [ ] **Step 4: Run tests**

Run: `cd /home/separk/workspace/infra && python -m pytest tests/benchmark/ -v`
Expected: All existing tests PASS (no behavior change, just refactor)

- [ ] **Step 5: Commit**

```bash
git add benchmark/evaluators/common.py benchmark/evaluators/cer.py benchmark/evaluators/anls.py
git commit -m "refactor(benchmark): extract common strip_thinking utility for evaluators"
```

---

### Task 2: Harden exact_match Evaluator

**Files:**
- Modify: `benchmark/evaluators/exact_match.py`
- Modify: `tests/benchmark/test_evaluators.py`

- [ ] **Step 1: Add robustness tests**

Append to `tests/benchmark/test_evaluators.py`:
```python
class TestExactMatchRobust:
    """Test _normalize handles various model output styles."""

    def test_answer_tag_format(self):
        ev = ExactMatchEvaluator()
        result = ev.evaluate(predictions=["Answer: B"], references=["B"])
        assert result["score"] == 1.0

    def test_answer_tag_with_number(self):
        ev = ExactMatchEvaluator()
        result = ev.evaluate(predictions=["Answer: 42"], references=["42"])
        assert result["score"] == 1.0

    def test_verbose_with_answer_tag(self):
        ev = ExactMatchEvaluator()
        result = ev.evaluate(
            predictions=["Let me think about this...\n\nThe correct option is C.\n\nAnswer: C"],
            references=["C"],
        )
        assert result["score"] == 1.0

    def test_thinking_tags_stripped(self):
        ev = ExactMatchEvaluator()
        result = ev.evaluate(
            predictions=["<think>reasoning here about option A and B</think>B"],
            references=["B"],
        )
        assert result["score"] == 1.0

    def test_claude_style_verbose(self):
        """Claude tends to say 'The answer is X because...'"""
        ev = ExactMatchEvaluator()
        result = ev.evaluate(
            predictions=["The answer is D because the formula requires..."],
            references=["D"],
        )
        assert result["score"] == 1.0

    def test_number_with_units_ignored(self):
        ev = ExactMatchEvaluator()
        result = ev.evaluate(
            predictions=["Answer: 84"], references=["84"],
        )
        assert result["score"] == 1.0

    def test_number_extraction_from_verbose_math(self):
        ev = ExactMatchEvaluator()
        result = ev.evaluate(
            predictions=["Step 1: 5*6=30\nStep 2: 30+54=84\n\nAnswer: 84"],
            references=["84"],
        )
        assert result["score"] == 1.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/benchmark/test_evaluators.py::TestExactMatchRobust -v`
Expected: Some FAIL (especially thinking_tags_stripped, claude_style)

- [ ] **Step 3: Rewrite _normalize in exact_match.py**

```python
import re

from benchmark.evaluators import register_evaluator
from benchmark.evaluators.common import strip_thinking, extract_answer_tag


def _normalize(text: str) -> str:
    """Extract the final answer from model output. Ordered by specificity."""
    text = strip_thinking(text)
    text = text.strip()

    if not text:
        return ""

    # 1. Explicit "Answer: X" tag (highest priority — our prompt format)
    answer = extract_answer_tag(text)
    if answer:
        # If answer is a single letter, return it
        if len(answer) == 1 and answer.upper() in "ABCD":
            return answer.upper()
        # If answer is numeric, return cleaned number
        num_match = re.match(r"^[-+]?\d[\d,]*\.?\d*$", answer.replace(",", ""))
        if num_match:
            return answer.replace(",", "")
        return answer

    # 2. Multiple-choice: explicit answer statements
    mc_patterns = [
        r"(?:answer|정답|답)[은는이\s:]*\**\s*([A-D])",
        r"(?:correct\s+answer|최종\s*답|결론)[은는이\s:]*\**\s*([A-D])\b",
        r"\*\*([A-D])\*\*",
        r"\b([A-D])\s*[\.\)]*\s*$",
    ]
    for pattern in mc_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE | re.MULTILINE)
        if matches:
            return matches[-1].upper()

    # 3. Short text with single letter
    if len(text) < 20:
        match = re.search(r"\b([A-D])\b", text, re.IGNORECASE)
        if match:
            return match.group(1).upper()

    # 4. Numeric: explicit final answer patterns
    num_patterns = [
        r"(?:답|answer|결과|결론|따라서|그러므로)[은는이\s:]*\**\s*([-+]?\d[\d,]*\.?\d*)",
        r"(?:=\s*)([-+]?\d[\d,]*\.?\d*)\s*$",
    ]
    for pattern in num_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE | re.MULTILINE)
        if matches:
            return matches[-1].replace(",", "")

    # 5. Fallback: last number in text
    numbers = re.findall(r"[-+]?\d[\d,]*\.?\d*", text)
    if numbers:
        return numbers[-1].replace(",", "")

    return text.strip().upper()


@register_evaluator("exact_match")
class ExactMatchEvaluator:
    def evaluate(self, predictions: list[str], references: list[str]) -> dict:
        correct = 0
        details = []
        for pred, ref in zip(predictions, references):
            pred_norm = _normalize(pred)
            ref_norm = _normalize(ref)
            match = pred_norm == ref_norm
            if match:
                correct += 1
            details.append({
                "prediction": pred,
                "reference": ref,
                "extracted": pred_norm,
                "correct": match,
            })

        total = len(references)
        return {
            "score": correct / total if total else 0.0,
            "correct": correct,
            "total": total,
            "details": details,
        }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/benchmark/test_evaluators.py -v`
Expected: All PASS (old + new tests)

- [ ] **Step 5: Commit**

```bash
git add benchmark/evaluators/exact_match.py tests/benchmark/test_evaluators.py
git commit -m "feat(benchmark): harden exact_match evaluator with ordered pattern extraction"
```

---

### Task 3: Harden f1_em Evaluator

**Files:**
- Modify: `benchmark/evaluators/f1_em.py`
- Modify: `tests/benchmark/test_evaluators.py`

- [ ] **Step 1: Add robustness tests**

Append to `tests/benchmark/test_evaluators.py`:
```python
class TestF1EMRobust:
    def test_answer_tag_extraction(self):
        ev = F1EMEvaluator()
        result = ev.evaluate(
            predictions=["답: 블로그"],
            references=["블로그"],
        )
        assert result["em"] == 1.0

    def test_thinking_stripped(self):
        ev = F1EMEvaluator()
        result = ev.evaluate(
            predictions=["<think>Let me analyze the context...</think>블로그"],
            references=["블로그"],
        )
        assert result["f1"] == 1.0

    def test_quoted_answer(self):
        ev = F1EMEvaluator()
        result = ev.evaluate(
            predictions=['"블로그"'],
            references=["블로그"],
        )
        assert result["f1"] == 1.0

    def test_verbose_with_answer_at_end(self):
        ev = F1EMEvaluator()
        result = ev.evaluate(
            predictions=["Based on the context, the answer is 베냐민 네타냐후."],
            references=["베냐민 네타냐후"],
        )
        assert result["f1"] > 0.5
```

- [ ] **Step 2: Rewrite _extract_answer in f1_em.py**

```python
import re
from collections import Counter

from benchmark.evaluators import register_evaluator
from benchmark.evaluators.common import strip_thinking, extract_answer_tag


def _extract_answer(text: str) -> str:
    """Extract the final answer from verbose model output."""
    text = strip_thinking(text)
    text = text.strip()

    if not text:
        return ""

    # 1. Explicit "Answer: X" or "답: X" tag
    answer = extract_answer_tag(text)
    if answer:
        return answer.strip('"').strip("'")

    # 2. Quoted answer
    quoted = re.findall(r'["\']([^"\']+)["\']', text)
    if quoted and len(quoted[-1]) < 100:
        return quoted[-1]

    # 3. Last non-empty line (skip reasoning prefixes)
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    if lines:
        return lines[-1].strip('"').strip("'")

    return text
```

Keep `_tokenize`, `_compute_f1`, `_compute_em`, `F1EMEvaluator` as-is (they already call `_extract_answer`).

- [ ] **Step 3: Run tests**

Run: `python -m pytest tests/benchmark/test_evaluators.py -v`
Expected: All PASS

- [ ] **Step 4: Commit**

```bash
git add benchmark/evaluators/f1_em.py tests/benchmark/test_evaluators.py
git commit -m "feat(benchmark): harden f1_em answer extraction with common utility"
```

---

### Task 4: Standardize Dataset Prompts

**Files:**
- Modify: `benchmark/datasets/kmmlu.py`
- Modify: `benchmark/datasets/mmlu.py`
- Modify: `benchmark/datasets/korquad.py`
- Modify: `benchmark/datasets/gsm8k.py`
- Modify: `benchmark/datasets/longbench.py`
- Modify: `benchmark/datasets/ruler.py`

- [ ] **Step 1: Update KMMLU prompt**

In `benchmark/datasets/kmmlu.py`, change `PROMPT_TEMPLATE`:

```python
PROMPT_TEMPLATE = """다음 질문에 대해 A, B, C, D 중 하나만 답하세요.

질문: {question}
A. {a}
B. {b}
C. {c}
D. {d}

Reply with ONLY the letter (A, B, C, or D). No explanation."""
```

- [ ] **Step 2: Update MMLU prompt**

In `benchmark/datasets/mmlu.py`, change `PROMPT_TEMPLATE`:

```python
PROMPT_TEMPLATE = """Answer the following multiple choice question.

Question: {question}
A. {a}
B. {b}
C. {c}
D. {d}

Reply with ONLY the letter (A, B, C, or D). No explanation."""
```

- [ ] **Step 3: Update KorQuAD prompt**

In `benchmark/datasets/korquad.py`, change `PROMPT_TEMPLATE`:

```python
PROMPT_TEMPLATE = """다음 문맥을 읽고 질문에 답하세요.

문맥: {context}

질문: {question}

답만 작성하세요. 설명이나 문장으로 감싸지 마세요."""
```

- [ ] **Step 4: Update GSM8K prompt**

In `benchmark/datasets/gsm8k.py`, change `PROMPT_TEMPLATE`:

```python
PROMPT_TEMPLATE = """다음 수학 문제를 단계별로 풀어주세요.

문제: {question}

풀이 후 마지막 줄에 'Answer: [숫자]' 형식으로만 작성하세요."""
```

- [ ] **Step 5: Update LongBench prompt**

In `benchmark/datasets/longbench.py`, change `PROMPT_TEMPLATE`:

```python
PROMPT_TEMPLATE = """Read the following document and answer the question.

{context}

Question: {question}
A. {a}
B. {b}
C. {c}
D. {d}

Reply with ONLY the letter (A, B, C, or D). No explanation."""
```

- [ ] **Step 6: Update RULER prompt**

In `benchmark/datasets/ruler.py`, change the prompt in `load_samples`:

```python
prompt = (
    f"Read the following document carefully and answer the question at the end.\n\n"
    f"{context}\n\n"
    f"Question: {needle_info['question']}\n"
    f"Answer concisely with only the answer, no explanation:"
)
```

- [ ] **Step 7: Run all tests**

Run: `python -m pytest tests/benchmark/ -v`
Expected: All PASS (prompt changes don't break mock-based tests)

- [ ] **Step 8: Commit**

```bash
git add benchmark/datasets/kmmlu.py benchmark/datasets/mmlu.py benchmark/datasets/korquad.py benchmark/datasets/gsm8k.py benchmark/datasets/longbench.py benchmark/datasets/ruler.py
git commit -m "feat(benchmark): standardize dataset prompts with strict output format"
```

---

### Task 5: Client Factory + run.py Refactor

**Files:**
- Modify: `benchmark/client.py`
- Modify: `benchmark/run.py`

- [ ] **Step 1: Add create_client factory to client.py**

Append to `benchmark/client.py`:

```python
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
```

- [ ] **Step 2: Update run.py imports and client creation**

In `benchmark/run.py`:

```python
from benchmark.client import BenchmarkClient, ClaudeNativeClient, GenerateResult, create_client
```

Replace the inline client creation block:
```python
# Old:
if model.get("api_type") == "claude":
    client = ClaudeNativeClient(...)
else:
    client = BenchmarkClient(...)

# New:
client = create_client(model, self.settings)
```

- [ ] **Step 3: Run tests**

Run: `python -m pytest tests/benchmark/ -v`
Expected: All PASS

- [ ] **Step 4: Commit**

```bash
git add benchmark/client.py benchmark/run.py
git commit -m "refactor(benchmark): add create_client factory, simplify run.py"
```

---

### Task 6: Result Analysis CLI

**Files:**
- Create: `benchmark/analyze.py`

- [ ] **Step 1: Implement analyze.py**

`benchmark/analyze.py`:
```python
"""Benchmark result analysis CLI.

Usage: python -m benchmark.analyze <results.json> [--dataset NAME] [--model NAME]
"""

import argparse
import json
import re
import sys

from benchmark.evaluators.common import strip_thinking


def _re_extract(text: str) -> str:
    """Re-run answer extraction on raw prediction for analysis."""
    text = strip_thinking(text).strip()
    # Try Answer: tag
    m = re.search(r"(?:Answer|답)\s*:\s*(.+?)(?:\n|$)", text, re.IGNORECASE)
    if m:
        return f"tag:{m.group(1).strip()}"
    # Try single letter
    if len(text) < 5 and re.match(r"^[A-D]$", text.strip(), re.IGNORECASE):
        return f"clean:{text.strip().upper()}"
    # Last line
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    if lines:
        last = lines[-1][:50]
        return f"last_line:{last}"
    return f"raw:{text[:50]}"


def _classify_prediction(pred: str) -> str:
    """Classify prediction into pattern categories."""
    pred = strip_thinking(pred).strip()
    if not pred:
        return "empty"
    if pred.startswith("ERROR:"):
        return "error"
    if re.match(r"^[A-D]\.?$", pred.strip(), re.IGNORECASE):
        return "clean_letter"
    if re.match(r"^[-+]?\d[\d,]*\.?\d*$", pred.strip()):
        return "clean_number"
    if re.search(r"(?:Answer|답)\s*:", pred, re.IGNORECASE):
        return "has_answer_tag"
    if len(pred) < 50:
        return "short_verbose"
    return "long_verbose"


def analyze_dataset(ds_name: str, model_results: dict, model_name: str):
    """Analyze results for one dataset-model pair."""
    for temp_key, result in model_results.items():
        temp = temp_key.replace("temperature_", "t=")
        details = result.get("details", [])
        if not details:
            continue

        # Determine score
        score_keys = ["score", "f1", "anls", "cer", "mean_score"]
        score_val = None
        for sk in score_keys:
            if sk in result:
                score_val = result[sk]
                break

        total = result.get("total", len(details))
        correct_count = sum(1 for d in details if d.get("correct", False))
        error_count = sum(1 for d in details if "error" in d)

        print(f"\n=== {ds_name.upper()} ({model_name}, {temp}) ===")
        if score_val is not None:
            if isinstance(score_val, float) and score_val <= 1.0:
                print(f"Score: {score_val:.1%} ({correct_count}/{total})")
            else:
                print(f"Score: {score_val}")

        # Wrong answers
        wrong = [d for d in details if not d.get("correct", True) and "error" not in d]
        if wrong:
            print(f"\nWrong answers ({len(wrong)}):")
            for i, d in enumerate(wrong[:10]):
                pred_short = str(d.get("prediction", ""))[:80].replace("\n", " ")
                ref = str(d.get("reference", ""))[:40]
                extracted = d.get("extracted", "?")
                print(f"  #{i+1}  Ref: {ref:<35} Extracted: {extracted}")
                print(f"       Pred: {pred_short}")

        # Errors
        errors = [d for d in details if "error" in d]
        if errors:
            print(f"\nErrors ({len(errors)}):")
            for d in errors[:5]:
                print(f"  {str(d['error'])[:100]}")

        # Pattern statistics
        preds = [d.get("prediction", "") for d in details]
        patterns = {}
        for p in preds:
            cat = _classify_prediction(p)
            patterns[cat] = patterns.get(cat, 0) + 1

        print(f"\nPattern stats:")
        for cat, count in sorted(patterns.items(), key=lambda x: -x[1]):
            print(f"  {cat:<20} {count}/{total} ({count/total:.0%})")

        # Potential eval issues: reference found in prediction but marked wrong
        potential_issues = []
        for d in details:
            if not d.get("correct", True):
                ref = str(d.get("reference", "")).lower()
                pred = strip_thinking(str(d.get("prediction", ""))).lower()
                if ref and ref in pred:
                    potential_issues.append(d)

        if potential_issues:
            print(f"\nPotential eval issues ({len(potential_issues)} — ref found in pred but scored wrong):")
            for d in potential_issues[:5]:
                ref = str(d.get("reference", ""))[:30]
                extracted = d.get("extracted", "?")
                print(f"  Ref: {ref}  Extracted: {extracted}")


def main():
    parser = argparse.ArgumentParser(description="Analyze benchmark results")
    parser.add_argument("results_file", help="Path to results JSON file")
    parser.add_argument("--dataset", help="Analyze only this dataset")
    parser.add_argument("--model", help="Analyze only this model")
    args = parser.parse_args()

    with open(args.results_file) as f:
        data = json.load(f)

    results = data.get("results", {})

    for ds_name, ds_results in results.items():
        if args.dataset and ds_name != args.dataset:
            continue
        for model_name, model_results in ds_results.items():
            if args.model and model_name != args.model:
                continue
            analyze_dataset(ds_name, model_results, model_name)

    print()


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Test with existing results**

Run: `python -m benchmark.analyze benchmark/results/2026-03-23T18-42-16_think_benchmark.json --dataset kmmlu`
Expected: Shows wrong answers, pattern stats, potential eval issues for KMMLU

- [ ] **Step 3: Commit**

```bash
git add benchmark/analyze.py
git commit -m "feat(benchmark): add result analysis CLI for debugging eval issues"
```

---

### Task 7: Final Verification

- [ ] **Step 1: Run full test suite**

Run: `python -m pytest tests/benchmark/ -v`
Expected: All PASS

- [ ] **Step 2: Quick smoke test with real model**

Run: `JUDGE_API_KEY=... python -m benchmark.run --dataset kmmlu --samples 5 --no-thinking`
Expected: Results display correctly, no errors

- [ ] **Step 3: Run analyzer on result**

Run: `python -m benchmark.analyze benchmark/results/<latest>.json`
Expected: Pattern stats show higher "clean_letter" or "has_answer_tag" percentages

- [ ] **Step 4: Commit all remaining changes**

```bash
git add -A
git commit -m "feat(benchmark): complete reliability improvements — prompts, evaluators, analyzer, client factory"
```
