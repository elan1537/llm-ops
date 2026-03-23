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

    # 3. Last non-empty line
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    if lines:
        return lines[-1].strip('"').strip("'")

    return text


def _tokenize(text: str) -> list[str]:
    text = text.strip().lower()
    text = re.sub(r"[^\w\s]", " ", text)
    tokens = text.split()
    # For CJK/Korean text, use character-level tokenization to handle morphology
    if any(re.search(r"[\u1100-\u11ff\uac00-\ud7af\u4e00-\u9fff]", t) for t in tokens):
        return list(re.sub(r"\s+", "", text))
    return tokens


def _compute_f1(prediction: str, reference: str) -> float:
    prediction = _extract_answer(prediction)
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
    prediction = _extract_answer(prediction)
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
