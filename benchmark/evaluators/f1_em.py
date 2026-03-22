import re
from collections import Counter

from benchmark.evaluators import register_evaluator


def _extract_answer(text: str) -> str:
    """Extract the final answer from verbose/thinking model output."""
    # Try to find explicit answer markers
    patterns = [
        r"(?:답|answer|답변)[은는이\s:]+(.+?)(?:\n|$)",
        r"(?:최종\s*답|final\s*answer)[은는이\s:]+(.+?)(?:\n|$)",
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            return matches[-1].strip()

    # If text has clear sections, take the last non-empty line
    lines = [l.strip() for l in text.strip().split("\n") if l.strip()]
    if len(lines) > 3:
        # Skip lines that look like thinking/reasoning
        for line in reversed(lines):
            if not any(line.lower().startswith(p) for p in
                       ["thinking", "process", "step", "1.", "2.", "3.", "*", "-", "#"]):
                return line
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
