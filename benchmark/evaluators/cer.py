"""Character Error Rate (CER) evaluator for OCR benchmarking.

CER = edit_distance(prediction, reference) / len(reference)
Lower is better (0.0 = perfect, 1.0+ = very bad).
"""

import re

from benchmark.evaluators import register_evaluator


def _normalize_for_cer(text: str) -> str:
    """Normalize text for CER comparison: strip thinking, markdown, normalize whitespace."""
    # Strip </think> reasoning if present
    if "</think>" in text:
        text = text.split("</think>")[-1]
    # Remove markdown formatting
    text = re.sub(r"[#*_`~\[\]()>|]", "", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _edit_distance(s1: str, s2: str) -> int:
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


def compute_cer(prediction: str, reference: str) -> float:
    pred = _normalize_for_cer(prediction)
    ref = _normalize_for_cer(reference)
    if not ref:
        return 0.0 if not pred else 1.0
    return _edit_distance(pred, ref) / len(ref)


@register_evaluator("cer")
class CEREvaluator:
    def evaluate(self, predictions: list[str], references: list[str]) -> dict:
        scores = []
        details = []
        for pred, ref in zip(predictions, references):
            cer = compute_cer(pred, ref)
            scores.append(cer)
            details.append({
                "prediction_len": len(pred),
                "reference_len": len(ref),
                "cer": cer,
            })

        n = len(references)
        mean_cer = sum(scores) / n if n else 0.0
        return {
            "cer": mean_cer,
            "accuracy": 1.0 - min(mean_cer, 1.0),
            "total": n,
            "details": details,
        }
