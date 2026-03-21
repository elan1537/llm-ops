import re

from benchmark.evaluators import register_evaluator


def _normalize(text: str) -> str:
    text = text.strip().upper()
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
