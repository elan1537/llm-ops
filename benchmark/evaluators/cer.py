"""Character Error Rate (CER) evaluator for OCR benchmarking.

CER = edit_distance(prediction, reference) / len(reference)
Lower is better (0.0 = perfect, 1.0+ = very bad).
"""

import re

from benchmark.evaluators import register_evaluator
from benchmark.evaluators.common import strip_thinking


def _normalize_for_cer(text: str) -> str:
    """Normalize text for CER comparison: strip thinking, markdown, normalize whitespace."""
    text = strip_thinking(text)
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
    def evaluate(self, predictions: list[str], references: list[str],
                 metadata: list[dict] | None = None) -> dict:
        scores = []
        details = []
        lang_scores: dict[str, list[float]] = {}

        for i, (pred, ref) in enumerate(zip(predictions, references)):
            cer = compute_cer(pred, ref)
            scores.append(cer)

            lang = "unknown"
            if metadata and i < len(metadata):
                lang = metadata[i].get("lang", "unknown")

            lang_scores.setdefault(lang, []).append(cer)
            details.append({
                "prediction_len": len(pred),
                "reference_len": len(ref),
                "cer": cer,
                "lang": lang,
            })

        n = len(references)
        mean_cer = sum(scores) / n if n else 0.0

        result = {
            "cer": mean_cer,
            "accuracy": 1.0 - min(mean_cer, 1.0),
            "total": n,
            "details": details,
        }

        # Per-language breakdown
        if len(lang_scores) > 1 or "unknown" not in lang_scores:
            per_lang = {}
            for lang, lscores in sorted(lang_scores.items()):
                lmean = sum(lscores) / len(lscores)
                per_lang[lang] = {"cer": lmean, "accuracy": 1.0 - min(lmean, 1.0), "count": len(lscores)}
            result["per_lang"] = per_lang

        return result
