from benchmark.evaluators import register_evaluator
from benchmark.evaluators.common import strip_thinking


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


def _anls_score(prediction: str, reference: str, threshold: float = 0.5) -> float:
    pred = strip_thinking(prediction).lower()
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
