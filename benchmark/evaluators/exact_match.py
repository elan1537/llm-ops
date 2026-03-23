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
        if len(answer) == 1 and answer.upper() in "ABCD":
            return answer.upper()
        num_match = re.match(r"^[-+]?\d[\d,]*\.?\d*$", answer.replace(",", ""))
        if num_match:
            return answer.replace(",", "")
        return answer

    # 2. Multiple-choice: explicit answer statements
    mc_patterns = [
        r"(?:answer|정답|답)[은는이\s:]*\**\s*([A-D])",
        r"(?:answer|정답|답)\s+is\s+\**\s*([A-D])\b",
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
