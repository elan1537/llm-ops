import re

from benchmark.evaluators import register_evaluator


def _normalize(text: str) -> str:
    text = text.strip()

    # For multiple-choice: extract the final answer letter from verbose/thinking responses
    # Search from the END of the text (most likely location of final answer)
    patterns = [
        # Explicit answer statements (Korean/English)
        r"(?:answer|정답|답)[은는이\s:]*\**\s*([A-D])",
        r"(?:answer|정답|답)\s*(?:is|는|은)\s*\**\s*([A-D])",
        # "The correct answer is X" / "정답: X"
        r"(?:correct\s+answer|최종\s*답|결론)[은는이\s:]*\**\s*([A-D])\b",
        # Standalone letter at the very end (common pattern)
        r"\b([A-D])\s*[\.\)]*\s*$",
        # **A** or **B** markdown bold (common in thinking models)
        r"\*\*([A-D])\*\*",
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE | re.MULTILINE)
        if matches:
            return matches[-1].upper()  # Take the LAST match (final answer)

    # Fallback: if text is very short and contains a single A-D letter
    if len(text) < 20:
        match = re.search(r"\b([A-D])\b", text, re.IGNORECASE)
        if match:
            return match.group(1).upper()

    # For numeric answers (GSM8K): extract the last standalone number from the text
    # Models typically output reasoning then the final number at the end
    # Look for explicit final answer patterns first
    final_num_patterns = [
        r"(?:답|answer|결과|결론|따라서|그러므로)[은는이\s:]*\**\s*([-+]?\d[\d,]*\.?\d*)",
        r"(?:=\s*)([-+]?\d[\d,]*\.?\d*)\s*$",
    ]
    for pattern in final_num_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE | re.MULTILINE)
        if matches:
            return matches[-1].replace(",", "")

    # Fallback: last number in the text
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
