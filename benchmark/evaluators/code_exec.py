"""Code execution evaluator for HumanEval.

Runs model-generated code against test cases to compute pass@1.
"""

import re
import signal
import traceback
from contextlib import contextmanager

from benchmark.evaluators import register_evaluator
from benchmark.evaluators.common import strip_thinking


@contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutError("Execution timed out")
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)


def _extract_code(text: str, function_prompt: str) -> str:
    """Extract function body from model output."""
    text = strip_thinking(text)

    # If model output contains the full function, extract just the body
    # Try to find code blocks
    code_blocks = re.findall(r"```(?:python)?\n(.*?)```", text, re.DOTALL)
    if code_blocks:
        text = code_blocks[-1]

    # If the output includes the function signature, use it as-is
    if "def " in text:
        return text

    # Otherwise, assume it's just the body — indent and prepend to prompt
    lines = text.split("\n")
    indented = "\n".join("    " + l if l.strip() else l for l in lines)
    return function_prompt + indented


def _run_test(code: str, test: str, entry_point: str, timeout: int = 10) -> bool:
    """Run code + test, return True if all assertions pass."""
    full_code = code + "\n\n" + test + f"\n\ncheck({entry_point})"
    try:
        with time_limit(timeout):
            exec(full_code, {})
        return True
    except Exception:
        return False


@register_evaluator("code_exec")
class CodeExecEvaluator:
    def evaluate(self, predictions: list[str], references: list[str],
                 metadata: list[dict] | None = None) -> dict:
        if not metadata:
            return {"score": 0.0, "pass_at_1": 0.0, "total": len(predictions), "details": []}

        passed = 0
        details = []

        for i, (pred, ref) in enumerate(zip(predictions, references)):
            meta = metadata[i] if i < len(metadata) else {}
            test = meta.get("test", "")
            entry_point = meta.get("entry_point", "")
            function_prompt = meta.get("function_prompt", "")

            code = _extract_code(pred, function_prompt)
            success = _run_test(code, test, entry_point)

            if success:
                passed += 1

            details.append({
                "prediction": pred[:200],
                "reference": ref[:200],
                "passed": success,
                "correct": success,
            })

        total = len(predictions)
        pass_rate = passed / total if total else 0.0

        return {
            "score": pass_rate,
            "pass_at_1": pass_rate,
            "passed": passed,
            "total": total,
            "details": details,
        }
