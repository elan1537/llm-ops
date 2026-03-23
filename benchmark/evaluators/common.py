"""Common utilities shared by all evaluators."""

import re


def strip_thinking(text: str) -> str:
    """Remove <think>...</think> blocks and </think> tags from model output."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    if "</think>" in text:
        text = text.split("</think>")[-1]
    return text.strip()


def extract_answer_tag(text: str) -> str | None:
    """Extract answer from 'Answer: X' format. Returns None if not found."""
    match = re.search(r"(?:Answer|답)\s*:\s*(.+?)(?:\n|$)", text, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return None
