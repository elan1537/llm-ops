import re
import random

from datasets import load_dataset

from benchmark.datasets import register_dataset
from benchmark.datasets.base import BaseDataset, Sample

PROMPT_TEMPLATE = """다음 수학 문제를 단계별로 풀어주세요. 최종 답은 마지막 줄에 숫자만 작성하세요.

문제: {question}

풀이:"""


def _extract_answer(answer_text: str) -> str:
    match = re.search(r"####\s*(.+)", answer_text)
    if match:
        return match.group(1).strip().replace(",", "")
    return answer_text.strip()


@register_dataset("gsm8k")
class GSM8KDataset(BaseDataset):
    @property
    def name(self) -> str:
        return "gsm8k"

    def load_samples(self, n: int, seed: int = 42) -> list[Sample]:
        ds = load_dataset("openai/gsm8k", "main")
        data = ds["test"]

        random.seed(seed)
        total = len(data)
        indices = random.sample(range(total), min(n, total))

        samples = []
        for idx in indices:
            row = data[idx]
            prompt = PROMPT_TEMPLATE.format(question=row["question"])
            reference = _extract_answer(row["answer"])

            samples.append(Sample(
                id=f"gsm8k_{idx}",
                prompt=prompt,
                reference=reference,
                metadata={"index": idx, "full_answer": row["answer"]},
            ))
        return samples
