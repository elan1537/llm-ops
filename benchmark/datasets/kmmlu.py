import random

from datasets import load_dataset

from benchmark.datasets import register_dataset
from benchmark.datasets.base import BaseDataset, Sample

INDEX_TO_LETTER = {1: "A", 2: "B", 3: "C", 4: "D", "1": "A", "2": "B", "3": "C", "4": "D"}

PROMPT_TEMPLATE = """다음 질문에 대해 A, B, C, D 중 하나만 답하세요.

질문: {question}
A. {a}
B. {b}
C. {c}
D. {d}

정답:"""


@register_dataset("kmmlu")
class KMMLUDataset(BaseDataset):
    @property
    def name(self) -> str:
        return "kmmlu"

    def load_samples(self, n: int, seed: int = 42) -> list[Sample]:
        ds = load_dataset("HAERAE-HUB/KMMLU", "all")
        data = ds["test"]

        random.seed(seed)
        total = len(data)
        indices = random.sample(range(total), min(n, total))

        samples = []
        for idx in indices:
            row = data[idx]
            prompt = PROMPT_TEMPLATE.format(
                question=row["input"],
                a=row["A"],
                b=row["B"],
                c=row["C"],
                d=row["D"],
            )
            ref_raw = row["output"]
            reference = INDEX_TO_LETTER.get(ref_raw, str(ref_raw).strip().upper())

            samples.append(Sample(
                id=f"kmmlu_{idx}",
                prompt=prompt,
                reference=reference,
                metadata={"index": idx},
            ))
        return samples
