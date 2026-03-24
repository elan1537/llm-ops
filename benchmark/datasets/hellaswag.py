"""HellaSwag — English commonsense reasoning benchmark.

Uses Rowan/hellaswag validation split. Tests ability to predict
the most plausible continuation of a scenario.
"""

import random

from datasets import load_dataset

from benchmark.datasets import register_dataset
from benchmark.datasets.base import BaseDataset, Sample

INDEX_TO_LETTER = {0: "A", 1: "B", 2: "C", 3: "D"}

PROMPT_TEMPLATE = """Choose the most plausible continuation of this scenario.

{context}

A. {a}
B. {b}
C. {c}
D. {d}

Reply with ONLY the letter (A, B, C, or D). No explanation."""


@register_dataset("hellaswag")
class HellaSwagDataset(BaseDataset):
    @property
    def name(self) -> str:
        return "hellaswag"

    def load_samples(self, n: int, seed: int = 42) -> list[Sample]:
        ds = load_dataset("Rowan/hellaswag", split="validation")

        random.seed(seed)
        total = len(ds)
        indices = random.sample(range(total), min(n, total))

        samples = []
        for idx in indices:
            row = ds[idx]
            endings = row["endings"]
            reference = INDEX_TO_LETTER.get(int(row["label"]), str(row["label"]))

            prompt = PROMPT_TEMPLATE.format(
                context=row["ctx"],
                a=endings[0],
                b=endings[1],
                c=endings[2],
                d=endings[3],
            )

            samples.append(Sample(
                id=f"hellaswag_{idx}",
                prompt=prompt,
                reference=reference,
                metadata={"index": idx, "activity": row["activity_label"]},
            ))
        return samples
