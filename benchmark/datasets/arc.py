"""ARC (AI2 Reasoning Challenge) — English science reasoning benchmark.

Uses allenai/ai2_arc ARC-Challenge split. Multiple-choice science questions
requiring logical reasoning beyond pattern matching.
"""

import random

from datasets import load_dataset

from benchmark.datasets import register_dataset
from benchmark.datasets.base import BaseDataset, Sample

PROMPT_TEMPLATE = """Answer the following science question.

Question: {question}
{choices}

Reply with ONLY the letter ({labels}). No explanation."""


@register_dataset("arc")
class ARCDataset(BaseDataset):
    @property
    def name(self) -> str:
        return "arc"

    def load_samples(self, n: int, seed: int = 42) -> list[Sample]:
        ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")

        random.seed(seed)
        total = len(ds)
        indices = random.sample(range(total), min(n, total))

        samples = []
        for idx in indices:
            row = ds[idx]
            labels = row["choices"]["label"]
            texts = row["choices"]["text"]
            choices_str = "\n".join(f"{l}. {t}" for l, t in zip(labels, texts))
            labels_str = ", ".join(labels)

            prompt = PROMPT_TEMPLATE.format(
                question=row["question"],
                choices=choices_str,
                labels=labels_str,
            )

            samples.append(Sample(
                id=f"arc_{row['id']}",
                prompt=prompt,
                reference=row["answerKey"],
                metadata={"index": idx},
            ))
        return samples
