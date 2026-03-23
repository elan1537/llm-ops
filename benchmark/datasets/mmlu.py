"""MMLU (Massive Multitask Language Understanding) — English knowledge benchmark.

Uses cais/mmlu dataset from HuggingFace. 57 subjects covering STEM, humanities,
social sciences, etc. Multiple choice (A/B/C/D).
"""

import random

from datasets import load_dataset, get_dataset_config_names, concatenate_datasets

from benchmark.datasets import register_dataset
from benchmark.datasets.base import BaseDataset, Sample

INDEX_TO_LETTER = {0: "A", 1: "B", 2: "C", 3: "D"}

PROMPT_TEMPLATE = """Answer the following multiple choice question. Reply with only A, B, C, or D.

Question: {question}
A. {a}
B. {b}
C. {c}
D. {d}

Answer:"""


@register_dataset("mmlu")
class MMLUDataset(BaseDataset):
    @property
    def name(self) -> str:
        return "mmlu"

    def load_samples(self, n: int, seed: int = 42) -> list[Sample]:
        configs = get_dataset_config_names("cais/mmlu")
        configs = [c for c in configs if c not in ("all", "auxiliary_train")]
        splits = []
        for cfg in configs:
            try:
                ds = load_dataset("cais/mmlu", cfg, split="test")
                splits.append(ds)
            except Exception:
                continue
        data = concatenate_datasets(splits)

        random.seed(seed)
        total = len(data)
        indices = random.sample(range(total), min(n, total))

        samples = []
        for idx in indices:
            row = data[idx]
            choices = row["choices"]
            prompt = PROMPT_TEMPLATE.format(
                question=row["question"],
                a=choices[0],
                b=choices[1],
                c=choices[2],
                d=choices[3],
            )
            reference = INDEX_TO_LETTER.get(row["answer"], str(row["answer"]))

            samples.append(Sample(
                id=f"mmlu_{idx}",
                prompt=prompt,
                reference=reference,
                metadata={"index": idx, "subject": row.get("subject", "")},
            ))
        return samples
