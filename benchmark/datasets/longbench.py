"""LongBench v2 — Long context understanding benchmark.

Uses THUDM/LongBench-v2: multiple-choice QA over very long documents.
Tests model's ability to comprehend and reason over long contexts (4K~2M tokens).
"""

import random

from datasets import load_dataset

from benchmark.datasets import register_dataset
from benchmark.datasets.base import BaseDataset, Sample

PROMPT_TEMPLATE = """Read the following document and answer the question. Choose only A, B, C, or D.

{context}

Question: {question}
A. {a}
B. {b}
C. {c}
D. {d}

Answer:

Reply with ONLY the letter (A, B, C, or D). No explanation."""


@register_dataset("longbench")
class LongBenchDataset(BaseDataset):
    @property
    def name(self) -> str:
        return "longbench"

    def load_samples(self, n: int, seed: int = 42) -> list[Sample]:
        ds = load_dataset("THUDM/LongBench-v2", split="train")

        # Filter by length category if configured
        length_filter = self.config.get("length", None)  # "short", "medium", "long"
        if length_filter:
            ds = ds.filter(lambda x: x["length"] == length_filter)

        random.seed(seed)
        total = len(ds)
        indices = random.sample(range(total), min(n, total))

        samples = []
        for idx in indices:
            row = ds[idx]
            context = row["context"]

            # Truncate context if it exceeds a reasonable limit for the prompt
            max_ctx_chars = self.config.get("max_context_chars", 500000)
            if len(context) > max_ctx_chars:
                context = context[:max_ctx_chars] + "\n\n[... document truncated ...]"

            prompt = PROMPT_TEMPLATE.format(
                context=context,
                question=row["question"],
                a=row["choice_A"],
                b=row["choice_B"],
                c=row["choice_C"],
                d=row["choice_D"],
            )

            samples.append(Sample(
                id=f"longbench_{row['_id']}",
                prompt=prompt,
                reference=row["answer"],
                metadata={
                    "domain": row["domain"],
                    "sub_domain": row["sub_domain"],
                    "difficulty": row["difficulty"],
                    "length": row["length"],
                    "context_chars": len(row["context"]),
                },
            ))
        return samples
