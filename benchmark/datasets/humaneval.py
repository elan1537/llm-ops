"""HumanEval — Code generation benchmark.

Uses openai/openai_humaneval. Model generates Python function body,
evaluated by running test cases (pass@1).
"""

import random
import re

from datasets import load_dataset

from benchmark.datasets import register_dataset
from benchmark.datasets.base import BaseDataset, Sample

PROMPT_TEMPLATE = """Complete the following Python function. Output ONLY the function body code, no explanation.

{prompt}"""


@register_dataset("humaneval")
class HumanEvalDataset(BaseDataset):
    @property
    def name(self) -> str:
        return "humaneval"

    def load_samples(self, n: int, seed: int = 42) -> list[Sample]:
        ds = load_dataset("openai/openai_humaneval", split="test")

        random.seed(seed)
        total = len(ds)
        indices = random.sample(range(total), min(n, total))

        samples = []
        for idx in indices:
            row = ds[idx]
            prompt = PROMPT_TEMPLATE.format(prompt=row["prompt"])

            samples.append(Sample(
                id=row["task_id"],
                prompt=prompt,
                reference=row["canonical_solution"],
                metadata={
                    "index": idx,
                    "test": row["test"],
                    "entry_point": row["entry_point"],
                    "function_prompt": row["prompt"],
                },
            ))
        return samples
