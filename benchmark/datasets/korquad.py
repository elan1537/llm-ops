import random

from datasets import load_dataset

from benchmark.datasets import register_dataset
from benchmark.datasets.base import BaseDataset, Sample

PROMPT_TEMPLATE = """다음 문맥을 읽고 질문에 답하세요. 답은 문맥에서 직접 찾아 간결하게 작성하세요.

문맥: {context}

질문: {question}

답:

답만 작성하세요. 설명이나 문장으로 감싸지 마세요."""


@register_dataset("korquad")
class KorQuADDataset(BaseDataset):
    @property
    def name(self) -> str:
        return "korquad"

    def load_samples(self, n: int, seed: int = 42) -> list[Sample]:
        ds = load_dataset("squad_kor_v1")
        data = ds["validation"]

        random.seed(seed)
        total = len(data)
        indices = random.sample(range(total), min(n, total))

        samples = []
        for idx in indices:
            row = data[idx]
            prompt = PROMPT_TEMPLATE.format(
                context=row["context"],
                question=row["question"],
            )
            reference = row["answers"]["text"][0]

            samples.append(Sample(
                id=f"korquad_{idx}",
                prompt=prompt,
                reference=reference,
                metadata={"index": idx},
            ))
        return samples
