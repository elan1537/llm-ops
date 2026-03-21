import random

from datasets import load_dataset

from benchmark.datasets import register_dataset
from benchmark.datasets.base import BaseDataset, Sample

PROMPT_TEMPLATE = """다음 텍스트를 핵심 내용 위주로 간결하게 요약하세요.

텍스트:
{text}

요약:"""


@register_dataset("xlsum_ko")
class XLSumKoDataset(BaseDataset):
    @property
    def name(self) -> str:
        return "xlsum_ko"

    def load_samples(self, n: int, seed: int = 42) -> list[Sample]:
        ds = load_dataset("csebuetnlp/xlsum", "korean")
        data = ds["test"]

        random.seed(seed)
        total = len(data)
        indices = random.sample(range(total), min(n, total))

        samples = []
        for idx in indices:
            row = data[idx]
            prompt = PROMPT_TEMPLATE.format(text=row["text"])

            samples.append(Sample(
                id=f"xlsum_ko_{idx}",
                prompt=prompt,
                reference=row["summary"],
                metadata={"index": idx, "source": row["text"]},
            ))
        return samples
