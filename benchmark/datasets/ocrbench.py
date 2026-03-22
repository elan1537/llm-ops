import base64
import io
import random

from datasets import load_dataset

from benchmark.datasets import register_dataset
from benchmark.datasets.base import BaseDataset, Sample


def _image_to_base64(image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


@register_dataset("ocrbench")
class OCRBenchDataset(BaseDataset):
    @property
    def name(self) -> str:
        return "ocrbench"

    @property
    def requires_vision(self) -> bool:
        return True

    def load_samples(self, n: int, seed: int = 42) -> list[Sample]:
        ds = load_dataset("echo840/ocrbench", split="test")

        random.seed(seed)
        total = len(ds)
        indices = random.sample(range(total), min(n, total))

        samples = []
        for idx in indices:
            row = ds[idx]
            image_uri = _image_to_base64(row["image"])

            prompt = [
                {"type": "text", "text": row["question"]},
                {"type": "image_url", "image_url": {"url": image_uri}},
            ]

            # answer is a list of acceptable answers; use first as primary
            answers = row["answer"]
            reference = answers[0] if answers else ""

            samples.append(Sample(
                id=f"ocrbench_{idx}",
                prompt=prompt,
                reference=reference,
                metadata={
                    "index": idx,
                    "all_answers": answers,
                    "question_type": row["question_type"],
                    "source_dataset": row["dataset"],
                },
            ))
        return samples
