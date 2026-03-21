import base64
import io
import random

from datasets import load_dataset

from benchmark.datasets import register_dataset
from benchmark.datasets.base import BaseDataset, Sample

QUESTION_PREFIX = "다음 문서 이미지를 보고 질문에 답하세요.\n\n질문: "


def _image_to_base64(image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


@register_dataset("docvqa")
class DocVQADataset(BaseDataset):
    @property
    def name(self) -> str:
        return "docvqa"

    @property
    def requires_vision(self) -> bool:
        return True

    def load_samples(self, n: int, seed: int = 42) -> list[Sample]:
        ds = load_dataset("lmms-lab/DocVQA")
        data = ds["test"]

        random.seed(seed)
        total = len(data)
        indices = random.sample(range(total), min(n, total))

        samples = []
        for idx in indices:
            row = data[idx]
            image_uri = _image_to_base64(row["image"])

            prompt = [
                {"type": "text", "text": f"{QUESTION_PREFIX}{row['question']}"},
                {"type": "image_url", "image_url": {"url": image_uri}},
            ]

            answers = row["answers"]
            reference = answers[0] if answers else ""

            samples.append(Sample(
                id=f"docvqa_{idx}",
                prompt=prompt,
                reference=reference,
                metadata={"index": idx, "all_answers": answers},
            ))
        return samples
