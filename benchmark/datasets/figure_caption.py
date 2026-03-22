"""Figure caption benchmark dataset.

Uses Leonardo6/ArXiv-AI-Figure-Caption: arxiv paper figures + ground truth captions.
Model generates a description, LLM Judge evaluates against the original caption.
"""

import base64
import io
import random

import httpx
from datasets import load_dataset
from PIL import Image

from benchmark.datasets import register_dataset
from benchmark.datasets.base import BaseDataset, Sample


MAX_IMAGE_SIZE = 1024


def _download_image(url: str) -> Image.Image | None:
    """Download image from URL, resize if too large, return PIL Image or None."""
    try:
        response = httpx.get(url, timeout=30, follow_redirects=True)
        response.raise_for_status()
        img = Image.open(io.BytesIO(response.content)).convert("RGB")
        # Resize if either dimension exceeds MAX_IMAGE_SIZE
        w, h = img.size
        if max(w, h) > MAX_IMAGE_SIZE:
            scale = MAX_IMAGE_SIZE / max(w, h)
            img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        return img
    except Exception:
        return None


def _image_to_base64(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


@register_dataset("figure_caption")
class FigureCaptionDataset(BaseDataset):
    @property
    def name(self) -> str:
        return "figure_caption"

    @property
    def requires_vision(self) -> bool:
        return True

    def load_samples(self, n: int, seed: int = 42) -> list[Sample]:
        ds = load_dataset("Leonardo6/ArXiv-AI-Figure-Caption", split="train", streaming=True)

        # Collect more candidates than needed (some images may fail to download)
        random.seed(seed)
        candidates = []
        for i, row in enumerate(ds):
            if i >= n * 3:  # over-sample to handle download failures
                break
            candidates.append(row)

        random.shuffle(candidates)

        samples = []
        for row in candidates:
            if len(samples) >= n:
                break

            img = _download_image(row["image"])
            if img is None:
                continue

            b64_uri = _image_to_base64(img)
            prompt = [
                {"type": "text", "text": "Describe this figure in exactly one sentence."},
                {"type": "image_url", "image_url": {"url": b64_uri}},
            ]

            # Clean caption: remove "Figure N:" prefix
            caption = row["text"]
            if caption.startswith("Figure"):
                colon_idx = caption.find(":")
                if colon_idx != -1 and colon_idx < 15:
                    caption = caption[colon_idx + 1:]

            samples.append(Sample(
                id=f"figcap_{row['paper_id']}_fig{row['figure_idx']}",
                prompt=prompt,
                reference=caption.strip(),
                metadata={
                    "paper_id": row["paper_id"],
                    "figure_idx": row["figure_idx"],
                    "image_url": row["image"],
                    "source": caption.strip(),  # for LLM Judge
                },
            ))

        return samples
