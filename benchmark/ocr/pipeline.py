"""PDF OCR Pipeline using vision models via OpenAI-compatible gateway.

Mirrors the DeepSeek OCR approach:
  PDF → page images (PyMuPDF) → vision model → markdown → post-process
"""

import asyncio
import base64
import io
import re
from dataclasses import dataclass, field

import fitz  # PyMuPDF
from PIL import Image

from benchmark.client import BenchmarkClient, GenerateResult


@dataclass
class PageResult:
    page_num: int
    markdown: str
    raw_output: str
    prompt_tokens: int = 0
    completion_tokens: int = 0


@dataclass
class OCRResult:
    pages: list[PageResult] = field(default_factory=list)
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0

    @property
    def full_markdown(self) -> str:
        return "\n\n<--- Page Split --->\n\n".join(p.markdown for p in self.pages)


OCR_PROMPT = """이 문서 이미지를 마크다운으로 변환하세요.
규칙:
- 원본 텍스트를 정확히 그대로 옮기세요
- 제목, 목록, 표 등의 구조를 마크다운으로 표현하세요
- 이미지나 그림은 무시하세요
- 추가 설명이나 해석 없이 텍스트만 출력하세요"""


def pdf_to_images(pdf_path: str, dpi: int = 144) -> list[Image.Image]:
    """Convert PDF pages to PIL Images using PyMuPDF."""
    doc = fitz.open(pdf_path)
    images = []
    zoom = dpi / 72
    matrix = fitz.Matrix(zoom, zoom)

    for page in doc:
        pix = page.get_pixmap(matrix=matrix, alpha=False)
        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        images.append(img)

    doc.close()
    return images


def image_to_base64(image: Image.Image) -> str:
    """Convert PIL Image to base64 data URI."""
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def postprocess(text: str) -> str:
    """Clean up model output to produce clean markdown."""
    # Strip </think> reasoning if present
    if "</think>" in text:
        text = text.split("</think>")[-1]

    # Remove DeepSeek-style reference tags
    text = re.sub(r"<\|ref\|>.*?<\|/ref\|><\|det\|>.*?<\|/det\|>", "", text)

    # Remove special tokens
    text = re.sub(r"<[｜|]end[▁_]of[▁_]sentence[｜|]>", "", text)
    text = re.sub(r"<\|.*?\|>", "", text)

    # Fix LaTeX artifacts
    text = text.replace("\\coloneqq", ":=")
    text = text.replace("\\eqqcolon", "=:")

    # Normalize whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


async def ocr_pdf(
    pdf_path: str,
    client: BenchmarkClient,
    model_name: str,
    prompt: str = OCR_PROMPT,
    max_tokens: int = 4096,
    dpi: int = 144,
) -> OCRResult:
    """Run OCR on a PDF file, returning markdown per page."""
    images = pdf_to_images(pdf_path, dpi=dpi)

    tasks = []
    for img in images:
        b64_uri = image_to_base64(img)
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": b64_uri}},
            ],
        }]
        tasks.append(client.generate(
            model=model_name,
            messages=messages,
            temperature=0.0,
            max_tokens=max_tokens,
        ))

    raw_results = await asyncio.gather(*tasks, return_exceptions=True)

    result = OCRResult()
    for i, r in enumerate(raw_results):
        if isinstance(r, Exception):
            page = PageResult(page_num=i, markdown=f"ERROR: {r}", raw_output=str(r))
        else:
            gen: GenerateResult = r
            cleaned = postprocess(gen.content)
            page = PageResult(
                page_num=i,
                markdown=cleaned,
                raw_output=gen.content,
                prompt_tokens=gen.prompt_tokens,
                completion_tokens=gen.completion_tokens,
            )
            result.total_prompt_tokens += gen.prompt_tokens
            result.total_completion_tokens += gen.completion_tokens
        result.pages.append(page)

    return result
