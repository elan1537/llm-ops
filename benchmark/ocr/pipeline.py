"""PDF OCR Pipeline — 2-stage architecture.

Stage 1: DeepSeek-OCR (localhost:8003) — OCR + grounding (bbox for figures)
Stage 2: Qwen3.5 (localhost:8000 gateway) — description for cropped figures

Flow:
  PDF → page images (PyMuPDF)
    → DeepSeek-OCR: text OCR + figure bbox detection
    → crop figures from bbox coordinates
    → Qwen: generate description for each figure
    → final markdown with figure descriptions
"""

import asyncio
import base64
import io
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import fitz  # PyMuPDF
import httpx
from PIL import Image

from benchmark.client import BenchmarkClient, GenerateResult

DEEPSEEK_OCR_URL = os.environ.get("DEEPSEEK_OCR_URL", "http://localhost:8003")

DEEPSEEK_GROUNDING_PROMPT = "<image>\n<|grounding|>Convert the document to markdown."

FIGURE_DESCRIPTION_PROMPT = "/no_think\nDescribe this figure in exactly one sentence."


@dataclass
class FigureInfo:
    page_num: int
    fig_idx: int
    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2 (pixel coords)
    image: Image.Image | None = None
    description: str = ""
    image_path: str = ""


@dataclass
class PageResult:
    page_num: int
    markdown: str
    raw_output: str
    figures: list[FigureInfo] = field(default_factory=list)
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


# --- Utility functions ---

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


# --- Stage 1: DeepSeek-OCR ---

async def _call_deepseek_ocr(pdf_path: str, ocr_url: str = DEEPSEEK_OCR_URL) -> list[str]:
    """Send PDF to DeepSeek-OCR, return raw output per page."""
    async with httpx.AsyncClient(timeout=300) as client:
        with open(pdf_path, "rb") as f:
            response = await client.post(
                f"{ocr_url}/ocr/pdf",
                files={"file": (Path(pdf_path).name, f, "application/pdf")},
                data={"prompt": DEEPSEEK_GROUNDING_PROMPT},
            )
        response.raise_for_status()
        data = response.json()

    if not data.get("success"):
        raise RuntimeError(f"DeepSeek-OCR failed: {data}")

    return [r["result"] for r in data["results"] if r.get("success")]


def _extract_figures(raw_text: str, page_image: Image.Image, page_num: int) -> tuple[str, list[FigureInfo]]:
    """Extract figure bboxes from DeepSeek grounding tags, crop images, clean text."""
    w, h = page_image.size
    figures = []
    cleaned = raw_text

    # Find image references: <|ref|>image<|/ref|><|det|>[[x1,y1,x2,y2]]<|/det|>
    pattern = r'<\|ref\|>image<\|/ref\|><\|det\|>(.*?)<\|/det\|>'
    matches = list(re.finditer(pattern, cleaned))

    for fig_idx, match in enumerate(reversed(matches)):  # reverse to preserve indices
        try:
            coords = eval(match.group(1))
            for points in coords:
                x1, y1, x2, y2 = points
                # Scale from 0-999 normalized to pixel coords
                px1 = int(x1 / 999 * w)
                py1 = int(y1 / 999 * h)
                px2 = int(x2 / 999 * w)
                py2 = int(y2 / 999 * h)

                cropped = page_image.crop((px1, py1, px2, py2))
                fig = FigureInfo(
                    page_num=page_num,
                    fig_idx=fig_idx,
                    bbox=(px1, py1, px2, py2),
                    image=cropped,
                )
                figures.append(fig)

                # Replace tag with placeholder
                placeholder = f"{{{{FIGURE_{page_num}_{fig_idx}}}}}"
                cleaned = cleaned[:match.start()] + placeholder + cleaned[match.end():]
        except Exception:
            continue

    # Remove remaining reference tags
    cleaned = re.sub(r'<\|ref\|>.*?<\|/ref\|><\|det\|>.*?<\|/det\|>', '', cleaned)

    # Clean special tokens
    cleaned = re.sub(r"<[｜|]end[▁_]of[▁_]sentence[｜|]>", "", cleaned)
    cleaned = re.sub(r"<\|.*?\|>", "", cleaned)
    cleaned = cleaned.replace("\\coloneqq", ":=").replace("\\eqqcolon", "=:")
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)

    figures.reverse()  # restore original order
    return cleaned.strip(), figures


# --- Stage 2: Qwen figure description ---

async def _describe_figures(
    figures: list[FigureInfo],
    qwen_client: BenchmarkClient,
    qwen_model: str,
) -> list[FigureInfo]:
    """Send cropped figures to Qwen for description."""
    if not figures:
        return figures

    tasks = []
    for fig in figures:
        b64_uri = image_to_base64(fig.image)
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": FIGURE_DESCRIPTION_PROMPT},
                {"type": "image_url", "image_url": {"url": b64_uri}},
            ],
        }]
        tasks.append(qwen_client.generate(
            model=qwen_model,
            messages=messages,
            temperature=0.0,
            max_tokens=512,
        ))

    results = await asyncio.gather(*tasks, return_exceptions=True)

    for fig, result in zip(figures, results):
        if isinstance(result, Exception):
            fig.description = "(Figure description unavailable)"
        else:
            desc = result.content
            if "</think>" in desc:
                desc = desc.split("</think>")[-1]
            fig.description = _clean_description(desc)


def _clean_description(text: str) -> str:
    """Extract a clean one-sentence description from possibly verbose model output."""
    # Remove markdown bold/italic markers and list bullets
    text = re.sub(r"\*+\s*", "", text)
    # Remove "Draft N:" patterns
    text = re.sub(r"Draft\s*\d+[^:]*:\s*", "", text, flags=re.IGNORECASE)
    # Remove numbered list prefixes
    text = re.sub(r"^\d+\.\s*", "", text, flags=re.MULTILINE)

    lines = [l.strip() for l in text.strip().split("\n") if l.strip()]

    # Find lines that look like actual descriptions (contain subject+verb patterns)
    desc_lines = [l for l in lines if len(l) > 30
                  and not l.lower().startswith(("the user", "let me", "i need", "identify", "analyze", "draft"))]

    if desc_lines:
        # Take the last good description (model refines iteratively)
        chosen = desc_lines[-1]
    elif lines:
        chosen = lines[-1]
    else:
        return ""

    # Take only the first sentence if multiple
    chosen = re.split(r'(?<=[.!?])\s+', chosen)[0]
    return chosen.strip('"').strip("'").strip()

    return figures


# --- Main pipeline ---

async def ocr_pdf(
    pdf_path: str,
    qwen_client: BenchmarkClient,
    qwen_model: str = "qwen3.5-122b",
    ocr_url: str = DEEPSEEK_OCR_URL,
    images_dir: str | None = None,
    dpi: int = 144,
) -> OCRResult:
    """2-stage PDF OCR pipeline.

    Args:
        pdf_path: Path to PDF file.
        qwen_client: BenchmarkClient for Qwen gateway.
        qwen_model: Model name for figure description.
        ocr_url: DeepSeek-OCR server URL.
        images_dir: Directory to save cropped figures. None = don't save.
        dpi: Resolution for PDF rendering.
    """
    # Render pages
    page_images = pdf_to_images(pdf_path, dpi=dpi)

    # Stage 1: DeepSeek-OCR
    raw_outputs = await _call_deepseek_ocr(pdf_path, ocr_url=ocr_url)

    if images_dir:
        os.makedirs(images_dir, exist_ok=True)

    result = OCRResult()
    all_figures: list[FigureInfo] = []

    for page_num, (raw, img) in enumerate(zip(raw_outputs, page_images)):
        cleaned, figures = _extract_figures(raw, img, page_num)

        # Save cropped figures if images_dir provided
        for fig in figures:
            if images_dir and fig.image:
                fname = f"page{page_num}_fig{fig.fig_idx}.jpg"
                fig.image_path = os.path.join(images_dir, fname)
                fig.image.save(fig.image_path)

        all_figures.extend(figures)
        result.pages.append(PageResult(
            page_num=page_num,
            markdown=cleaned,
            raw_output=raw,
            figures=figures,
        ))

    # Stage 2: Qwen figure descriptions
    await _describe_figures(all_figures, qwen_client, qwen_model)

    # Insert descriptions into markdown
    for page in result.pages:
        md = page.markdown
        for fig in page.figures:
            placeholder = f"{{{{FIGURE_{fig.page_num}_{fig.fig_idx}}}}}"
            img_ref = fig.image_path if fig.image_path else f"page{fig.page_num}_fig{fig.fig_idx}.jpg"
            md = md.replace(placeholder, f"![{fig.description}]({img_ref})")
        page.markdown = md

    return result


# --- Simple single-stage pipeline (Qwen only, no DeepSeek-OCR) ---

OCR_PROMPT = """이 문서 이미지를 마크다운으로 변환하세요.
규칙:
- 원본 텍스트를 정확히 그대로 옮기세요
- 제목, 목록, 표 등의 구조를 마크다운으로 표현하세요
- 이미지나 그림은 무시하세요
- 추가 설명이나 해석 없이 텍스트만 출력하세요"""


def postprocess(text: str) -> str:
    """Clean up model output to produce clean markdown."""
    if "</think>" in text:
        text = text.split("</think>")[-1]
    text = re.sub(r"<\|ref\|>.*?<\|/ref\|><\|det\|>.*?<\|/det\|>", "", text)
    text = re.sub(r"<[｜|]end[▁_]of[▁_]sentence[｜|]>", "", text)
    text = re.sub(r"<\|.*?\|>", "", text)
    text = text.replace("\\coloneqq", ":=").replace("\\eqqcolon", "=:")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


async def ocr_pdf_simple(
    pdf_path: str,
    client: BenchmarkClient,
    model_name: str,
    prompt: str = OCR_PROMPT,
    max_tokens: int = 4096,
    dpi: int = 144,
) -> OCRResult:
    """Single-stage OCR using Qwen only (no figure extraction)."""
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
                page_num=i, markdown=cleaned, raw_output=gen.content,
                prompt_tokens=gen.prompt_tokens, completion_tokens=gen.completion_tokens,
            )
            result.total_prompt_tokens += gen.prompt_tokens
            result.total_completion_tokens += gen.completion_tokens
        result.pages.append(page)

    return result
