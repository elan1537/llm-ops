"""PDF OCR benchmark dataset.

Two modes:
1. synthetic: Generates PDFs from known text, OCRs them, compares with original
2. directory: Uses user-provided PDFs + ground truth .txt files

For directory mode, place files in a folder like:
  test_docs/
    document1.pdf
    document1.txt   # ground truth text for each page, separated by ---
    document2.pdf
    document2.txt
"""

import os
import random
import tempfile

from benchmark.datasets import register_dataset
from benchmark.datasets.base import BaseDataset, Sample

# Sample texts for synthetic PDF generation (Korean + English mixed)
SYNTHETIC_TEXTS = [
    {
        "title": "회의록",
        "content": (
            "제목: 2024년 1분기 경영 전략 회의\n"
            "일시: 2024년 3월 15일 오후 2시\n"
            "장소: 본사 대회의실\n\n"
            "1. 매출 현황 보고\n"
            "   - 1분기 매출: 152억원 (전년 대비 12% 증가)\n"
            "   - 영업이익: 23억원 (전년 대비 8% 증가)\n"
            "   - 해외 매출 비중: 35%\n\n"
            "2. 신규 사업 계획\n"
            "   - AI 기반 문서 자동화 서비스 런칭 예정 (6월)\n"
            "   - 동남아 시장 진출 검토 중\n"
            "   - R&D 투자 확대: 연매출의 15% 목표"
        ),
    },
    {
        "title": "Technical Specification",
        "content": (
            "Product: LLM Inference Server v2.0\n"
            "Date: 2024-03-20\n\n"
            "System Requirements:\n"
            "- GPU: NVIDIA A100 80GB x4 (NVLink)\n"
            "- RAM: 512GB DDR5 ECC\n"
            "- Storage: 2TB NVMe SSD\n"
            "- Network: 100GbE\n\n"
            "Performance Metrics:\n"
            "- Throughput: 150 tokens/sec (batch=32)\n"
            "- Latency P99: 120ms (single request)\n"
            "- Max concurrent users: 200\n"
            "- GPU utilization target: 92%"
        ),
    },
    {
        "title": "계약서",
        "content": (
            "소프트웨어 라이선스 계약서\n\n"
            "제1조 (목적)\n"
            "본 계약은 갑(주식회사 테크솔루션)이 을(주식회사 데이터랩)에게\n"
            "소프트웨어 라이선스를 부여하는 조건을 정한다.\n\n"
            "제2조 (계약 기간)\n"
            "본 계약의 유효 기간은 2024년 4월 1일부터 2025년 3월 31일까지로 한다.\n\n"
            "제3조 (라이선스 비용)\n"
            "월 라이선스 비용: 5,000,000원 (부가세 별도)\n"
            "연간 총액: 60,000,000원\n"
            "지급 방법: 매월 말일 익월 10일까지 계좌 이체"
        ),
    },
    {
        "title": "Research Abstract",
        "content": (
            "Title: Efficient Large Language Model Serving with Quantization\n\n"
            "Abstract:\n"
            "We present a novel approach to serving large language models (LLMs)\n"
            "using GPTQ-Int4 quantization on NVIDIA A100 GPUs. Our method achieves\n"
            "95% of the original model's accuracy while reducing memory footprint\n"
            "by 75%. Experiments on Korean and English benchmarks show that\n"
            "quantized models maintain strong performance across tasks including\n"
            "question answering (F1=87.3%), summarization (ROUGE-L=42.1%),\n"
            "and mathematical reasoning (GSM8K accuracy=78.5%).\n\n"
            "Keywords: LLM, quantization, GPTQ, inference optimization, vLLM"
        ),
    },
    {
        "title": "데이터 분석 보고서",
        "content": (
            "월간 서버 모니터링 리포트 (2024년 3월)\n\n"
            "1. 서버 가용성\n"
            "   - 전체 가동률: 99.97%\n"
            "   - 계획된 점검: 2회 (총 45분)\n"
            "   - 비계획 장애: 0회\n\n"
            "2. GPU 사용률\n"
            "   - GPU 0,1 (Qwen 122B): 평균 78%, 피크 95%\n"
            "   - GPU 2 (Qwen 35B): 평균 65%, 피크 88%\n"
            "   - GPU 3 (Nemotron): 평균 42%, 피크 72%\n\n"
            "3. API 호출 통계\n"
            "   - 총 요청 수: 1,234,567건\n"
            "   - 평균 응답 시간: 2.3초\n"
            "   - 에러율: 0.02%"
        ),
    },
]


def _create_synthetic_pdf(text: str, title: str) -> str:
    """Create a simple PDF from text using PyMuPDF. Returns temp file path."""
    import fitz

    doc = fitz.open()
    page = doc.new_page(width=595, height=842)  # A4

    # Title
    title_rect = fitz.Rect(50, 50, 545, 90)
    page.insert_textbox(title_rect, title, fontsize=16, fontname="helv")

    # Body text
    body_rect = fitz.Rect(50, 100, 545, 792)
    page.insert_textbox(body_rect, text, fontsize=10, fontname="helv")

    tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
    doc.save(tmp.name)
    doc.close()
    tmp.close()
    return tmp.name


@register_dataset("pdf_ocr")
class PDFOCRDataset(BaseDataset):
    @property
    def name(self) -> str:
        return "pdf_ocr"

    @property
    def requires_vision(self) -> bool:
        return True

    def load_samples(self, n: int, seed: int = 42) -> list[Sample]:
        mode = self.config.get("mode", "synthetic")

        if mode == "directory":
            return self._load_from_directory(n, seed)
        else:
            return self._load_synthetic(n, seed)

    def _load_synthetic(self, n: int, seed: int = 42) -> list[Sample]:
        random.seed(seed)
        selected = random.choices(SYNTHETIC_TEXTS, k=min(n, len(SYNTHETIC_TEXTS)))

        samples = []
        for i, doc in enumerate(selected):
            pdf_path = _create_synthetic_pdf(doc["content"], doc["title"])

            # Convert to page images for the vision model
            from benchmark.ocr.pipeline import pdf_to_images, image_to_base64
            images = pdf_to_images(pdf_path, dpi=144)

            for page_num, img in enumerate(images):
                b64_uri = image_to_base64(img)
                prompt = [
                    {"type": "text", "text": (
                        "이 문서 이미지의 텍스트를 정확히 그대로 읽어주세요. "
                        "마크다운 포맷 없이 원문 텍스트만 출력하세요."
                    )},
                    {"type": "image_url", "image_url": {"url": b64_uri}},
                ]
                samples.append(Sample(
                    id=f"pdf_ocr_synth_{i}_p{page_num}",
                    prompt=prompt,
                    reference=doc["content"],
                    metadata={
                        "title": doc["title"],
                        "page_num": page_num,
                        "pdf_path": pdf_path,
                        "mode": "synthetic",
                    },
                ))

            # Clean up temp PDF
            os.unlink(pdf_path)

        return samples[:n]

    def _load_from_directory(self, n: int, seed: int = 42) -> list[Sample]:
        """Load PDFs + ground truth .txt files from a directory."""
        doc_dir = self.config.get("doc_dir", "benchmark/test_docs")
        if not os.path.isdir(doc_dir):
            raise FileNotFoundError(f"Document directory not found: {doc_dir}")

        from benchmark.ocr.pipeline import pdf_to_images, image_to_base64

        pdf_files = sorted(f for f in os.listdir(doc_dir) if f.endswith(".pdf"))
        samples = []

        for pdf_file in pdf_files:
            pdf_path = os.path.join(doc_dir, pdf_file)
            txt_path = os.path.join(doc_dir, pdf_file.replace(".pdf", ".txt"))

            if not os.path.exists(txt_path):
                continue

            with open(txt_path, encoding="utf-8") as f:
                ground_truths = f.read().split("---")

            images = pdf_to_images(pdf_path, dpi=144)

            for page_num, img in enumerate(images):
                gt = ground_truths[page_num].strip() if page_num < len(ground_truths) else ""
                if not gt:
                    continue

                b64_uri = image_to_base64(img)
                prompt = [
                    {"type": "text", "text": (
                        "이 문서 이미지의 텍스트를 정확히 그대로 읽어주세요. "
                        "마크다운 포맷 없이 원문 텍스트만 출력하세요."
                    )},
                    {"type": "image_url", "image_url": {"url": b64_uri}},
                ]
                samples.append(Sample(
                    id=f"pdf_ocr_{pdf_file}_p{page_num}",
                    prompt=prompt,
                    reference=gt,
                    metadata={
                        "pdf_file": pdf_file,
                        "page_num": page_num,
                        "mode": "directory",
                    },
                ))

        random.seed(seed)
        random.shuffle(samples)
        return samples[:n]
