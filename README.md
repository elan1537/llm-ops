# LLM-Ops

LLM serving infrastructure on NVIDIA A100 GPUs with vLLM, FastAPI gateway, and OCR pipeline.

## Architecture

```
Client (OpenAI SDK)
    ↓
Gateway (:8000) — FastAPI proxy, health check, model routing
    ↓
┌──────────────────────┬──────────────────────┐
│ vLLM Model Server    │ DeepSeek-OCR (:8003) │
│ (OpenAI-compatible)  │ (grounding + OCR)    │
└──────────────────────┴──────────────────────┘
         GPU 0,1                 GPU 3
```

## Quick Start

### Prerequisites

- NVIDIA GPU(s) with CUDA support
- Docker + Docker Compose
- Python 3.11+
- HuggingFace account (for model downloads)

### 1. Clone

```bash
git clone https://github.com/elan1537/llm-ops.git
cd llm-ops
```

### 2. Download Models

```bash
pip install huggingface_hub

# Main model: Qwen3.5-27B Dense BF16 (recommended, ~54GB)
python -c "
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen3.5-27B', local_dir='<MODEL_BASE_PATH>/Qwen3.5-27B')
"
```

### 3. Configure

Edit `models.yaml` to match your hardware:

```yaml
global:
  model_base_path: /path/to/your/models   # where you downloaded models
  vllm_image: vllm/vllm-openai:latest
  gateway_port: 8000

models:
  qwen3.5-27b:
    enabled: true
    model_id: Qwen/Qwen3.5-27B
    model_path: Qwen3.5-27B
    port: 8001
    gpus: [0, 1]          # adjust to your GPU setup
    tensor_parallel: 2     # number of GPUs
    dtype: bfloat16
    quantization: null
    max_model_len: 131072  # 128K context
    gpu_memory_utilization: 0.92
    max_num_seqs: 64
    max_num_batched_tokens: 131072
    swap_space: 8
    extra_args:
      - "--trust-remote-code"
      - '--limit-mm-per-prompt={"image":10}'
```

### 4. Environment Variables

```bash
cp .env.example .env
# Edit .env with your values
```

### 5. Generate & Start

```bash
# Generate docker-compose.yml from models.yaml
python generate.py --config models.yaml --output docker-compose.yml

# Start all services
docker compose up -d

# Verify
curl http://localhost:8000/health
```

## Configuration

### models.yaml

Single source of truth for all model configurations. `generate.py` reads this and produces `docker-compose.yml`.

Key fields per model:
| Field | Description |
|-------|-------------|
| `gpus` | GPU device IDs (e.g. `[0, 1]`) |
| `tensor_parallel` | Number of GPUs for model parallelism |
| `dtype` | `bfloat16` (recommended) or `float16` |
| `quantization` | `null` (BF16), `gptq`, or `awq` |
| `max_model_len` | Max context length in tokens |
| `gpu_memory_utilization` | VRAM fraction to use (0.0-1.0) |

### GPU Sizing Guide

| Model | BF16 VRAM | GPUs Needed (A100-80GB) | max_model_len |
|-------|-----------|------------------------|---------------|
| Qwen3.5-27B (Dense) | ~54GB | 1 (64K) or 2 (128K) | up to 131072 |
| Qwen3.5-35B-A3B (MoE) | ~70GB | 1 (8K) or 2 (128K) | up to 131072 |
| Qwen3.5-122B-A10B (MoE) | ~244GB | 4 (BF16) or 2 (Int8) | up to 131072 |

## API Usage

Gateway provides OpenAI-compatible API on port 8000.

### Python (OpenAI SDK)

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="unused")

response = client.chat.completions.create(
    model="qwen3.5-27b",
    messages=[{"role": "user", "content": "Hello"}],
    max_tokens=2048,
)
print(response.choices[0].message.content)
```

### Vision (Image Input)

```python
import base64

with open("document.png", "rb") as f:
    image_b64 = base64.b64encode(f.read()).decode()

response = client.chat.completions.create(
    model="qwen3.5-27b",
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "Describe this image"},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
        ],
    }],
)
```

### Thinking Mode Control

Qwen3.5 outputs reasoning by default. Disable for concise answers:

```python
response = client.chat.completions.create(
    model="qwen3.5-27b",
    messages=[{"role": "user", "content": "What is 2+2?"}],
    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
)
```

### Streaming

```python
stream = client.chat.completions.create(
    model="qwen3.5-27b",
    messages=[{"role": "user", "content": "Tell me a story"}],
    stream=True,
)
for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

### curl

```bash
# Health check
curl http://localhost:8000/health

# Chat
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3.5-27b","messages":[{"role":"user","content":"Hello"}]}'

# Model list
curl http://localhost:8000/v1/models
```

## API Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /health` | Service health (healthy/degraded/unhealthy) |
| `GET /v1/models` | List available models |
| `POST /v1/chat/completions` | Chat completion (OpenAI-compatible) |
| `POST /v1/completions` | Text completion (OpenAI-compatible) |

## OCR Pipeline

2-stage PDF OCR using DeepSeek-OCR (grounding) + Qwen (figure description):

```
PDF → page images (PyMuPDF)
  → Stage 1: DeepSeek-OCR — text OCR + figure bbox detection
  → Stage 2: Qwen — figure description generation
  → Final markdown with embedded figure descriptions
```

Setup DeepSeek-OCR:
```bash
cd DeepSeek-OCR---Dockerized-API
docker compose up -d    # runs on GPU 3, port 8003
```

Usage:
```python
import asyncio
from benchmark.client import BenchmarkClient
from benchmark.ocr.pipeline import ocr_pdf

client = BenchmarkClient(base_url="http://localhost:8000/v1")
result = asyncio.run(ocr_pdf("paper.pdf", client, "qwen3.5-27b"))
print(result.full_markdown)
```

## Benchmark Suite

Built-in benchmark framework for evaluating model performance.

### Available Benchmarks

| Benchmark | Language | Type | Evaluator |
|-----------|----------|------|-----------|
| MMLU | EN | Knowledge (57 subjects) | exact_match |
| KMMLU | KO | Knowledge (44 subjects) | exact_match |
| KorQuAD | KO | Reading comprehension | F1/EM |
| GSM8K | EN | Math reasoning | exact_match |
| ARC | EN | Science reasoning | exact_match |
| HellaSwag | EN | Commonsense reasoning | exact_match |
| HumanEval | EN | Code generation | pass@1 |
| LongBench v2 | EN | Long context QA | exact_match |
| RULER | EN | Needle in haystack | F1/EM |
| PDF OCR | KO/EN | OCR accuracy | CER |
| Figure Caption | EN | Image description | LLM Judge |
| OCRBench | EN | OCR tasks | ANLS |
| DocVQA | EN | Document VQA | ANLS |

### Running Benchmarks

```bash
# Install dependencies
pip install -r benchmark/requirements.txt

# Run specific datasets
python -m benchmark.run --dataset mmlu gsm8k --samples 50

# Run with thinking disabled
python -m benchmark.run --dataset kmmlu --samples 100 --no-thinking

# Filter by model
python -m benchmark.run --model qwen3.5-27b --dataset arc hellaswag

# Analyze results
python -m benchmark.analyze benchmark/results/<file>.json
```

### Benchmark Results (Qwen3.5-27B BF16, A100)

| Benchmark | Score | Notes |
|-----------|-------|-------|
| MMLU | 84.0% | English knowledge |
| KMMLU | 68.0% | Korean knowledge |
| KorQuAD F1 | 95.2% | Korean reading comp |
| GSM8K | 94.0% | Math reasoning |
| ARC | 88.0% | Science reasoning |
| HellaSwag | 88.0% | Commonsense |
| HumanEval | 70.0% | Code generation (pass@1) |
| RULER | 100% F1 | 128K needle search |
| PDF OCR | 95.1% accuracy | CER=4.9% |

## Project Structure

```
.
├── models.yaml          # Model configuration (source of truth)
├── generate.py          # YAML → docker-compose.yml generator
├── gateway/             # FastAPI proxy service
│   ├── main.py
│   └── Dockerfile
├── benchmark/           # Benchmark framework
│   ├── run.py           # CLI runner
│   ├── analyze.py       # Result analysis
│   ├── client.py        # API clients (OpenAI + Claude native)
│   ├── config.yaml      # Benchmark config
│   ├── datasets/        # 13 dataset loaders
│   ├── evaluators/      # 6 evaluators
│   ├── ocr/             # PDF OCR pipeline
│   └── results/         # JSON results
├── tests/               # Test suite
└── docs/                # Design specs and plans
```

## Management

```bash
# Start services
python generate.py && docker compose up -d

# Stop services
docker compose down

# View logs
docker logs -f <container-name>

# Add/change models: edit models.yaml, then
python generate.py && docker compose up -d

# Restart gateway after config change
docker restart llm-gateway
```

## License

MIT
