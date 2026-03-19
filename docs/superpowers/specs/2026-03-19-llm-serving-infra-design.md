# LLM Serving Infrastructure Design

## Overview

A100 80GB x4 (SXM4, NVLink4) 서버에서 양자화된 LLM 모델들을 vLLM으로 서빙하는 확장 가능한 인프라.

## Goals

- YAML 설정 파일 하나로 모델 추가/제거/관리
- OpenAI-compatible API로 통일된 엔드포인트 제공
- 비전+텍스트 통합 모델(Qwen3.5)로 OCR/비전/범용 모두 커버
- A100 SXM4 80GB에 최적화된 파라미터

## Non-Goals

- Web UI (챗 인터페이스)
- API 인증 (구조만 준비, 초기에는 미적용)
- Grafana 등 모니터링 대시보드 (초기에는 Docker 로그로 충분)

---

## Hardware

| 항목 | 사양 |
|---|---|
| GPU | NVIDIA A100-SXM4-80GB x 4 |
| GPU 토폴로지 | NVLink4 전체 연결 |
| CUDA | 12.2 |
| Driver | 535.288.01 |
| 총 VRAM | 320GB |

### A100 제약 사항

- FP8 텐서코어 미지원 → `dtype: bfloat16` 사용
- GPU당 80GB → 모델 크기 + KV 캐시가 80GB 이내여야 함

---

## Architecture

```
Client (:8000) → FastAPI Gateway → vLLM:8001 (qwen3.5-122b, GPU 0,1)
                  - model 라우팅     → vLLM:8002 (qwen3.5-35b, GPU 2)
                  - /v1/models 통합
                  - /health 통합     GPU 3: 여유 (추후 모델 추가)
                  - SSE 스트리밍

models.yaml → generate.py → docker-compose.yml

/mnt/models/ (설정 가능한 외부 경로)
  ├── Qwen3.5-122B-A10B-GPTQ-Int4/
  └── Qwen3.5-35B-A3B-GPTQ-Int4/
```

### 구성 요소

1. **models.yaml** — 유일한 사용자 설정 파일. 모델 정의, GPU 배정, vLLM 파라미터.
2. **generate.py** — models.yaml을 읽어 docker-compose.yml을 자동 생성. GPU/포트 충돌 검증.
3. **gateway/** — FastAPI 기반 프록시. request body의 model 필드로 라우팅.
4. **vLLM 컨테이너** — 모델별 1개 컨테이너, `vllm/vllm-openai` 공식 이미지 사용.

---

## Models

### Initial Models

| 모델명 | HuggingFace ID | 크기 | GPU | TP | 용도 |
|---|---|---|---|---|---|
| qwen3.5-122b | Qwen/Qwen3.5-122B-A10B-GPTQ-Int4 | 122B MoE (10B active) | 0,1 | 2 | 비전+범용 메인 |
| qwen3.5-35b | Qwen/Qwen3.5-35B-A3B-GPTQ-Int4 | 35B MoE (3B active) | 2 | 1 | 경량 빠른 응답 |

### Model Selection Rationale

- **Qwen3.5 시리즈**: 모든 모델이 비전+텍스트 통합 (early fusion). 별도 VL 버전 불필요.
- **GPTQ-Int4**: A100에서 안정적, 공식 양자화 모델 제공, vLLM 호환 검증됨.
- **122B MoE**: GPTQ-Int4 기준 총 ~79GB (tp=2 시 GPU당 ~40GB). 활성 파라미터 10B로 효율적.
- **35B MoE**: GPTQ-Int4 기준 총 ~22GB, GPU 1장으로 충분. 활성 파라미터 3B로 빠른 응답.

### VRAM Budget

**qwen3.5-122b (GPU 0,1, tp=2)**

| 항목 | GPU당 |
|---|---|
| 가용 VRAM (80GB x 0.92) | ~73.6GB |
| 모델 가중치 (79GB / 2) | ~39.5GB |
| KV 캐시 여유 | ~34.1GB |
| swap_space (CPU) | 4GB |

**qwen3.5-35b (GPU 2, tp=1)**

| 항목 | GPU당 |
|---|---|
| 가용 VRAM (80GB x 0.92) | ~73.6GB |
| 모델 가중치 | ~22GB |
| KV 캐시 여유 | ~51.6GB |
| swap_space (CPU) | 4GB |

> 비전 요청(이미지 임베딩)은 추가 VRAM을 소비하므로 `--limit-mm-per-prompt`로 제한 필요.

---

## Configuration

### models.yaml

```yaml
global:
  model_base_path: /mnt/models
  vllm_image: vllm/vllm-openai:v0.8.5    # 버전 고정, latest 사용 금지
  gateway_port: 8000
  api_key: ""

models:
  qwen3.5-122b:
    enabled: true
    model_id: Qwen/Qwen3.5-122B-A10B-GPTQ-Int4
    model_path: Qwen3.5-122B-A10B-GPTQ-Int4
    port: 8001
    gpus: [0, 1]
    tensor_parallel: 2
    dtype: bfloat16
    quantization: gptq
    max_model_len: 32768
    gpu_memory_utilization: 0.92
    max_num_seqs: 64
    max_num_batched_tokens: 32768
    swap_space: 4
    extra_args:
      - "--trust-remote-code"
      - "--reasoning-parser=qwen3"
      - "--limit-mm-per-prompt=image=5"

  qwen3.5-35b:
    enabled: true
    model_id: Qwen/Qwen3.5-35B-A3B-GPTQ-Int4
    model_path: Qwen3.5-35B-A3B-GPTQ-Int4
    port: 8002
    gpus: [2]
    tensor_parallel: 1
    dtype: bfloat16
    quantization: gptq
    max_model_len: 65536
    gpu_memory_utilization: 0.92
    max_num_seqs: 128
    max_num_batched_tokens: 32768
    swap_space: 4
    extra_args:
      - "--trust-remote-code"
      - "--reasoning-parser=qwen3"
      - "--limit-mm-per-prompt=image=5"
```

### Parameter Rationale

| 파라미터 | 값 | 근거 |
|---|---|---|
| dtype | bfloat16 | A100 FP8 미지원, BF16 최적 |
| gpu_memory_utilization | 0.92 | 기본 0.9보다 살짝 높여 KV 캐시 극대화, 0.95+ OOM 위험 |
| max_model_len | 32768 (122B) / 65536 (35B) | 모델 크기에 따라 KV 캐시 여유 확보 |
| max_num_seqs | 64 (122B) / 128 (35B) | 모델 크기에 따라 차등, 과다하면 레이턴시 증가 |
| swap_space | 4 | CPU 메모리 스왑으로 OOM 안전망 |
| quantization | gptq | A100에서 INT4 GPTQ 안정적, 공식 모델 사용 |

---

## Components

### generate.py

models.yaml → docker-compose.yml 자동 생성.

**검증 로직:**
1. GPU 충돌 검사 — 두 모델이 같은 GPU를 사용하면 에러
2. 포트 충돌 검사 — 같은 포트를 쓰는 모델이 있으면 에러
3. 모델 경로 존재 확인 — model_base_path/model_path 디렉토리 존재 여부
4. VRAM 예상 검사 — 모델 크기가 할당된 GPU 총 VRAM을 초과하면 경고

**생성물:**
- docker-compose.yml (vLLM 컨테이너 + gateway 컨테이너)

### gateway/ (FastAPI Proxy)

**기능:**
1. 모델 라우팅 — request body의 model 필드로 올바른 vLLM 컨테이너에 프록시
2. /v1/models 통합 — 모든 vLLM 컨테이너의 모델 목록을 합쳐서 반환
3. /health 통합 — 각 vLLM 헬스 상태를 종합. 일부 모델 다운 시 `degraded` 상태 반환, 전체 다운 시 503
4. 스트리밍 지원 — SSE 스트리밍 응답 그대로 전달
5. 에러 처리 — 잘못된 model명이면 사용 가능한 모델 목록과 함께 422 반환. 대상 모델 다운 시 503
6. 요청 제한 — 최대 request body 크기 제한 (base64 이미지 포함 시 대용량 가능)

**라우팅 테이블:** models.yaml에서 자동 로드.

**Docker 이미지:** python:3.11-slim + fastapi, uvicorn, httpx, pyyaml

### vLLM Containers

- 공식 이미지 사용 (버전 고정, `models.yaml`의 `vllm_image`로 관리)
- 커스텀 Dockerfile 불필요
- 모델별 1개 컨테이너
- `restart: unless-stopped` — 크래시/OOM/재부팅 시 자동 재시작
- healthcheck: /health 엔드포인트, start_period 300s (모델 로딩 대기)
- Docker 로그 로테이션: `max-size: 100m`, `max-file: 3` (디스크 고갈 방지)

---

## Directory Structure

```
infra/
├── models.yaml              # 모델 설정 (유일한 사용자 편집 파일)
├── generate.py              # models.yaml → docker-compose.yml 생성
├── docker-compose.yml       # (자동 생성됨, .gitignore 대상)
├── gateway/
│   ├── Dockerfile           # FastAPI 프록시 이미지
│   ├── main.py              # 라우팅 + /v1/models + /health
│   └── requirements.txt     # fastapi, uvicorn, httpx, pyyaml
├── DeepSeek-OCR---Dockerized-API/  # 기존 레퍼런스 (별도 유지)
└── docs/
    └── superpowers/
        └── specs/            # 설계 문서
```

---

## Usage

### Initial Setup

```bash
# 1. 모델 가중치 다운로드
huggingface-cli download Qwen/Qwen3.5-122B-A10B-GPTQ-Int4 --local-dir /mnt/models/Qwen3.5-122B-A10B-GPTQ-Int4
huggingface-cli download Qwen/Qwen3.5-35B-A3B-GPTQ-Int4 --local-dir /mnt/models/Qwen3.5-35B-A3B-GPTQ-Int4

# 2. 설정 확인/수정
vi models.yaml

# 3. docker-compose.yml 생성
python generate.py

# 4. 서비스 시작
docker compose up -d

# 5. 확인
curl http://localhost:8000/v1/models
curl http://localhost:8000/health
```

### Adding a New Model

```bash
# 1. models.yaml에 새 모델 블록 추가
# 2. 모델 가중치 다운로드
# 3. 재생성 & 재시작
python generate.py
docker compose up -d
```

### Client Usage (OpenAI SDK)

```python
from openai import OpenAI

client = OpenAI(base_url="http://server:8000/v1", api_key="unused")

# 비전 + 텍스트 (122B)
response = client.chat.completions.create(
    model="qwen3.5-122b",
    messages=[
        {"role": "user", "content": [
            {"type": "text", "text": "이 이미지를 마크다운으로 변환해줘"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
        ]}
    ]
)

# 경량 텍스트 (35B)
response = client.chat.completions.create(
    model="qwen3.5-35b",
    messages=[{"role": "user", "content": "간단한 질문"}]
)
```

---

## Model Upgrade Procedure

모델 가중치 또는 양자화 변경 시:

```bash
# 1. 새 모델 다운로드 (기존 모델과 별도 디렉토리)
huggingface-cli download Qwen/Qwen3.5-122B-A10B-GPTQ-Int4 --revision v2 --local-dir /mnt/models/Qwen3.5-122B-A10B-GPTQ-Int4-v2

# 2. models.yaml의 model_path 변경
# 3. 재생성 & 재시작
python generate.py
docker compose up -d
```

> 모델 이름(qwen3.5-122b)은 API 인터페이스이므로 가급적 유지. 내부 가중치만 교체.

---

## Known Risks

- **Gateway SPOF**: 단일 FastAPI 프로세스. 크래시 시 전체 접근 불가. 향후 복수 인스턴스 고려.
- **비전 메모리 스파이크**: 고해상도 이미지 다수 포함 요청 시 OOM 가능. `--limit-mm-per-prompt`로 완화.
- **NUMA 경계**: GPU 0 (NUMA 0)과 GPU 1 (NUMA 1)이 다른 NUMA 노드. NVLink으로 통신하므로 영향 미미하나, CPU 피닝 시 주의.

---

## Future Extensions

- **인증**: api_key 설정 시 gateway에서 Bearer 토큰 검증
- **모니터링**: Prometheus metrics 엔드포인트 추가, Grafana 연동
- **모델 추가**: GPU 3에 추가 모델 배치
- **로드 밸런싱**: 같은 모델의 복수 인스턴스 지원
- **Gateway HA**: 복수 gateway 인스턴스 + 로드 밸런서
