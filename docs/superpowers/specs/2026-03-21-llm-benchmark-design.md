# LLM Benchmark Framework Design

## 목적

서버에서 서빙 중인 3개 모델(Qwen3.5-122B, Qwen3.5-35B, Nemotron-3-Nano-30B)의 정확도와 품질을 평가하는 벤치마크 프레임워크.

**목표:**
- 모델 간 성능 비교 → 용도별 라우팅 전략 수립
- GPTQ 양자화 + vLLM 환경에서의 실제 성능 확인
- 실무 task별 답변 품질 확인

## 아키텍처

```
benchmark/
├── config.yaml          # 벤치마크 설정 (모델, 데이터셋, Judge 설정)
├── run.py               # CLI 진입점
├── datasets/            # 데이터셋 로더 (HuggingFace datasets 활용)
│   ├── base.py          # 공통 인터페이스
│   ├── kmmlu.py         # 한국어 QA (KMMLU)
│   ├── korquad.py       # 한국어 독해/QA
│   ├── summarization.py # 요약 (XL-Sum Korean)
│   ├── gsm8k.py         # 수학 추론
│   └── docvqa.py        # Vision/OCR
├── evaluators/          # 채점 로직
│   ├── exact_match.py   # 객관식/단답형 자동 채점
│   ├── f1_em.py         # F1/EM score (KorQuAD)
│   ├── anls.py          # ANLS score (DocVQA)
│   └── llm_judge.py     # Claude Judge 호출
├── client.py            # OpenAI-compatible API 호출 (모델 + Judge 공용)
└── results/             # 결과 저장 (JSON)
```

**흐름:**
```
config.yaml → run.py → 데이터셋 로드 → 모델 API 호출 → 채점 (auto/judge) → JSON 결과 + CLI 출력
```

## 대상 모델

| 모델 | 엔드포인트 | Vision |
|------|-----------|--------|
| Qwen3.5-122B-A10B-GPTQ-Int4 | localhost:8000 (gateway) | O |
| Qwen3.5-35B-A3B-GPTQ-Int4 | localhost:8000 (gateway) | O |
| NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 | localhost:8000 (gateway) | X |

Gateway(8000)를 통해 `model_id`로 라우팅. 모든 호출은 OpenAI-compatible API 형식.

## 데이터셋 & 채점

| Task | 데이터셋 | HuggingFace ID | 샘플 수 | 채점 방식 |
|------|----------|----------------|---------|-----------|
| 한국어 QA | KMMLU | `HAERAE-HUB/KMMLU` | 200 | 자동 (객관식 정답 매칭) |
| 한국어 독해 | KorQuAD 1.0 | `squad_kor_v1` | 200 | 자동 (F1/EM score) |
| 문서 요약 | XL-Sum Korean | `csebuetnlp/xlsum` (lang=korean) | 100 | LLM Judge (1~5점) |
| 수학 추론 | GSM8K | `openai/gsm8k` | 200 | 자동 (최종 숫자 정답 매칭) |
| Vision/OCR | DocVQA | `lmms-lab/DocVQA` | 100 | 자동 (ANLS) + LLM Judge 보조 |

### 채점 세부

- **자동 채점**: 객관식 선택지 매칭, 숫자 추출 비교, F1/EM, ANLS
- **LLM Judge**: Claude Opus — reference + model output을 주고 rubric 기반 1~5점 평가 (충실도, 간결성, 유창성)
- **Vision**: ANLS(Average Normalized Levenshtein Similarity) 자동 채점, 애매한 케이스는 Judge 보조
- **vision_only 데이터셋**: vision 미지원 모델은 자동 스킵

## LLM Judge

- CLIProxyAPI (별도 서버)를 통해 Claude Opus를 OpenAI-compatible 엔드포인트로 호출
- 벤치마크 코드에서는 OpenAI client의 `base_url`만 변경
- 요약 평가 rubric: 충실도(원문 충실), 간결성(불필요한 내용 제거), 유창성(자연스러운 문장)

## config.yaml

```yaml
models:
  - name: qwen3.5-122b
    base_url: http://localhost:8000/v1
    model_id: Qwen/Qwen3.5-122B-A10B-GPTQ-Int4
    vision: true
  - name: qwen3.5-35b
    base_url: http://localhost:8000/v1
    model_id: Qwen/Qwen3.5-35B-A3B-GPTQ-Int4
    vision: true
  - name: nemotron-3-nano
    base_url: http://localhost:8000/v1
    model_id: nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16
    vision: false

judge:
  base_url: http://<cliforproxy-host>:<port>/v1
  model_id: claude-opus-4-20250514
  api_key: ""

datasets:
  kmmlu:
    enabled: true
    samples: 200
    evaluator: exact_match
  korquad:
    enabled: true
    samples: 200
    evaluator: f1_em
  xlsum_ko:
    enabled: true
    samples: 100
    evaluator: llm_judge
  gsm8k:
    enabled: true
    samples: 200
    evaluator: exact_match
  docvqa:
    enabled: true
    samples: 100
    evaluator: anls
    vision_only: true

settings:
  temperatures: [0.0, 0.5]
  stochastic_runs: 3
  concurrent_requests: 4
  timeout: 120
  max_tokens: 2048
  results_dir: benchmark/results
```

## Temperature 전략

- `temperatures` 리스트로 여러 온도에서 비교 가능
- `temperature: 0.0` — baseline (재현성 보장)
- `temperature > 0` — 동일 샘플을 `stochastic_runs`회 반복, 평균 ± 표준편차 기록
- 실무에서 쓸 온도(0.5 등)에서도 성능이 유지되는지 확인 가능

## CLI 인터페이스

```bash
# 전체 벤치마크
python -m benchmark.run

# 특정 데이터셋만
python -m benchmark.run --dataset kmmlu gsm8k

# 특정 모델만
python -m benchmark.run --model qwen3.5-122b

# 샘플 수 오버라이드 (빠른 테스트)
python -m benchmark.run --samples 10
```

**CLI 출력 예시:**
```
[KMMLU] qwen3.5-122b  t=0.0: 72.5%  t=0.5: 71.8% (±1.2%)
[KMMLU] qwen3.5-35b   t=0.0: 65.0%  t=0.5: 64.2% (±1.5%)
[KMMLU] nemotron-3-nano t=0.0: 58.5%  t=0.5: 57.1% (±2.0%)

[DocVQA] qwen3.5-122b  t=0.0: 78.3% ANLS
[DocVQA] nemotron-3-nano ... skipped (no vision)

Results saved: benchmark/results/2026-03-21T14:30_full.json
```

## 결과 JSON 구조

```json
{
  "timestamp": "2026-03-21T14:30:00",
  "config_snapshot": { ... },
  "results": {
    "kmmlu": {
      "qwen3.5-122b": {
        "temperature_0.0": {
          "score": 0.725,
          "correct": 145,
          "total": 200,
          "errors": 0,
          "per_sample": [...]
        },
        "temperature_0.5": {
          "mean_score": 0.718,
          "std": 0.012,
          "runs": [0.72, 0.71, 0.724],
          "errors": 0
        }
      }
    }
  }
}
```

## 에러 처리

- **API 호출 실패**: 3회 재시도 (exponential backoff), 실패 시 해당 샘플 `error`로 기록, 다음 진행
- **Judge 호출 실패**: 자동 채점 데이터셋은 정상 진행, Judge 필수 데이터셋은 해당 부분만 스킵
- **결과에 에러율 표시**: `72.5% (145/200, 3 errors)`

## Vision 처리

- 이미지는 base64 인코딩으로 OpenAI vision API 형식에 맞춰 전송
- DocVQA 이미지는 HuggingFace에서 다운로드 후 로컬 캐시
- `vision_only: true` 데이터셋은 `vision: false` 모델 자동 스킵

## 재현성

- `temperature: 0.0` baseline
- 데이터셋 샘플링 시 고정 seed (`random.seed(42)`)
- config.yaml 스냅샷을 결과 JSON에 포함

## 의존성

- `openai` — API 클라이언트 (모델 + Judge 공용)
- `datasets` — HuggingFace 데이터셋 로딩
- `tqdm` — progress bar
- `Pillow` — 이미지 처리 (DocVQA)
