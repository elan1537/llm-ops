# Benchmark Reliability Improvements Design

## 목적

벤치마크 측정의 신뢰성을 높이기 위한 4가지 개선.
모델 답변 스타일 차이, 파싱 실패, 디버깅 어려움 등의 문제를 해결.

## 배경 — 발생한 문제들

1. **답변 추출 실패** — thinking 출력 섞임, `</think>` 미처리, 장황한 답변에서 A/B/C/D 못 잡음
2. **순서 셔플** — `asyncio.as_completed`로 prediction-reference 불일치 (수정 완료)
3. **모델별 답변 스타일 차이** — Qwen은 "A", Opus는 "The answer is A because..."
4. **evaluator-프롬프트 불일치** — 요약 프롬프트로 figure caption 평가, CER에서 thinking 미제거
5. **점수 표시 버그** — stochastic 집계에서 judge 스케일 오인 (수정 완료)

## 1. 프롬프트 표준화

모든 데이터셋 프롬프트 끝에 엄격한 출력 형식 지시를 추가.

### 객관식 (KMMLU, MMLU, LongBench)

```
기존: "A, B, C, D 중 하나만 답하세요."
변경: "Reply with ONLY the letter (A, B, C, or D). No explanation."
```

### 단답형 (KorQuAD)

```
기존: "답은 문맥에서 직접 찾아 간결하게 작성하세요."
변경: "답만 작성하세요. 설명이나 문장으로 감싸지 마세요."
```

### 수학 (GSM8K)

```
기존: "최종 답은 마지막 줄에 숫자만 작성하세요."
변경: "풀이 후 마지막 줄에 'Answer: [숫자]' 형식으로만 작성하세요."
```

Claude 네이티브 API에서는 thinking이 별도 block이므로 text block에만 간결한 답이 옴.
프롬프트 강제 + API 분리로 이중 보장.

## 2. Evaluator 로버스트화

### exact_match._normalize 개선

추출 순서 — 가장 명시적인 패턴부터 fallback으로 관대하게:

1. `</think>` 태그 제거
2. `Answer: X` 패턴 추출 (새 프롬프트 형식)
3. `The answer is X` / `정답은 X` 패턴
4. markdown bold `**X**` 추출
5. 마지막 줄에서 단독 A/B/C/D 추출
6. 숫자: `Answer: 42` 패턴 우선, fallback으로 마지막 숫자

### f1_em._extract_answer 개선

1. `</think>` 제거
2. `답: X` / `Answer: X` 패턴 우선
3. 인용부호 안의 텍스트 추출 (`"블로그"` → `블로그`)
4. 마지막 줄 추출 (fallback)

### 공통

모든 evaluator에서 `</think>` 제거를 공통 유틸로 분리.

## 3. 결과 분석 CLI

`python -m benchmark.analyze <results.json>`

### 출력

```
=== KMMLU (qwen3.5-27b, t=0.0) ===
Score: 70.0% (35/50)

Wrong answers (15):
  #3  Ref: B  Pred: "The answer is D because..."  → Extracted: D  ✗
  #7  Ref: A  Pred: "ERROR: timeout"              → Error

Pattern stats:
  Clean (A/B/C/D only): 38/50 (76%)
  Verbose but extractable: 9/50 (18%)
  Failed extraction: 3/50 (6%)

Potential eval issues: 2 samples where extraction may have failed
```

### 기능

- **오답 상세** — prediction 원문 + 추출 결과 + reference 비교
- **패턴 통계** — 모델 답변 간결성, 추출 실패율
- **eval 이슈 감지** — reference가 prediction에 포함되는데 점수가 0인 케이스 자동 탐지

## 4. 범용 API client 구조

### config 형식

```yaml
models:
  - name: qwen3.5-27b
    base_url: http://localhost:8000/v1
    api_type: openai          # vLLM, OpenAI호환
    vision: true

  - name: claude-opus-4-6
    base_url: http://sehyeon-macstudio:8317/v1
    api_type: claude          # Claude 네이티브
    api_key: ${JUDGE_API_KEY}
    thinking_budget: 8192

  - name: gpt-5
    base_url: http://sehyeon-macstudio:8317/v1
    api_type: openai          # CLIProxyAPI 경유
    api_key: ${JUDGE_API_KEY}
```

### client 선택 로직

```python
def create_client(model_config, settings):
    api_type = model_config.get("api_type", "openai")
    if api_type == "claude":
        return ClaudeNativeClient(...)
    else:
        return BenchmarkClient(...)
```

- `openai` — BenchmarkClient. vLLM, GPT, Gemini 전부 커버
- `claude` — ClaudeNativeClient. thinking 지원, `/v1/messages`
- 확장: 필요 시 `api_type` 추가 가능

### run.py 변경

모델별 client 생성을 `create_client()` 팩토리 함수로 통일.
`enable_thinking`은 `api_type: claude`일 때만 thinking block 분리, `openai`일 때는 `chat_template_kwargs` 전달.

## 파일 변경 목록

- `benchmark/client.py` — `create_client()` 팩토리, ClaudeNativeClient 정리
- `benchmark/evaluators/common.py` — `strip_thinking()` 공통 유틸
- `benchmark/evaluators/exact_match.py` — `_normalize` 패턴 보강
- `benchmark/evaluators/f1_em.py` — `_extract_answer` 패턴 보강
- `benchmark/evaluators/cer.py` — 공통 유틸 사용
- `benchmark/datasets/kmmlu.py` — 프롬프트 강화
- `benchmark/datasets/mmlu.py` — 프롬프트 강화
- `benchmark/datasets/korquad.py` — 프롬프트 강화
- `benchmark/datasets/gsm8k.py` — 프롬프트 강화
- `benchmark/datasets/longbench.py` — 프롬프트 강화
- `benchmark/datasets/ruler.py` — 프롬프트 강화
- `benchmark/analyze.py` — 결과 분석 CLI (신규)
- `benchmark/run.py` — `create_client()` 사용, api_type 기반 분기
