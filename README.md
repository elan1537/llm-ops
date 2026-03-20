# LLM Serving Infrastructure

A100-SXM4-80GB x4 서버에서 양자화된 LLM 모델들을 vLLM으로 서빙하는 인프라.
OpenAI-compatible API를 제공하므로 OpenAI SDK로 바로 사용 가능.

## 현재 서빙 중인 모델

| 모델 | GPU | 용도 | 컨텍스트 |
|---|---|---|---|
| `qwen3.5-122b` | GPU 0,1 | 비전+범용 메인 (122B MoE, 활성 10B) | 32K |
| `qwen3.5-35b` | GPU 2 | 경량 빠른 응답 (35B MoE, 활성 3B) | 64K |
| `nemotron-3-nano` | GPU 3 | NVIDIA 에이전트용 (30B MoE, 활성 3B) | 32K |

## API 엔드포인트

| 엔드포인트 | 설명 |
|---|---|
| `http://<서버IP>:8000/v1/models` | 사용 가능한 모델 목록 |
| `http://<서버IP>:8000/v1/chat/completions` | 채팅 (OpenAI 호환) |
| `http://<서버IP>:8000/v1/completions` | 텍스트 생성 (OpenAI 호환) |
| `http://<서버IP>:8000/health` | 서버 상태 확인 |

---

## 사용법

### Python (OpenAI SDK)

```bash
pip install openai
```

#### 기본 텍스트 대화

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://<서버IP>:8000/v1",
    api_key="unused",  # 인증 미적용 상태
)

response = client.chat.completions.create(
    model="qwen3.5-35b",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Python으로 피보나치 함수 작성해줘"},
    ],
    max_tokens=2048,
)

print(response.choices[0].message.content)
```

#### 비전 (이미지 분석)

Qwen3.5는 비전+텍스트 통합 모델이므로 이미지를 직접 입력할 수 있습니다.

```python
import base64

# 이미지를 base64로 인코딩
with open("document.png", "rb") as f:
    image_b64 = base64.b64encode(f.read()).decode()

response = client.chat.completions.create(
    model="qwen3.5-122b",
    messages=[
        {"role": "user", "content": [
            {"type": "text", "text": "이 문서를 마크다운으로 변환해줘"},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
        ]},
    ],
    max_tokens=4096,
)

print(response.choices[0].message.content)
```

#### 스트리밍

```python
stream = client.chat.completions.create(
    model="qwen3.5-35b",
    messages=[{"role": "user", "content": "긴 이야기를 하나 해줘"}],
    max_tokens=2048,
    stream=True,
)

for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

#### Thinking 모드 제어

Qwen3.5는 기본적으로 thinking(추론 과정)을 출력합니다.
thinking 없이 바로 답변만 받으려면:

```python
response = client.chat.completions.create(
    model="qwen3.5-35b",
    messages=[{"role": "user", "content": "1+1은?"}],
    max_tokens=512,
    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
)
```

### curl

```bash
# 모델 목록
curl http://<서버IP>:8000/v1/models

# 채팅
curl http://<서버IP>:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.5-35b",
    "messages": [{"role": "user", "content": "안녕하세요"}],
    "max_tokens": 1024
  }'

# 헬스 체크
curl http://<서버IP>:8000/health
```

### JavaScript / TypeScript

```typescript
import OpenAI from "openai";

const client = new OpenAI({
  baseURL: "http://<서버IP>:8000/v1",
  apiKey: "unused",
});

const response = await client.chat.completions.create({
  model: "qwen3.5-35b",
  messages: [{ role: "user", content: "Hello" }],
  max_tokens: 1024,
});

console.log(response.choices[0].message.content);
```

---

## 모델 선택 가이드

| 상황 | 추천 모델 | 이유 |
|---|---|---|
| 복잡한 추론, 코드 생성 | `qwen3.5-122b` | 가장 높은 품질 |
| 이미지/문서 분석, OCR | `qwen3.5-122b` | 비전 성능 최고 |
| 빠른 응답, 간단한 질문 | `qwen3.5-35b` | 경량, 빠름 |
| 에이전트, 도구 호출 | `nemotron-3-nano` | 에이전트 특화 |
| 대량 배치 처리 | `qwen3.5-35b` | 동시 요청 많이 처리 |

---

## 파라미터 참고

| 파라미터 | 설명 | 기본값 | 범위 |
|---|---|---|---|
| `max_tokens` | 생성할 최대 토큰 수 | 모델 기본값 | 1 ~ max_model_len |
| `temperature` | 무작위성 (낮을수록 결정적) | 1.0 | 0.0 ~ 2.0 |
| `top_p` | 누적 확률 샘플링 | 1.0 | 0.0 ~ 1.0 |
| `stream` | 스트리밍 응답 | false | true/false |
| `stop` | 생성 중단 문자열 | null | 문자열 또는 배열 |
| `n` | 생성할 응답 수 | 1 | 1+ |

---

## 서버 관리 (관리자용)

```bash
cd /home/separk/workspace/infra

# 서비스 상태 확인
docker ps

# 서비스 시작
python generate.py && docker compose up -d

# 서비스 중지
docker compose down

# 로그 확인
docker logs -f vllm-qwen3.5-122b
docker logs -f vllm-qwen3.5-35b
docker logs -f vllm-nemotron-3-nano
docker logs -f llm-gateway

# 모델 추가/변경: models.yaml 수정 후
python generate.py && docker compose up -d
```
