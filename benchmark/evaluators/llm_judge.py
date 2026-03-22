import asyncio
import re

from benchmark.evaluators import register_evaluator

JUDGE_PROMPT_TEMPLATE = """당신은 텍스트 요약 품질을 평가하는 전문가입니다.

아래 원문과 모델이 생성한 요약을 비교하여 평가해주세요.

## 원문
{source}

## 참조 요약
{reference}

## 모델 생성 요약
{prediction}

## 평가 기준 (각 1~5점)
- 충실도: 원문의 핵심 내용을 정확히 반영하는가
- 간결성: 불필요한 내용 없이 핵심만 담았는가
- 유창성: 문장이 자연스럽고 읽기 쉬운가

## 응답 형식
충실도: [점수]
간결성: [점수]
유창성: [점수]

종합 점수: [1~5 정수]"""

FIGURE_JUDGE_PROMPT_TEMPLATE = """당신은 이미지 설명(figure description) 품질을 평가하는 전문가입니다.

논문의 그림(figure)에 대해, 모델이 이미지만 보고 생성한 설명과 논문 원본 캡션을 비교합니다.
모델은 이미지에서 보이는 것만 설명할 수 있으므로, 논문 캡션의 전문 용어나 맥락 정보가 없어도 감점하지 마세요.

## 논문 원본 캡션 (참고용)
{reference}

## 모델 생성 설명
{prediction}

## 평가 기준 (각 1~5점)
- 정확성: 이미지에 실제로 보이는 내용을 정확히 설명하는가 (차트 유형, 축, 데이터 등)
- 완전성: 이미지의 주요 요소를 빠짐없이 포함하는가
- 명확성: 이미지를 보지 않은 사람도 이해할 수 있게 설명하는가

## 응답 형식
정확성: [점수]
완전성: [점수]
명확성: [점수]

종합 점수: [1~5 정수]"""

PROMPT_TEMPLATES = {
    "default": JUDGE_PROMPT_TEMPLATE,
    "figure_caption": FIGURE_JUDGE_PROMPT_TEMPLATE,
}


def _parse_score(text: str) -> int | None:
    match = re.search(r"종합\s*점수[:\s]*(\d)", text)
    if match:
        score = int(match.group(1))
        if 1 <= score <= 5:
            return score

    numbers = re.findall(r"\b([1-5])\b", text)
    if numbers:
        return int(numbers[-1])

    return None


@register_evaluator("llm_judge")
class LLMJudgeEvaluator:
    async def evaluate_async(
        self,
        client,
        judge_model: str,
        predictions: list[str],
        references: list[str],
        sources: list[str] | None = None,
        prompt_type: str = "default",
    ) -> dict:
        template = PROMPT_TEMPLATES.get(prompt_type, JUDGE_PROMPT_TEMPLATE)
        if sources is None:
            sources = [""] * len(predictions)

        scores = []
        errors = 0
        details = []

        for pred, ref, src in zip(predictions, references, sources):
            prompt = template.format(
                source=src, reference=ref, prediction=pred,
            )
            try:
                result = await client.generate(
                    model=judge_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=512,
                )
                response_text = result.content if hasattr(result, 'content') else str(result)
                score = _parse_score(response_text)
                if score is None:
                    errors += 1
                scores.append(score)
                details.append({
                    "prediction": pred, "reference": ref,
                    "judge_response": response_text, "score": score,
                })
            except Exception as e:
                scores.append(None)
                errors += 1
                details.append({
                    "prediction": pred, "reference": ref,
                    "error": str(e), "score": None,
                })

        valid_scores = [s for s in scores if s is not None]
        mean = sum(valid_scores) / len(valid_scores) if valid_scores else 0.0

        return {
            "mean_score": mean,
            "scores": scores,
            "errors": errors,
            "total": len(predictions),
            "details": details,
        }
