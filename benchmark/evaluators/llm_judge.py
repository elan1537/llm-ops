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
        sources: list[str],
    ) -> dict:
        scores = []
        errors = 0
        details = []

        for pred, ref, src in zip(predictions, references, sources):
            prompt = JUDGE_PROMPT_TEMPLATE.format(
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
