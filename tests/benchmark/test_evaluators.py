import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from benchmark.evaluators.exact_match import ExactMatchEvaluator
from benchmark.evaluators.f1_em import F1EMEvaluator
from benchmark.evaluators.anls import ANLSEvaluator


class TestExactMatch:
    def test_perfect_score(self):
        ev = ExactMatchEvaluator()
        result = ev.evaluate(
            predictions=["A", "B", "C"],
            references=["A", "B", "C"],
        )
        assert result["score"] == 1.0
        assert result["correct"] == 3
        assert result["total"] == 3

    def test_partial_score(self):
        ev = ExactMatchEvaluator()
        result = ev.evaluate(
            predictions=["A", "B", "D"],
            references=["A", "B", "C"],
        )
        assert result["score"] == pytest.approx(2 / 3)

    def test_case_insensitive(self):
        ev = ExactMatchEvaluator()
        result = ev.evaluate(predictions=["a"], references=["A"])
        assert result["score"] == 1.0

    def test_strips_whitespace(self):
        ev = ExactMatchEvaluator()
        result = ev.evaluate(predictions=["  A  "], references=["A"])
        assert result["score"] == 1.0

    def test_extracts_answer_letter_from_text(self):
        ev = ExactMatchEvaluator()
        result = ev.evaluate(
            predictions=["The answer is B.", "정답은 A입니다"],
            references=["B", "A"],
        )
        assert result["score"] == 1.0


class TestF1EM:
    def test_exact_match(self):
        ev = F1EMEvaluator()
        result = ev.evaluate(
            predictions=["서울은 한국의 수도입니다"],
            references=["서울은 한국의 수도입니다"],
        )
        assert result["em"] == 1.0
        assert result["f1"] == 1.0

    def test_partial_overlap(self):
        ev = F1EMEvaluator()
        result = ev.evaluate(
            predictions=["서울"],
            references=["서울은 한국의 수도"],
        )
        assert result["em"] == 0.0
        assert result["f1"] > 0.0

    def test_no_overlap(self):
        ev = F1EMEvaluator()
        result = ev.evaluate(
            predictions=["부산"],
            references=["서울"],
        )
        assert result["f1"] == 0.0


class TestANLS:
    def test_perfect_match(self):
        ev = ANLSEvaluator()
        result = ev.evaluate(predictions=["hello"], references=["hello"])
        assert result["anls"] == 1.0

    def test_partial_match(self):
        ev = ANLSEvaluator()
        result = ev.evaluate(predictions=["helo"], references=["hello"])
        assert result["anls"] > 0.5

    def test_completely_wrong(self):
        ev = ANLSEvaluator()
        result = ev.evaluate(predictions=["xyz"], references=["abcdef"])
        assert result["anls"] < 0.5

    def test_empty_prediction(self):
        ev = ANLSEvaluator()
        result = ev.evaluate(predictions=[""], references=["hello"])
        assert result["anls"] == 0.0

    def test_threshold_applied(self):
        ev = ANLSEvaluator()
        result = ev.evaluate(predictions=["completely wrong answer here"], references=["x"])
        assert result["anls"] == 0.0


from benchmark.evaluators.llm_judge import LLMJudgeEvaluator


class TestLLMJudge:
    def test_parses_numeric_score(self):
        mock_client = MagicMock()
        mock_client.generate = AsyncMock(return_value="4")

        ev = LLMJudgeEvaluator()
        result = asyncio.run(ev.evaluate_async(
            client=mock_client,
            judge_model="claude-opus",
            predictions=["모델의 요약"],
            references=["원문 요약"],
            sources=["원문 텍스트"],
        ))
        assert result["mean_score"] == 4.0
        assert len(result["scores"]) == 1

    def test_parses_score_from_verbose_response(self):
        mock_client = MagicMock()
        mock_client.generate = AsyncMock(
            return_value="충실도: 4\n간결성: 3\n유창성: 5\n\n종합 점수: 4"
        )

        ev = LLMJudgeEvaluator()
        result = asyncio.run(ev.evaluate_async(
            client=mock_client,
            judge_model="claude-opus",
            predictions=["요약"],
            references=["참조"],
            sources=["원문"],
        ))
        assert result["mean_score"] == 4.0

    def test_handles_unparseable_score(self):
        mock_client = MagicMock()
        mock_client.generate = AsyncMock(return_value="I cannot evaluate this")

        ev = LLMJudgeEvaluator()
        result = asyncio.run(ev.evaluate_async(
            client=mock_client,
            judge_model="claude-opus",
            predictions=["요약"],
            references=["참조"],
            sources=["원문"],
        ))
        assert result["scores"][0] is None
        assert result["errors"] == 1
