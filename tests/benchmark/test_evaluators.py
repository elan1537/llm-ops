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
