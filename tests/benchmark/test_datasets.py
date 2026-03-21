import pytest
from unittest.mock import patch, MagicMock

from benchmark.datasets.base import Sample, BaseDataset
from benchmark.datasets.kmmlu import KMMLUDataset


class TestSample:
    def test_text_sample(self):
        s = Sample(id="1", prompt="Hello", reference="A", metadata={})
        assert s.prompt == "Hello"
        assert s.reference == "A"

    def test_vision_sample(self):
        content = [
            {"type": "text", "text": "What is this?"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
        ]
        s = Sample(id="1", prompt=content, reference="cat", metadata={})
        assert isinstance(s.prompt, list)


class TestKMMLU:
    @pytest.fixture
    def mock_kmmlu(self):
        """Mock get_dataset_config_names, load_dataset, and concatenate_datasets for KMMLU."""
        from datasets import Dataset

        rows = [
            {"question": "한국의 수도는?", "answer": 1, "A": "서울", "B": "부산", "C": "대전", "D": "인천"},
            {"question": "1+1=?", "answer": 2, "A": "1", "B": "2", "C": "3", "D": "4"},
            {"question": "물의 화학식은?", "answer": 2, "A": "CO2", "B": "H2O", "C": "NaCl", "D": "O2"},
        ]
        ds = Dataset.from_list(rows)

        return patch("benchmark.datasets.kmmlu.get_dataset_config_names", return_value=["Math"]), \
               patch("benchmark.datasets.kmmlu.load_dataset", return_value=ds), \
               patch("benchmark.datasets.kmmlu.concatenate_datasets", return_value=ds)

    def test_load_samples(self, mock_kmmlu):
        p1, p2, p3 = mock_kmmlu
        with p1, p2, p3:
            ds = KMMLUDataset({"samples": 2})
            samples = ds.load_samples(n=2, seed=42)
            assert len(samples) == 2
            assert all(isinstance(s, Sample) for s in samples)

    def test_prompt_format(self, mock_kmmlu):
        p1, p2, p3 = mock_kmmlu
        with p1, p2, p3:
            ds = KMMLUDataset({"samples": 1})
            samples = ds.load_samples(n=1, seed=42)
            prompt = samples[0].prompt
            assert "A." in prompt
            assert "B." in prompt
            assert "C." in prompt
            assert "D." in prompt

    def test_reference_is_letter(self, mock_kmmlu):
        p1, p2, p3 = mock_kmmlu
        with p1, p2, p3:
            ds = KMMLUDataset({"samples": 1})
            samples = ds.load_samples(n=1, seed=42)
            assert samples[0].reference in ("A", "B", "C", "D")

    def test_name(self):
        ds = KMMLUDataset({})
        assert ds.name == "kmmlu"

    def test_requires_vision_false(self):
        ds = KMMLUDataset({})
        assert ds.requires_vision is False


from benchmark.datasets.korquad import KorQuADDataset
from benchmark.datasets.gsm8k import GSM8KDataset
from benchmark.datasets.summarization import XLSumKoDataset
from benchmark.datasets.docvqa import DocVQADataset
from PIL import Image
import io
import base64


class TestKorQuAD:
    @pytest.fixture
    def mock_hf_dataset(self):
        rows = [
            {
                "context": "서울은 대한민국의 수도이며 가장 큰 도시이다.",
                "question": "대한민국의 수도는 어디인가?",
                "answers": {"text": ["서울"], "answer_start": [0]},
            },
            {
                "context": "파이썬은 1991년에 만들어진 프로그래밍 언어이다.",
                "question": "파이썬은 언제 만들어졌는가?",
                "answers": {"text": ["1991년"], "answer_start": [4]},
            },
        ]
        mock_ds = MagicMock()
        mock_ds.__len__ = lambda self: len(rows)
        mock_ds.__getitem__ = lambda self, idx: rows[idx]
        mock_ds.shuffle = lambda seed: mock_ds
        return mock_ds

    def test_load_samples(self, mock_hf_dataset):
        with patch("benchmark.datasets.korquad.load_dataset", return_value={"validation": mock_hf_dataset}):
            ds = KorQuADDataset({"samples": 2})
            samples = ds.load_samples(n=2, seed=42)
            assert len(samples) == 2

    def test_prompt_contains_context_and_question(self, mock_hf_dataset):
        with patch("benchmark.datasets.korquad.load_dataset", return_value={"validation": mock_hf_dataset}):
            ds = KorQuADDataset({"samples": 1})
            samples = ds.load_samples(n=1, seed=42)
            assert "서울은 대한민국의 수도" in samples[0].prompt
            assert "수도는 어디" in samples[0].prompt

    def test_reference_is_answer_text(self, mock_hf_dataset):
        with patch("benchmark.datasets.korquad.load_dataset", return_value={"validation": mock_hf_dataset}):
            ds = KorQuADDataset({"samples": 1})
            samples = ds.load_samples(n=1, seed=42)
            assert samples[0].reference == "서울"


class TestGSM8K:
    @pytest.fixture
    def mock_hf_dataset(self):
        rows = [
            {"question": "3+5=?", "answer": "3+5=8\n#### 8"},
            {"question": "10*2=?", "answer": "10*2=20\n#### 20"},
        ]
        mock_ds = MagicMock()
        mock_ds.__len__ = lambda self: len(rows)
        mock_ds.__getitem__ = lambda self, idx: rows[idx]
        mock_ds.shuffle = lambda seed: mock_ds
        return mock_ds

    def test_load_samples(self, mock_hf_dataset):
        with patch("benchmark.datasets.gsm8k.load_dataset", return_value={"test": mock_hf_dataset}):
            ds = GSM8KDataset({"samples": 2})
            samples = ds.load_samples(n=2, seed=42)
            assert len(samples) == 2

    def test_reference_is_final_number(self, mock_hf_dataset):
        with patch("benchmark.datasets.gsm8k.load_dataset", return_value={"test": mock_hf_dataset}):
            ds = GSM8KDataset({"samples": 1})
            samples = ds.load_samples(n=1, seed=42)
            assert samples[0].reference in ("8", "20")

    def test_prompt_asks_for_step_by_step(self, mock_hf_dataset):
        with patch("benchmark.datasets.gsm8k.load_dataset", return_value={"test": mock_hf_dataset}):
            ds = GSM8KDataset({"samples": 1})
            samples = ds.load_samples(n=1, seed=42)
            assert "단계별" in samples[0].prompt or "step" in samples[0].prompt.lower()


class TestXLSumKo:
    @pytest.fixture
    def mock_hf_dataset(self):
        rows = [
            {"text": "서울에서 대규모 축제가 열렸다. 수만 명의 시민이 참여했다.", "summary": "서울 대규모 축제에 수만 명 참여"},
            {"text": "새로운 AI 기술이 발표되었다.", "summary": "새 AI 기술 발표"},
        ]
        mock_ds = MagicMock()
        mock_ds.__len__ = lambda self: len(rows)
        mock_ds.__getitem__ = lambda self, idx: rows[idx]
        mock_ds.shuffle = lambda seed: mock_ds
        return mock_ds

    def test_load_samples(self, mock_hf_dataset):
        with patch("benchmark.datasets.summarization.load_dataset", return_value={"test": mock_hf_dataset}):
            ds = XLSumKoDataset({"samples": 2})
            samples = ds.load_samples(n=2, seed=42)
            assert len(samples) == 2

    def test_prompt_contains_source_text(self, mock_hf_dataset):
        with patch("benchmark.datasets.summarization.load_dataset", return_value={"test": mock_hf_dataset}):
            ds = XLSumKoDataset({"samples": 1})
            samples = ds.load_samples(n=1, seed=42)
            assert "서울에서 대규모 축제" in samples[0].prompt

    def test_reference_is_summary(self, mock_hf_dataset):
        with patch("benchmark.datasets.summarization.load_dataset", return_value={"test": mock_hf_dataset}):
            ds = XLSumKoDataset({"samples": 1})
            samples = ds.load_samples(n=1, seed=42)
            assert "축제" in samples[0].reference or "AI" in samples[0].reference

    def test_metadata_has_source(self, mock_hf_dataset):
        with patch("benchmark.datasets.summarization.load_dataset", return_value={"test": mock_hf_dataset}):
            ds = XLSumKoDataset({"samples": 1})
            samples = ds.load_samples(n=1, seed=42)
            assert "source" in samples[0].metadata


class TestDocVQA:
    @pytest.fixture
    def mock_hf_dataset(self):
        img = Image.new("RGB", (100, 100), color="white")

        rows = [
            {"image": img, "question": "What is the title?", "answers": ["Annual Report"]},
            {"image": img, "question": "What is the date?", "answers": ["2024-01-01"]},
        ]
        mock_ds = MagicMock()
        mock_ds.__len__ = lambda self: len(rows)
        mock_ds.__getitem__ = lambda self, idx: rows[idx]
        mock_ds.shuffle = lambda seed: mock_ds
        return mock_ds

    def test_load_samples(self, mock_hf_dataset):
        with patch("benchmark.datasets.docvqa.load_dataset", return_value={"test": mock_hf_dataset}):
            ds = DocVQADataset({"samples": 2})
            samples = ds.load_samples(n=2, seed=42)
            assert len(samples) == 2

    def test_prompt_is_vision_format(self, mock_hf_dataset):
        with patch("benchmark.datasets.docvqa.load_dataset", return_value={"test": mock_hf_dataset}):
            ds = DocVQADataset({"samples": 1})
            samples = ds.load_samples(n=1, seed=42)
            prompt = samples[0].prompt
            assert isinstance(prompt, list)
            types = [item["type"] for item in prompt]
            assert "text" in types
            assert "image_url" in types

    def test_image_is_base64_encoded(self, mock_hf_dataset):
        with patch("benchmark.datasets.docvqa.load_dataset", return_value={"test": mock_hf_dataset}):
            ds = DocVQADataset({"samples": 1})
            samples = ds.load_samples(n=1, seed=42)
            prompt = samples[0].prompt
            img_item = [i for i in prompt if i["type"] == "image_url"][0]
            assert img_item["image_url"]["url"].startswith("data:image/png;base64,")

    def test_requires_vision_true(self):
        ds = DocVQADataset({})
        assert ds.requires_vision is True

    def test_reference_is_first_answer(self, mock_hf_dataset):
        with patch("benchmark.datasets.docvqa.load_dataset", return_value={"test": mock_hf_dataset}):
            ds = DocVQADataset({"samples": 1})
            samples = ds.load_samples(n=1, seed=42)
            assert samples[0].reference in ("Annual Report", "2024-01-01")
