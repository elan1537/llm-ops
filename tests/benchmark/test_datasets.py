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
    def mock_hf_dataset(self):
        rows = [
            {"input": "한국의 수도는?", "A": "서울", "B": "부산", "C": "대전", "D": "인천", "output": "1"},
            {"input": "1+1=?", "A": "1", "B": "2", "C": "3", "D": "4", "output": "2"},
            {"input": "물의 화학식은?", "A": "CO2", "B": "H2O", "C": "NaCl", "D": "O2", "output": "2"},
        ]
        mock_ds = MagicMock()
        mock_ds.__len__ = lambda self: len(rows)
        mock_ds.__getitem__ = lambda self, idx: rows[idx] if isinstance(idx, int) else rows
        mock_ds.select = lambda indices: MagicMock(
            __iter__=lambda self: iter([rows[i] for i in indices]),
            __len__=lambda self: len(indices),
        )
        mock_ds.shuffle = lambda seed: mock_ds
        return mock_ds

    def test_load_samples(self, mock_hf_dataset):
        with patch("benchmark.datasets.kmmlu.load_dataset", return_value={"test": mock_hf_dataset}):
            ds = KMMLUDataset({"samples": 2})
            samples = ds.load_samples(n=2, seed=42)
            assert len(samples) == 2
            assert all(isinstance(s, Sample) for s in samples)

    def test_prompt_format(self, mock_hf_dataset):
        with patch("benchmark.datasets.kmmlu.load_dataset", return_value={"test": mock_hf_dataset}):
            ds = KMMLUDataset({"samples": 1})
            samples = ds.load_samples(n=1, seed=42)
            prompt = samples[0].prompt
            assert "A." in prompt
            assert "B." in prompt
            assert "C." in prompt
            assert "D." in prompt

    def test_reference_is_letter(self, mock_hf_dataset):
        with patch("benchmark.datasets.kmmlu.load_dataset", return_value={"test": mock_hf_dataset}):
            ds = KMMLUDataset({"samples": 1})
            samples = ds.load_samples(n=1, seed=42)
            assert samples[0].reference in ("A", "B", "C", "D")

    def test_name(self):
        ds = KMMLUDataset({})
        assert ds.name == "kmmlu"

    def test_requires_vision_false(self):
        ds = KMMLUDataset({})
        assert ds.requires_vision is False
