import asyncio
import json
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml

from benchmark.run import BenchmarkRunner


@pytest.fixture
def minimal_config(tmp_path):
    config = {
        "models": [
            {"name": "test-model", "base_url": "http://localhost:8000/v1",
             "model_id": "test/model", "vision": False}
        ],
        "judge": {"base_url": "http://judge:8080/v1", "model_id": "claude-opus", "api_key": ""},
        "datasets": {
            "kmmlu": {"enabled": True, "samples": 2, "evaluator": "exact_match"},
        },
        "settings": {
            "temperatures": [0.0],
            "stochastic_runs": 1,
            "concurrent_requests": 2,
            "timeout": 10,
            "max_tokens": 100,
            "results_dir": str(tmp_path / "results"),
        },
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.dump(config))
    return str(path), config


class TestBenchmarkRunner:
    def test_init_creates_results_dir(self, minimal_config):
        config_path, config = minimal_config
        runner = BenchmarkRunner(config_path)
        assert os.path.isdir(config["settings"]["results_dir"])

    def test_filter_models_for_vision_dataset(self, minimal_config):
        config_path, _ = minimal_config
        runner = BenchmarkRunner(config_path)
        models = runner._filter_models(vision_only=True)
        assert len(models) == 0

    def test_filter_models_for_text_dataset(self, minimal_config):
        config_path, _ = minimal_config
        runner = BenchmarkRunner(config_path)
        models = runner._filter_models(vision_only=False)
        assert len(models) == 1

    def test_run_stores_results_json(self, minimal_config):
        config_path, config = minimal_config

        from benchmark.datasets.base import Sample
        mock_samples = [
            Sample(id="1", prompt="Q1", reference="A", metadata={}),
            Sample(id="2", prompt="Q2", reference="B", metadata={}),
        ]

        with patch("benchmark.run.DATASET_REGISTRY", {
            "kmmlu": MagicMock(return_value=MagicMock(
                load_samples=MagicMock(return_value=mock_samples),
                name="kmmlu",
                requires_vision=False,
            ))
        }), patch("benchmark.run.EVALUATOR_REGISTRY", {
            "exact_match": MagicMock(return_value=MagicMock(
                evaluate=MagicMock(return_value={"score": 0.5, "correct": 1, "total": 2, "details": []})
            ))
        }), patch("benchmark.run.BenchmarkClient") as MockClient:
            mock_client_instance = MagicMock()
            mock_client_instance.generate = AsyncMock(return_value="A")
            MockClient.return_value = mock_client_instance

            runner = BenchmarkRunner(config_path)
            asyncio.run(runner.run())

            results_dir = config["settings"]["results_dir"]
            result_files = os.listdir(results_dir)
            json_files = [f for f in result_files if f.endswith(".json")]
            assert len(json_files) == 1

            with open(os.path.join(results_dir, json_files[0])) as f:
                data = json.load(f)
            assert "kmmlu" in data["results"]


class TestIntegration:
    """Smoke test: loads real config, mocks only API calls."""

    def test_full_pipeline_with_mocked_api(self, tmp_path):
        config = {
            "models": [
                {"name": "model-a", "base_url": "http://localhost:8000/v1",
                 "model_id": "test/a", "vision": True},
                {"name": "model-b", "base_url": "http://localhost:8000/v1",
                 "model_id": "test/b", "vision": False},
            ],
            "judge": {"base_url": "http://judge:8080/v1", "model_id": "claude-opus", "api_key": ""},
            "datasets": {
                "kmmlu": {"enabled": True, "samples": 3, "evaluator": "exact_match"},
                "gsm8k": {"enabled": True, "samples": 3, "evaluator": "exact_match"},
                "docvqa": {"enabled": True, "samples": 2, "evaluator": "anls",
                           "vision_only": True, "anls_judge_threshold": 0.5},
            },
            "settings": {
                "temperatures": [0.0],
                "stochastic_runs": 1,
                "concurrent_requests": 2,
                "timeout": 10,
                "max_tokens": 100,
                "results_dir": str(tmp_path / "results"),
            },
        }
        config_path = str(tmp_path / "config.yaml")
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        from benchmark.datasets.base import Sample

        kmmlu_samples = [
            Sample(id="k1", prompt="Q?", reference="A", metadata={}),
            Sample(id="k2", prompt="Q?", reference="B", metadata={}),
            Sample(id="k3", prompt="Q?", reference="C", metadata={}),
        ]
        gsm8k_samples = [
            Sample(id="g1", prompt="Math?", reference="42", metadata={}),
            Sample(id="g2", prompt="Math?", reference="7", metadata={}),
            Sample(id="g3", prompt="Math?", reference="100", metadata={}),
        ]
        docvqa_samples = [
            Sample(id="d1", prompt=[{"type": "text", "text": "Q?"}], reference="yes", metadata={}),
            Sample(id="d2", prompt=[{"type": "text", "text": "Q?"}], reference="no", metadata={}),
        ]

        mock_datasets = {
            "kmmlu": MagicMock(return_value=MagicMock(
                load_samples=MagicMock(return_value=kmmlu_samples),
                name="kmmlu", requires_vision=False,
            )),
            "gsm8k": MagicMock(return_value=MagicMock(
                load_samples=MagicMock(return_value=gsm8k_samples),
                name="gsm8k", requires_vision=False,
            )),
            "docvqa": MagicMock(return_value=MagicMock(
                load_samples=MagicMock(return_value=docvqa_samples),
                name="docvqa", requires_vision=True,
            )),
        }

        with patch("benchmark.run.DATASET_REGISTRY", mock_datasets), \
             patch("benchmark.run.BenchmarkClient") as MockClient:

            mock_instance = MagicMock()
            mock_instance.generate = AsyncMock(return_value="A")
            MockClient.return_value = mock_instance

            runner = BenchmarkRunner(config_path)
            asyncio.run(runner.run())

            result_files = [f for f in os.listdir(str(tmp_path / "results")) if f.endswith(".json")]
            assert len(result_files) == 1

            with open(str(tmp_path / "results" / result_files[0])) as f:
                data = json.load(f)

            # kmmlu: both models ran
            assert "model-a" in data["results"]["kmmlu"]
            assert "model-b" in data["results"]["kmmlu"]
            # docvqa: only model-a (vision=True), model-b skipped
            assert "model-a" in data["results"]["docvqa"]
            assert "model-b" not in data["results"]["docvqa"]
