import pytest
import yaml

from benchmark.config import load_config, validate_config, ConfigError


@pytest.fixture
def write_yaml(tmp_path):
    def _write(data):
        path = tmp_path / "config.yaml"
        path.write_text(yaml.dump(data, default_flow_style=False))
        return str(path)
    return _write


VALID_CONFIG = {
    "models": [
        {"name": "test-model", "base_url": "http://localhost:8000/v1",
         "model_id": "test/model", "vision": False}
    ],
    "judge": {"base_url": "http://judge:8080/v1", "model_id": "claude-opus-4-20250514", "api_key": ""},
    "datasets": {
        "kmmlu": {"enabled": True, "samples": 10, "evaluator": "exact_match"}
    },
    "settings": {
        "temperatures": [0.0],
        "stochastic_runs": 3,
        "concurrent_requests": 4,
        "timeout": 120,
        "max_tokens": 2048,
        "results_dir": "benchmark/results",
    },
}


class TestLoadConfig:
    def test_loads_valid_config(self, write_yaml):
        path = write_yaml(VALID_CONFIG)
        config = load_config(path)
        assert len(config["models"]) == 1
        assert config["models"][0]["name"] == "test-model"

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/config.yaml")

    def test_missing_models_section_raises(self, write_yaml):
        path = write_yaml({"datasets": {}, "settings": {}})
        with pytest.raises(ConfigError, match="models"):
            load_config(path)

    def test_missing_settings_section_raises(self, write_yaml):
        data = {**VALID_CONFIG}
        del data["settings"]
        path = write_yaml(data)
        with pytest.raises(ConfigError, match="settings"):
            load_config(path)


class TestValidateConfig:
    def test_valid_config_passes(self, write_yaml):
        path = write_yaml(VALID_CONFIG)
        config = load_config(path)
        validate_config(config)

    def test_duplicate_model_names_raises(self, write_yaml):
        data = {**VALID_CONFIG, "models": [
            {"name": "dup", "base_url": "http://a/v1", "model_id": "a", "vision": False},
            {"name": "dup", "base_url": "http://b/v1", "model_id": "b", "vision": False},
        ]}
        path = write_yaml(data)
        config = load_config(path)
        with pytest.raises(ConfigError, match="Duplicate model name"):
            validate_config(config)

    def test_unknown_evaluator_raises(self, write_yaml):
        data = {**VALID_CONFIG, "datasets": {
            "test": {"enabled": True, "samples": 10, "evaluator": "nonexistent"}
        }}
        path = write_yaml(data)
        config = load_config(path)
        with pytest.raises(ConfigError, match="Unknown evaluator"):
            validate_config(config)
