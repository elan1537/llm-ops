import pytest
import yaml

def write_yaml(tmp_path, data):
    path = tmp_path / "models.yaml"
    path.write_text(yaml.dump(data, default_flow_style=False))
    return str(path)

class TestLoadConfig:
    def test_loads_valid_config(self, tmp_path):
        from generate import load_config
        data = {
            "global": {"model_base_path": "/mnt/models", "vllm_image": "vllm/vllm-openai:v0.8.5", "gateway_port": 8000, "api_key": ""},
            "models": {"test-model": {"enabled": True, "model_id": "org/model", "model_path": "model-dir", "port": 8001, "gpus": [0], "tensor_parallel": 1, "dtype": "bfloat16", "quantization": "gptq", "max_model_len": 8192, "gpu_memory_utilization": 0.9, "max_num_seqs": 64, "max_num_batched_tokens": 8192, "swap_space": 4, "extra_args": []}},
        }
        config = load_config(write_yaml(tmp_path, data))
        assert config["global"]["gateway_port"] == 8000
        assert "test-model" in config["models"]

    def test_missing_global_raises(self, tmp_path):
        from generate import load_config
        with pytest.raises(SystemExit):
            load_config(write_yaml(tmp_path, {"models": {}}))

    def test_missing_models_raises(self, tmp_path):
        from generate import load_config
        with pytest.raises(SystemExit):
            load_config(write_yaml(tmp_path, {"global": {"model_base_path": "/mnt/models", "vllm_image": "img", "gateway_port": 8000, "api_key": ""}}))

class TestValidation:
    def _base_config(self):
        return {"global": {"model_base_path": "/tmp/models", "vllm_image": "vllm/vllm-openai:v0.8.5", "gateway_port": 8000, "api_key": ""}, "models": {}}

    def _model(self, **overrides):
        base = {"enabled": True, "model_id": "org/model", "model_path": "model-dir", "port": 8001, "gpus": [0], "tensor_parallel": 1, "dtype": "bfloat16", "quantization": "gptq", "max_model_len": 8192, "gpu_memory_utilization": 0.9, "max_num_seqs": 64, "max_num_batched_tokens": 8192, "swap_space": 4, "extra_args": []}
        base.update(overrides)
        return base

    def test_gpu_conflict_raises(self, tmp_path):
        from generate import validate_config
        config = self._base_config()
        config["models"]["a"] = self._model(port=8001, gpus=[0, 1])
        config["models"]["b"] = self._model(port=8002, gpus=[1, 2])
        with pytest.raises(SystemExit):
            validate_config(config)

    def test_port_conflict_raises(self, tmp_path):
        from generate import validate_config
        config = self._base_config()
        config["models"]["a"] = self._model(port=8001, gpus=[0])
        config["models"]["b"] = self._model(port=8001, gpus=[1])
        with pytest.raises(SystemExit):
            validate_config(config)

    def test_no_conflict_passes(self, tmp_path):
        from generate import validate_config
        config = self._base_config()
        config["models"]["a"] = self._model(port=8001, gpus=[0])
        config["models"]["b"] = self._model(port=8002, gpus=[1])
        validate_config(config)

    def test_disabled_model_skipped(self, tmp_path):
        from generate import validate_config
        config = self._base_config()
        config["models"]["a"] = self._model(port=8001, gpus=[0])
        config["models"]["b"] = self._model(port=8001, gpus=[0], enabled=False)
        validate_config(config)
