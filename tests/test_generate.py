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

class TestGenerateCompose:
    def _base_config(self):
        return {"global": {"model_base_path": "/mnt/models", "vllm_image": "vllm/vllm-openai:v0.8.5", "gateway_port": 8000, "api_key": ""}, "models": {"test-model": {"enabled": True, "model_id": "org/model", "model_path": "model-dir", "port": 8001, "gpus": [0, 1], "tensor_parallel": 2, "dtype": "bfloat16", "quantization": "gptq", "max_model_len": 8192, "gpu_memory_utilization": 0.92, "max_num_seqs": 64, "max_num_batched_tokens": 8192, "swap_space": 4, "extra_args": ["--trust-remote-code"]}}}

    def test_generates_valid_yaml(self):
        from generate import generate_compose
        parsed = yaml.safe_load(generate_compose(self._base_config()))
        assert "services" in parsed

    def test_vllm_service_has_correct_image(self):
        from generate import generate_compose
        parsed = yaml.safe_load(generate_compose(self._base_config()))
        assert parsed["services"]["test-model"]["image"] == "vllm/vllm-openai:v0.8.5"

    def test_vllm_service_has_gpu_reservation(self):
        from generate import generate_compose
        parsed = yaml.safe_load(generate_compose(self._base_config()))
        devices = parsed["services"]["test-model"]["deploy"]["resources"]["reservations"]["devices"]
        assert devices[0]["device_ids"] == ["0", "1"]
        assert "gpu" in devices[0]["capabilities"]

    def test_vllm_service_has_correct_port(self):
        from generate import generate_compose
        parsed = yaml.safe_load(generate_compose(self._base_config()))
        assert "8001:8000" in parsed["services"]["test-model"]["ports"]

    def test_vllm_command_contains_model_path(self):
        from generate import generate_compose
        parsed = yaml.safe_load(generate_compose(self._base_config()))
        command = parsed["services"]["test-model"]["command"]
        assert "--model" in command
        assert "/models/model-dir" in command

    def test_vllm_command_contains_tensor_parallel(self):
        from generate import generate_compose
        parsed = yaml.safe_load(generate_compose(self._base_config()))
        command = parsed["services"]["test-model"]["command"]
        idx = command.index("--tensor-parallel-size")
        assert command[idx + 1] == "2"

    def test_gateway_service_exists(self):
        from generate import generate_compose
        parsed = yaml.safe_load(generate_compose(self._base_config()))
        assert "gateway" in parsed["services"]
        assert "8000:8000" in parsed["services"]["gateway"]["ports"]

    def test_gateway_depends_on_models(self):
        from generate import generate_compose
        parsed = yaml.safe_load(generate_compose(self._base_config()))
        assert "test-model" in parsed["services"]["gateway"]["depends_on"]

    def test_vllm_has_restart_policy(self):
        from generate import generate_compose
        parsed = yaml.safe_load(generate_compose(self._base_config()))
        assert parsed["services"]["test-model"]["restart"] == "unless-stopped"

    def test_vllm_has_log_rotation(self):
        from generate import generate_compose
        parsed = yaml.safe_load(generate_compose(self._base_config()))
        assert parsed["services"]["test-model"]["logging"]["options"]["max-size"] == "100m"

    def test_vllm_has_healthcheck(self):
        from generate import generate_compose
        parsed = yaml.safe_load(generate_compose(self._base_config()))
        hc = parsed["services"]["test-model"]["healthcheck"]
        assert "localhost:8000/health" in hc["test"]

    def test_disabled_model_excluded(self):
        from generate import generate_compose
        config = self._base_config()
        config["models"]["disabled-model"] = {**config["models"]["test-model"], "enabled": False, "port": 8002, "gpus": [2]}
        parsed = yaml.safe_load(generate_compose(config))
        assert "disabled-model" not in parsed["services"]
