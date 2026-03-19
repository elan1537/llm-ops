# LLM Serving Infrastructure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a YAML-driven LLM serving infrastructure that auto-generates Docker Compose configs for vLLM containers with a FastAPI gateway proxy.

**Architecture:** `models.yaml` defines models and GPU assignments. `generate.py` reads it and produces `docker-compose.yml`. A FastAPI gateway container routes OpenAI-compatible API requests to the correct vLLM backend by parsing the `model` field from the request body.

**Tech Stack:** Python 3.11, FastAPI, httpx, uvicorn, PyYAML, Docker Compose, vLLM (vllm/vllm-openai)

**Spec:** `docs/superpowers/specs/2026-03-19-llm-serving-infra-design.md`

---

## File Map

| File | Responsibility |
|---|---|
| `models.yaml` | User-facing config: model definitions, GPU assignments, vLLM params |
| `generate.py` | Reads models.yaml, validates config, writes docker-compose.yml |
| `gateway/__init__.py` | Package marker (empty) |
| `gateway/main.py` | FastAPI proxy: model routing, /v1/models aggregation, /health, SSE streaming |
| `gateway/Dockerfile` | Container image for the gateway |
| `gateway/requirements.txt` | Python dependencies for the gateway |
| `.gitignore` | Ignore generated docker-compose.yml |
| `tests/__init__.py` | Package marker (empty) |
| `tests/test_generate.py` | Tests for generate.py validation and output |
| `tests/test_gateway.py` | Tests for gateway routing, health, error handling, streaming, 503 |

---

## Task 1: models.yaml

**Files:**
- Create: `models.yaml`

- [ ] **Step 1: Create models.yaml**

```yaml
global:
  model_base_path: /mnt/models
  vllm_image: vllm/vllm-openai:v0.8.5
  gateway_port: 8000
  api_key: ""

models:
  qwen3.5-122b:
    enabled: true
    model_id: Qwen/Qwen3.5-122B-A10B-GPTQ-Int4
    model_path: Qwen3.5-122B-A10B-GPTQ-Int4
    port: 8001
    gpus: [0, 1]
    tensor_parallel: 2
    dtype: bfloat16
    quantization: gptq
    max_model_len: 32768
    gpu_memory_utilization: 0.92
    max_num_seqs: 64
    max_num_batched_tokens: 32768
    swap_space: 4
    extra_args:
      - "--trust-remote-code"
      - "--reasoning-parser=qwen3"
      - "--limit-mm-per-prompt=image=5"

  qwen3.5-35b:
    enabled: true
    model_id: Qwen/Qwen3.5-35B-A3B-GPTQ-Int4
    model_path: Qwen3.5-35B-A3B-GPTQ-Int4
    port: 8002
    gpus: [2]
    tensor_parallel: 1
    dtype: bfloat16
    quantization: gptq
    max_model_len: 65536
    gpu_memory_utilization: 0.92
    max_num_seqs: 128
    max_num_batched_tokens: 32768
    swap_space: 4
    extra_args:
      - "--trust-remote-code"
      - "--reasoning-parser=qwen3"
      - "--limit-mm-per-prompt=image=5"
```

- [ ] **Step 2: Create .gitignore**

```
docker-compose.yml
__pycache__/
*.pyc
.pytest_cache/
```

- [ ] **Step 3: Create package marker files**

```bash
touch gateway/__init__.py tests/__init__.py
```

- [ ] **Step 4: Commit**

```bash
mkdir -p gateway tests
git add models.yaml .gitignore gateway/__init__.py tests/__init__.py
git commit -m "feat: add models.yaml config, package markers, and .gitignore"
```

---

## Task 2: generate.py — Config Loading & Validation

**Files:**
- Create: `generate.py`
- Create: `tests/test_generate.py`

- [ ] **Step 1: Write failing tests for config loading and validation**

```python
# tests/test_generate.py
import pytest
import yaml
import tempfile
import os
from pathlib import Path


def write_yaml(tmp_path, data):
    """Write a YAML config to a temp file and return the path."""
    path = tmp_path / "models.yaml"
    path.write_text(yaml.dump(data, default_flow_style=False))
    return str(path)


# --- Import will be tested after implementation ---

class TestLoadConfig:
    def test_loads_valid_config(self, tmp_path):
        from generate import load_config
        data = {
            "global": {
                "model_base_path": "/mnt/models",
                "vllm_image": "vllm/vllm-openai:v0.8.5",
                "gateway_port": 8000,
                "api_key": "",
            },
            "models": {
                "test-model": {
                    "enabled": True,
                    "model_id": "org/model",
                    "model_path": "model-dir",
                    "port": 8001,
                    "gpus": [0],
                    "tensor_parallel": 1,
                    "dtype": "bfloat16",
                    "quantization": "gptq",
                    "max_model_len": 8192,
                    "gpu_memory_utilization": 0.9,
                    "max_num_seqs": 64,
                    "max_num_batched_tokens": 8192,
                    "swap_space": 4,
                    "extra_args": [],
                },
            },
        }
        config = load_config(write_yaml(tmp_path, data))
        assert config["global"]["gateway_port"] == 8000
        assert "test-model" in config["models"]

    def test_missing_global_raises(self, tmp_path):
        from generate import load_config
        data = {"models": {}}
        with pytest.raises(SystemExit):
            load_config(write_yaml(tmp_path, data))

    def test_missing_models_raises(self, tmp_path):
        from generate import load_config
        data = {"global": {"model_base_path": "/mnt/models", "vllm_image": "img", "gateway_port": 8000, "api_key": ""}}
        with pytest.raises(SystemExit):
            load_config(write_yaml(tmp_path, data))


class TestValidation:
    def _base_config(self):
        return {
            "global": {
                "model_base_path": "/tmp/models",
                "vllm_image": "vllm/vllm-openai:v0.8.5",
                "gateway_port": 8000,
                "api_key": "",
            },
            "models": {},
        }

    def _model(self, **overrides):
        base = {
            "enabled": True,
            "model_id": "org/model",
            "model_path": "model-dir",
            "port": 8001,
            "gpus": [0],
            "tensor_parallel": 1,
            "dtype": "bfloat16",
            "quantization": "gptq",
            "max_model_len": 8192,
            "gpu_memory_utilization": 0.9,
            "max_num_seqs": 64,
            "max_num_batched_tokens": 8192,
            "swap_space": 4,
            "extra_args": [],
        }
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
        validate_config(config)  # should not raise

    def test_disabled_model_skipped(self, tmp_path):
        from generate import validate_config
        config = self._base_config()
        config["models"]["a"] = self._model(port=8001, gpus=[0])
        config["models"]["b"] = self._model(port=8001, gpus=[0], enabled=False)
        validate_config(config)  # disabled model should be ignored
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /home/separk/workspace/infra && python -m pytest tests/test_generate.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'generate'`

- [ ] **Step 3: Implement load_config and validate_config**

```python
# generate.py
"""
Reads models.yaml and generates docker-compose.yml for vLLM serving infrastructure.
Usage: python generate.py [--config models.yaml]
"""
import sys
import argparse
from pathlib import Path

import yaml


def load_config(config_path: str) -> dict:
    """Load and validate basic structure of models.yaml."""
    with open(config_path) as f:
        config = yaml.safe_load(f)

    if "global" not in config:
        print("Error: 'global' section missing in config", file=sys.stderr)
        sys.exit(1)
    if "models" not in config:
        print("Error: 'models' section missing in config", file=sys.stderr)
        sys.exit(1)

    required_global = ["model_base_path", "vllm_image", "gateway_port"]
    for key in required_global:
        if key not in config["global"]:
            print(f"Error: 'global.{key}' missing in config", file=sys.stderr)
            sys.exit(1)

    return config


def _enabled_models(config: dict) -> dict:
    """Return only enabled models."""
    return {
        name: model
        for name, model in config["models"].items()
        if model.get("enabled", True)
    }


def validate_config(config: dict) -> None:
    """Validate GPU and port assignments across enabled models."""
    models = _enabled_models(config)

    # Check GPU conflicts
    gpu_assignments: dict[int, str] = {}
    for name, model in models.items():
        for gpu_id in model["gpus"]:
            if gpu_id in gpu_assignments:
                other = gpu_assignments[gpu_id]
                print(
                    f"Error: GPU {gpu_id} conflict between '{other}' and '{name}'",
                    file=sys.stderr,
                )
                sys.exit(1)
            gpu_assignments[gpu_id] = name

    # Check port conflicts
    port_assignments: dict[int, str] = {}
    for name, model in models.items():
        port = model["port"]
        if port in port_assignments:
            other = port_assignments[port]
            print(
                f"Error: Port {port} conflict between '{other}' and '{name}'",
                file=sys.stderr,
            )
            sys.exit(1)
        port_assignments[port] = name
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /home/separk/workspace/infra && python -m pytest tests/test_generate.py -v
```

Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add generate.py tests/test_generate.py
git commit -m "feat: add config loading and validation for generate.py"
```

---

## Task 3: generate.py — Docker Compose Generation

**Files:**
- Modify: `generate.py`
- Modify: `tests/test_generate.py`

- [ ] **Step 1: Write failing tests for docker-compose generation**

Add to `tests/test_generate.py`:

```python
class TestGenerateCompose:
    def _base_config(self):
        return {
            "global": {
                "model_base_path": "/mnt/models",
                "vllm_image": "vllm/vllm-openai:v0.8.5",
                "gateway_port": 8000,
                "api_key": "",
            },
            "models": {
                "test-model": {
                    "enabled": True,
                    "model_id": "org/model",
                    "model_path": "model-dir",
                    "port": 8001,
                    "gpus": [0, 1],
                    "tensor_parallel": 2,
                    "dtype": "bfloat16",
                    "quantization": "gptq",
                    "max_model_len": 8192,
                    "gpu_memory_utilization": 0.92,
                    "max_num_seqs": 64,
                    "max_num_batched_tokens": 8192,
                    "swap_space": 4,
                    "extra_args": ["--trust-remote-code"],
                },
            },
        }

    def test_generates_valid_yaml(self):
        from generate import generate_compose
        config = self._base_config()
        result = generate_compose(config)
        parsed = yaml.safe_load(result)
        assert "services" in parsed

    def test_vllm_service_has_correct_image(self):
        from generate import generate_compose
        config = self._base_config()
        parsed = yaml.safe_load(generate_compose(config))
        assert parsed["services"]["test-model"]["image"] == "vllm/vllm-openai:v0.8.5"

    def test_vllm_service_has_gpu_reservation(self):
        from generate import generate_compose
        config = self._base_config()
        parsed = yaml.safe_load(generate_compose(config))
        devices = parsed["services"]["test-model"]["deploy"]["resources"]["reservations"]["devices"]
        assert devices[0]["device_ids"] == ["0", "1"]
        assert "gpu" in devices[0]["capabilities"]

    def test_vllm_service_has_correct_port(self):
        from generate import generate_compose
        config = self._base_config()
        parsed = yaml.safe_load(generate_compose(config))
        assert "8001:8000" in parsed["services"]["test-model"]["ports"]

    def test_vllm_command_contains_model_path(self):
        from generate import generate_compose
        config = self._base_config()
        parsed = yaml.safe_load(generate_compose(config))
        command = parsed["services"]["test-model"]["command"]
        assert "--model" in command
        assert "/models/model-dir" in command

    def test_vllm_command_contains_tensor_parallel(self):
        from generate import generate_compose
        config = self._base_config()
        parsed = yaml.safe_load(generate_compose(config))
        command = parsed["services"]["test-model"]["command"]
        idx = command.index("--tensor-parallel-size")
        assert command[idx + 1] == "2"

    def test_gateway_service_exists(self):
        from generate import generate_compose
        config = self._base_config()
        parsed = yaml.safe_load(generate_compose(config))
        assert "gateway" in parsed["services"]
        assert "8000:8000" in parsed["services"]["gateway"]["ports"]

    def test_gateway_depends_on_models(self):
        from generate import generate_compose
        config = self._base_config()
        parsed = yaml.safe_load(generate_compose(config))
        deps = parsed["services"]["gateway"]["depends_on"]
        assert "test-model" in deps

    def test_vllm_has_restart_policy(self):
        from generate import generate_compose
        config = self._base_config()
        parsed = yaml.safe_load(generate_compose(config))
        assert parsed["services"]["test-model"]["restart"] == "unless-stopped"

    def test_vllm_has_log_rotation(self):
        from generate import generate_compose
        config = self._base_config()
        parsed = yaml.safe_load(generate_compose(config))
        logging = parsed["services"]["test-model"]["logging"]
        assert logging["options"]["max-size"] == "100m"

    def test_vllm_has_healthcheck(self):
        from generate import generate_compose
        config = self._base_config()
        parsed = yaml.safe_load(generate_compose(config))
        hc = parsed["services"]["test-model"]["healthcheck"]
        assert "localhost:8000/health" in hc["test"]

    def test_disabled_model_excluded(self):
        from generate import generate_compose
        config = self._base_config()
        config["models"]["disabled-model"] = {
            **config["models"]["test-model"],
            "enabled": False,
            "port": 8002,
            "gpus": [2],
        }
        parsed = yaml.safe_load(generate_compose(config))
        assert "disabled-model" not in parsed["services"]
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /home/separk/workspace/infra && python -m pytest tests/test_generate.py::TestGenerateCompose -v
```

Expected: FAIL — `ImportError: cannot import name 'generate_compose'`

- [ ] **Step 3: Implement generate_compose**

Add to `generate.py`:

```python
def _build_vllm_command(model_name: str, model: dict, base_path: str) -> list[str]:
    """Build the vLLM CLI command list for a model."""
    model_full_path = f"/models/{model['model_path']}"
    cmd = [
        "--model", model_full_path,
        "--served-model-name", model_name,
        "--tensor-parallel-size", str(model["tensor_parallel"]),
        "--dtype", model["dtype"],
        "--max-model-len", str(model["max_model_len"]),
        "--gpu-memory-utilization", str(model["gpu_memory_utilization"]),
        "--max-num-seqs", str(model["max_num_seqs"]),
        "--max-num-batched-tokens", str(model["max_num_batched_tokens"]),
        "--swap-space", str(model["swap_space"]),
    ]
    if model.get("quantization"):
        cmd.extend(["--quantization", model["quantization"]])
    for arg in model.get("extra_args", []):
        cmd.append(arg)
    return cmd


def generate_compose(config: dict) -> str:
    """Generate docker-compose.yml content from config."""
    global_cfg = config["global"]
    models = _enabled_models(config)
    base_path = global_cfg["model_base_path"]

    services = {}

    # vLLM services
    for name, model in models.items():
        gpu_ids = [str(g) for g in model["gpus"]]
        services[name] = {
            "image": global_cfg["vllm_image"],
            "container_name": f"vllm-{name}",
            "ports": [f"{model['port']}:8000"],
            "volumes": [f"{base_path}:/models:ro"],
            "command": _build_vllm_command(name, model, base_path),
            "deploy": {
                "resources": {
                    "reservations": {
                        "devices": [{
                            "driver": "nvidia",
                            "device_ids": gpu_ids,
                            "capabilities": ["gpu"],
                        }]
                    }
                }
            },
            "restart": "unless-stopped",
            "logging": {
                "driver": "json-file",
                "options": {"max-size": "100m", "max-file": "3"},
            },
            "healthcheck": {
                "test": ["CMD-SHELL", "python -c \"import urllib.request; urllib.request.urlopen('http://localhost:8000/health')\""],
                "interval": "30s",
                "timeout": "10s",
                "retries": 3,
                "start_period": "300s",
            },
        }

    # Gateway service
    services["gateway"] = {
        "build": "./gateway",
        "container_name": "llm-gateway",
        "ports": [f"{global_cfg['gateway_port']}:8000"],
        "volumes": ["./models.yaml:/app/models.yaml:ro"],
        "depends_on": list(models.keys()),
        "restart": "unless-stopped",
        "logging": {
            "driver": "json-file",
            "options": {"max-size": "100m", "max-file": "3"},
        },
    }

    compose = {"services": services}
    return yaml.dump(compose, default_flow_style=False, sort_keys=False)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /home/separk/workspace/infra && python -m pytest tests/test_generate.py -v
```

Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add generate.py tests/test_generate.py
git commit -m "feat: add docker-compose.yml generation in generate.py"
```

---

## Task 4: generate.py — CLI Entry Point & File Writing

**Files:**
- Modify: `generate.py`
- Modify: `tests/test_generate.py`

- [ ] **Step 1: Write failing test for main function**

Add to `tests/test_generate.py`:

```python
class TestMain:
    def test_writes_compose_file(self, tmp_path):
        from generate import load_config, validate_config, generate_compose

        config_data = {
            "global": {
                "model_base_path": "/mnt/models",
                "vllm_image": "vllm/vllm-openai:v0.8.5",
                "gateway_port": 8000,
                "api_key": "",
            },
            "models": {
                "test-model": {
                    "enabled": True,
                    "model_id": "org/model",
                    "model_path": "model-dir",
                    "port": 8001,
                    "gpus": [0],
                    "tensor_parallel": 1,
                    "dtype": "bfloat16",
                    "quantization": "gptq",
                    "max_model_len": 8192,
                    "gpu_memory_utilization": 0.9,
                    "max_num_seqs": 64,
                    "max_num_batched_tokens": 8192,
                    "swap_space": 4,
                    "extra_args": [],
                },
            },
        }

        config_path = write_yaml(tmp_path, config_data)
        output_path = tmp_path / "docker-compose.yml"

        config = load_config(config_path)
        validate_config(config)
        content = generate_compose(config)

        output_path.write_text(content)
        assert output_path.exists()

        parsed = yaml.safe_load(output_path.read_text())
        assert "test-model" in parsed["services"]
        assert "gateway" in parsed["services"]
```

- [ ] **Step 2: Run test to verify it passes** (should pass with existing code)

```bash
cd /home/separk/workspace/infra && python -m pytest tests/test_generate.py::TestMain -v
```

Expected: PASS

- [ ] **Step 3: Add CLI entry point to generate.py**

Add to the bottom of `generate.py`:

```python
def main():
    parser = argparse.ArgumentParser(description="Generate docker-compose.yml from models.yaml")
    parser.add_argument("--config", default="models.yaml", help="Path to models.yaml")
    parser.add_argument("--output", default="docker-compose.yml", help="Output path")
    args = parser.parse_args()

    config = load_config(args.config)
    validate_config(config)

    # Check model paths exist (warning only)
    base_path = Path(config["global"]["model_base_path"])
    for name, model in _enabled_models(config).items():
        model_dir = base_path / model["model_path"]
        if not model_dir.exists():
            print(f"Warning: Model path not found: {model_dir} (model: {name})", file=sys.stderr)

    content = generate_compose(config)

    with open(args.output, "w") as f:
        f.write(content)

    models = _enabled_models(config)
    print(f"Generated {args.output} with {len(models)} model(s):")
    for name, model in models.items():
        print(f"  - {name}: GPU {model['gpus']}, port {model['port']}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Test CLI manually**

```bash
cd /home/separk/workspace/infra && python generate.py --config models.yaml --output /tmp/test-compose.yml && cat /tmp/test-compose.yml
```

Expected: Valid docker-compose.yml printed, with warning about missing model paths.

- [ ] **Step 5: Commit**

```bash
git add generate.py tests/test_generate.py
git commit -m "feat: add CLI entry point to generate.py"
```

---

## Task 5: Gateway — FastAPI Proxy with Model Routing

**Files:**
- Create: `gateway/requirements.txt`
- Create: `gateway/main.py`
- Create: `tests/test_gateway.py`

- [ ] **Step 1: Create gateway/requirements.txt**

```
fastapi==0.115.0
uvicorn[standard]==0.30.0
httpx==0.27.0
pyyaml==6.0.1
```

- [ ] **Step 2: Write failing tests for gateway routing**

```python
# tests/test_gateway.py
import pytest
import httpx
import json
from unittest.mock import AsyncMock, patch, MagicMock
from fastapi.testclient import TestClient


MOCK_ROUTES = {
    "model-a": "http://model-a:8000",
    "model-b": "http://model-b:8000",
}


@pytest.fixture
def client():
    with patch("gateway.main.load_routes", return_value=MOCK_ROUTES.copy()):
        from gateway.main import create_app
        app = create_app()
        yield TestClient(app)


class TestModelsEndpoint:
    def test_models_returns_list(self, client):
        response = client.get("/v1/models")
        assert response.status_code == 200
        data = response.json()
        assert "data" in data
        model_ids = [m["id"] for m in data["data"]]
        assert "model-a" in model_ids
        assert "model-b" in model_ids


class TestHealthEndpoint:
    def test_health_all_healthy(self, client):
        mock_response = httpx.Response(200, json={"status": "ok"})
        with patch("httpx.AsyncClient.get", new_callable=AsyncMock, return_value=mock_response):
            response = client.get("/health")
            assert response.status_code == 200
            assert response.json()["status"] == "healthy"

    def test_health_partial_down_returns_degraded(self, client):
        async def mock_get(url):
            if "model-a" in url:
                return httpx.Response(200, json={"status": "ok"})
            raise httpx.ConnectError("Connection refused")

        with patch("httpx.AsyncClient.get", new_callable=AsyncMock, side_effect=mock_get):
            response = client.get("/health")
            assert response.status_code == 200
            assert response.json()["status"] == "degraded"
            assert response.json()["models"]["model-a"] == "healthy"
            assert response.json()["models"]["model-b"] == "down"

    def test_health_all_down_returns_503(self, client):
        with patch("httpx.AsyncClient.get", new_callable=AsyncMock, side_effect=httpx.ConnectError("refused")):
            response = client.get("/health")
            assert response.status_code == 503
            assert response.json()["status"] == "unhealthy"


class TestRouting:
    def test_unknown_model_returns_422(self, client):
        response = client.post(
            "/v1/chat/completions",
            json={"model": "nonexistent", "messages": [{"role": "user", "content": "hi"}]},
        )
        assert response.status_code == 422
        available = response.json()["available_models"]
        assert "model-a" in available
        assert "model-b" in available

    def test_missing_model_field_returns_422(self, client):
        response = client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hi"}]},
        )
        assert response.status_code == 422

    def test_backend_down_returns_503(self, client):
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock, side_effect=httpx.ConnectError("refused")):
            response = client.post(
                "/v1/chat/completions",
                json={"model": "model-a", "messages": [{"role": "user", "content": "hi"}]},
            )
            assert response.status_code == 503
            assert "unavailable" in response.json()["error"]

    def test_proxy_forwards_to_backend(self, client):
        mock_resp = httpx.Response(200, json={"id": "chatcmpl-123", "choices": [{"message": {"content": "hello"}}]})
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_resp):
            response = client.post(
                "/v1/chat/completions",
                json={"model": "model-a", "messages": [{"role": "user", "content": "hi"}]},
            )
            assert response.status_code == 200
            assert response.json()["id"] == "chatcmpl-123"
```

- [ ] **Step 3: Run tests to verify they fail**

```bash
cd /home/separk/workspace/infra && python -m pytest tests/test_gateway.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'gateway'`

- [ ] **Step 4: Implement gateway/main.py**

```python
# gateway/main.py
"""
FastAPI gateway proxy for vLLM model serving.
Routes requests to the correct vLLM backend based on the 'model' field.
"""
import os
from pathlib import Path

import httpx
import yaml
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse


def load_routes(config_path: str = "/app/models.yaml") -> dict[str, str]:
    """Load routing table from models.yaml. Returns {model_name: upstream_url}."""
    with open(config_path) as f:
        config = yaml.safe_load(f)

    routes = {}
    for name, model in config.get("models", {}).items():
        if model.get("enabled", True):
            # Inside Docker network, service name = model name from docker-compose
            routes[name] = f"http://{name}:8000"
    return routes


def create_app(config_path: str | None = None) -> FastAPI:
    app = FastAPI(title="LLM Gateway", version="1.0.0")

    if config_path is None:
        config_path = os.environ.get("CONFIG_PATH", "/app/models.yaml")

    routes = load_routes(config_path)

    @app.get("/health")
    async def health():
        statuses = {}
        async with httpx.AsyncClient(timeout=5.0) as client:
            for name, url in routes.items():
                try:
                    resp = await client.get(f"{url}/health")
                    statuses[name] = "healthy" if resp.status_code == 200 else "unhealthy"
                except httpx.RequestError:
                    statuses[name] = "down"

        healthy_count = sum(1 for s in statuses.values() if s == "healthy")
        total = len(statuses)

        if healthy_count == total:
            status = "healthy"
            code = 200
        elif healthy_count > 0:
            status = "degraded"
            code = 200
        else:
            status = "unhealthy"
            code = 503

        return JSONResponse(
            status_code=code,
            content={"status": status, "models": statuses},
        )

    @app.get("/v1/models")
    async def list_models():
        data = [
            {"id": name, "object": "model", "owned_by": "local"}
            for name in routes
        ]
        return {"object": "list", "data": data}

    @app.api_route("/v1/{path:path}", methods=["POST"])
    async def proxy(request: Request, path: str):
        body = await request.json()
        model = body.get("model")

        if not model:
            return JSONResponse(
                status_code=422,
                content={
                    "error": "Missing 'model' field in request body",
                    "available_models": list(routes.keys()),
                },
            )

        upstream = routes.get(model)
        if not upstream:
            return JSONResponse(
                status_code=422,
                content={
                    "error": f"Unknown model: {model}",
                    "available_models": list(routes.keys()),
                },
            )

        upstream_url = f"{upstream}/v1/{path}"
        is_stream = body.get("stream", False)

        async with httpx.AsyncClient(timeout=httpx.Timeout(300.0)) as client:
            if is_stream:
                req = client.build_request(
                    "POST",
                    upstream_url,
                    json=body,
                    headers={"Content-Type": "application/json"},
                )
                resp = await client.send(req, stream=True)

                async def stream_response():
                    try:
                        async for chunk in resp.aiter_bytes():
                            yield chunk
                    finally:
                        await resp.aclose()

                return StreamingResponse(
                    stream_response(),
                    status_code=resp.status_code,
                    media_type="text/event-stream",
                )
            else:
                try:
                    resp = await client.post(
                        upstream_url,
                        json=body,
                        headers={"Content-Type": "application/json"},
                    )
                    return JSONResponse(
                        status_code=resp.status_code,
                        content=resp.json(),
                    )
                except httpx.RequestError:
                    return JSONResponse(
                        status_code=503,
                        content={"error": f"Backend '{model}' is unavailable"},
                    )

    return app


def _get_app():
    """Lazy app factory for uvicorn. Only called at container startup."""
    return create_app()
```

> Note: The Dockerfile CMD uses `main:app` — add this to the bottom of main.py for uvicorn:
> `app = _get_app()` is NOT at module level. Instead, update the Dockerfile CMD to use the factory:
> `CMD ["uvicorn", "main:_get_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]`

- [ ] **Step 5: Run tests to verify they pass**

```bash
cd /home/separk/workspace/infra && python -m pytest tests/test_gateway.py -v
```

Expected: All PASS

- [ ] **Step 6: Commit**

```bash
git add gateway/main.py gateway/requirements.txt tests/test_gateway.py
git commit -m "feat: add FastAPI gateway with model routing and health aggregation"
```

---

## Task 6: Gateway — Dockerfile

**Files:**
- Create: `gateway/Dockerfile`

- [ ] **Step 1: Create gateway/Dockerfile**

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .

EXPOSE 8000

CMD ["uvicorn", "main:_get_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
```

- [ ] **Step 2: Verify Dockerfile builds**

```bash
cd /home/separk/workspace/infra && docker build -t llm-gateway:test ./gateway
```

Expected: Build succeeds.

- [ ] **Step 3: Commit**

```bash
git add gateway/Dockerfile
git commit -m "feat: add gateway Dockerfile"
```

---

## Task 7: End-to-End Integration Test

**Files:**
- Modify: `tests/test_generate.py`

This task validates the full pipeline: models.yaml → generate.py → docker-compose.yml → valid structure.

- [ ] **Step 1: Write integration test**

Add to `tests/test_generate.py`:

```python
class TestEndToEnd:
    def test_full_pipeline_with_real_config(self):
        """Test with the actual models.yaml from the project."""
        from generate import load_config, validate_config, generate_compose

        config = load_config("models.yaml")
        validate_config(config)
        content = generate_compose(config)
        parsed = yaml.safe_load(content)

        # Should have both models + gateway
        assert "qwen3.5-122b" in parsed["services"]
        assert "qwen3.5-35b" in parsed["services"]
        assert "gateway" in parsed["services"]

        # 122b should have 2 GPUs
        devices_122b = parsed["services"]["qwen3.5-122b"]["deploy"]["resources"]["reservations"]["devices"]
        assert devices_122b[0]["device_ids"] == ["0", "1"]

        # 35b should have 1 GPU
        devices_35b = parsed["services"]["qwen3.5-35b"]["deploy"]["resources"]["reservations"]["devices"]
        assert devices_35b[0]["device_ids"] == ["2"]

        # Gateway should depend on both models
        deps = parsed["services"]["gateway"]["depends_on"]
        assert "qwen3.5-122b" in deps
        assert "qwen3.5-35b" in deps
```

- [ ] **Step 2: Run full test suite**

```bash
cd /home/separk/workspace/infra && python -m pytest tests/ -v
```

Expected: All PASS

- [ ] **Step 3: Generate the actual docker-compose.yml**

```bash
cd /home/separk/workspace/infra && python generate.py
```

Expected: `docker-compose.yml` created with correct content.

- [ ] **Step 4: Commit**

```bash
git add tests/test_generate.py
git commit -m "test: add end-to-end integration test for full pipeline"
```

---

## Task 8: Final Verification & Cleanup

**Files:**
- No new files

- [ ] **Step 1: Run full test suite one final time**

```bash
cd /home/separk/workspace/infra && python -m pytest tests/ -v --tb=short
```

Expected: All tests PASS.

- [ ] **Step 2: Verify generated docker-compose.yml is correct**

```bash
cd /home/separk/workspace/infra && python generate.py && docker compose config
```

Expected: `docker compose config` validates the generated file without errors.

- [ ] **Step 3: Verify directory structure**

```bash
ls -la /home/separk/workspace/infra/{models.yaml,generate.py,gateway/,.gitignore}
ls -la /home/separk/workspace/infra/gateway/{Dockerfile,main.py,requirements.txt}
ls -la /home/separk/workspace/infra/tests/
```

Expected: All files present.

- [ ] **Step 4: Final commit (if any uncommitted changes)**

```bash
git status
# If clean, no action needed
```

---

## Post-Implementation Notes

### To actually serve models (not part of this plan):

1. Download model weights to `/mnt/models/`
2. `python generate.py`
3. `docker compose up -d`
4. `curl http://localhost:8000/v1/models`

### Future tasks (separate plans):

- API key authentication in gateway
- Prometheus metrics endpoint
- Adding more models to GPU 3
