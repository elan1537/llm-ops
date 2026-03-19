"""
Reads models.yaml and generates docker-compose.yml for vLLM serving infrastructure.
Usage: python generate.py [--config models.yaml]
"""
import sys
import argparse
from pathlib import Path
import yaml


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        config = yaml.safe_load(f)
    if "global" not in config:
        print("Error: 'global' section missing in config", file=sys.stderr)
        sys.exit(1)
    if "models" not in config:
        print("Error: 'models' section missing in config", file=sys.stderr)
        sys.exit(1)
    for key in ["model_base_path", "vllm_image", "gateway_port"]:
        if key not in config["global"]:
            print(f"Error: 'global.{key}' missing in config", file=sys.stderr)
            sys.exit(1)
    return config


def _enabled_models(config: dict) -> dict:
    return {name: model for name, model in config["models"].items() if model.get("enabled", True)}


def validate_config(config: dict) -> None:
    models = _enabled_models(config)
    gpu_assignments = {}
    for name, model in models.items():
        for gpu_id in model["gpus"]:
            if gpu_id in gpu_assignments:
                print(f"Error: GPU {gpu_id} conflict between '{gpu_assignments[gpu_id]}' and '{name}'", file=sys.stderr)
                sys.exit(1)
            gpu_assignments[gpu_id] = name
    port_assignments = {}
    for name, model in models.items():
        port = model["port"]
        if port in port_assignments:
            print(f"Error: Port {port} conflict between '{port_assignments[port]}' and '{name}'", file=sys.stderr)
            sys.exit(1)
        port_assignments[port] = name


def _build_vllm_command(model_name: str, model: dict, base_path: str) -> list[str]:
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
    global_cfg = config["global"]
    models = _enabled_models(config)
    base_path = global_cfg["model_base_path"]
    services = {}
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
                "test": "python -c \"import urllib.request; urllib.request.urlopen('http://localhost:8000/health')\"",
                "interval": "30s",
                "timeout": "10s",
                "retries": 3,
                "start_period": "300s",
            },
        }
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
    return yaml.dump({"services": services}, default_flow_style=False, sort_keys=False)


def main():
    parser = argparse.ArgumentParser(description="Generate docker-compose.yml from models.yaml")
    parser.add_argument("--config", default="models.yaml", help="Path to models.yaml")
    parser.add_argument("--output", default="docker-compose.yml", help="Output path")
    args = parser.parse_args()
    config = load_config(args.config)
    validate_config(config)
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
