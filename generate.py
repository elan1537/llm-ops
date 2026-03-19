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
