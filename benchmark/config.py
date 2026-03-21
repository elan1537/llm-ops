import yaml

VALID_EVALUATORS = {"exact_match", "f1_em", "anls", "llm_judge"}


class ConfigError(Exception):
    pass


def load_config(path: str) -> dict:
    with open(path) as f:
        config = yaml.safe_load(f)

    for section in ("models", "settings"):
        if section not in config:
            raise ConfigError(f"Missing required section: {section}")

    return config


def validate_config(config: dict) -> None:
    names = [m["name"] for m in config["models"]]
    if len(names) != len(set(names)):
        dupes = [n for n in names if names.count(n) > 1]
        raise ConfigError(f"Duplicate model name: {dupes[0]}")

    for ds_name, ds_cfg in config.get("datasets", {}).items():
        if not ds_cfg.get("enabled", False):
            continue
        evaluator = ds_cfg.get("evaluator")
        if evaluator not in VALID_EVALUATORS:
            raise ConfigError(f"Unknown evaluator '{evaluator}' in dataset '{ds_name}'")
