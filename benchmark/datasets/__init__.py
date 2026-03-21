DATASET_REGISTRY: dict[str, type] = {}


def register_dataset(name: str):
    def decorator(cls):
        DATASET_REGISTRY[name] = cls
        return cls
    return decorator
