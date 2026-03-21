EVALUATOR_REGISTRY: dict[str, type] = {}


def register_evaluator(name: str):
    def decorator(cls):
        EVALUATOR_REGISTRY[name] = cls
        return cls
    return decorator
