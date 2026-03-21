from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class Sample:
    id: str
    prompt: str | list  # str for text, list[dict] for vision (OpenAI content format)
    reference: str
    metadata: dict = field(default_factory=dict)


class BaseDataset(ABC):
    def __init__(self, config: dict):
        self.config = config

    @abstractmethod
    def load_samples(self, n: int, seed: int = 42) -> list[Sample]:
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @property
    def requires_vision(self) -> bool:
        return False
