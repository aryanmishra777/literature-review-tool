from abc import ABC, abstractmethod
from models import CSLRecord


class BaseTranslator(ABC):
    @abstractmethod
    def search(self, query: str, limit: int | None = None, workers: int = 2) -> list[CSLRecord]:
        ...
