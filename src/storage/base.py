from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from langchain_core.documents import Document

class VectorStore(ABC):
    @abstractmethod
    def upsert(self, documents: List[Document]) -> None:
        pass

    @abstractmethod
    def search(self, query: str, k: int = 10) -> List[Document]:
        pass

class ShortTermMemoryStore(ABC):
    @abstractmethod
    def add(self, data: Dict[str, Any]) -> None:
        pass

    @abstractmethod
    def get(self, limit: int = 10) -> List[Dict[str, Any]]:
        pass

class DocumentStore(ABC):
    """Optional: for storing raw large docs if not in vector DB"""
    @abstractmethod
    def save(self, doc_id: str, content: Any) -> None:
        pass

    @abstractmethod
    def load(self, doc_id: str) -> Any:
        pass
