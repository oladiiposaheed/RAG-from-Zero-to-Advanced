"""
RAG service for AgriVoice.

Uses Protocol-based typing so any object with .fetch() and .generate()
can be plugged in — real or fake.
"""

from typing import Protocol, runtime_checkable


@runtime_checkable
class RetrieverProtocol(Protocol):
    """Any object with a .fetch(query) method returning a list of strings."""

    def fetch(self, query: str) -> list[str]:
        ...


@runtime_checkable
class LLMProtocol(Protocol):
    """Any object with a .generate(prompt) method returning a string."""

    def generate(self, prompt: str) -> str:
        ...


class RAGService:
    """RAG service typed with Protocols."""

    def __init__(
        self,
        retriever: RetrieverProtocol,
        llm: LLMProtocol,
    ):
        self._retriever = retriever
        self._llm = llm
        self._is_ready = False

    @property
    def is_ready(self) -> bool:
        return self._is_ready

    def warm_up(self) -> None:
        self._is_ready = True
        print('RAGService warmed up')

    def ask(self, question: str) -> str:
        if not self._is_ready:
            raise RuntimeError('Call warm_up() first')
        chunks = self._retriever.fetch(question)
        prompt = f'Context: {chunks}. Question: {question}'
        return self._llm.generate(prompt)
