"""
Tests for the RAGService class.

Uses fake dependencies to test the service in isolation.
"""

import pytest

from agrivoice.services.rag_service import RAGService


# --- Fake dependencies ---

class FakeRetriever:
    """Minimal retriever — matches RetrieverProtocol."""

    def fetch(self, query: str) -> list[str]:
        return [f'doc about {query}']


class FakeLLM:
    """Minimal LLM — matches LLMProtocol."""

    def generate(self, prompt: str) -> str:
        return f'Answer: {prompt[:40]}...'


# --- Fixture ---

@pytest.fixture
def service():
    """Warmed-up RAGService for every test that needs one."""
    s = RAGService(retriever=FakeRetriever(), llm=FakeLLM())
    s.warm_up()
    return s


# --- Tests ---

def test_service_starts_not_ready():
    """A fresh service should not be ready."""
    s = RAGService(retriever=FakeRetriever(), llm=FakeLLM())
    assert s.is_ready is False


def test_warm_up_makes_ready():
    """warm_up() should set is_ready to True."""
    s = RAGService(retriever=FakeRetriever(), llm=FakeLLM())
    s.warm_up()
    assert s.is_ready is True


def test_ask_before_warm_up_raises():
    """Calling ask() before warm_up() should raise."""
    s = RAGService(retriever=FakeRetriever(), llm=FakeLLM())
    with pytest.raises(RuntimeError):
        s.ask('anything')


def test_ask_returns_string(service):
    """A warmed-up service should return a string answer."""
    answer = service.ask('How do I treat cassava mosaic?')
    assert isinstance(answer, str)
    assert len(answer) > 0


def test_ask_includes_question(service):
    """The answer should include the original question."""
    answer = service.ask('What is maize smut?')
    assert 'maize smut' in answer
