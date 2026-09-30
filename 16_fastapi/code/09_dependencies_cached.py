"""
Dependency injection with caching.

Shows the production pattern: load a service ONCE, reuse it forever.
"""

from functools import lru_cache
from fastapi import FastAPI, Depends


app = FastAPI(title='AgriVoice API')


class RAGService:
    """A fake RAG service — in production, this loads Chroma."""

    def __init__(self, top_k: int = 4):
        print(f'🔧 Loading RAGService (top_k={top_k})...')
        self.top_k = top_k
        self.loaded = True

    def ask(self, question: str) -> str:
        return f'[top_k={self.top_k}] Answer to: {question}'


@lru_cache
def get_rag_service() -> RAGService:
    """Loaded once, cached forever — no arguments, no problems."""
    return RAGService(top_k=4)


@app.post('/ask')
def ask(question: str, service: RAGService = Depends(get_rag_service)):
    """Answer using the shared, cached RAG service."""
    return {
        'question': question,
        'answer': service.ask(question),
        'service_loaded': service.loaded,
    }