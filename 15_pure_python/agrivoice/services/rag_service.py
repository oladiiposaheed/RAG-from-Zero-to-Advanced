
class RAGError(Exception):
    """Custom exception for RAG errors."""
    pass


class RAGService:
    """Production RAG service using composition."""

    def __init__(self, retriever, llm):
        self._retriever = retriever
        self._llm = llm
        self._request_count = 0
        self._is_ready = False

    @property
    def is_ready(self) -> bool:
        return self._is_ready

    @property
    def request_count(self) -> int:
        return self._request_count

    def warm_up(self) -> None:
        self._is_ready = True
        print('RAGService warmed up')

    def ask(self, question: str) -> str:
        if not self._is_ready:
            raise RAGError('Call warm_up() first')
        chunks = self._retriever.fetch(question)
        prompt = f'Context: {chunks}. Question: {question}'
        answer = self._llm.generate(prompt)
        self._request_count += 1
        return answer
