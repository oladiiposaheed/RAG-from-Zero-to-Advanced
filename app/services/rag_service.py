"""
RAG service.

Uses the vector store retriever and an LLM to answer questions
with context retrieved from stored documents.
Supports simple retrieval, multi-query, contextual compression, self-query,
HyDE, query decomposition, and combinations.
"""

from typing import List

from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.chains import HypotheticalDocumentEmbedder

from app.config.config import config
from app.logger.custom_logger import get_logger
from app.exceptions.custom_exceptions import RAGError

logger = get_logger(__name__)


class RAGService:
    """Service that answers questions using retrieved context."""

    def __init__(
        self,
        chroma_storage_dir: str | None = None,
        embeddings: OpenAIEmbeddings | None = None,
        llm: ChatOpenAI | None = None,
        search_k: int | None = None,
        use_multi_query: bool = False,
        use_compression: bool = False,
        use_self_query: bool = False,
        use_hyde: bool = False,
        use_decomposition: bool = False
    ):
        """Initialise the RAG service.

        Args:
            chroma_storage_dir: Directory of the persisted vector store.
            embeddings: Optional embedding model. If None, uses config.
            llm: Optional chat model. If None, uses config.
            search_k: Number of chunks to retrieve per query.
            use_multi_query: If True, use multi-query retrieval.
            use_compression: If True, use contextual compression.
            use_self_query: If True, use self-query retrieval.
            use_hyde: If True, use HyDE retrieval.
            use_decomposition: If True, decompose compound questions.
        """
        self.use_multi_query = use_multi_query
        self.use_compression = use_compression
        self.use_self_query = use_self_query
        self.use_hyde = use_hyde
        self.use_decomposition = use_decomposition

        try:
            # Determine storage directory
            self.chroma_storage_dir = chroma_storage_dir or config.chroma_storage_dir

            # Determine number of retrieved chunks
            self.search_k = search_k or config.top_k

            # Create or use embeddings
            if embeddings:
                self.embeddings = embeddings
            else:
                self.embeddings = OpenAIEmbeddings(
                    model=config.embedding_model,
                    openai_api_key=config.openai_api_key
                )

            # Create or use LLM
            if llm:
                self.llm = llm
            else:
                self.llm = ChatOpenAI(
                    model=config.llm_model,
                    openai_api_key=config.openai_api_key,
                    temperature=0
                )

            # Create HyDE embedder (built-in LangChain component)
            self.hyde_embedder = HypotheticalDocumentEmbedder.from_llm(
                self.llm,
                self.embeddings,
                prompt_key="web_search"
            )

            # Create decompose chain (for query decomposition)
            decompose_prompt = ChatPromptTemplate.from_messages([
                ('system', 'Break the user question into 2-4 simpler sub-questions. Output each sub-question on a new line. Do not answer them.'),
                ('human', '{question}')
            ])
            self.decompose_chain = decompose_prompt | self.llm | StrOutputParser()

            # Load vector store and create base retriever
            self.vectorstore = Chroma(
                persist_directory=self.chroma_storage_dir,
                embedding_function=self.embeddings
            )
            self.retriever = self.vectorstore.as_retriever(
                search_kwargs={'k': self.search_k}
            )

            # Build retrieval pipeline step by step
            retrieval = self.retriever

            # Self-query (optional)
            if self.use_self_query:
                metadata_field_info = [
                    AttributeInfo(
                        name='source',
                        description='The file path of the document. Contains keywords like "nigeria_health" or "crop_disease" or "agriculture".',
                        type='string',
                    ),
                    AttributeInfo(
                        name='doc_type',
                        description='The type of document: pdf, html, txt',
                        type='string',
                    ),
                    AttributeInfo(
                        name='language',
                        description='Language of the document, e.g., English',
                        type='string',
                    ),
                ]
                document_content_description = 'Documents about agriculture, public health, and diseases in Nigeria'
                retrieval = SelfQueryRetriever.from_llm(
                    llm=self.llm,
                    vectorstore=self.vectorstore,
                    document_contents=document_content_description,
                    metadata_field_info=metadata_field_info,
                    verbose=False
                )
                logger.info('Enabled self-query retrieval.')

            # Multi-query (optional)
            if self.use_multi_query:
                retrieval = MultiQueryRetriever.from_llm(
                    retriever=retrieval,
                    llm=self.llm
                )
                logger.info('Enabled multi-query retrieval.')

            # Contextual compression (optional)
            if self.use_compression:
                compressor = LLMChainExtractor.from_llm(self.llm)
                retrieval = ContextualCompressionRetriever(
                    base_compressor=compressor,
                    base_retriever=retrieval
                )
                logger.info('Enabled contextual compression.')

            # Set final retrieval component (used when not using HyDE or decomposition)
            self.retrieval = retrieval

            # Log final mode
            if self.use_self_query or self.use_multi_query or self.use_compression:
                logger.info('Final retrieval mode: combination of enabled techniques.')
            else:
                logger.info('Using simple retrieval.')

            # Define RAG prompt template
            self.rag_prompt = ChatPromptTemplate.from_messages([
                ('system', 'You are a helpful assistant. Answer the question using only the provided context. If you don\'t know, say you don\'t know.'),
                ('human', 'Context:\n{context}\n\nQuestion: {question}')
            ])

            # Build standard RAG chain (used when decomposition and HyDE are not active)
            self.chain = (
                {
                    'context': self.retrieval | self._deduplicate_docs | self._format_docs,
                    'question': RunnablePassthrough()
                }
                | self.rag_prompt
                | self.llm
                | StrOutputParser()
            )

            logger.info('RAGService initialised.')

        except Exception as e:
            logger.error(f'Failed to initialise RAGService: {e}')
            raise RAGError('Failed to initialise RAG service', str(e)) from e

    def _deduplicate_docs(self, docs: List[Document]) -> List[Document]:
        """Remove duplicate documents based on page_content."""
        seen = set()
        unique = []
        for doc in docs:
            text = doc.page_content.strip()
            if text not in seen:
                seen.add(text)
                unique.append(doc)

        logger.debug(f'Deduplication: {len(docs)} -> {len(unique)} unique documents.')
        return unique

    def _format_docs(self, docs: List[Document]) -> str:
        """Combine chunk texts into a single context string."""
        combined = '\n\n'.join(doc.page_content for doc in docs)
        logger.info(f'Formatted context from {len(docs)} documents (length={len(combined)} chars)')
        return combined

    def _hyde_retrieve(self, question: str, k: int | None = None) -> List[Document]:
        """Use HyDE to retrieve documents.

        Generates a hypothetical answer, embeds it, and uses that embedding
        to search the vector store directly.

        Args:
            question: The user query.
            k: Number of documents to retrieve. Defaults to self.search_k.

        Returns:
            List of retrieved Document objects.
        """
        k = k or self.search_k

        # Generate a hypothetical embedding for the question
        hyde_embedding = self.hyde_embedder.embed_query(question)

        # Search the vector store using that embedding
        docs = self.vectorstore.similarity_search_by_vector(hyde_embedding, k=k)
        logger.debug(f'HyDE retrieval returned {len(docs)} documents.')
        return docs

    def _decompose_and_retrieve(self, question: str, k: int | None = None) -> List[Document]:
        """Break a compound question into sub-questions, retrieve for each, and combine.

        Args:
            question: The user query (may contain multiple parts).
            k: Number of documents to retrieve per sub-question.

        Returns:
            List of unique Document objects from all sub-queries.
        """
        k = k or self.search_k

        # Decompose the question into sub-questions
        sub_questions_text = self.decompose_chain.invoke({'question': question})
        sub_questions = [q.strip() for q in sub_questions_text.split('\n') if q.strip()]
        logger.info(f'Decomposed into {len(sub_questions)} sub-questions: {sub_questions}')

        all_docs = []
        seen = set()
        for sub_q in sub_questions:
            if self.use_hyde:
                # Use HyDE retrieval for each sub-question
                docs = self._hyde_retrieve(sub_q, k)
            else:
                # Use the normal retrieval component
                docs = self.retrieval.invoke(sub_q)
            for doc in docs:
                key = doc.page_content.strip()
                if key not in seen:
                    seen.add(key)
                    all_docs.append(doc)

        logger.debug(f'Decomposition combined {len(all_docs)} unique documents.')
        return all_docs

    def answer(self, question: str) -> str:
        """Answer a question using the appropriate retrieval mode."""
        try:
            if self.use_decomposition:
                # Retrieve combined docs from decomposition
                docs = self._decompose_and_retrieve(question)
                unique_docs = self._deduplicate_docs(docs)
                context = self._format_docs(unique_docs)
                # Build prompt manually
                prompt = self.rag_prompt.format(context=context, question=question)
                result = self.llm.invoke(prompt).content
            elif self.use_hyde:
                # Use HyDE retrieval only
                docs = self._hyde_retrieve(question)
                unique_docs = self._deduplicate_docs(docs)
                context = self._format_docs(unique_docs)
                prompt = self.rag_prompt.format(context=context, question=question)
                result = self.llm.invoke(prompt).content
            else:
                # Use standard chain
                result = self.chain.invoke(question)

            logger.info(f'Answered question: "{question}"')
            return result
        except Exception as e:
            logger.error(f'RAG generation failed: {e}')
            raise RAGError('RAG generation failed', str(e)) from e

    def retrieve_sources(self, question: str) -> List[Document]:
        """Retrieve relevant documents for a question."""
        try:
            if self.use_decomposition:
                raw_docs = self._decompose_and_retrieve(question)
            elif self.use_hyde:
                raw_docs = self._hyde_retrieve(question)
            else:
                raw_docs = self.retrieval.invoke(question)

            unique_docs = self._deduplicate_docs(raw_docs)
            logger.info(f'Retrieved {len(unique_docs)} sources for question: "{question}"')
            return unique_docs
        except Exception as e:
            logger.error(f'Source retrieval failed: {e}')
            raise RAGError('Source retrieval failed', str(e)) from e