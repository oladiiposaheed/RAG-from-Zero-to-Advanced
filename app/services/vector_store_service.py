'''
Vector store service.

Handles embedding documents and storing them in Chroma,
as well as loading and searching.
'''

from pathlib import Path
from typing import List
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from app.config.config import config
from app.logger.custom_logger import get_logger
from app.exceptions.custom_exceptions import VectoStoreError
from app.exceptions.custom_exceptions import VectorStoreError


# Create a logger for this module
logger = get_logger(__name__)


# Define VectorStoreService
class VectorStoreService:
    '''Service for managing a Chroma vector store.'''
    
    def __init__(
        self,
        embeddings: OpenAIEmbeddings | None = None,
        chroma_storage_dir: str | None = None
    ):
        '''
        Initialise the vector store service.

        Args:
            embeddings: Optional embedding model. If None, use default OpenAIEmbeddings.
            chroma_storage_dir: Optional directory for Chroma persistence.
        '''
        
        if embeddings:
            self.embeddings = embeddings
        
        else:
            self.embeddings = OpenAIEmbeddings(
                model=config.embedding_model,
                openai_api_type=config.openai_api_key
            )
            
        # Use provided storage dir or fallback to config
        self.chroma_storage_dir = chroma_storage_dir or config.chroma_storage_dir
        
        self.vectorestore = None
        
        logger.info('VectorStoreService initialised.')
        
        
    def create_embed_and_store(
        self,
        documents: List[Document],
        chroma_storage_dir: str | None = None
    ) -> None:
        '''
        Embed documents and store them in Chroma

        Args:
            documents: List of Document objects to embed and store.
            chroma_storage_dir: Optional directory for persistence.
                If not given, uses the one from __init__.
        '''
        
        try:
            # Define which directory to use
            storage_dir = chroma_storage_dir or self.chroma_storage_dir
            
            # Create the Chroma vector store from documents
            self.vectorestore = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                persist_directory=storage_dir
            )
            logger.info(f'Created vector store with {len(documents)} documents.')
        
        except Exception as e:
            logger.error(f'Failed to create vector store: {e}')
            raise VectorStoreError('Failed to create vector store', str(e)) from e


    def load(self, chtoma_dir: str | None = None) -> None:
        '''
        Load an existing vector store from disk.

        Args:
            chroma_storage_dir: Optional directory. If not given, uses
                the one set in __init__ (self.chroma_storage_dir).
        '''
        
        try:
            # Determine which directory to load from
            storage_dir = self.chroma_storage_dir or self.chroma_storage_dir
        
            if not storage_dir:
                raise ValueError('No chroma_storage_dir provided.')
            
            # Load Chroma from disk
            self.vectorestore = Chroma(
                persist_directory=storage_dir,
                embedding_function=self.embeddings
            )
            
            logger.info(f'Loaded vector store from {storage_dir}.')
        except Exception as e:
            logger.error(f'Failed to load vector store: {e}')
            raise VectoStoreError('Failed to load vector store', str(e)) from e
        

    def similarity_search(self, query: str, k: int | None = None) -> List[Document]:
        '''
        Search for the most similar documents to a query.

        Args:
            query: The user query string.
            k: Number of results to return. If None, uses config.top_k.

        Returns:
            List of Document objects.
        '''
        
        if k is None:
            k = config.top_k
            
        # Ensure the vector store is already created or loaded
        if self.vectorestore is None:
            raise VectorStoreError('Vector store is not initialised.')
        
        try:
            # Perform the similarity search
            results = self.vectorestore.similarity_search(query, k=k)
            logger.info(f'Performed similarity search for query: "{query}"')
            return results
        
        except Exception as e:
            logger.error(f'Similarity search failed: {e}')
            raise VectorStoreError('Similarity search failed', str(e)) from e
        
        