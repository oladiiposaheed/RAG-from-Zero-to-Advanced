'''
Text splitting service.

Splits loaded documents into chunks using a recursive splitter.
Adds metadata and saves chunks to JSON.
'''

from pathlib import Path
from typing import List
import json
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.logger.logger import get_logger
from app.exceptions.custom_exceptions import TextSplitterError
from app.config.config import config

logger = get_logger(__name__)

class TextSplitterService:
    '''
    Service that splits documents into chunks and adds metadata.
    '''
    
    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        separators: list[str] | None = None
    ):
        
        # Store the chunk size, chunk overlap
        self.chunk_size = chunk_size if chunk_size is not None else config.chunk_size
        self.chunk_overlap = chunk_overlap if chunk_overlap is not None else config.chunk_overlap
    
        # Use default list if no separators provided
        self.separators = separators or ['\n\n', '\n', '.', ' ', '']

        # Create splitter object
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=self.separators
        )
        
    
    # Split documents
    def split_documents(self, docs: List[Document]) -> List[Document]:
        '''Split a list of documents into chunks.'''    

        try:
            # Create chunks from splitter
            logger.info(f'Splitting {len(docs)} document(s)...')
            chunks = self.splitter.split_documents(docs)
            logger.info(f'Created {len(chunks)} chunks')        
            return chunks
        
        except Exception as e:
            logger.error(f'Failed to split documents: {e}')
            raise TextSplitterError('Failed to split documents', str(e)) from e    
    
    # Add metadata function
    def add_metadata(self, chunks: List[Document]) -> List[Document]:
        '''Add custom metadata to each chunk'''
        
        # Loop through each chunk
        for i, chunk in enumerate(chunks):
            
            # Get the original source path from existing metadata
            source = chunk.metadata.get('source', 'unknown')
            
            # Extract file name from full path
            file_name = Path(source).name
            
            # Add file name to metadata
            chunk.metadata['file_name'] = file_name
            
            # Add doc_type('txt', 'pdf', 'csv')
            chunk.metadata['doc_type'] = Path(file_name).suffix.lower().lstrip('.')
            
            # Add language
            chunk.metadata['language'] = 'English'
            
            # Add a unique chunk_id
            chunk.metadata['chunk_id'] = f'{file_name}_{i+1:03d}'
            
        return chunks
    
    # Save list of chunks in json format
    def save_chunks_json(self, chunks: List[Document], output_path: str | Path) -> None:
        '''Save chunks to a JSON file.'''
        
        try:
            # Convert output_path to a Path object if it is a string
            output_path = Path(output_path)
                    
            # Create the parent directory if not exist
            output_path.parent.mkdir(parents=True, exist_ok=True)
                    
            # Build a list of dicts with content, metadata
            data = [
                {'content': chunk.page_content, 'metadata': chunk.metadata}
                for chunk in chunks
            ]
                    
            # Write the data to the JSON file
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
                        
            # Save chunks
            logger.info(f'Save {len(chunks)} chunks to {output_path}')
            
        except Exception as e:
            logger.error(f'Failed to save chunks: {e}')
            raise TextSplitterError('Failed to save chunks', str(e)) from e