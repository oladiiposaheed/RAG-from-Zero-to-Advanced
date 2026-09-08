'''
Text splitter service.

Splits loaded documents into chunks using RecursiveCharacterTextSplitter,
adds metadata, and optionally saves chunks to JSON.
'''

import json
from pathlib import Path
from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.config.config import config
from app.logger.custom_logger import get_logger
from app.exceptions.custom_exceptions import TextSplitterError


# Create a logger
logger = get_logger(__name__)

class TextSplitterService:
    '''Service that splits documents into chunks and adds metadata'''
    
    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        separators: list[str] | None = None
    ):
        '''Initialise the splitter with settings.'''
        
        # Use provided values or fallback to config
        self.chunk_size = chunk_size if chunk_size is not None else config.chunk_size
        self.chunk_overlap = chunk_overlap if chunk_overlap is not None else config.chunk_overlap
        
        # If no separators provided, use default list
        self.separators = separators or ['\n\n', '\n', '.', ' ', '']
        
        # Create the splitter object
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=self.separators
        )
        
        logger.info(f'TextSplitterService initialised with chunk_size={self.chunk_size}, chunk_overlap={self.chunk_overlap}')
        
        
    # Split list of documents
    def split_documents(self, docs: List[Document]) -> List[Document]:
        '''
        Split a list of documents into chunks.
        '''
        
        try:
            logger.info(f'Splittng {len(docs)} documents(s)...')
            
            chunks = self.splitter.split_documents(docs)
            
            logger.info(f'Created {len(chunks)} chunks.')
            return chunks
            
        except Exception as e:
            logger.error(f'Failed to split documents: {e}')
            raise TextSplitterError('Failed to split documents', str(e)) from e
            
            
    # Add custom metadata
    def add_metadata(self, chunks: List[Document]) -> List[Document]:
        '''
        Add custom metadata to each chunk.
        '''
        
        # Loop through chunks
        for i, chunk in enumerate(chunks):
            
            source = chunk.metadata.get('source', 'unknown')
            file_name = Path(source).name
            
            # Set extracted file name metadata
            chunk.metadata['file_name'] = file_name
            
            # Set doc_type based on file extension
            chunk.metadata['doc_type'] = Path(file_name).suffix.lower().lstrip('.')
            chunk.metadata['language'] = 'English'
            
            # Set a unique chunk_id 
            chunk.metadata['chunk_id'] = f'{file_name}_{i+1:03d}'    
        
        return chunks
    
    
    # Save chunks to a JSON file
    def save_chunks_json(self, chunks: List[Document], output_path: str | Path) -> None:
        '''
        Save chunks to a JSON file.

        Args:
            chunks: List of chunk Document objects.
            output_path: Path to the JSON output file.
        '''
        
        try:
            # Convert output_path to Path object
            output_path = Path(output_path)
            
            # Create parent directories if they don't exist
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Build a list of dicts with content and metadata
            data = [
                {'content': chunk.page_content, 'metadata': chunk.metadata}
                for chunk in chunks
            ]
            
            # Write data to JSON file
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
                
            logger.info(f'Saved {len(chunks)} chunks to {output_path}')
        
        except Exception as e:
            logger.error(f'Failed to save chunk: {e}')
            raise TextSplitterError('Failed to save chunks', str(e)) from e
        
        