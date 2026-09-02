'''
 Document loader service.

Loads supported file types from the data directory using LangChain loaders.
Uses shared config, custom exceptions, and the advanced logger.
'''

from pathlib import Path
from typing import List
from langchain_core.documents import Document
from langchain_community.document_loaders import (
    TextLoader, CSVLoader, JSONLoader, PyPDFLoader, BSHTMLLoader
)
from app.config.config import config
from app.exceptions.custom_exceptions import (
    DataDirectoryNotFoundError, UnsupportedFileTypeError, DocumentLoadError
)
from app.logger.logger import get_logger

logger = get_logger(__name__)

class DocumentLoaderService:
    
    # Validate file extensions
    SUPPORTED_EXTENSIONS = {
        '.txt': 'text',
        '.csv': 'csv',
        '.json': 'json',
        '.pdf': 'pdf',
        '.html': 'html'
    }
    
    def __init__(self, data_dir: Path | None = None):
        '''
        Initialise the loader with a data directory.

        If no data_dir is provided, use the one from the shared config.
        '''
        
        self.data_dir = Path(data_dir) if data_dir else config.data_dir
        
        # Check if the data directory exists
        if not self.data_dir.exists():
            raise DataDirectoryNotFoundError(
                'Data directory not found',
                str(self.data_dir)
            )

    def load_file(self, file_path: Path) -> List[Document]:
        '''
        Load a single file based on its extension.

        Args:
            file_path: Path to the file.

        Returns:
            List of Document objects.
        '''
        
        ext = file_path.suffix.lower()
        
        # Validate the extension
        if ext not in self.SUPPORTED_EXTENSIONS:
            raise UnsupportedFileTypeError('Unsupported file type', ext)
        
        logger.info(f'Loading: {file_path.name}')
        
        try:
            # Select the correct loader based on extension
            if ext == '.txt':
                loader = TextLoader(str(file_path), encoding='utf-8')
                
            elif ext == '.csv':
                loader = CSVLoader(str(file_path), encoding='utf-8')
                
            elif ext == '.json':
                loader = JSONLoader(
                    file_path=str(file_path),
                    jq_schema='.[]',
                    content_key='body'
                )
                
            elif ext == '.pdf':
                loader = PyPDFLoader(str(file_path))
                
            elif ext == '.html':
                loader = BSHTMLLoader(
                    str(file_path),
                    open_encoding='utf-8',
                    bs_kwargs={'features': 'html.parser'}
                )
                
            else:
                raise UnsupportedFileTypeError('Unsupported file type', ext)
            
            # Return the loaded documents
            return loader.load()
                
        except Exception as e:
            raise DocumentLoadError('Failed to load document', str(file_path)) from e
        
    
    def load_all(self) -> List[Document]:
        '''
        Load all supported files in the data directory.

        Returns:
            Combined list of Document objects.
        '''
        
        all_docs = []
        
        # Loop through every file pathin the data dir
        for file_path in self.data_dir.iterdir():
            if file_path.is_file():
                docs = self.load_file(file_path)
                all_docs.extend(docs)
                
        logger.info(f'Loaded {len(all_docs)} documents from {self.data_dir}')
        
        return all_docs
        
