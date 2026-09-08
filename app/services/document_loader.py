'''
Document loader service.

Loads supported file types from a directory and returns a list of Document objects.
Supported types: .txt, .csv, .json, .pdf, .html
'''

from pathlib import Path
from typing import List
from langchain_core.documents import Document
from langchain_community.document_loaders import (
    TextLoader, CSVLoader, JSONLoader, PyPDFLoader, BSHTMLLoader
)
from app.config.config import config
from app.logger.custom_logger import get_logger
from app.exceptions.custom_exceptions import (
    DataDirectoryNotFoundError,
    UnsupportedFileTypeError,
    DocumentLoadError
)

# Create a logger
logger = get_logger(__name__)

class DocumentLoaderService:
    '''Service class that loads documents from a directory.'''

    SUPPORTED_EXTENSIONS = {
        '.txt': 'text',
        '.csv': 'csv',
        '.json': 'json',
        '.pdf': 'pdf',
        '.html': 'html'
    }
    
    def __init__(self, data_dir: Path | None = None):
        '''
        Initialize with a data directory.

        If no data_dir is given, use the one from config.
        '''
        
        self.data_dir = Path(data_dir) if data_dir else config.data_dir
        
        # Check if the directory exists
        if not self.data_dir.exists():
            raise DataDirectoryNotFoundError('Data directory not found', str(self.data_dir))
        
        logger.info(f'DocumentLoaderService initialised for: {self.data_dir}')
        
        
    # Load one file based on its extension
    def load_file(self, file_path: Path) -> List[Document]:
        '''
        Read a single file and return its documents.
        Checks the file extension, selects the correct loader,
        and returns the loaded Document objects.
        '''
        
        # Get the file extension in lowercase
        ext = file_path.suffix.lower()
        
        if ext not in self.SUPPORTED_EXTENSIONS:
            raise UnsupportedFileTypeError('Unsupported file type', ext)
        
        logger.info(f'Loading: {file_path.name}')
        
        # Select loader and return documents
        try:
            if ext == '.txt':
                loader = TextLoader(str(file_path), encoding='utf-8')
            elif ext == '.csv':
                loader = CSVLoader(str(file_path), encoding='utf-8')
            elif ext == '.json':
                loader = JSONLoader(
                    file_path=str(file_path), jq_schema='.[]', content_key='body'
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
            
            # Load and return the documents
            return loader.load()
        
        except Exception as e:
            raise DocumentLoadError('Failed to load document', str(file_path)) from e
        
        
    # Load all supported files
    def load_all(self) -> List[Document]:
        '''Load all supported files in the directory.'''
            
        all_docs = []
        for file_path in self.data_dir.iterdir():
            if file_path.is_file():
                docs = self.load_file(file_path)
                all_docs.extend(docs)
                
        # Log the total number of loaded documents        
        # logger.info(f'Loaded {len(all_docs)} documents from {self.data_dir}')
        
        return all_docs
    
    