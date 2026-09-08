class BaseAppError(Exception):
    '''
    Base class for all custom application errors.
    '''
    pass

class  DocumentLoadError(BaseAppError):
    '''Raised when a document cannot be loaded.'''
    pass

class UnsupportedFileTypeError(BaseAppError):
    '''Raised when a file type is not supported.'''
    pass

class DataDirectoryNotFoundError(BaseAppError):
    '''Raised when the data directory does not exist.'''
    pass

class TextSplitterError(BaseAppError):
    '''Raised when text splitting fails.'''
    pass

class VectoStoreError(BaseAppError):
    '''Raised when vector store operations fail.'''
    pass

class RAGError(BaseAppError):
    '''Raised when the RAG pipeline fails.'''
    pass

class VectorStoreError(BaseAppError):
    """Raised when vector store operations fail."""
    pass

class RAGError(BaseAppError):
    '''Raised when the RAG pipeline fails.'''
    pass