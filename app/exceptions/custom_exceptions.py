class BaseAppError(Exception):
    '''Base class for all application errors.'''
    
    def __init__(self, message: str, details: str = ''):
        
        self.message = message   # main error message
        self.datails = details
        
        full_message = f'{full_message} | {details}' if details else full_message
        super().__init__(full_message)
        
        
class DocumentLoadError(BaseAppError):
    '''Raised when a document cannot be loaded.'''
    
class UnsupportedFileTypeError(BaseAppError):
    '''Raised when a file type is not supported.'''
    
class DataDirectoryNotFoundError(BaseAppError):
    '''Raised when the data directory does not exist.'''
    
class TextSplitterError(BaseAppError):
    '''Raised when text splitting fails'''
    
    