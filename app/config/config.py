'''
Application configuration using pydantic-settings.
'''

from pathlib import Path
from pydantic_settings import BaseSettings

# Settings class that reads .env
class Config(BaseSettings):
    '''
    Application configuration.

    Attributes:
        data_dir: Path to the folder containing documents.
        log_level: Logging level (INFO, DEBUG, WARNING, etc.).
        openai_api_key: API key for OpenAI (loaded from .env).
    '''
    
    data_dir: Path = Path('04_data_ingestion_document_processing/data')
    log_level: str = 'INFO'
    openai_api_key: str = ''
    
    # Log file path
    log_file: str | None = None
    
    # Environment: 'development' or 'production'
    environment: str = 'development'
    
    # Chunk settings for text splitting
    chunk_size: int = 300
    chunk_overlap: int = 50
   
    # Pydantic model config 
    class Config:
        '''Pydantic model configuration.'''
        
        # Ignore undefined variables found in .env
        extra = 'ignore'
        
        # .env file name
        env_file = '.env'
        
        # .env encoding
        env_file_encoding = 'utf-8'   
        
# Create instance of Config
config = Config()

    