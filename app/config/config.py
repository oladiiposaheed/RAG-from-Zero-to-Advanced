'''
Central application configuration using pydantic-settings.
'''

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Config(BaseSettings):
    '''
    Application configuration model.
    '''
    
    # Document processing
    data_dir: Path = Field(
        default=Path('04_data_ingestion_document_processing/data'),
        description='Directory containing input documents'
    )
    chunk_size: int = Field(default=300, description='Maximum characters per chunk')
    chunk_overlap: int = Field(default=50, description='Overlap between chunks')
    embedding_model: str = Field(default='text-embedding-3-small', description='Embedding model name')
    llm_model: str = Field(default='gpt-4o-mini', description='LLM model name for generation')
    openai_api_key: str = Field(default='', description='OpenAI API key')
    chroma_storage_dir: str = Field(default='chroma_db', description='Chroma persistence directory')
    top_k: int = Field(default=4, description='Number of retrieved documents')
    
    # Logging
    log_level: str = Field(default='INFO', description='Logging level')
    log_file: str | None = Field(default=None, description='Optional log file path')
    environment: str = Field(default='development', description='development/production')
    
    # Pydantic settings config
    model_config = SettingsConfigDict(env_file='.env', env_file_encoding='utf-8', extra='ignore')
    

config = Config()
    
    
    
    
