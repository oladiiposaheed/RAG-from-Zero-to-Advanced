'''
Document indexing pipeline.

Combines document loading, text splitting, metadata enrichment,
and saving into one reusable pipeline.
'''

from pathlib import Path
from app.services.document_loader import DocumentLoaderService
from app.services.text_splitter_service import TextSplitterService
from app.logger.logger import get_logger

logger = get_logger(__name__)


class DocumentIndexingPipeline:
    '''
    Runs the full indexing pipeline: load → split → enrich → save.
    '''
    
    # Step 2: Add the __init__ method
    def __init__(
        self,
        data_dir: Path | None = None,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None
    ):
        
        # Store chunk settings 
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Create document loader with given data_dir or config default
        self.loader = DocumentLoaderService(data_dir=data_dir)
        
        # Create the text splitter service with given chunk settings
        self.spliter = TextSplitterService(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
    
    # Step 3: Add the run method
    def run(self, output_path: str | Path) -> None:
        '''
        Execute the full indexing pipeline.

        Args:
            output_path: Where to save the final chunks JSON.
        '''
        
        try:
            # Log the start of the pipeline
            logger.info('Starting indexing pipeline....')
                    
            # 1. Load all documents from the data directory
            docs = self.loader.load_all()
                    
            # 2. Split documents into chunks
            chunks = self.spliter.split_documents(docs)
                    
            # 3. Add metadata to each chunk
            chunks = self.spliter.add_metadata(chunks)
                    
            # 4. Save chunks to JSON
            self.spliter.save_chunks_json(chunks, output_path)
                    
            logger.info(f'Indexing complete. Total chunks: {len(chunks)}')
        
        except Exception as e:
            logger.error(f'Indexing pipeline failed: {e}')
            raise