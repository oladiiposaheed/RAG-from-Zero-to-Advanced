from app.services.document_loader import DocumentLoaderService
from app.services.text_splitter_service import TextSplitterService
from app.services.vector_store_service import VectorStoreService

from app.logger.custom_logger import get_logger
from pathlib import Path


logger = get_logger(__name__)

# Pipeline for Loading all documents from the data dir
class DocumentPipeline:
    '''Pipeline that runs document loading.'''
    
    def __init__(self):
        '''Initialise the pipeline and its services.'''
        
        # Create the loader, splitter service (uses config defaults)
        self.loader = DocumentLoaderService()
        self.splitter = TextSplitterService()
        
        # Create the vector store service
        self.vector_store_service = VectorStoreService()
        
    def run(self):
        '''
        Execute the full document processing pipeline.

        Steps:
            1. Load documents
            2. Split into chunks
            3. Add metadata
            4. Save chunks to JSON
            
        Returns:
            tuple: (raw_documents, processed_chunks)
        '''
        
        logger.info('Starting document processing pipeline...')
        
        # Step 1: Load documents
        docs = self.loader.load_all()
        
        # Step 2: Split documents into chunks
        chunks = self.splitter.split_documents(docs)
        
        # Step 3: Add metadata to chunks
        chunks = self.splitter.add_metadata(chunks)
        
        # Step 4: Save chunks to JSON
        output_path = Path('04_data_ingestion_document_processing/data/chunks/all_chunks.json')
        self.splitter.save_chunks_json(chunks, output_path)
        
        # Step 5: Embed chunks and store them in Chroma
        self.vector_store_service.create_embed_and_store(chunks)
        
        logger.info(f'Pipeline complete. Total documents: {len(docs)}, Total chunks: {len(chunks)}')
        
        return docs, chunks
        
        # logger.info(f'Total documents loaded: {len(docs)}')
        
        # return docs