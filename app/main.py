'''
Entry point for the document indexing pipeline.
'''

from app.logger.logger import get_logger
from app.services.document_indexing_pipeline import DocumentIndexingPipeline
from app.config.config import config

logger = get_logger(__name__, log_file='logs/indexing_pipeline.log')

def main() -> None:
    '''
    Run the indexing pipeline.
    '''
    
    logger.info('Starting application pipeline...')
    
    # Create the pipeline
    pipeline = DocumentIndexingPipeline(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap
    )
    
    # Run the pipeline and save output
    output_path = '04_data_ingestion_document_processing/data/chunks/indexed_chunks.json'
    
    pipeline.run(output_path)
    
    logger.info('Document indexing pipeline completed successfully.')
    
    
if __name__=='__main__':
    main()