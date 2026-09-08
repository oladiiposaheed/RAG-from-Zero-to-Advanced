'''
Main entry point for the application.
'''

from app.pipeline import DocumentPipeline
from app.logger.custom_logger import setup_logging, get_logger
from dotenv import load_dotenv

load_dotenv()

# Configure logging(root logger)
setup_logging(log_file='logs/app.log')

# Create a logger
logger = get_logger(__name__)

def main():
    '''Run the document pipeline.'''
    
    logger.info('Starting application...')
    
    # Create the pipeline
    pipeline = DocumentPipeline()
    
    docs, chunks = pipeline.run()
    
    # Run the pipeline (load, split, add metadata, save)
    print(f'Loaded {len(docs)} documents.')
    print(f'Total chunks created: {len(chunks)}')
    

if __name__=='__main__':
    main()