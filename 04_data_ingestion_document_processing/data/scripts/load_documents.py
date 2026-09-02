'''
Module 4 – Task 1: Document Loader

This script loads all supported files from the data folder and returns
a list of LangChain Document objects with metadata.

Supported types: .txt, .csv, .json, .pdf, .html
'''

import logging
from pathlib import Path
from typing import List, Optional
from langchain_core.documents import Document
from langchain_community.document_loaders import (
    TextLoader,
    CSVLoader,
    JSONLoader,
    PyPDFLoader,
    BSHTMLLoader
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

class DocumentLoaderService:
    '''
    Service class for loading documents from a folder.
    '''
    
    # Define supported file extensions and their categories
    SUPPORTED_EXTENSIONS = {
        '.txt': 'text',
        '.csv': 'csv',
        '.json': 'json',
        '.pdf': 'pdf',
        '.html': 'html'
    }

# Create the Main Loading Function

