'''
    Module 3 – Task 3: Advanced Pydantic Features

    This script demonstrates:
    - Nested models (Author inside Document)
    - Lists of strings (tags)
    - Optional fields (pages)
    - Automatic validation and type coercion
'''

from typing import List, Optional
from pydantic import BaseModel, Field, ValidationError

class Author(BaseModel):
    '''Author model representing a person who wrote a document.
    '''

    name: str
    email: str
    

class Document(BaseModel):
    '''
        Document model with nested Author, tags, and optional pages.
    '''
    
    title: str = Field(description='Document title', min_length=1)
    author: Author = Field(description='Author of the document')
    tags: List[str] = Field(default=[], description='List of tags')
    pages: Optional[int] = Field(default=None, description='Number of pages')
    
    
def create_document(
    title: str,
    author_name: str,
    author_email: str,
    tags: Optional[List[str]] = None,
    pages: Optional[int] = None
) -> Document:
    
    '''
        Create a Document instance with nested Author.
    '''
    
    # Create the nested Author object
    author = Author(name=author_name, email=author_email)
    
    # Build Document with optional fields
    doc_data = {
        'title': title,
        'author': author
    }
    
    if tags is not None:
        doc_data['tags'] = tags
    
    if pages is not None:
        doc_data['pages'] = pages
        
    return Document(**doc_data)


def main() -> None:
    
    doc1 = create_document(
        title='Bsasic RAG',
        author_name='Saheed',
        author_email='saheed@example.com',
        tags = ['rag', 'llm', 'agent-ai'],
        pages=600
    )

    print('Document 1:')
    print(doc1)
    print(f'Author type: {type(doc1.author).__name__}')
    print(f'Tags: {doc1.tags}')
    print(f'Pages: {doc1.pages}')

    # Document with no tags or pages
    doc2 = create_document(
        title='Introduction to Pydantic',
        author_name='Fatimah',
        author_email='tima@example.com'
    )
    
    print('\nDocument 2:')
    print(doc2)
    print(f'Tags: {doc2.tags}')
    print(f'Pages: {doc2.pages}')
    
    # Document with Invalid title
    try:
        doc3 = create_document(
            title='',
            author_name='Grace',
            author_email='grace@example.com'
        )
        print(f'\nDocument 3: {doc3}')
        
    except ValidationError as e:
        print('\nValidation failed for title:')
        print(e)
        
if __name__=='__main__':
    main()