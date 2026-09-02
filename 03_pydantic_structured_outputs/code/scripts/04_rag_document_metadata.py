'''
RAG Document Metadata

This script demonstrates how to use Pydantic models to validate
and manage metadata for documents retrieved from a knowledge base.
'''

from typing import List, Optional
from pydantic import BaseModel, Field, ValidationError

class Author(BaseModel):
    '''
    Author of a knowledge article.

    Attributes:
        name: Full name of the author.
        email: Email address of the author.
    '''
    name: str
    email: str
    

class KnowledgeArticle(BaseModel):
    
    title: str = Field(description='Title of the article', min_length=2)
    author: Author = Field(description='Who wrote the article')
    tags: List[str] = Field(default=[], description='List of topics/tags')
    language: str = Field(default='English', description='Language of the article')
    pages: Optional[int] = Field(default=None, description='Number of pages if PDF')
    url: Optional[str] = Field(default=None, description='Original source URL')


def create_article(
    title: str,
    author_name: str,
    author_email: str,
    tags: Optional[List[str]] = None,
    language: str = 'English',
    pages: Optional[int] = None,
    url: Optional[str] = None
) -> KnowledgeArticle:
    '''
    Create a validated KnowledgeArticle instance.
    
    Args:
        title: Article title.
        author_name: Author's full name.
        author_email: Author's email.
        tags: Optional list of tags.
        language: Language (default English).
        pages: Optional number of pages.
        url: Optional source URL.

    Returns:
        A validated KnowledgeArticle object.
    '''
    # Create the nested Author object
    author = Author(name=author_name, email=author_email)
    
    # Build the article with all provided fields
    article = KnowledgeArticle(
        title=title,
        author=author,
        tags=tags if tags is not None else [],
        language=language,
        pages=pages,
        url=url
    )
    
    return article


def display_article(article: KnowledgeArticle) -> None:
    '''
    Print article metadata in a readable format.
    
    Args:
        article: The KnowledgeArticle to display.
    '''
    print(f'Title: {article.title}')
    print(f'Author: {article.author.name} ({article.author.email})')
    print(f'Tags: {", ".join(article.tags) if article.tags else "None"}')
    print(f'Language: {article.language}')
    
    if article.pages:
        print(f'Pages: {article.pages}')
        
    if article.url:
        print(f'URL: {article.url}')
    
    print('-' * 70)
    

def main() -> None:
    '''Test RAG metadata validation.'''
    
    # Create articles with different metadata
    article1 = create_article(
        title='How to Use RAG for Customer Support',
        author_name='Saheed',
        author_email='saheed@company.com',
        tags=['rag', 'customer-support', 'llm'],
        pages=15,
        url='https://internal.company.com/docs/rag-support'
    )
    
    article2 = create_article(
        title='Guide to Vector Databases',
        author_name='Aisha',
        author_email='aisha@company.com',
        tags=['vector-db', 'faiss', 'pinecone'],
    )
    
    # Simulate a retrieval list
    retrieved_docs = [article1, article2]
    
    print('Retrieved Documents:')
    
    for doc in retrieved_docs:
        display_article(doc)
        
    # Validation error example
    try:
        invalid_article = create_article(
            title='',
            author_name='Fatimah',
            author_email='tima@example.com'
        )
        print(f'Invalid article created: {invalid_article}')
        
    except Exception as e:
        print('\nValidation failed ')
        print(e)
        
if __name__=='__main__':
    main()