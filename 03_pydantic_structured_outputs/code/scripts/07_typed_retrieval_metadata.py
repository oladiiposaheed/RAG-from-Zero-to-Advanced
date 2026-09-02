'''
Module 3 – Task 6: Typed Retrieval and Metadata Models

This script demonstrates how to:
- Define nested Pydantic models for retrieval results and metadata.
- Create validated retrieval objects.
- Access nested fields cleanly.
- Catch invalid metadata with Pydantic validation.

'''

from pydantic import BaseModel, Field, ValidationError

# Define metadata schema for a retrieved document
class DocumentMetadata(BaseModel):
    """Metadata for a retrieved document.

    Attributes:
        source: Source file or URL.
        page: Page number (optional).
        language: Language of content (default English).
    """
    source: str = Field(description='Source file or URL')
    page: int | None = Field(default=None, description='Page number')
    language: str = Field(default='English', description='Language of content')


# Define schema for a single retrieval result
class RetrievalResult(BaseModel):
    """A single retrieval result with content, metadata, and score.

    Attributes:
        content: The text content of the document.
        metadata: DocumentMetadata object.
        score: Similarity score from retriever.
    """
    content: str = Field(description='The text content of the document')
    metadata: DocumentMetadata = Field(description='Metadata of the document')
    score: float = Field(description='Similarity score from retriever')
    
    
    # Define Helper Function to Display Results
def display_result(result: RetrievalResult, index: int) -> None:
    '''
    Print a retrieval result in a clean format.

    Args:
        result: The RetrievalResult to display.
        index: The result number (starting from 1).
    '''
    print(f'Result: {index}')
    print(f'  Content : {result.content}')
    print(f'  Source  : {result.metadata.source}')
    print(f'  Page    : {result.metadata.page}')
    print(f'  Language: {result.metadata.language}')
    print(f'  Score   : {result.score}')
    print('-' * 40)
    
    
# Define the main() Function
def main() -> None:
    '''Run the typed retrieval example.'''
    
     # Create two valid retrieval results with metadata
    result1 = RetrievalResult(
        content='RAG is a technique that combines retrieval with generation.',
        metadata=DocumentMetadata(source='doc1.pdf', page=2, language='English'),
        score=0.87
    )

    result2 = RetrievalResult(
        content='Vector stores enable fast similarity search.',
        metadata=DocumentMetadata(source='doc2.pdf', page=10, language='English'),
        score=0.72
    )

    # Combine results into a list
    results = [result1, result2]

    # Display each result
    for i, res in enumerate(results, start=1):
        display_result(res, i)

    # validation error: invalid result
    try:
        bad_result = RetrievalResult(
            content='Invalid result',
            metadata=DocumentMetadata(source='doc3.pdf', page='not a number', language='English'),
            score=0.5
        )
        print('Created:', bad_result)
    except ValidationError as e:
        print('\nValidation failed as expected:')
        print(e)


if __name__ == '__main__':
    main()
    
        