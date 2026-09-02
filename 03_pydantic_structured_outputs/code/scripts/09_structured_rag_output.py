"""
Mini-Project: Structured RAG Output with Citations

This script simulates a RAG pipeline that:
- Retrieves a document from a small in-memory store.
- Uses the retrieved text as context for the LLM.
- Returns a structured RAGResponse object with question, answer, source, and confidence.
- Uses PydanticOutputParser and OutputFixingParser for robust parsing.
"""


# Step 1: Imports

from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.output_parsers import OutputFixingParser
from dotenv import load_dotenv

# Step 2: Define the RAGResponse Model

# Define the structured output model for the RAG response
class RAGResponse(BaseModel):
    """Structured RAG response.

    Attributes:
        question: The original user question.
        answer: The generated answer.
        source: The source document used.
        confidence: Confidence score between 0 and 1.
    """
    question: str = Field(description='The original user question')
    answer: str = Field(description='The generated answer')
    source: str = Field(description='The source document used')
    confidence: float = Field(description='Confidence score between 0 and 1')
    
# Simulated in-memory document store (list of dicts with text and source)
DOCUMENTS = [
    {
        'text': 'RAG is a technique that combines retrieval of relevant documents with language generation.',
        'source': 'doc1_rag_intro.txt'
    },
    {
        'text': 'Vector databases store embeddings and perform fast similarity search.',
        'source': 'doc2_vector_db.txt'
    },
    {
        'text': 'LangChain provides tools for building RAG pipelines easily.',
        'source': 'doc3_langchain.txt'
    }
]

# Step 3: Simulated Document Store and Retriever

def retrieve(query: str) -> dict:
    """Simulate a retriever.
    Return the first document for any query.

    Args:
        query: User question (unused in this simulation).

    Returns:
        A dictionary with 'text' and 'source'.
    """
    return DOCUMENTS[0]


# Step 4: Build the Structured RAG Chain
def build_chain():
    '''
    Build a LangChain chain that returns a RAGResponse object.

    Returns:
        A chain that takes context, source, and question, and returns a RAGResponse.
    '''
    
    # Create the Pydantic parser for RAGResponse
    parser = PydanticOutputParser(pydantic_object=RAGResponse)
    
    # Create a fixing parser to auto-correct invalid output if needed
    fixing_parser = OutputFixingParser.from_llm(
        parser=parser,
        llm=ChatOpenAI(model='gpt-4o-mini', temperature=0)
    )
    
    # Create the chat model
    llm = ChatOpenAI(model='gpt-4o-mini', temperature=0)

    # Build the prompt template with context, source, question, and format instructions
    prompt = ChatPromptTemplate.from_messages([
        ('system', 'You are a helpful assistant. Answer the question using only the provided context.'),
        ('human',
         'Context: {context}\n'
         'Source: {source}\n\n'
         'Question: {question}\n\n'
         '{format_instructions}'
        )
    ])

    # Pre-fill format instructions, only pass context, source, and question
    
    prompt = prompt.partial(format_instructions=parser.get_format_instructions())
    
    # Return the full chain: prompt -> llm -> fixing_parser
    return prompt | llm | fixing_parser


# Step 5: Define the main() Function
def main() -> None:
    '''Run the structured RAG mini-project.'''

    load_dotenv()
    
    # Build the RAG chain
    chain = build_chain()
    
    # Simulate a user question
    question = 'What is RAG?'
    
    # Retrieve a document (simulated)
    doc = retrieve(question)
    
    # Invoke the chain with context, source, and question
    result = chain.invoke({
        'context': doc['text'],
        'source': doc['source'],
        'question': question
    })
    
    # Display the structured result
    print('Structured RAG Result:')
    print(result)
    print('Type:', type(result).__name__)

    # Access individual fields
    print('\nFields:')
    print(f'Question: {result.question}')
    print(f'Answer  : {result.answer}')
    print(f'Source  : {result.source}')
    print(f'Confidence: {result.confidence}')


if __name__ == '__main__':
    main()