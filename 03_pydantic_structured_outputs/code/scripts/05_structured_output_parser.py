'''
Module 3 – Task 4: Structured Output Parsers in LangChain

This script demonstrates how to use PydanticOutputParser to get
a validated, typed object directly from a LangChain chain.
'''

from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from dotenv import load_dotenv


# Define the output schema using Pydantic
class AnswerWithCitation(BaseModel):
    '''
    Structured answer with source and confidence.

    Attributes:
        answer: The answer text.
        source: The source used to answer.
        confidence: Confidence score between 0 and 1.
    '''
    
    answer: str = Field(description='The answer to the question')
    source: str = Field(description='The source document or section used')
    confidence: float = Field(description='Confidence score between 0 and 1')


def create_structured_chain():
    '''
    Build a LangChain chain that returns AnswerWithCitation objects.

    Returns:
        A chain that takes a question and returns a validated Pydantic object.
    '''
    
    # Create the parser from the Pydantic model
    parser = PydanticOutputParser(pydantic_object=AnswerWithCitation)
    
    format_instructions = parser.get_format_instructions()
    
    # Create chat model
    llm = ChatOpenAI(model='gpt-4o-mini', temperature=0)

    # Build prompt with placeholders
    prompt = ChatPromptTemplate.from_messages([
        ('system', 'You are a helpful assistant. Answer the user question using the provided format.'),
        ('human', 'Question: {question}\n\n{format_instructions}')
    ])    
    
    # Pre-fill the format instructions (constant across calls)
    prompt = prompt.partial(format_instructions=format_instructions)
    
    # Build and return the chain
    return prompt | llm | parser


def main() -> None:
    '''
    Run the structured output chain with a sample question.
    '''
    
    load_dotenv()
    
    # Create chain
    chain = create_structured_chain()
    
    # Invoke the chain with a question
    response = chain.invoke({'question': 'Define RAG'})
    
    # Display the result
    print('Structured result:')
    print(response)
    print(f'Type: {type(response).__name__}')
    
    # Access individual fields
    print('\nFields:')
    print(f'Answer: {response.answer}')
    print(f'Source: {response.source}')
    print(f'Confidence: {response.confidence}')

    # Convert to dictionary
    print('\nAs dictionary:')
    print(response.model_dump())
    
    
if __name__ == '__main__':
    main()
    