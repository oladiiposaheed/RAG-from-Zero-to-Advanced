'''
Task 5: Function Calling and Tool Schemas

This script demonstrates how to:
- Define a tool schema using Pydantic.
- Bind the schema to a chat model.
- Trigger a tool call from a user query.
- Execute the corresponding function with the returned arguments.
'''

from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

# Define the schema(Pydantic model) for the search tool
class SearchQuery(BaseModel):
    '''
    Schema for search tool arguments.

    Attributes:
        query: The search query string.
        top_k: Number of results to return (default 5).
    '''
    
    query: str = Field(description='The search query string')
    top_k: int = Field(description='Number of results to return', default=5)

# Define the Fake Search Function
def fake_search(query: str, top_k: int = 5) -> list[str]:
    '''
    Simulate a search engine.

    Args:
        query: The search query.
        top_k: Number of results to return.

    Returns:
        A list of dummy result titles.
    '''
    dummy_results = [
        'Introduction to RAG',
        'Advanced RAG Techniques',
        'Building RAG with LangChain',
        'RAG Evaluation Methods',
        'Multilingual RAG Systems'
    ]
    
    return dummy_results[:top_k]


# Bind Tools to the Model
def bind_tools_to_model():
    '''
    Create a chat model and bind the SearchQuery tool schema.

    Returns:
        A chat model with the tool schema bound.
    '''
    llm = ChatOpenAI(model='gpt-4o-mini', temperature=0)
    
    return llm.bind_tools([SearchQuery])
    
    
#  Execute the Tool Call
def execute_tool_call(response):
    '''
    Execute the function based on the model's tool call.

    Args:
        response: The AIMessage from the model.

    Returns:
        The result of executing the tool, or None if no tool call.
    '''
    
    if not response.tool_calls:
        print('No tool call made.')
        return None
    
    # Get the first tool call
    tool_call = response.tool_calls[0]
    tool_name = tool_call['name']
    args = tool_call['args']
    
    print(f'Tool called: {tool_name}')
    print(f'Arguments: {args}')
    
    # Execute the correct function based on tool name
    if tool_name == 'SearchQuery':
        results = fake_search(**args)
        
        return results
    
    else:
        raise ValueError(f'Unknown tool: {tool_name}')
        

def main() -> None:
    '''Run the function calling example.'''
    
    load_dotenv()
    
    # Bind tools to the model
    llm_with_tools = bind_tools_to_model()
    
    # Ask a question
    response = llm_with_tools.invoke('Search for RAG tutorials, top 3')
    
    # Execute the tool call
    results = execute_tool_call(response)
    
    # Display results
    if results:
        print('\nSearch results:')
        
        for i, res in enumerate(results, start=1):
            print(f'{i}. {res}')
            

# Entry Point
if __name__ == '__main__':
    main()