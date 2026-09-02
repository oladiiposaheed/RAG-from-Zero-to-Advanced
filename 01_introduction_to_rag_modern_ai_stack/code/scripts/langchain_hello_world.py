from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from pathlib import Path

# Load .env
ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent

load_dotenv(ROOT_DIR / '.env')

def main():
    '''
    Main function that creates and runs a simple LangChain chain:
    Prompt -> LLM -> Answer
    '''
    
    # Create a ChatOpenAI instance
    llm = ChatOpenAI(model='gpt-4o-mini', temperature=0)
    
    # Define a prompt template with two messages:
    # - system: sets the assistant's behavior
    # - human: contains the user's question
    prompt = ChatPromptTemplate.from_messages([
        ('system', 'You are a helpful assistant.'),
        ('human', 'Explain what RAG is in one sentence.')
    ])
    
    # Combine the prompt and the LLM into a chain
    chain = prompt | llm
    
    # Invoke the chain with an empty dictionary {} because the prompt has no variables.
    response = chain.invoke({})
    
    # Print the content of the response (the actual text answer)
    print(response.content)
    

if __name__ == '__main__':
    main()