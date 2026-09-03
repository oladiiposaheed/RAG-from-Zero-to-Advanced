Let’s restart Module 6 cleanly. I’ll guide you from the very beginning, one small step at a time, with explanations and outputs.

We will build a **Core RAG** system that:

- Loads a persisted vector store (from Module 5)
- Retrieves relevant chunks
- Deduplicates them
- Generates an answer using an LLM
- Displays the answer

We will do it in a fresh notebook or clear the current one.

---

## Step 1: Imports

Create a new notebook or clear current cells.  
Then run this cell:

```python
# Import required libraries
from pathlib import Path
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

print('✅ Imports successful.')
```

**What this does:**  
- Imports all needed components for Core RAG.
- Loads API keys.

Run this and share output. Then we’ll load the vector store.


You’re right, we need `Path` to use `.exists()`. Let’s fix that.

## Corrected Step 2: Load Vector Store and Create Retriever

We’ll use `Path` correctly this time:

```python
from pathlib import Path

# Define the path to the persisted vector store (go up 3 levels to course root)
chroma_storage_dir = Path('../../../chroma_db')

# Check if the folder exists
if not chroma_storage_dir.exists():
    print(f'❌ Vector store not found at {chroma_storage_dir}')
else:
    print(f'✅ Vector store found at {chroma_storage_dir}')

# Create embedding model
embeddings = OpenAIEmbeddings()

# Load Chroma from disk
vectorstore = Chroma(
    persist_directory=str(chroma_storage_dir),
    embedding_function=embeddings
)

# Create retriever
retriever = vectorstore.as_retriever(search_kwargs={'k': 4})

print('✅ Retriever created.')
```

Run this corrected cell. It should now either print that the vector store is found or not. Share the output, and we'll continue.


Let's now test the retriever directly to see if it returns relevant chunks.

## Step 3: Test the Retriever

Run this cell:

```python
# Define a test question
question = 'What are common crop diseases and how can they be controlled?'

# Retrieve the top 4 chunks without any deduplication yet
raw_docs = retriever.invoke(question)

print(f'Retrieved {len(raw_docs)} chunks:\n')
for i, doc in enumerate(raw_docs, start=1):
    print(f'Chunk {i}:')
    print(doc.page_content)
    print('-' * 50)
```

**What this does:**
- Sends the question to the retriever.
- Retrieves up to 4 chunks from Chroma.
- Prints each chunk so you can see what was found.

Run this and share the output. If you see chunks, we’ll proceed to **deduplicate** and build the full RAG chain. If it returns 0 chunks, we’ll adjust the path again, but it should work now.


We need to remove duplicate chunks so the LLM receives clean, unique context. We'll define a deduplication function and test it on the retrieved chunks.

## Step 4: Deduplicate Retrieved Chunks

Run this cell:

```python
def deduplicate_docs(docs):
    """Return only documents with unique page_content."""
    seen = set()
    unique_docs = []
    for doc in docs:
        text = doc.page_content.strip()   # remove leading/trailing whitespace
        if text not in seen:
            seen.add(text)
            unique_docs.append(doc)
    return unique_docs

# Use the function on the raw retrieved docs
unique_docs = deduplicate_docs(raw_docs)

print(f'Before deduplication: {len(raw_docs)} chunks')
print(f'After deduplication: {len(unique_docs)} chunks')
```

**What this does:**  
- Keeps only the first occurrence of each unique text.  
- Reduces duplicate chunks so the context is cleaner.

Run this cell and share the output. The `After deduplication` number should be smaller. Then we'll build the RAG chain using this deduplication step.



Now we'll build the RAG chain using the deduplicated chunks. We'll define a function to format the documents into context, then create the prompt, LLM, and chain.

## Step 5: Define `format_docs` and Create Prompt

Run this cell:

```python
def format_docs(docs):
    """Combine a list of documents into a single context string."""
    return '\n\n'.join(doc.page_content for doc in docs)

# Create the RAG prompt template
rag_prompt = ChatPromptTemplate.from_messages([
    ('system', 'You are a helpful assistant. Answer the question using only the provided context. If you don\'t know, say you don\'t know.'),
    ('human', 'Context:\n{context}\n\nQuestion: {question}')
])

print('✅ Prompt and formatting function ready.')
```

## Step 6: Create LLM and Build RAG Chain

```python
# Create the LLM (generator)
llm = ChatOpenAI(model='gpt-4o-mini', temperature=0)

# Build the RAG chain with deduplication
rag_chain = (
    {
        'context': retriever | deduplicate_docs | format_docs,
        'question': RunnablePassthrough()
    }
    | rag_prompt
    | llm
    | StrOutputParser()
)

print('✅ RAG chain created with deduplication.')
```

## Step 7: Test the RAG Chain

```python
# Ask the same question
answer = rag_chain.invoke(question)

print('Answer:')
print(answer)
```

Run these three cells in order. The chain will:
- Retrieve 4 chunks
- Deduplicate to 2 unique chunks
- Format them into context
- Send context + question to the LLM
- Output a grounded answer

Share the final answer.


Now that the core RAG chain works in the notebook, we'll build a modular **`RAGService`** class for production use, just like we did for loaders, splitters, and vector store.

We'll create `app/services/rag_service.py` step by step.

## Step 1: Imports and Class Setup

Create the file and add this first part:

```python
"""
RAG service.

Provides a reusable class that combines a retriever and a generator (LLM)
to answer questions using retrieved context from a vector store.
"""

from typing import List
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from app.logger.logger import get_logger
from app.exceptions.custom_exceptions import RAGError  # we'll add this if not exists

logger = get_logger(__name__)
```

**What this does:**  
- Imports required LangChain components.  
- Imports logger and a custom `RAGError` exception (we'll add that shortly).  
- Prepares for the `RAGService` class.

Would you like me to first add the `RAGError` exception to `custom_exceptions.py`, or continue with the `RAGService` class definition? I suggest adding the exception first for clean error handling.



You're right — we need to add `RAGError` to `custom_exceptions.py` before using it. Let's do that first.

## Step 1: Add `RAGError` to `app/exceptions/custom_exceptions.py`

Open the file and add this class at the end:

```python
class RAGError(BaseAppError):
    """Raised when a RAG pipeline operation fails."""
```

This inherits from `BaseAppError`, so it already has the improved `__init__` that accepts `message` and `details`.

Save the file.

Let me know when you've added it, and then we'll continue with `RAGService`.


