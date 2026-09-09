## Module 9: RAG Evaluation & Observability

### Why Evaluate RAG?

When you build a RAG system, it's not enough to just see if it "works" on a few test questions. You need a **systematic way to measure quality** and catch problems early. Evaluation helps you:

- Know if the LLM is **faithful** to the retrieved context.
- Check if the answer is **relevant** to the user question.
- See if the retrieved chunks are **relevant** to the question.
- Debug failures and improve retrieval/generation.
- Compare different models, prompts, or retrieval strategies.

### What to Evaluate?

There are three main metrics for RAG:

1. **Faithfulness** – Is the answer grounded in the retrieved context?  
   The answer should only use information from the provided chunks. If it invents facts, faithfulness is low.

2. **Answer Relevance** – Does the answer directly address the user question?  
   Even if faithful, the answer might be off‑topic.

3. **Context Relevance** – Are the retrieved chunks relevant to the question?  
   This measures the quality of the retriever.

### How LangSmith Helps

LangSmith is a platform for tracing, testing, and evaluating LLM applications. It can:

- **Trace** every step of your chain (retrieval, prompt, LLM call) automatically.
- **Log** inputs, outputs, metadata, and intermediate steps.
- **Evaluate** runs using built‑in or custom evaluators.
- **Build datasets** of question‑answer pairs for regression testing.
- **Compare experiments** when you change a component.

### Visualisation

```
User Query
   │
   ▼
[RAG Chain] → Answer + Retrieved Context
   │
   ▼
[LangSmith] captures trace:
   - retrieved documents
   - prompt
   - LLM output
   │
   ▼
[Evaluators] score:
   - Faithfulness (answer grounded?)
   - Answer Relevance (answer on‑topic?)
   - Context Relevance (retrieval good?)
   │
   ▼
[Dashboard] metrics & reports
```

### What We'll Do in Module 9

We'll learn how to:

1. **Enable tracing** in LangSmith (we've already done basic tracing).
2. **Create evaluation datasets** from our sample questions.
3. **Run evaluations** using LangSmith’s evaluators.
4. **Inspect and interpret** the evaluation results.
5. **Use evaluation feedback** to improve retrieval or prompts.

We'll follow the same pattern: concept → notebook → modular code.

---

## First Topic: Setting Up LangSmith Tracing and Evaluation

We already have tracing enabled through environment variables. Now we'll go deeper and create a small evaluation workflow.

**Would you like to proceed with the first notebook?**  
If yes, I'll guide you step by step. We'll create a notebook in `09_rag_evaluation/notebooks/` and start with setting up LangSmith and a simple QA evaluator.



Let’s start **Module 9: RAG Evaluation & Observability** with the first code step.

## Step 1: Imports and Environment

Create the notebook `01_evaluation_setup.ipynb` inside `09_rag_evaluation/code/notebooks/`. Then add and run this cell:

```python
# Import necessary libraries
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv

# Load environment variables (API keys, LangSmith config)
load_dotenv()

print('✅ Imports and environment ready.')
```

**What this does:**  
- Imports core LangChain components.  
- Loads `.env` file for API keys.  
- Prints confirmation.

Run this cell and share output. Then we’ll load the vector store and build a simple RAG chain for evaluation.



Great! Now we'll load the vector store and build a simple RAG chain that we can evaluate.

## Step 2: Load Vector Store and Build RAG Chain

Add this cell and run it:

```python
# Define the path to the persisted vector store.
# Since this notebook is in 09_rag_evaluation/code/notebooks,
# go up three levels to reach the course root, then into chroma_db.
chroma_storage_dir = '../../../chroma_db'

# Create embedding model
embeddings = OpenAIEmbeddings()

# Load the vector store
vectorstore = Chroma(
    persist_directory=chroma_storage_dir,
    embedding_function=embeddings
)

# Create a retriever
retriever = vectorstore.as_retriever(search_kwargs={'k': 4})

# Create an LLM
llm = ChatOpenAI(model='gpt-4o-mini', temperature=0)

# Define a RAG prompt
rag_prompt = ChatPromptTemplate.from_messages([
    ('system', 'You are a helpful assistant. Answer the question using only the provided context. If you don\'t know, say you don\'t know.'),
    ('human', 'Context:\n{context}\n\nQuestion: {question}')
])

# Helper to format retrieved documents
def format_docs(docs):
    return '\n\n'.join(doc.page_content for doc in docs)

# Build the RAG chain
rag_chain = (
    {
        'context': retriever | format_docs,
        'question': RunnablePassthrough()
    }
    | rag_prompt
    | llm
    | StrOutputParser()
)

print('✅ RAG chain ready for evaluation.')
```

**What this does:**  
- Loads the persisted Chroma vector store.  
- Creates a retriever that returns top 4 chunks.  
- Creates the LLM and prompt.  
- Builds the RAG chain using LCEL.  
- This chain will be used for evaluation.

Run this cell and share output. Then we'll import LangSmith evaluation tools.



Now we'll import LangSmith tools and verify the connection.

## Step 3: Import LangSmith Evaluation Tools

Add this cell and run it:

```python
# Import LangSmith evaluation utilities
from langsmith import Client
from langsmith.evaluation import evaluate

# Create a LangSmith client (uses environment variables)
client = Client()

print('✅ LangSmith evaluation tools ready.')
```

**What this does:**  
- Imports `Client` to interact with LangSmith.  
- Imports `evaluate` to run evaluations on datasets.  
- Creates a client using your API key from `.env`.  
- Confirms setup.

Run this cell and share output. If you see any error about missing API key, we'll verify your `.env`.  

Once successful, we'll create a small evaluation dataset and run the RAG chain on a few questions.



Here's the corrected cell with **single quotes** for the examples list.

```python
# Define a small list of example questions (using single quotes)
examples = [
    'What are the common crop diseases and their control methods?',
    'What are the top causes of death in Nigeria?',
    'What are the symptoms of Cassava Mosaic Disease?',
    'What is Mastitis and how is it managed in dairy animals?',
    "What product has the comment 'Spacious and strong'?",
]

# Use LangSmith evaluate to run the chain on these examples
results = evaluate(
    rag_chain.invoke,           # the function to evaluate (our RAG chain)
    data=examples,              # list of input strings
    evaluators=[],              # we'll add evaluators in the next step
    experiment_prefix='rag-eval-basic'
)

print('✅ Evaluation run completed.')
print(results)
```

**Note:** The last question contains a single quote (`'Spacious and strong'`), so we used double quotes for that string to avoid escaping.

Run this cell and share the output. After that, we'll add evaluators to score the answers.



The problem is clear: your persisted vector store (`chroma_db`) was built from **all raw files**, including the placeholder `ecommerce.json` and `ecommerce.csv`. That’s why retrieval returns Latin text and product data instead of health/agriculture chunks.

The fix is to **rebuild the vector store using only the relevant documents** (health PDF, crop PDF, agriculture HTML, and agriculture TXT). We’ll do that inside the evaluation notebook, then re‑test.

---

## Step A: Import Document Loaders and Splitter

Add and run this cell:

```python
# Import loaders and splitter for rebuilding a clean vector store
from langchain_community.document_loaders import PyPDFLoader, BSHTMLLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
```

---

## Step B: Load Only Relevant Documents

Use this cell (paths are three levels up from `09_rag_evaluation/code/notebooks`):

```python
# Load only health and agriculture documents
health_pdf = PyPDFLoader('../../../04_data_ingestion_document_processing/data/nigeria_health_diseases_and_prevention.pdf').load()
crop_pdf = PyPDFLoader('../../../04_data_ingestion_document_processing/data/crop_disease.pdf').load()
agri_html = BSHTMLLoader('../../../04_data_ingestion_document_processing/data/agriculture.html', open_encoding='utf-8', bs_kwargs={'features': 'html.parser'}).load()
agri_txt = TextLoader('../../../04_data_ingestion_document_processing/data/agriculture.txt', encoding='utf-8').load()

all_docs = health_pdf + crop_pdf + agri_html + agri_txt
print(f'Loaded {len(all_docs)} relevant documents.')
```

---

## Step C: Split and Add Metadata

```python
splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
chunks = splitter.split_documents(all_docs)

for i, chunk in enumerate(chunks):
    source = chunk.metadata.get('source', '')
    file_name = source.split('\\')[-1] if '\\' in source else source.split('/')[-1]
    chunk.metadata['file_name'] = file_name
    chunk.metadata['doc_type'] = (
        'pdf' if file_name.endswith('.pdf')
        else 'html' if file_name.endswith('.html')
        else 'txt'
    )
    chunk.metadata['language'] = 'English'
    chunk.metadata['chunk_id'] = f'{file_name}_{i+1:03d}'

print(f'Created {len(chunks)} clean chunks.')
```

---

## Step D: Create a New Clean Vector Store

We'll persist to a **new directory** `chroma_db_clean` to avoid mixing with the old bad store.

```python
# Create new embeddings and vectorstore
embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory='../../../chroma_db_clean'   # new clean store
)

# Create retriever from clean store
retriever = vectorstore.as_retriever(search_kwargs={'k': 4})

print('✅ Clean vector store ready.')
```

---

## Step E: Rebuild the RAG Chain (if it was defined earlier)

If `rag_chain` was built before using the old retriever, rebuild it now:

```python
# LLM and prompt (if not already defined)
llm = ChatOpenAI(model='gpt-4o-mini', temperature=0)
rag_prompt = ChatPromptTemplate.from_messages([
    ('system', 'You are a helpful assistant. Answer the question using only the provided context. If you don\'t know, say you don\'t know.'),
    ('human', 'Context:\n{context}\n\nQuestion: {question}')
])

def format_docs(docs):
    return '\n\n'.join(doc.page_content for doc in docs)

rag_chain = (
    {
        'context': retriever | format_docs,
        'question': RunnablePassthrough()
    }
    | rag_prompt
    | llm
    | StrOutputParser()
)

print('✅ RAG chain rebuilt with clean retriever.')
```

---

## Step F: Re‑test Manual Evaluation

```python
for q in examples:
    print(f'Question: {q}')
    ans = rag_chain.invoke(q)
    print(f'Answer: {ans}\n---')
```

Now you should get relevant answers from health/agriculture documents. Run these cells and share the output. After that, we can continue with LangSmith evaluation if you want, or proceed with custom evaluators.

Let me know the results.



We'll finish the **manual evaluation** by adding the third key RAG metric: **Context Relevance**. This checks if the retrieved chunks are actually relevant to the question.

---

## Step: Add Manual Context Relevance Evaluator

### 1. Define the prompt

```python
# Prompt for context relevance
context_relevance_prompt = ChatPromptTemplate.from_messages([
    ('system', 'You are an evaluator. Given the QUESTION and RETRIEVED CONTEXT, judge if the RETRIEVED CONTEXT is relevant to the QUESTION. Respond with only "yes" or "no".'),
    ('human', 'QUESTION:\n{question}\n\nRETRIEVED CONTEXT:\n{context}')
])
```

### 2. Define the evaluation function

```python
def evaluate_context_relevance(question, docs):
    context = '\n\n'.join(doc.page_content for doc in docs)
    chain = context_relevance_prompt | llm | StrOutputParser()
    result = chain.invoke({'question': question, 'context': context})
    return result.strip().lower() == 'yes'
```

### 3. Run the full evaluation loop with all three metrics

Update the loop to include context relevance:

```python
for q in examples:
    if 'Spacious' in q:
        continue

    docs = retriever.invoke(q)
    answer = rag_chain.invoke(q)

    faithful = evaluate_faithfulness(q, answer, docs)
    relevant = evaluate_answer_relevance(q, answer)
    context_rel = evaluate_context_relevance(q, docs)

    print(f'Question: {q}')
    print(f'Answer: {answer}')
    print(f'Faithful: {faithful}')
    print(f'Relevant: {relevant}')
    print(f'Context Relevant: {context_rel}')
    print('-' * 100)
```

---

### What This Adds

You now have three RAG metrics:

- **Faithfulness** – answer grounded in context?
- **Answer Relevance** – answer directly addresses the question?
- **Context Relevance** – retrieved chunks are useful for the question?

This completes the manual evaluation of your RAG chain. The results will show which component (retriever vs generator) needs improvement.

---

Run this updated loop and share the output. After that, we can move to the built‑in evaluator version (Option 2) and then LangSmith integration.



We'll now create the new notebook for **built-in evaluators**. This will keep your manual evaluation separate from the production-ready approach.

## Create the Notebook

Create:
```
09_rag_evaluation/code/notebooks/02_builtin_evaluators.ipynb
```

Then add the concept markdown below, followed by the code cells.

---

## Concept Markdown (smaller font)

```html
<div style="font-size: 0.85em; line-height: 1.5;">

<h3>Built-in RAG Evaluators in LangChain</h3>

<p><strong>Why use built-in evaluators?</strong><br>
LangChain provides standard evaluation chains that automatically score answers for common criteria. They save time and ensure consistency.</p>

<p><strong>Two key evaluators for RAG</strong></p>
<ul>
  <li><strong><code>ContextQAEvalChain</code></strong> – measures <em>faithfulness</em>: whether the answer is grounded in the retrieved context.</li>
  <li><strong><code>CriteriaEvalChain</code></strong> – measures a chosen criterion, such as <em>relevance</em>: whether the answer directly addresses the question.</li>
</ul>

<p><strong>How they work</strong></p>
<ol>
  <li>Create an evaluator instance from an LLM.</li>
  <li>Call <code>evaluate_strings()</code> with the answer, question, and (for faithfulness) the context.</li>
  <li>Receive a dict with a <code>score</code> (1 = yes, 0 = no) and <code>reasoning</code>.</li>
</ol>

<p><strong>Why this matters</strong><br>
Built-in evaluators give a standard, reproducible way to measure RAG quality. They can be used in tests and with LangSmith experiments.</p>

</div>
```

---

## First Code Cell: Imports

After the markdown, add a code cell and run:

```python
# Import built-in evaluation chains
from langchain.evaluation import ContextQAEvalChain, CriteriaEvalChain
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv

load_dotenv()

print('✅ Imports ready for built-in evaluators.')
```

**What this does:**  
- Imports the evaluation chains and RAG components.  
- Loads environment variables.  
- Prepares for building the RAG chain and evaluators.

Run this cell and share output. Then we'll recreate the clean RAG chain in this notebook.



Now we'll load the **clean vector store** and rebuild the RAG chain in this notebook. This ensures we use the same good data as before.

## Cell 2: Load Clean Vector Store and Build RAG Chain

Add and run this cell:

```python
# Path to the clean vector store (created in the previous notebook)
chroma_storage_dir = '../../../chroma_db_clean'

# Create embedding model
embeddings = OpenAIEmbeddings()

# Load the clean vector store
vectorstore = Chroma(
    persist_directory=chroma_storage_dir,
    embedding_function=embeddings
)

# Create a retriever
retriever = vectorstore.as_retriever(search_kwargs={'k': 4})

# Create LLM
llm = ChatOpenAI(model='gpt-4o-mini', temperature=0)

# Define RAG prompt
rag_prompt = ChatPromptTemplate.from_messages([
    ('system', 'You are a helpful assistant. Answer the question using only the provided context. If you don\'t know, say you don\'t know.'),
    ('human', 'Context:\n{context}\n\nQuestion: {question}')
])

# Helper to format retrieved documents
def format_docs(docs):
    return '\n\n'.join(doc.page_content for doc in docs)

# Build RAG chain
rag_chain = (
    {
        'context': retriever | format_docs,
        'question': RunnablePassthrough()
    }
    | rag_prompt
    | llm
    | StrOutputParser()
)

print('✅ RAG chain ready using clean vector store.')
```

Run this cell and share output. Then we'll create the built-in evaluators.



We'll define the example questions we'll evaluate using the built‑in evaluators.

## Cell 3: Define Example Questions

Add and run this cell:

```python
# Define a small list of evaluation questions
examples = [
    'What are the common crop diseases and their control methods?',
    'What are the top causes of death in Nigeria?',
    'What are the symptoms of Cassava Mosaic Disease?',
    'What is Mastitis and how is it managed in dairy animals?',
]

print('✅ Examples defined.')
```

**What this does:**  
- Creates a list of health/agriculture questions.  
- We'll use this list for the built‑in evaluators.

Run this cell and share output. Then I'll give the next step: creating the evaluator instances.


