# 📘 RAG Course – Modules 1–4 Interview Summary

## Module 1: Introduction to RAG and Modern AI Stack

### Definition

**Retrieval-Augmented Generation (RAG)** is an AI architecture that combines **information retrieval** with **language generation**. It retrieves relevant documents from a knowledge base and uses them as context for a large language model (LLM) to generate accurate, grounded answers.

### Key Components

- **Retriever** – finds relevant documents using embeddings and similarity search.
- **Knowledge Base** – collection of documents, usually split into chunks.
- **Generator** – an LLM that produces the final answer based on retrieved context.

### Visualisation

```
User Query
   │
   ▼
[Retriever] → Top‑k relevant documents
   │
   ▼
[Prompt = Query + Retrieved Documents]
   │
   ▼
[LLM Generator] → Answer
```

### Why Use RAG?

- ✅ Up-to-date knowledge without retraining
- ✅ No fine-tuning needed
- ✅ Grounded answers with citations
- ✅ Handles private / proprietary data

### Tools Used

- **LangChain** – framework for building LLM applications
- **LangGraph** – for stateful, agentic workflows
- **LangSmith** – observability, tracing, evaluation

### Interview Points

- Explain the difference between RAG, fine-tuning, and prompt engineering.
- What are the three core components of a RAG pipeline?
- Why is RAG often preferred over fine-tuning for enterprise knowledge bases?

---

## Module 2: LangChain Core Fundamentals

### 2A. Core Concepts

- **LLM vs Chat Model**
  - LLM: takes a string, returns a string (e.g., `gpt-3.5-turbo-instruct`)
  - Chat Model: takes a list of messages, returns a message (e.g., `gpt-4o-mini`)
- **Prompt Templates** – reusable prompts with placeholders (`{topic}`).
- **Output Parsers** – `StrOutputParser` (plain text), `CommaSeparatedListOutputParser` (list).
- **LCEL** – LangChain Expression Language; components are connected with the pipe `|` operator.
- **Runnables** – `RunnablePassthrough` (passes input unchanged), `RunnableLambda` (custom functions), `RunnableParallel` (run multiple branches concurrently).

### 2B. Advanced

- **Few-shot prompting** – providing examples in the prompt to guide the output.
- **Pydantic Output Parser** – returns validated, structured objects.
- **RunnableBranch** – conditional routing (like `if/else` for chains).
- **`.bind()`, `.with_config()`, `.with_retry()`, `.with_fallbacks()`** – production-level utilities.
- **Mini-project**: Customer Support Assistant that classifies intent, routes to a specialist, and returns a structured response.

### Visualisation – Basic Chain

```
Input dict → Prompt Template → Chat Model → Output Parser → Final Output
```

### Visualisation – Advanced Chain with Branching

```
Input dict
   │
   ▼
[RunnableBranch]
   ├── condition 1 ──> Chain A
   ├── condition 2 ──> Chain B
   └── default      ──> Chain C
```

### Interview Points

- What is LCEL and why is the pipe operator useful?
- How do you get structured output from an LLM? (Use Pydantic parsers)
- Explain the purpose of `RunnableBranch` and give an example.
- What are fallbacks and retries, and why are they important in production?

---

## Module 3: Pydantic & Structured Outputs

### Definition

**Pydantic** is a Python library for data validation using type hints. It ensures that data (like LLM outputs) is valid, correctly typed, and well-structured.

### Key Features

- `BaseModel` – define data models.
- `Field()` – add descriptions, defaults, constraints.
- Nested models, lists, optional fields.
- `PydanticOutputParser` – converts LLM JSON output into validated Pydantic objects.
- Function calling / tool schemas with `bind_tools()`.
- Error handling: `OutputFixingParser`, `try/except`, retries.

### Visualisation – Structured Output Flow

```
Pydantic Model → Parser → Format Instructions → Prompt → LLM → JSON → Pydantic Object
```

### Interview Points

- Why use Pydantic in RAG? (validation, type safety)
- What does `get_format_instructions()` do?
- How does `OutputFixingParser` work and when would you use it?
- Explain function calling and tool schemas.

---

## Module 4: Data Ingestion & Document Processing

### Definition

**Document ingestion** is the process of loading raw files (PDF, CSV, JSON, HTML, TXT) and preparing them for retrieval by splitting them into chunks and adding metadata.

### Key Tasks

1. **Document Loaders** – `PyPDFLoader`, `CSVLoader`, `JSONLoader`, `TextLoader`, etc.
2. **Text Splitting** – `RecursiveCharacterTextSplitter` is the production standard.
3. **Chunk Size & Overlap** – control chunk size and overlap to preserve context.
4. **Metadata** – source, file name, doc type, language, chunk ID.
5. **Indexing Pipeline** – load → split → enrich metadata → save chunks.

### Visualisation – Indexing Pipeline

```
Raw files → Document Loader → Text Splitter → Metadata Enrichment → Save Chunks (JSON)
```

### Visualisation – Chunking with Overlap

```
Document: "A B C D E F G H I J"
chunk_size = 4, overlap = 2

Chunk 1: A B C D
Chunk 2: C D E F
Chunk 3: E F G H
Chunk 4: G H I J
```

### Interview Points

- Which splitter is best for production? **`RecursiveCharacterTextSplitter`**
- Why is chunk overlap important? It prevents context loss at boundaries.
- What metadata would you add? Source, page, language, chunk ID.
- Explain the steps of a document indexing pipeline.

---

## Pending (Not Yet Covered)

- Web scraping (Task 2/3 of Module 4) – postponed
- Multilingual / Nigerian languages – Module 8
- Embeddings & Vector Stores – Module 5 (next)
