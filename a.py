## Contextual Compression / Reranking – Concept

In RAG, after the retriever finds chunks, some chunks may be partially relevant or contain extra text that isn’t useful for answering the query. Passing all that text to the LLM can increase token usage and reduce answer quality.

**Contextual compression** solves this by taking the retrieved chunks and **compressing or filtering** them to keep only the parts that are most relevant to the query before sending them to the generator. This is often done by an LLM that reads each chunk and extracts or keeps only the relevant sentences.

**Reranking** is a related technique: instead of compressing, you **reorder** the retrieved chunks by relevance score (using a reranking model) so the most useful chunks appear first. Some systems combine both—compress first, then rerank.

### Visualisation

```
Retriever → Many chunks (some noisy)
        │
        ▼
[Contextual Compressor] → Smaller, focused excerpts
        │
        ▼
[Reranker (optional)] → Sorted by relevance
        │
        ▼
LLM Generator → Better answer with less irrelevant context
```

### Why Use It?

- **Reduces token usage** – less irrelevant text means smaller prompts.
- **Improves answer quality** – the LLM sees only the most useful context.
- **Reduces hallucination** – fewer distractors in context.
- **Faster and cheaper** – especially with long chunks.

### Example

If retrieved chunk is a long paragraph about agriculture and the query is “What is maize smut?”, contextual compression keeps only the sentence(s) about maize smut and drops the rest.

We will use LangChain’s `ContextualCompressionRetriever` and `LLMChainExtractor` (or `LLMChainFilter`) to implement this.

---

Ready to proceed to the notebook? We'll do it step by step. Just say **"next"** and I'll provide the first cell.



We'll create a new notebook for **Contextual Compression / Reranking**.

## Notebook File

Create a new notebook in:

```
07_advanced_retrieval/notebooks/
```

Name it:

```
04_contextual_compression.ipynb
```

Then select the `rag_env` kernel as usual.

---

## Step 1: Add the Concept Markdown (Small Size)

Copy and paste this HTML into a Markdown cell at the top of the notebook:

```html
<div style="font-size: 0.85em;">

<h3>Contextual Compression / Reranking</h3>

<p><strong>What is it?</strong><br>
After retrieving chunks, some may be partly irrelevant or contain extra text.<br>
Contextual compression reduces each chunk to only the parts that are relevant to the query.<br>
Reranking reorders chunks by relevance before sending them to the LLM.</p>

<p><strong>Visualisation</strong></p>
<pre>
Retriever → Many chunks (some noisy)
        │
        ▼
[Contextual Compressor] → Smaller, focused excerpts
        │
        ▼
[Reranker (optional)] → Sorted by relevance
        │
        ▼
LLM Generator → Better answer with less irrelevant context
</pre>

<p><strong>Why use it?</strong></p>
<ul>
  <li><strong>Reduces token usage</strong> – smaller prompts, lower cost.</li>
  <li><strong>Improves answer quality</strong> – LLM sees only useful context.</li>
  <li><strong>Reduces hallucination</strong> – fewer distractors.</li>
  <li><strong>Faster and cheaper</strong> – especially with long documents.</li>
</ul>

<p><strong>Example</strong><br>
If a retrieved chunk is a long paragraph about agriculture, and the query is “What is maize smut?”, contextual compression keeps only the sentence(s) about maize smut and drops the rest.</p>

<p><strong>Implementation</strong><br>
We will use LangChain’s <code>ContextualCompressionRetriever</code> and <code>LLMChainExtractor</code> (or <code>LLMChainFilter</code>).</p>

</div>
```

---

## Step 2: First Code Cell – Imports

Add a new code cell and run it:

```python
# Import necessary libraries for contextual compression
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

print('✅ Imports successful.')
```

**What this does:**  
- Imports `ContextualCompressionRetriever` and `LLMChainExtractor` for compression.  
- Imports other components we’ll need later.

Run this cell and share the output. Then we’ll create a base retriever using the clean vector store.



We’ll now build the **clean vector store** inside this notebook, using only the relevant documents. This ensures the contextual compression experiment works on good data.

## Step 2: Load Relevant Documents

Add a new code cell and run it:

```python
from langchain_community.document_loaders import PyPDFLoader, BSHTMLLoader, TextLoader

# Load only health and agriculture documents (same as before)
health_pdf = PyPDFLoader('../../04_data_ingestion_document_processing/data/nigeria_health_diseases_and_prevention.pdf').load()
crop_pdf = PyPDFLoader('../../04_data_ingestion_document_processing/data/crop_disease.pdf').load()
agri_html = BSHTMLLoader('../../04_data_ingestion_document_processing/data/agriculture.html', open_encoding='utf-8', bs_kwargs={'features': 'html.parser'}).load()
agri_txt = TextLoader('../../04_data_ingestion_document_processing/data/agriculture.txt', encoding='utf-8').load()

# Combine all documents
all_docs = health_pdf + crop_pdf + agri_html + agri_txt

print(f'Loaded {len(all_docs)} relevant documents.')
```

**What this does:**  
- Loads the PDFs, HTML, and text file from your data folder.  
- Combines them into one list for splitting.

Run this and share the output. Then we’ll split and add metadata.



## Step 3: Split and Add Metadata

Add a new cell and run:

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Create a text splitter
splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50,
    separators=['\n\n', '\n', '.', ' ', '']
)

# Split the loaded documents into chunks
chunks = splitter.split_documents(all_docs)

# Add metadata to each chunk
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

print(f'Created {len(chunks)} chunks.')
```

**What this does:**  
- Splits documents into chunks of 300 characters with 50 overlap.  
- Adds useful metadata like `file_name`, `doc_type`, `language`, `chunk_id`.

Run this and share output. Then we'll create the vector store and base retriever.



**Vector store** and **base retriever** are the core components that make retrieval possible in RAG.

### Vector Store
- It’s a database that holds your chunk embeddings (the numeric representations of text).
- In this notebook, it’s Chroma. When you add documents, it automatically embeds each chunk using the embedding model.
- You can later search this store by similarity – it returns the chunks whose vectors are closest to the query vector.

### Base Retriever
- It’s a wrapper around the vector store that gives you a simple method to query it.
- When you call `retriever.invoke(query)`, it performs the similarity search and returns the top‑k most relevant chunks.
- We set `k=4` in the search kwargs, so it returns 4 chunks per query.

### Why they matter
- Without them, you cannot find which parts of your documents are relevant to a user question.
- They form the **retrieval** half of RAG. The generator (LLM) will later use the retrieved chunks as context to produce an answer.

Next we will create the vector store from the 185 chunks, then create a base retriever. Once we have that, we’ll add contextual compression to filter the retrieved chunks and improve answer quality.


We'll now create the **vector store** and **base retriever** using the chunks.

## Step 4: Create Vector Store and Base Retriever

Add and run this cell:

```python
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

# Create embedding model
embeddings = OpenAIEmbeddings()

# Create an in-memory Chroma vector store from the chunks
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=None      # keep in memory for this notebook
)

# Create a base retriever that returns top 4 chunks per query
base_retriever = vectorstore.as_retriever(search_kwargs={'k': 4})

print('✅ Vector store and base retriever ready.')
```

**What this does:**  
- Embeds all 185 chunks and stores them in memory.  
- Creates a retriever that can return the 4 most similar chunks for any query.

Run this cell and share the output. Then we’ll add **contextual compression** on top of this base retriever.



We’ll now add **contextual compression** on top of the base retriever.

## Step 5: Create Contextual Compression Retriever

Add and run this cell:

```python
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

# Create an LLM for extraction
llm = ChatOpenAI(model='gpt-4o-mini', temperature=0)

# Create a compressor that keeps only relevant parts of each chunk
compressor = LLMChainExtractor.from_llm(llm)

# Wrap the base retriever with the compressor
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=base_retriever
)

print('✅ Contextual compression retriever ready.')
```

**What this does:**  
- `LLMChainExtractor` uses the LLM to extract only the relevant content from each retrieved chunk.  
- `ContextualCompressionRetriever` applies that compressor after the base retriever fetches chunks.

Now we’ll test the same query we used earlier to see how compression changes the retrieved content.

## Step 6: Test Retrieval with and without Compression

Add and run this cell:

```python
# Choose a query
query = 'What are Nigeria communicable and infectious diseases?'

# Without compression
raw_docs = base_retriever.invoke(query)
print('Without compression:')
for i, doc in enumerate(raw_docs, start=1):
    print(f'{i}. {doc.page_content[:200]}')
    print('   Source:', doc.metadata.get('source', 'unknown'))

# With compression
compressed_docs = compression_retriever.invoke(query)
print('\nWith compression:')
for i, doc in enumerate(compressed_docs, start=1):
    print(f'{i}. {doc.page_content[:200]}')
    print('   Source:', doc.metadata.get('source', 'unknown'))
```

Run this cell and share the output. You should see that the compressed docs are shorter and more focused than the raw ones, containing only the most relevant sentences.



Your output shows that **contextual compression is working correctly**—it shortened the retrieved chunks, keeping only the most relevant sentence(s). However, you still don't see the specific disease list because the **base retriever did not retrieve the chunks that contain that list** in the top‑4 results. Contextual compression only filters what was already retrieved; it does **not** find new information.

So the issue is not the compression, but the underlying retrieval. In our earlier experiment, **multi‑query retrieval** successfully retrieved those disease‑list chunks. The best approach is to **combine multi‑query retrieval with contextual compression**:

1. Use multi‑query to generate several question variations and retrieve many chunks.
2. Apply contextual compression to those chunks to keep only the most relevant parts.
3. Pass the compressed context to the LLM for a better answer.

This gives both **high recall** (from multi‑query) and **high precision** (from compression).

---

### Next Step: Build a Combined Retriever (Multi‑Query + Compression)

We can do this in the same notebook. Here’s how to implement it:

#### Step 1: Create a MultiQueryRetriever (as before)

```python
from langchain.retrievers.multi_query import MultiQueryRetriever

# Use the same LLM (already defined)
multi_query_retriever = MultiQueryRetriever.from_llm(
    retriever=base_retriever,
    llm=llm
)
```

#### Step 2: Wrap the MultiQueryRetriever with Contextual Compression

```python
# Create the compression retriever on top of multi_query_retriever
compressed_multi_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=multi_query_retriever
)
```

#### Step 3: Test the Same Query

```python
query = 'What are Nigeria communicable and infectious diseases?'

# Retrieve with combined retriever
docs = compressed_multi_retriever.invoke(query)

print('Retrieved compressed chunks from multi-query:')
for i, doc in enumerate(docs, start=1):
    print(f'{i}. {doc.page_content}')
    print(f'    Source: {doc.metadata.get("source", "unknown")}')
```

Now you should see chunks that contain the actual disease list, but compressed to only the essential sentences.

---

After running, let me know the output. We can then integrate this combined retriever into the modular `RAGService` or continue to the next advanced technique.


Great! 🎉 The combination of **multi‑query + contextual compression** worked. Now the retrieval contains the actual disease list (HIV/AIDS, tuberculosis, malaria, etc.) in compressed form. This is exactly what we wanted.

### What You Achieved

- **Multi‑query** generated multiple question variations, increasing the chance of retrieving the disease list.
- **Contextual compression** removed irrelevant text from the retrieved chunks, keeping only the essential disease list.
- The final retrieved chunks are now **short, focused, and contain the answer**.

This is a powerful production pattern: **high recall (multi‑query) + high precision (compression)**.

---

## Next Steps

We can now:

1. **Integrate this combined retriever into the modular `RAGService`** – add a flag like `use_multi_query` and `use_compression`, or combine them.
2. **Continue to the next advanced retrieval technique** – e.g., **Self‑Query Retriever** or **Parent Document Retriever**.
3. **Move to Module 9 – Evaluation & Observability** – to measure and improve retrieval quality systematically.

I recommend **option 1** because it makes your production service much stronger and directly builds on what we just tested.

Would you like to proceed with integration, or do you prefer to explore another technique? Let me know, and I’ll guide you step by step.



### Simple Explanation

- **`use_multi_query`**  
  When set to `True`, the RAG service will **ask the same question in multiple different ways** before searching for documents.  
  This helps find more relevant chunks that a single question might miss.  
  Example: If you ask “What are Nigeria’s diseases?”, it might also search “List common illnesses in Nigeria” and “What are the major health problems in Nigeria?”.

- **`use_compression`**  
  When set to `True`, the service will **shrink the retrieved chunks** to keep only the sentences that are actually useful for answering the question.  
  This removes filler text and reduces noise, making the final answer more accurate and cheaper (less tokens).

Both are optional and default to `False`, so the default behaviour is simple retrieval. You can enable them individually or together for better results.

---

Now, let’s continue with **Step 2: Build the retriever chain** based on these flags. I’ll provide the code.



Great! We'll now load the relevant documents and prepare them for the vector store.

## Step 2: Load Relevant Documents and Create Chunks

Add this cell and run it:

```python
# Load only the health and agriculture documents (same as clean setup)
health_pdf = PyPDFLoader('../../04_data_ingestion_document_processing/data/nigeria_health_diseases_and_prevention.pdf').load()
crop_pdf = PyPDFLoader('../../04_data_ingestion_document_processing/data/crop_disease.pdf').load()
agri_html = BSHTMLLoader('../../04_data_ingestion_document_processing/data/agriculture.html', open_encoding='utf-8', bs_kwargs={'features': 'html.parser'}).load()
agri_txt = TextLoader('../../04_data_ingestion_document_processing/data/agriculture.txt', encoding='utf-8').load()

# Combine all documents
all_docs = health_pdf + crop_pdf + agri_html + agri_txt
print(f'Loaded {len(all_docs)} documents.')

# Split into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
chunks = splitter.split_documents(all_docs)

# Add metadata
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

print(f'Created {len(chunks)} chunks.')
```

**What this does:**  
- Loads four relevant documents (health, crop, agriculture).  
- Splits them into chunks.  
- Adds useful metadata: `file_name`, `doc_type`, `language`, `chunk_id`.

Run this cell and share output. Then we'll build the vector store.


We'll now continue from **Step 4: Define Metadata Field Info**.

## Step 4: Define Metadata Field Info

Add this cell and run it:

```python
# Describe the metadata fields available in the chunks
metadata_field_info = [
    AttributeInfo(
        name='source',
        description='The file path of the document. Contains keywords like "nigeria_health" or "crop_disease" or "agriculture".',
        type='string',
    ),
    AttributeInfo(
        name='doc_type',
        description='The type of document: pdf, html, txt',
        type='string',
    ),
    AttributeInfo(
        name='language',
        description='Language of the document, e.g., English',
        type='string',
    ),
]
```

**What this does:**  
- Informs the self‑query retriever which metadata fields exist and what they mean.  
- The LLM will use this information to decide filters automatically.

Run this cell. Then we'll create the `SelfQueryRetriever` in Step 5.


We'll now create the **SelfQueryRetriever** using the metadata field info.

## Step 5: Create the SelfQueryRetriever

Add and run this cell:

```python
# LLM for parsing queries (same as before)
llm = ChatOpenAI(model='gpt-4o-mini', temperature=0)

# Document content description
document_content_description = 'Documents about agriculture, public health, and diseases in Nigeria'

# Create self-query retriever
self_query_retriever = SelfQueryRetriever.from_llm(
    llm=llm,
    vectorstore=vectorstore,
    document_contents=document_content_description,
    metadata_field_info=metadata_field_info,
    verbose=True    # shows the generated query and filter
)

print('✅ SelfQueryRetriever ready.')
```

**What this does:**  
- Uses the LLM to automatically infer the semantic query and metadata filter from the user question.  
- `verbose=True` prints the filter so you can see how it decides to restrict retrieval.  
- The retriever is now ready to use.

Run this cell. You may see a log about the retriever initialising. Share the output, then we'll test it with a question.



We’ll now test the self‑query retriever with a question that should automatically filter by source or document type.

## Step 6: Test the SelfQueryRetriever

Add and run this cell:

```python
# Ask a question that should retrieve only from health documents
question = 'What are the common diseases in Nigeria according to the health document?'

# Retrieve using self-query retriever
docs = self_query_retriever.invoke(question)

print(f'Retrieved {len(docs)} chunks:\n')
for i, doc in enumerate(docs, start=1):
    print(f'{i}. {doc.page_content[:200]}')
    print(f'   Source: {doc.metadata.get("source", "unknown")}')
    print(f'   Type: {doc.metadata.get("doc_type", "unknown")}')
```

**What to expect:**  
- The `verbose=True` in the retriever will print a line like:
  ```
  Query: common diseases in Nigeria
  Filter: {"source": {"$contains": "nigeria_health"}}
  ```
- The returned chunks should come only from the health PDF, not from agriculture or ecommerce.
- This demonstrates automatic metadata filtering.

Run this cell and share the output. After that, we can try another question to see different filters, then integrate this into the modular `RAGService`.



We'll now integrate **Self-Query Retriever** into the modular `RAGService`, step by step.

## Step 1: Add Imports

Open `app/services/rag_service.py` and add these imports at the top (with the other LangChain imports):

```python
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo
```

These are needed to create the self-query retriever.

Let me know when you've added these imports, and I'll give Step 2 (adding the `use_self_query` parameter).



We'll now integrate **Self-Query Retriever** into the modular `RAGService`, step by step.

## Step 1: Add Imports

Open `app/services/rag_service.py` and add these imports at the top (with the other LangChain imports):

```python
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo
```

These are needed to create the self-query retriever.

Let me know when you've added these imports, and I'll give Step 2 (adding the `use_self_query` parameter).


You put it **inside the `__init__` method**, **after** the `elif self.use_compression:` branch and **before** the final `else:` branch.

Here is the exact placement in your `rag_service.py`:

```python
            elif self.use_compression:
                compressor = LLMChainExtractor.from_llm(self.llm)
                self.retrieval = ContextualCompressionRetriever(
                    base_compressor=compressor,
                    base_retriever=self.retriever
                )
                logger.info('Using contextual compression retrieval.')

            # ---- INSERT THE SELF-QUERY BRANCH HERE ----
            elif self.use_self_query:
                metadata_field_info = [
                    AttributeInfo(
                        name='source',
                        description='The file path of the document. Contains keywords like "nigeria_health" or "crop_disease" or "agriculture".',
                        type='string',
                    ),
                    AttributeInfo(
                        name='doc_type',
                        description='The type of document: pdf, html, txt',
                        type='string',
                    ),
                    AttributeInfo(
                        name='language',
                        description='Language of the document, e.g., English',
                        type='string',
                    ),
                ]

                document_content_description = 'Documents about agriculture, public health, and diseases in Nigeria'

                self.retrieval = SelfQueryRetriever.from_llm(
                    llm=self.llm,
                    vectorstore=self.vectorstore,
                    document_contents=document_content_description,
                    metadata_field_info=metadata_field_info,
                    verbose=False
                )
                logger.info('Using self-query retrieval.')

            else:
                self.retrieval = self.retriever
                logger.info('Using simple retrieval.')
```

Make sure the indentation matches (everything is inside the `try` block, aligned with the other `elif` branches). Also ensure the imports for `SelfQueryRetriever` and `AttributeInfo` are at the top of the file.

After adding this, the retrieval mode selection will support `--self-query` once we update `query.py` in the next step. Let me know when you've inserted it, and we'll proceed.



We'll start by updating `query.py` to support a `--all` flag that enables all three techniques.

## Step 1: Add `--all` flag to `query.py`

Replace the current `argparse` section in `query.py` with:

```python
    parser = argparse.ArgumentParser()
    parser.add_argument('--multi-query', action='store_true', help='Use multi-query retrieval')
    parser.add_argument('--compression', action='store_true', help='Use contextual compression')
    parser.add_argument('--self-query', action='store_true', help='Use self-query retrieval')
    parser.add_argument('--all', action='store_true', help='Use all retrieval techniques combined')
    args = parser.parse_args()

    # If --all is set, enable all techniques
    if args.all:
        use_multi_query = True
        use_compression = True
        use_self_query = True
    else:
        use_multi_query = args.multi_query
        use_compression = args.compression
        use_self_query = args.self_query

    logger.info(
        f'Starting RAG query (multi-query={use_multi_query}, '
        f'compression={use_compression}, self-query={use_self_query})...'
    )

    rag = RAGService(
        use_multi_query=use_multi_query,
        use_compression=use_compression,
        use_self_query=use_self_query
    )
```

This way, running `python query.py --all` will set all three flags to `True` before passing them to `RAGService`.

Now we need to update `RAGService` to handle the combined case. Let's proceed to Step 2.



We’ll now update `RAGService` to support the combined mode.

## Step 2: Add Combined Retrieval Branch

In `app/services/rag_service.py`, inside `__init__`, **before** the existing `if self.use_multi_query and self.use_compression:` branch, insert a new branch for all three flags.

Here’s the code to insert:

```python
            if self.use_self_query and self.use_multi_query and self.use_compression:
                # --- Combined: self-query -> multi-query -> compression ---

                # 1. Self-query retriever (automatic metadata filter)
                metadata_field_info = [
                    AttributeInfo(
                        name='source',
                        description='The file path of the document. Contains keywords like "nigeria_health" or "crop_disease" or "agriculture".',
                        type='string',
                    ),
                    AttributeInfo(
                        name='doc_type',
                        description='The type of document: pdf, html, txt',
                        type='string',
                    ),
                    AttributeInfo(
                        name='language',
                        description='Language of the document, e.g., English',
                        type='string',
                    ),
                ]
                document_content_description = 'Documents about agriculture, public health, and diseases in Nigeria'

                self_query_retriever = SelfQueryRetriever.from_llm(
                    llm=self.llm,
                    vectorstore=self.vectorstore,
                    document_contents=document_content_description,
                    metadata_field_info=metadata_field_info,
                    verbose=False
                )

                # 2. Multi-query on top of self-query
                multi_query_retriever = MultiQueryRetriever.from_llm(
                    retriever=self_query_retriever,
                    llm=self.llm
                )

                # 3. Compression on top of multi-query
                compressor = LLMChainExtractor.from_llm(self.llm)
                self.retrieval = ContextualCompressionRetriever(
                    base_compressor=compressor,
                    base_retriever=multi_query_retriever
                )
                logger.info('Using combined self-query + multi-query + compression retrieval.')
```

**What this does:**  
- Builds the full pipeline: filter by metadata, generate multiple queries, compress retrieved chunks.  
- Assigns the combined retriever to `self.retrieval`.

After adding this, your `RAGService` will handle `--all` correctly.

Let me know when you've inserted this branch, and I'll provide any final cleanup or test commands. Then we can run `python query.py --all` to see the results.


Here are **5 questions** to test the combined retrieval pipeline (`--all` or all three flags). They cover different document types and topics, so you can see how well the system filters, expands, and compresses.

### 1. Health-specific, with filter by source
> "What are the epidemic prone diseases in Nigeria according to the health PDF?"

**Expected behavior:** Self-query should filter `source` to `nigeria_health...pdf`; multi-query expands; compression trims to the disease list.

### 2. Agriculture-specific, with filter by doc_type
> "What are the symptoms and control of maize smut in the HTML document?"

**Expected:** Filter `doc_type` = `html`; retrieve only from `agriculture.html`; answer focuses on maize smut.

### 3. Mixed / broad question
> "What are the major diseases affecting crops and livestock in Nigeria, and how are they managed?"

**Expected:** Self-query may not filter narrowly; multi-query generates variations for both crops and livestock; compression extracts relevant parts from multiple sources (PDF, HTML, TXT).

### 4. E-commerce data query (if you still have ecommerce chunks)
> "What product has the comment 'Spacious and strong'? Give its name, price, category, stock, and rating."

**Expected:** Filter may target `doc_type` = `csv` or source `ecommerce.csv`; retrieve the Bag product info.

### 5. Public health statistics
> "What are the top causes of death in Nigeria and their percentages?"

**Expected:** Health PDF chunks; multi-query retrieves the list with percentages; compression keeps only the relevant statistics.

Run with:

```bash
python query.py --all
```

or

```bash
python query.py --self-query --multi-query --compression
```

Then enter each question. This will show how well the combined pipeline handles different scenarios. Let me know the results.


## Step 1: Add Concept Markdown (Small Font)

Copy and paste this into the first cell of `06_parent_document_retriever.ipynb`:

```html
<div style="font-size: 0.85em; line-height: 1.5;">

<h3>Parent Document Retriever</h3>

<p><strong>What is it?</strong><br>
A retriever that stores small chunks for search but returns larger parent chunks when answering. This gives the LLM more context and helps for broad questions.</p>

<p><strong>How it works</strong></p>
<ol>
  <li>Split documents into large parent chunks.</li>
  <li>Further split each parent into small child chunks.</li>
  <li>Embed and store child chunks in the vector store.</li>
  <li>When querying, retrieve child chunks, then return the parent chunk(s) instead.</li>
  <li>Pass the parent chunk(s) as context to the LLM.</li>
</ol>

<p><strong>Why use it?</strong></p>
<ul>
  <li><strong>Better context</strong> – small chunks for search, large chunks for answer.</li>
  <li><strong>Improves recall</strong> – captures full answer even if only part matched.</li>
  <li><strong>Great for long documents</strong> – PDFs, articles, manuals.</li>
</ul>

<p><strong>Visualisation</strong></p>
<pre>
Indexing:
[Large parent chunks] → split into small child chunks → embed child chunks

Retrieval:
Query → find similar child chunk(s)
         ↓
Return corresponding parent chunk(s)
         ↓
LLM uses parent chunk(s) as context
</pre>

<p><strong>Implementation</strong><br>
We will use LangChain’s <code>ParentDocumentRetriever</code> with a vector store and a document splitter.</p>

</div>
```

---

## Step 2: First Code Cell – Imports

Add a code cell and run it:

```python
# Import required libraries for parent document retriever
from langchain.retrievers import ParentDocumentRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, BSHTMLLoader, TextLoader
from dotenv import load_dotenv

load_dotenv()

print('✅ Imports ready.')
```

**What this does:**  
- Imports `ParentDocumentRetriever` and text splitters.  
- Imports embeddings, vector store, and document loaders.  
- Loads environment variables.

Run this cell and share output. Then we'll load documents and prepare parent/child chunks.


Your loop is well structured, but there is a small **bug** in the HTML loader arguments.

### Error in this line:

```python
loader = BSHTMLLoader(path, open_encoding='utf-8', bs_kwargs={'features': 'html_parser'})
```

The `features` value should be **`'html.parser'`**, not `'html_parser'`.  
If you use `'html_parser'`, BeautifulSoup will raise an error like:

```
FeatureNotFound: Couldn't find a tree builder with the features you requested: html_parser.
```

---

## Corrected Code

Replace the loop cell with this:

```python
files = [
    ('../../04_data_ingestion_document_processing/data/nigeria_health_diseases_and_prevention.pdf', 'pdf'),
    ('../../04_data_ingestion_document_processing/data/crop_disease.pdf', 'pdf'),
    ('../../04_data_ingestion_document_processing/data/agriculture.html', 'html'),
    ('../../04_data_ingestion_document_processing/data/agriculture.txt', 'txt'),
]

all_docs = []

for path, file_type in files:
    if file_type == 'pdf':
        loader = PyPDFLoader(path)
    elif file_type == 'html':
        loader = BSHTMLLoader(
            path,
            open_encoding='utf-8',
            bs_kwargs={'features': 'html.parser'}   # fixed: dot, not underscore
        )
    elif file_type == 'txt':
        loader = TextLoader(path, encoding='utf-8')
    else:
        continue
    all_docs.extend(loader.load())

print(f'Loaded {len(all_docs)} documents.')
```

Now run the corrected cell. It should load all four documents without errors. Share the output, and we'll continue with parent and child splitters.



Below is a **line‑by‑line explanation** of the corrected Step 5 code for `ParentDocumentRetriever`.

```python
# Import the InMemoryByteStore class from LangChain's storage module.
# It provides a simple in‑memory key‑value store to hold parent documents.
from langchain.storage import InMemoryByteStore

# Create an embedding model instance using OpenAI embeddings.
# This will convert text into numerical vectors for similarity search.
embeddings = OpenAIEmbeddings()

# Create an in‑memory byte store.
# This will store the larger parent chunks (or their identifiers) so we can
# retrieve them later when a matching child chunk is found.
byte_store = InMemoryByteStore()

# Create an in‑memory Chroma vector store.
# It will store the small child chunks and their embeddings.
vectorstore = Chroma(
    collection_name="parent_doc_demo",       # give this collection a unique name
    embedding_function=embeddings,           # use the embeddings we created
    persist_directory=None                   # keep data in memory (no disk save)
)

# Instantiate the ParentDocumentRetriever.
# This combines the vector store (child chunks) and byte store (parent chunks).
parent_retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,                 # child chunks are searched here
    byte_store=byte_store,                   # parent chunks are stored here
    child_splitter=child_splitter,           # how to split documents into small child chunks
    parent_splitter=parent_splitter          # how to split documents into large parent chunks
)

# Add all loaded documents to the retriever.
# Internally, the retriever will:
#   1. Split each document into parent chunks using parent_splitter.
#   2. For each parent chunk, split it into child chunks using child_splitter.
#   3. Embed and store child chunks in the vectorstore.
#   4. Store the parent chunks in the byte_store keyed by child IDs.
parent_retriever.add_documents(all_docs, ids=None)

# Print a success message to confirm setup.
print('✅ ParentDocumentRetriever ready.')
```

### In short

- `embeddings` = turns text into vectors.
- `byte_store` = stores the big parent chunks.
- `vectorstore` = stores the small child chunks (searchable).
- `parent_retriever` = ties them together: search child chunks, return parent chunks.
- `add_documents` = populates both stores with your documents.

After running this cell, you can query the retriever and it will return larger parent chunks, giving more context than the small child chunks would. This helps answer broad questions more completely.

Let me know when you're ready to test the retrieval.



## Step 6: Test ParentDocumentRetriever with a Broad Question

### Concept

Now we’ll test the retriever with a question that previously returned “I don’t know” because the matching child chunk was too small or didn’t contain enough information. The parent retriever should return a larger parent chunk that provides the necessary context.

### Code to run

```python
# Define a broad question
query = 'What are the major diseases affecting crops and livestock in Nigeria, and how are they managed?'

# Retrieve documents using the parent retriever
docs = parent_retriever.invoke(query)

print(f'Retrieved {len(docs)} parent chunk(s):\n')
for i, doc in enumerate(docs, start=1):
    print(f'Chunk {i} (length={len(doc.page_content)} characters):')
    print(doc.page_content)
    print('-' * 60)
```

**What to observe:**  
- The returned `docs` should be larger than the child chunks used for matching.  
- The parent chunks should contain more complete information, possibly listing multiple diseases for both crops and livestock.  
- You may get one or more parent chunks, depending on how many child matches occur.

---

Run this cell and share the output. If the parent retriever returns useful context, we can then integrate it into the modular `RAGService` as another retrieval mode.



The parent retriever returned **general agriculture sections** rather than the specific disease lists. This can happen when the child chunks matching the query are too generic or when the question is very broad. We can improve by asking a more focused question and/or adjusting chunk sizes.

Let’s test with a more specific question that should match disease‑related child chunks.

---

## Step 6 (Alternative): Test with a Focused Question

Run this cell:

```python
# More focused question about crop diseases only
query = 'What are the common crop diseases and their control methods?'

# Retrieve parent chunks
docs = parent_retriever.invoke(query)

print(f'Retrieved {len(docs)} parent chunk(s):\n')
for i, doc in enumerate(docs, start=1):
    print(f'Chunk {i} (length={len(doc.page_content)} characters):')
    print(doc.page_content)
    print('-' * 60)
```

**Expected improvement:**  
The parent chunks should now contain sections with lists of crop diseases (e.g., Cassava Mosaic, Maize Smut, Rice Blast) and their control measures. The context should be more relevant than before.

Run and share the output. If the results are still too broad, we can adjust the child splitter (e.g., increase child chunk size) or the number of retrieved child chunks (`search_k`). We'll also consider integrating this into the modular service later.



We’ll now cover **Fusion / Reciprocal Rank Fusion (RRF)**—a technique that combines results from multiple retrievers to improve recall, especially for broad or ambiguous queries.

## Fusion (Reciprocal Rank Fusion) – Concept

### What is it?

Reciprocal Rank Fusion (RRF) is a method that **combines search results from multiple retrievers** into one ranked list. It is simple but effective:

- Each retriever (e.g., multi‑query, self‑query, parent‑document) returns its own ranked list of documents.
- For each document, we calculate a score based on its rank in each list: `score = 1 / (k + rank)`, where `k` is a constant (often 60).
- Scores from all retrievers are summed for each document.
- The final documents are sorted by total score (descending).

This often surfaces documents that are not top in any single retriever but appear in several lists, improving recall.

### Visualisation

```
Retriever A → [doc1, doc3, doc5]
Retriever B → [doc2, doc3, doc4]
Retriever C → [doc1, doc2, doc3]
        │
        ▼
Combine using RRF:
  doc1: score from A (rank1) + score from C (rank1)
  doc2: score from B (rank1) + score from C (rank2)
  doc3: score from A (rank2) + score from B (rank2) + score from C (rank3)
  ...
        │
        ▼
Final ranked list of unique documents
```

### Why use it?

- **Improves recall** – a document missed by one retriever may be found by another.
- **No training needed** – simple mathematical fusion.
- **Works well with diverse retrievers** (different strengths).

### Implementation

We will implement RRF manually in Python, using retrievers you already built:

- Base retriever
- MultiQueryRetriever
- SelfQueryRetriever
- ParentDocumentRetriever

We can combine any number of them.

---

## Markdown to Copy (smaller font)

```html
<div style="font-size: 0.85em; line-height: 1.5;">

<h3>Fusion – Reciprocal Rank Fusion (RRF)</h3>

<p><strong>What is it?</strong><br>
A method to combine results from multiple retrievers into one ranked list using the formula: <code>score = 1 / (k + rank)</code>, where <code>k</code> is a constant (often 60). The final documents are sorted by the sum of scores from all retrievers.</p>

<p><strong>How it works</strong></p>
<ol>
  <li>Run several retrievers on the same query.</li>
  <li>For each retriever, record the rank of every returned document.</li>
  <li>Calculate RRF score for each document across all retrievers.</li>
  <li>Sort documents by total score (descending).</li>
  <li>Return the top documents for answer generation.</li>
</ol>

<p><strong>Why use it?</strong></p>
<ul>
  <li><strong>Improves recall</strong> – captures documents missed by single retrievers.</li>
  <li><strong>Simple and effective</strong> – no training required.</li>
  <li><strong>Works with diverse retrievers</strong> – combines different strengths.</li>
</ul>

<p><strong>Visualisation</strong></p>
<pre>
Retriever A → [doc1, doc3]
Retriever B → [doc2, doc3]
        │
        ▼
RRF combination:
  doc1: score from A
  doc2: score from B
  doc3: score from A + B
        │
        ▼
Final ranked list
</pre>

<p><strong>Implementation</strong><br>
We will write a simple Python function to compute RRF scores from multiple retrievers.</p>

</div>
```

---

## Create Notebook

Create a new notebook in `07_advanced_retrieval/notebooks/`:

```
07_fusion_rrf.ipynb
```

Select the `rag_env` kernel.

## Step 1: Imports

Add and run this cell:

```python
# Import necessary libraries
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo
from langchain_community.document_loaders import PyPDFLoader, BSHTMLLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

load_dotenv()
print('✅ Imports ready.')
```

Run and share output. Then we'll build the base retrievers and implement the RRF function.



We’ll now prepare the **base retriever** and the vector store from the clean documents. This is needed before we can combine multiple retrievers.

## Step 2: Load Documents, Create Chunks, and Base Retriever

### Explanation

We will:

- Load the same health/agriculture documents we used earlier.
- Split them into chunks and add metadata (source, doc_type, language).
- Create an in‑memory Chroma vector store.
- Create a base retriever that returns the top‑k chunks.

This base retriever will be the foundation for the other retrievers we’ll use in fusion.

### Code cell

```python
# Load relevant documents (same as previous notebooks)
files = [
    ('../../04_data_ingestion_document_processing/data/nigeria_health_diseases_and_prevention.pdf', 'pdf'),
    ('../../04_data_ingestion_document_processing/data/crop_disease.pdf', 'pdf'),
    ('../../04_data_ingestion_document_processing/data/agriculture.html', 'html'),
    ('../../04_data_ingestion_document_processing/data/agriculture.txt', 'txt'),
]

all_docs = []
for path, file_type in files:
    if file_type == 'pdf':
        loader = PyPDFLoader(path)
    elif file_type == 'html':
        loader = BSHTMLLoader(path, open_encoding='utf-8', bs_kwargs={'features': 'html.parser'})
    elif file_type == 'txt':
        loader = TextLoader(path, encoding='utf-8')
    else:
        continue
    all_docs.extend(loader.load())

# Split into chunks and add metadata
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

print(f'Created {len(chunks)} chunks.')

# Create embeddings and vectorstore (in-memory)
embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=None
)

# Create base retriever
base_retriever = vectorstore.as_retriever(search_kwargs={'k': 4})

print('✅ Base retriever ready.')
```

Run this cell and share output. Then we’ll create the other retrievers (multi‑query, self‑query, etc.) for fusion.


No, `base_retriever = vectorstore.as_retriever(search_kwargs={'k': 4})` does **not** retrieve documents with equal similarity. It retrieves the **top‑4 most similar** chunks to the query, ranked by similarity (distance).

### What `search_kwargs={'k': 4}` means

- `k` is the number of results to return.
- The vector store calculates the similarity (or distance) between the query vector and every stored document vector.
- Then it returns the **4 documents with the highest similarity** (or smallest distance) to the query, not documents that are equally similar.

So it’s a **top‑k retrieval**, where `k=4` asks for the 4 most relevant chunks. The similarity scores are not equal; the retriever returns the best matches in order.

If you want to see the similarity scores, you can use:

```python
results = vectorstore.similarity_search_with_score(query, k=4)
```

This will show you the actual scores (lower is better for Chroma's default distance). That will confirm that the retrieved documents have different similarity values, not equal.

---

Now, let's continue with fusion. We'll create other retrievers and then combine them using Reciprocal Rank Fusion. Would you like to proceed?


We'll now create the additional retrievers that will be used in fusion. We'll build three different retrievers:

1. **MultiQueryRetriever** – generates multiple query variations and merges results.
2. **SelfQueryRetriever** – automatically infers metadata filters from the question.
3. **ContextualCompressionRetriever** – compresses retrieved chunks to only relevant parts.

These retrievers have different strengths, so fusing them can improve overall retrieval.

## Step 3: Create LLM and Additional Retrievers

### Explanation

We need a shared LLM for the retrievers that require one. We'll use `gpt-4o-mini` with `temperature=0` for consistency.

Then:

- **MultiQueryRetriever** is built from the base retriever.
- **SelfQueryRetriever** requires metadata field info and document content description.
- **ContextualCompressionRetriever** wraps the base retriever with a compressor.

We'll create all three and store them in a list or dictionary for later fusion.

### Code cell

```python
# Create a shared LLM
llm = ChatOpenAI(model='gpt-4o-mini', temperature=0)

# --- MultiQueryRetriever ---
multi_query_retriever = MultiQueryRetriever.from_llm(
    retriever=base_retriever,
    llm=llm
)
print('✅ MultiQueryRetriever ready.')

# --- SelfQueryRetriever ---
metadata_field_info = [
    AttributeInfo(
        name='source',
        description='The file path of the document. Contains keywords like "nigeria_health" or "crop_disease" or "agriculture".',
        type='string',
    ),
    AttributeInfo(
        name='doc_type',
        description='The type of document: pdf, html, txt',
        type='string',
    ),
    AttributeInfo(
        name='language',
        description='Language of the document, e.g., English',
        type='string',
    ),
]

document_content_description = 'Documents about agriculture, public health, and diseases in Nigeria'

self_query_retriever = SelfQueryRetriever.from_llm(
    llm=llm,
    vectorstore=vectorstore,
    document_contents=document_content_description,
    metadata_field_info=metadata_field_info,
    verbose=False
)
print('✅ SelfQueryRetriever ready.')

# --- ContextualCompressionRetriever ---
compressor = LLMChainExtractor.from_llm(llm)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=base_retriever
)
print('✅ ContextualCompressionRetriever ready.')
```

Run this cell and share output. Once all retrievers are ready, we'll implement the Reciprocal Rank Fusion function.



Now we'll implement the **Reciprocal Rank Fusion (RRF)** function and test it.

## Step 4: Implement Reciprocal Rank Fusion

### Explanation

The RRF formula is simple. For each document in the ranked list from a retriever, we compute:

```
score = 1 / (k + rank)
```

where:
- `rank` starts at 1 for the top result.
- `k` is a constant (often 60) to reduce the impact of very high ranks.

We sum these scores across all retrievers for each document. Then we sort documents by total score descending and return the top `top_n` unique documents.

### Code cell

```python
def reciprocal_rank_fusion(retriever_results, k=60, top_n=5):
    """
    Combine multiple ranked lists of documents using Reciprocal Rank Fusion.

    Args:
        retriever_results: A list of lists, where each inner list is a ranked
                           list of Document objects from one retriever.
        k: constant to control rank influence (default 60).
        top_n: number of top documents to return after fusion.

    Returns:
        List of Document objects sorted by fused scores (descending).
    """
    # Dictionary to store cumulative RRF score for each unique document
    fused_scores = {}
    # Dictionary to keep the actual Document object for each unique content
    doc_map = {}

    # Iterate over each retriever's ranked list
    for ranked_list in retriever_results:
        # Iterate over documents and their ranks (starting at 1)
        for rank, doc in enumerate(ranked_list, start=1):
            # Use the page_content as the unique key (or metadata id if available)
            key = doc.page_content.strip()

            # Compute RRF score for this rank
            score = 1 / (k + rank)

            # Sum the score
            if key in fused_scores:
                fused_scores[key] += score
            else:
                fused_scores[key] = score
                doc_map[key] = doc

    # Sort keys by total score descending
    sorted_keys = sorted(fused_scores, key=fused_scores.get, reverse=True)

    # Return the top_n documents
    return [doc_map[key] for key in sorted_keys[:top_n]]
```

This function is generic and reusable. It will accept a list of retrieval result lists.

Run this cell to define the function. Then we'll use it in the next step.


Yes, LangChain provides several **advanced retrievers** and techniques that can improve handling of broad or ambiguous questions. Here are some notable ones:

### 1. `MultiQueryRetriever` ✅ (already used)
Generates multiple rephrased versions of the user query and retrieves for each. Good for broad questions because it covers different angles.

### 2. **HyDE** (Hypothetical Document Embeddings)
HyDE uses an LLM to **generate a hypothetical answer** to the question, then embeds that answer and uses it as the search query. This can dramatically improve retrieval for broad or open‑ended questions, because the hypothetical answer contains more specific terms that match document content.

### 3. `SelfQueryRetriever` ✅ (already used)
Automatically infers metadata filters from the question. Useful when the query mentions a document type, source, or language.

### 4. `ParentDocumentRetriever` ✅ (already used)
Returns larger parent chunks for more context, which helps when the answer spans multiple sentences.

### 5. `EnsembleRetriever`
Combines multiple retrievers and merges their results. It supports **weighted fusion** or plain concatenation. This is similar to our manual RRF but built‑in.

### 6. `ReciprocalRankFusionRetriever`
Built‑in version of RRF (but not available in your current version). It merges retrievers automatically.

### 7. `MultiVectorRetriever`
Stores multiple vectors per document (e.g., summaries + chunks) to improve retrieval. Good for long documents where a single chunk may not represent the whole document.

### 8. `TimeWeightedVectorStoreRetriever`
Combines semantic similarity with a **recency factor**. Useful when recent documents should be prioritised.

### 9. `ContextualCompressionRetriever` ✅ (already used)
Compresses retrieved chunks to only the most relevant parts, which helps prevent information overload for broad questions.

---

### Recommendation for Broad Questions

For broad queries like *“major diseases affecting crops and livestock”*, the best combination is:

- **HyDE** – to generate a rich hypothetical answer and search with it.
- **MultiQueryRetriever** – to cover multiple aspects.
- **Fusion** (EnsembleRetriever or RRF) – to combine keyword and semantic results.
- **ParentDocumentRetriever** – to return enough context once relevant chunks are found.

**HyDE** is particularly powerful because it expands a vague query into a detailed pseudo‑document, increasing the chance of matching relevant chunks. We haven’t covered HyDE yet. Would you like to learn and implement **HyDE** next? It’s a natural progression after fusion and would likely improve your broad question results.

The error indicates that `langchain.retrievers.hyde` module is not available in your installed version. However, `HypotheticalDocumentEmbedder` from `langchain.chains` **is** likely available. We'll use that built‑in component to generate a hypothetical embedding, and then perform retrieval using the vector store’s built‑in `similarity_search_by_vector` method.

That still uses LangChain’s built‑in chain, with only a small manual retrieval call.

---

## Step 1 (Fixed): Imports

Replace the previous import cell with this:

```python
# Import built-in HypotheticalDocumentEmbedder
from langchain.chains import HypotheticalDocumentEmbedder

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, BSHTMLLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

load_dotenv()

print('✅ Imports ready (using built-in HypotheticalDocumentEmbedder).')
```

Run this cell. It should import successfully. If `HypotheticalDocumentEmbedder` also fails, please share the error.

---

## Step 2: Load Documents and Create Chunks

```python
# Load relevant documents
files = [
    ('../../04_data_ingestion_document_processing/data/nigeria_health_diseases_and_prevention.pdf', 'pdf'),
    ('../../04_data_ingestion_document_processing/data/crop_disease.pdf', 'pdf'),
    ('../../04_data_ingestion_document_processing/data/agriculture.html', 'html'),
    ('../../04_data_ingestion_document_processing/data/agriculture.txt', 'txt'),
]

all_docs = []
for path, ftype in files:
    if ftype == 'pdf':
        loader = PyPDFLoader(path)
    elif ftype == 'html':
        loader = BSHTMLLoader(path, open_encoding='utf-8', bs_kwargs={'features': 'html.parser'})
    elif ftype == 'txt':
        loader = TextLoader(path, encoding='utf-8')
    else:
        continue
    all_docs.extend(loader.load())

# Split and add metadata
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

print(f'Loaded {len(all_docs)} docs, created {len(chunks)} chunks.')
```

---

## Step 3: Create Vector Store

```python
embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=None
)

print('✅ Vector store ready.')
```

---

## Step 4: Create the HypotheticalDocumentEmbedder

We'll use the built‑in chain. It will generate a hypothetical answer and embed it.

```python
# LLM for generating hypothetical answer
llm = ChatOpenAI(model='gpt-4o-mini', temperature=0)

# Built-in HypotheticalDocumentEmbedder
hyde_embedder = HypotheticalDocumentEmbedder.from_llm(
    llm=llm,
    base_embeddings=embeddings,
    prompt_key="web_search"   # or use default 'web_search' prompt style
)

print('✅ HypotheticalDocumentEmbedder ready.')
```

**What this does:**  
- `HypotheticalDocumentEmbedder` uses the LLM to generate a hypothetical answer to the query.  
- It then embeds that answer using the provided embeddings.  
- The result is a vector that can be used for similarity search.

---

## Step 5: Use HyDE to Retrieve Documents

We'll generate the hypothetical embedding for a broad question, then search the vector store.

```python
# Broad question that previously failed
query = 'What are the major diseases affecting crops and livestock in Nigeria, and how are they managed?'

# Generate hypothetical embedding
hyde_embedding = hyde_embedder.embed_query(query)

# Retrieve documents using the hypothetical embedding
docs = vectorstore.similarity_search_by_vector(hyde_embedding, k=5)

print(f'Retrieved {len(docs)} documents using HyDE:\n')
for i, doc in enumerate(docs, start=1):
    print(f'{i}. {doc.page_content[:250]}')
    print(f'   Source: {doc.metadata.get("source", "unknown")}')
    print('-' * 60)
```

Run these cells and share the output. HyDE should retrieve more focused disease-related chunks for the broad question.

If `HypotheticalDocumentEmbedder` gives any errors, let me know. Otherwise, we can later integrate this into `RAGService`.



We'll now integrate **query decomposition** and **HyDE** into the modular `RAGService`. This will let you run commands like:

```bash
python query.py --hyde
python query.py --decompose
python query.py --hyde --decompose
```

We'll do it step by step.

---

## Step 1: Add imports to `RAGService`

Open `app/services/rag_service.py` and add these imports at the top:

```python
from langchain.chains import HypotheticalDocumentEmbedder
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
```

**What these are for:**
- `HypotheticalDocumentEmbedder` – for HyDE.
- `ChatPromptTemplate` and `StrOutputParser` – for query decomposition.

Let me know when you've added these imports, and I'll give Step 2 (adding parameters and storing flags).



## Step 2: Add parameters and store flags

In `app/services/rag_service.py`, find the `__init__` method signature. Add two new parameters at the end:

```python
use_hyde: bool = False,
use_decomposition: bool = False
```

So the signature becomes:

```python
def __init__(
    self,
    chroma_storage_dir: str | None = None,
    embeddings: OpenAIEmbeddings | None = None,
    llm: ChatOpenAI | None = None,
    search_k: int | None = None,
    use_multi_query: bool = False,
    use_compression: bool = False,
    use_self_query: bool = False,
    use_hyde: bool = False,                     # NEW
    use_decomposition: bool = False             # NEW
):
```

Inside the `__init__` body, near the top where other flags are stored (after `self.use_self_query = use_self_query`), add:

```python
self.use_hyde = use_hyde
self.use_decomposition = use_decomposition
```

**What this does:**  
- Adds two new options to enable HyDE and query decomposition.

Let me know when you've added these parameters and stored them, and I'll give Step 3 (adding helper methods for HyDE retrieval and query decomposition).



Good catch! We haven't yet created `self.hyde_embedder` and `self.decompose_chain` inside `__init__`. We need to add them so the helper methods have access.

## Step 3.5: Create HyDE Embedder and Decompose Chain in `__init__`

Inside `__init__`, after the LLM is created (`self.llm = ...`), add this block:

```python
            # Create HyDE embedder (used if use_hyde or use_decomposition with hyde)
            self.hyde_embedder = HypotheticalDocumentEmbedder.from_llm(
                llm=self.llm,
                base_embeddings=self.embeddings,
                prompt_key="web_search"
            )

            # Create decompose chain (used if use_decomposition)
            decompose_prompt = ChatPromptTemplate.from_messages([
                ('system', 'Break the user question into 2-4 simpler sub-questions. Output each sub-question on a new line. Do not answer them.'),
                ('human', '{question}')
            ])
            self.decompose_chain = decompose_prompt | self.llm | StrOutputParser()
```

**What this does:**
- `self.hyde_embedder` – built-in LangChain `HypotheticalDocumentEmbedder` for generating hypothetical embeddings.
- `self.decompose_chain` – a simple chain that decomposes compound questions into sub-questions.

Now `_hyde_retrieve` and `_decompose_and_retrieve` can use these attributes.

---

After adding this, the service will be complete. Then we'll update `query.py` with the new flags and test. Let me know when you've inserted this block.


We'll now add the helper methods for **HyDE** and **query decomposition** inside `RAGService`.

## Step 3: Add Helper Methods

Add these methods after `_format_docs` (or before `answer`). They will be used only when the corresponding flags are enabled.

```python
    def _hyde_retrieve(self, question: str, k: int | None = None) -> List[Document]:
        """Use HyDE to retrieve documents.

        Generates a hypothetical answer, embeds it, and uses that embedding
        to search the vector store directly.

        Args:
            question: The user query.
            k: Number of documents to retrieve. Defaults to self.search_k.

        Returns:
            List of retrieved Document objects.
        """
        k = k or self.search_k

        # Generate a hypothetical embedding for the question
        hyde_embedding = self.hyde_embedder.embed_query(question)

        # Search the vector store using that embedding
        docs = self.vectorstore.similarity_search_by_vector(hyde_embedding, k=k)
        logger.debug(f'HyDE retrieval returned {len(docs)} documents.')
        return docs

    def _decompose_and_retrieve(self, question: str, k: int | None = None) -> List[Document]:
        """Break a compound question into sub-questions, retrieve for each, and combine.

        Args:
            question: The user query (may contain multiple parts).
            k: Number of documents to retrieve per sub-question.

        Returns:
            List of unique Document objects from all sub-queries.
        """
        k = k or self.search_k

        # Decompose the question into sub-questions
        sub_questions_text = self.decompose_chain.invoke({'question': question})
        sub_questions = [q.strip() for q in sub_questions_text.split('\n') if q.strip()]
        logger.info(f'Decomposed into {len(sub_questions)} sub-questions: {sub_questions}')

        # Retrieve for each sub-question
        all_docs = []
        seen = set()
        for sub_q in sub_questions:
            if self.use_hyde:
                # Use HyDE retrieval for each sub-question
                docs = self._hyde_retrieve(sub_q, k)
            else:
                # Use the normal retrieval component
                docs = self.retrieval.invoke(sub_q)
            for doc in docs:
                key = doc.page_content.strip()
                if key not in seen:
                    seen.add(key)
                    all_docs.append(doc)

        logger.debug(f'Decomposition combined {len(all_docs)} unique documents.')
        return all_docs
```

**What this does:**
- `_hyde_retrieve` generates a hypothetical embedding and searches the vector store.
- `_decompose_and_retrieve` splits the question, retrieves for each part (using HyDE if enabled), and merges unique documents.

Now I'll provide Step 4 (update `answer` and `retrieve_sources` to use these helpers). Let me know when you've added these methods.



Now we'll update `query.py` to accept the new flags `--hyde` and `--decompose`, and pass them to `RAGService`.

## Step 5: Update `query.py`

Replace your current `query.py` with this version:

```python
"""
Query script for the RAG system.

Usage:
    python query.py                            # simple retrieval
    python query.py --multi-query              # multi-query retrieval
    python query.py --compression              # contextual compression
    python query.py --self-query               # self-query retrieval
    python query.py --hyde                     # HyDE retrieval
    python query.py --decompose                # query decomposition
    python query.py --all                      # enable all techniques
"""

import argparse
from dotenv import load_dotenv
from app.services.rag_service import RAGService
from app.logger.custom_logger import setup_logging, get_logger

load_dotenv()
setup_logging(log_file='logs/query.log')
logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--multi-query', action='store_true', help='Use multi-query retrieval')
    parser.add_argument('--compression', action='store_true', help='Use contextual compression')
    parser.add_argument('--self-query', action='store_true', help='Use self-query retrieval')
    parser.add_argument('--hyde', action='store_true', help='Use HyDE retrieval')
    parser.add_argument('--decompose', action='store_true', help='Use query decomposition')
    parser.add_argument('--all', action='store_true', help='Enable all retrieval techniques')
    args = parser.parse_args()

    # Determine flags based on --all or individual arguments
    if args.all:
        use_multi_query = True
        use_compression = True
        use_self_query = True
        use_hyde = True
        use_decomposition = True
    else:
        use_multi_query = args.multi_query
        use_compression = args.compression
        use_self_query = args.self_query
        use_hyde = args.hyde
        use_decomposition = args.decompose

    logger.info(
        f'Starting RAG query (multi-query={use_multi_query}, '
        f'compression={use_compression}, self-query={use_self_query}, '
        f'hyde={use_hyde}, decompose={use_decomposition})...'
    )

    # Create RAG service with selected options
    rag = RAGService(
        use_multi_query=use_multi_query,
        use_compression=use_compression,
        use_self_query=use_self_query,
        use_hyde=use_hyde,
        use_decomposition=use_decomposition
    )

    question = input('Your question: ')
    answer = rag.answer(question)
    sources = rag.retrieve_sources(question)

    print(f'\nAnswer: {answer}')
    if sources:
        print('\nSources used:')
        for i, doc in enumerate(sources, start=1):
            print(f'\nSource {i}:')
            print(doc.page_content[:300])
            print(f'Metadata: {doc.metadata}')
    else:
        print('\nNo sources retrieved.')


if __name__ == '__main__':
    main()
```

Now run:

```bash
python query.py --decompose --hyde
```

and ask the compound question:

> List Common Crop Diseases and Control, also list top 10 causes of death in Nigeria

Let me know the output. This should now retrieve documents for both parts and produce a comprehensive answer. If you encounter any errors, share them and we'll fix.



Your RAG system has improved significantly with **query decomposition + HyDE**. Most of your test questions now produce accurate, well‑sourced answers, including compound and broad queries. For example:

- ✅ “List common crop diseases and control methods, and also list the top 10 causes of death” – now retrieves both topics and gives a combined answer with sources.
- ✅ “What are the major health problems and agricultural challenges?” – decomposes into sub‑questions and covers both domains.
- ✅ “What are the symptoms and control of Cassava Mosaic Disease?” – correct and focused.
- ✅ “What is Mastitis and how is it managed?” – correct with sources.
- ✅ Product lookup by comment – correct.
- ✅ “Leading causes of death and percentage of malaria” – correct with percentages.
- ✅ “Define hypertrophy and mention an example” – correct.

Remaining limitations:

- For a very broad query like *“Tell me about crop diseases and the top causes of death in Nigeria,”* the system often returns the causes of death but not the actual crop disease list, even after decomposition. This is because the retrieved documents for the crop‑disease sub‑question are generic sections rather than the specific list. This is a retrieval precision issue that can be improved with:
  - **Higher `search_k`** – retrieve more chunks per sub‑question.
  - **BM25 fusion** – combine keyword search with HyDE to capture exact disease names.
  - **Better chunking** – ensure each disease list is in its own chunk.

Recommended next steps:

1. **Commit your progress to GitHub** – your modular code, notebooks, and query script are now quite advanced.
2. **Move to Module 9: RAG Evaluation & Observability** – use LangSmith to measure retrieval quality and tune parameters systematically.
3. **Optionally, integrate BM25 fusion into `RAGService`** – add a `--fusion` flag to improve broad questions, using the `BM25Retriever` and the manual RRF function we built.

Which would you like to do? I recommend **committing progress** first, then **Module 9** to evaluate and refine further. Let me know your preference, and I’ll guide you through it.





