## 📓 Module 10b — GraphRAG

**Location:** `10b_graph_rag/code/notebooks/`

---

### 📘 Concept: Why GraphRAG?

Recall these failures from your earlier testing:

> **Q:** *"What do malaria and rice blast have in common?"*
> **A:** *"I don't know."*

> **Q:** *"How does agriculture affect health in Nigeria?"*
> **A:** *"I don't know."*

These are **bridge questions** — they ask about **relationships between two things**. Standard vector RAG fails on them because:

1. **Retrieval finds isolated chunks** about malaria OR rice blast — not both.
2. **No single chunk states the relationship** between them.
3. **The LLM has nothing to connect**, so it honestly says *"I don't know."*

### 🕸️ What GraphRAG changes

Instead of storing only vectors, we also build a **knowledge graph**:

```
        [Malaria]                      [Rice Blast]
           │                                │
           ├── caused_by ──→ [Plasmodium]   ├── caused_by ──→ [Fungus]
           │                                │
           ├── affects ──→ [Children]       ├── affects ──→ [Rice crops]
           │                                │
           └── controlled_by → [ACT drugs]  └── controlled_by → [Fungicides]
```

Now bridge questions can be answered by **traversing the graph**:

> *"What do malaria and rice blast have in common?"*
> → Both are **diseases**, both **have causes**, both **have treatments**.
> → The graph reveals the common structure.

### 🧩 How GraphRAG works — 4 steps

1. **Extract** entities and relationships from chunks using an LLM.
2. **Store** them as a graph (nodes = entities, edges = relationships).
3. **Query** the graph when the user asks a question — traverse to find connected facts.
4. **Combine** graph results + vector results for the final answer.

---

### 📂 Step 1: Create the folders

Run this in your terminal:

```bash
cd C:\Users\USER\rag_course
mkdir 10b_graph_rag\code\notebooks
mkdir 10b_graph_rag\data
```

You should end up with:

```
10b_graph_rag/
├── code/
│   └── notebooks/
└── data/
```

### 📓 Step 2: Create the notebook

In Jupyter:
- Navigate to `10b_graph_rag/code/notebooks/`
- New → Python 3 (ipykernel)
- Rename to `01_concept.ipynb`

---

### 🧩 Cell 1: Imports and setup

```python
import os
import logging
from pathlib import Path
from dotenv import load_dotenv

os.environ['ANONYMIZED_TELEMETRY'] = 'False'                 # Silence Chroma telemetry
logging.getLogger('httpx').setLevel(logging.WARNING)          # Silence OpenAI HTTP logs

import networkx as nx                                          # Graph library (in-memory)
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
print('✅ Imports ready')
print(f'   NetworkX version: {nx.__version__}')
```

**Expected output:**

```
✅ Imports ready
   NetworkX version: 3.x.x
```

### 🧠 What's new here

- **`networkx`** — pure‑Python graph library. No server, no setup. Perfect for learning GraphRAG before moving to a production graph DB like Neo4j.

**If `networkx` isn't installed**, run:

```bash
pip install networkx
```

Then **restart the kernel** and re‑run Cell 1.

---

Run it. When you see **`✅ Imports ready`** and the version, tell me **"next"** and I'll give you Cell 2 — a live demo showing exactly what standard RAG fails at.



## 📓 Module 10b — GraphRAG

**Location:** `10b_graph_rag/code/notebooks/`

---

### 📘 Concept: Why GraphRAG?

Recall these failures from your earlier testing:

> **Q:** *"What do malaria and rice blast have in common?"*
> **A:** *"I don't know."*

> **Q:** *"How does agriculture affect health in Nigeria?"*
> **A:** *"I don't know."*

These are **bridge questions** — they ask about **relationships between two things**. Standard vector RAG fails on them because:

1. **Retrieval finds isolated chunks** about malaria OR rice blast — not both.
2. **No single chunk states the relationship** between them.
3. **The LLM has nothing to connect**, so it honestly says *"I don't know."*

### 🕸️ What GraphRAG changes

Instead of storing only vectors, we also build a **knowledge graph**:

```
        [Malaria]                      [Rice Blast]
           │                                │
           ├── caused_by ──→ [Plasmodium]   ├── caused_by ──→ [Fungus]
           │                                │
           ├── affects ──→ [Children]       ├── affects ──→ [Rice crops]
           │                                │
           └── controlled_by → [ACT drugs]  └── controlled_by → [Fungicides]
```

Now bridge questions can be answered by **traversing the graph**:

> *"What do malaria and rice blast have in common?"*
> → Both are **diseases**, both **have causes**, both **have treatments**.
> → The graph reveals the common structure.

### 🧩 How GraphRAG works — 4 steps

1. **Extract** entities and relationships from chunks using an LLM.
2. **Store** them as a graph (nodes = entities, edges = relationships).
3. **Query** the graph when the user asks a question — traverse to find connected facts.
4. **Combine** graph results + vector results for the final answer.

---

### 📂 Step 1: Create the folders

Run this in your terminal:

```bash
cd C:\Users\USER\rag_course
mkdir 10b_graph_rag\code\notebooks
mkdir 10b_graph_rag\data
```

You should end up with:

```
10b_graph_rag/
├── code/
│   └── notebooks/
└── data/
```

### 📓 Step 2: Create the notebook

In Jupyter:
- Navigate to `10b_graph_rag/code/notebooks/`
- New → Python 3 (ipykernel)
- Rename to `01_concept.ipynb`

---

### 🧩 Cell 1: Imports and setup

```python
import os
import logging
from pathlib import Path
from dotenv import load_dotenv

os.environ['ANONYMIZED_TELEMETRY'] = 'False'                 # Silence Chroma telemetry
logging.getLogger('httpx').setLevel(logging.WARNING)          # Silence OpenAI HTTP logs

import networkx as nx                                          # Graph library (in-memory)
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
print('✅ Imports ready')
print(f'   NetworkX version: {nx.__version__}')
```

**Expected output:**

```
✅ Imports ready
   NetworkX version: 3.x.x
```

### 🧠 What's new here

- **`networkx`** — pure‑Python graph library. No server, no setup. Perfect for learning GraphRAG before moving to a production graph DB like Neo4j.

**If `networkx` isn't installed**, run:

```bash
pip install networkx
```

Then **restart the kernel** and re‑run Cell 1.

---

Run it. When you see **`✅ Imports ready`** and the version, tell me **"next"** and I'll give you Cell 2 — a live demo showing exactly what standard RAG fails at.



