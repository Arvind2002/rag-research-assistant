# 🔬 arXiv Research Assistant

A Retrieval Augmented Generation (RAG) system that answers questions grounded 
in real arXiv research papers, built as part of a 7-month AI/ML job search plan.

**[Live Demo →](https://arxiv-paper-research.streamlit.app/)**

---

## What it does

Ask any question about AI/ML research and the app will:
1. Embed your question using a local sentence-transformers model
2. Retrieve the most semantically relevant chunks from a Pinecone vector store
3. Generate a grounded answer using Llama 3.1 (via Groq) citing the retrieved papers
4. Show you the source chunks and arXiv links so you can verify every claim

---

## Architecture

User Query
    │
    ▼
Embedding Model (all-MiniLM-L6-v2)
    │
    ▼
Pinecone Vector Store ──► Top-K Relevant Chunks
    │
    ▼
Llama 3.1 via Groq API
    │
    ▼
Grounded Answer + Sources

---

## Evaluation

The system is evaluated using a custom LLM-as-judge pipeline across 4 metrics:

| Metric | Score |
|---|---|
| Faithfulness | 0.91 |
| Answer Relevancy | 0.89 |
| Context Precision | 0.98 |
| Context Recall | 0.92 |

---

## Stack

| Component | Tool |
|---|---|
| Vector Store | Pinecone |
| Embeddings | all-MiniLM-L6-v2 (sentence-transformers) |
| LLM | Llama 3.1 8B via Groq |
| Paper Source | arXiv API |
| Evaluation | Custom LLM-as-judge |
| Frontend | Streamlit |

---

## Run locally

```bash
git clone https://github.com/arvind2002/rag-research-assistant
cd rag-research-assistant
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file:
```
PINECONE_API_KEY=your_key
GROQ_API_KEY=your_key
```

Ingest papers:
```bash
python ingest.py
```

Run the app:
```bash
streamlit run app.py
```

---

## Project structure
```
├── ingest.py        # arXiv ingestion + Pinecone upsert
├── retriever.py     # Semantic search against vector store
├── generator.py     # Groq/Llama generation with context
├── rag.py           # CLI interface
├── evaluate.py      # LLM-as-judge eval pipeline
├── app.py           # Streamlit frontend
└── eval_results.csv # Evaluation output
```
---

*Built by Arvind Ram*