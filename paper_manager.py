import os
import arxiv
import hashlib
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
import time

load_dotenv()

def get_secret(key):
    try:
        import streamlit as st
        return st.secrets[key]
    except:
        return os.getenv(key)

pc = Pinecone(api_key=get_secret("PINECONE_API_KEY"))
index = pc.Index("arxiv-rag")
model = SentenceTransformer("all-MiniLM-L6-v2")

def chunk_text(text, chunk_size=500, overlap=50):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

def get_all_papers():
    """Fetch unique papers currently stored in Pinecone."""
    results = index.query(
        vector=[0.0] * 384,  # dummy vector to fetch any results
        top_k=100,
        include_metadata=True
    )

    seen = {}
    for match in results["matches"]:
        url = match["metadata"].get("url", "")
        if url not in seen:
            seen[url] = {
                "title": match["metadata"].get("title", "Unknown"),
                "url": url,
            }

    return list(seen.values())

def add_paper_by_query(query, max_results=3, retries=3):
    """Search arXiv and add matching papers to Pinecone with retry on rate limit."""
    
    for attempt in range(retries):
        try:
            client = arxiv.Client(delay_seconds=5)
            search = arxiv.Search(
                query=query,
                max_results=max_results,
                sort_by=arxiv.SortCriterion.Relevance
            )

            added = []
            for paper in client.results(search):
                full_text = f"{paper.title}\n\n{paper.summary}"
                chunks = chunk_text(full_text)
                vectors = []

                for i, chunk in enumerate(chunks):
                    embedding = model.encode(chunk).tolist()
                    chunk_id = hashlib.md5(f"{paper.entry_id}_{i}".encode()).hexdigest()
                    vectors.append({
                        "id": chunk_id,
                        "values": embedding,
                        "metadata": {
                            "title": paper.title,
                            "url": paper.entry_id,
                            "chunk_index": i,
                            "text": chunk
                        }
                    })

                for i in range(0, len(vectors), 100):
                    index.upsert(vectors=vectors[i:i+100])

                added.append({
                    "title": paper.title,
                    "url": paper.entry_id
                })

            return added

        except Exception as e:
            if "429" in str(e) and attempt < retries - 1:
                wait = 10 * (attempt + 1)  # 10s, 20s, 30s
                print(f"Rate limited. Waiting {wait}s before retry {attempt + 1}/{retries}...")
                time.sleep(wait)
                continue
            raise e

def add_paper_by_arxiv_id(arxiv_id, retries=3):
    """Add a specific paper by its arXiv ID with retry on rate limit."""
    
    for attempt in range(retries):
        try:
            client = arxiv.Client(delay_seconds=3)
            search = arxiv.Search(id_list=[arxiv_id])
            results = list(client.results(search))

            if not results:
                return None

            paper = results[0]
            full_text = f"{paper.title}\n\n{paper.summary}"
            chunks = chunk_text(full_text)
            vectors = []

            for i, chunk in enumerate(chunks):
                embedding = model.encode(chunk).tolist()
                chunk_id = hashlib.md5(f"{paper.entry_id}_{i}".encode()).hexdigest()
                vectors.append({
                    "id": chunk_id,
                    "values": embedding,
                    "metadata": {
                        "title": paper.title,
                        "url": paper.entry_id,
                        "chunk_index": i,
                        "text": chunk
                    }
                })

            for i in range(0, len(vectors), 100):
                index.upsert(vectors=vectors[i:i+100])

            return {"title": paper.title, "url": paper.entry_id}

        except Exception as e:
            if "429" in str(e) and attempt < retries - 1:
                wait = 5 * (attempt + 1)  # 5s, 10s, 15s
                time.sleep(wait)
                continue
            raise e