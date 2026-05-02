import arxiv
import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
import hashlib

load_dotenv()

# Init
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("arxiv-rag")
model = SentenceTransformer("all-MiniLM-L6-v2")  # 384-dim, fast, free

def chunk_text(text, chunk_size=500, overlap=50):
    """Split text into overlapping chunks by word count."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

def ingest_papers(query, max_results=10):
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance
    )

    vectors = []
    for paper in search.results():
        full_text = f"{paper.title}\n\n{paper.summary}"
        chunks = chunk_text(full_text)

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
        print(f"Chunked: {paper.title[:60]}...")

    # Upsert in batches of 100
    for i in range(0, len(vectors), 100):
        index.upsert(vectors=vectors[i:i+100])

    print(f"\n✅ Upserted {len(vectors)} chunks from {max_results} papers.")

if __name__ == "__main__":
    ingest_papers("large language models RAG retrieval augmented generation", max_results=15)