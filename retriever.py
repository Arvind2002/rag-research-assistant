import os
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

load_dotenv()

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("arxiv-rag")
model = SentenceTransformer("all-MiniLM-L6-v2")

def retrieve(query, top_k=5):
    embedding = model.encode(query).tolist()
    results = index.query(vector=embedding, top_k=top_k, include_metadata=True)
    
    chunks = []
    for match in results["matches"]:
        chunks.append({
            "text": match["metadata"]["text"],
            "title": match["metadata"]["title"],
            "url": match["metadata"]["url"],
            "score": match["score"]
        })
    return chunks