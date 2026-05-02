import os
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

load_dotenv()
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("arxiv-rag")
model = SentenceTransformer("all-MiniLM-L6-v2")

query = "How does RAG improve LLM factual accuracy?"
embedding = model.encode(query).tolist()

results = index.query(vector=embedding, top_k=3, include_metadata=True)
for match in results["matches"]:
    print(f"\nScore: {match['score']:.3f}")
    print(f"Title: {match['metadata']['title']}")
    print(f"Excerpt: {match['metadata']['text'][:200]}...")