import os
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
import os
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

def get_secret(key):
    try:
        return st.secrets[key]
    except:
        return os.getenv(key)

pc = Pinecone(api_key=get_secret("PINECONE_API_KEY"))
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