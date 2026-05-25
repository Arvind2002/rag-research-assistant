import os
from groq import Groq
from dotenv import load_dotenv
import os
import streamlit as st

load_dotenv()

def get_secret(key):
    try:
        return st.secrets[key]
    except:
        return os.getenv(key)

client = Groq(api_key=get_secret("GROQ_API_KEY"))

def generate(query, chunks):
    # Build context from retrieved chunks
    context = ""
    for i, chunk in enumerate(chunks):
        context += f"[{i+1}] {chunk['title']}\n{chunk['text']}\n\n"

    prompt = f"""You are a research assistant. Answer the question using ONLY the context provided below.
If the context doesn't contain enough information, say so clearly.
Cite the paper titles when you use information from them.

CONTEXT:
{context}

QUESTION: {query}

ANSWER:"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )

    return response.choices[0].message.content