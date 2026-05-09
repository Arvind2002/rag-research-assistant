from retriever import retrieve
from generator import generate

def ask(query):
    print(f" Query: {query}")
    print("Retrieving relevant chunks...")
    
    chunks = retrieve(query, top_k=5)
    
    print(f"Found {len(chunks)} relevant chunks:")
    for i, chunk in enumerate(chunks):
        print(f"  [{i+1}] {chunk['title'][:60]}... (score: {chunk['score']:.3f})")
    
    print("\nGenerating answer...\n")
    answer = generate(query, chunks)
    
    print("=" * 60)
    print("ANSWER:")
    print("=" * 60)
    print(answer)
    print("=" * 60)
    
    return {"query": query, "chunks": chunks, "answer": answer}

if __name__ == "__main__":
    # Test with a few questions
    ask("What are the main challenges in retrieval augmented generation?")
    ask("How does fine-tuning differ from RAG for improving LLM accuracy?")