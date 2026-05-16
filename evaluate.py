import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from langchain_openai import ChatOpenAI
import pandas as pd

from retriever import retrieve
from generator import generate

load_dotenv()

groq_llm = ChatOpenAI(
    model="llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)

embedder = SentenceTransformer("all-MiniLM-L6-v2")

test_questions = [
    "What are the main challenges in retrieval augmented generation?",
    "How does fine-tuning differ from RAG for improving LLM accuracy?",
    "What is the role of vector databases in RAG systems?",
    "How does chunking strategy affect RAG performance?",
    "What metrics are used to evaluate RAG systems?"
]

def score_faithfulness(answer, contexts):
    """Ask the LLM: is every claim in the answer supported by the context?"""
    context_str = "\n\n".join(contexts)
    prompt = f"""Given the following context and answer, score how faithful the answer is to the context.
Score from 0.0 to 1.0 where:
1.0 = every claim in the answer is directly supported by the context
0.5 = some claims are supported, some are not
0.0 = the answer contradicts or ignores the context entirely

Respond with ONLY a number between 0.0 and 1.0, nothing else.

CONTEXT:
{context_str}

ANSWER:
{answer}

SCORE:"""
    response = groq_llm.invoke(prompt)
    try:
        return float(response.content.strip())
    except:
        return 0.0

def score_relevancy(question, answer):
    """Ask the LLM: does the answer actually address the question?"""
    prompt = f"""Score how relevant this answer is to the question.
Score from 0.0 to 1.0 where:
1.0 = answer directly and completely addresses the question
0.5 = answer is somewhat related but incomplete
0.0 = answer does not address the question at all

Respond with ONLY a number between 0.0 and 1.0, nothing else.

QUESTION: {question}
ANSWER: {answer}

SCORE:"""
    response = groq_llm.invoke(prompt)
    try:
        return float(response.content.strip())
    except:
        return 0.0

def score_context_precision(question, contexts):
    """What fraction of retrieved chunks are actually relevant to the question?"""
    relevant = 0
    for ctx in contexts:
        prompt = f"""Is the following context chunk relevant to answering this question?
Answer with ONLY 'yes' or 'no'.

QUESTION: {question}
CONTEXT CHUNK: {ctx[:500]}

ANSWER:"""
        response = groq_llm.invoke(prompt)
        if "yes" in response.content.strip().lower():
            relevant += 1
    return relevant / len(contexts) if contexts else 0.0

def score_recall(question, answer, contexts):
    """Did the retrieved context contain enough info to produce a complete answer?"""
    context_str = "\n\n".join(contexts)
    prompt = f"""Given a question and the context that was retrieved, score how well the 
context contains the information needed to fully answer the question.
Score from 0.0 to 1.0 where:
1.0 = context contains all the information needed for a complete answer
0.5 = context contains some relevant info but is missing key details
0.0 = context is missing almost all information needed to answer the question

Respond with ONLY a number between 0.0 and 1.0, nothing else.

QUESTION: {question}
CONTEXT: {context_str}
ANSWER: {answer}

SCORE:"""
    response = groq_llm.invoke(prompt)
    try:
        return float(response.content.strip())
    except:
        return 0.0

def run_evaluation():
    rows = []

    for question in test_questions:
        print(f"\nEvaluating: {question[:60]}...")

        chunks = retrieve(question, top_k=5)
        answer = generate(question, chunks)
        contexts = [c["text"] for c in chunks]

        print("  Scoring faithfulness...")
        faith = score_faithfulness(answer, contexts)

        print("  Scoring relevancy...")
        relev = score_relevancy(question, answer)

        print("  Scoring context precision...")
        prec = score_context_precision(question, contexts)
        
        print("  Scoring context Recall...")
        rec = score_recall(question, answer, contexts)

        print(f"  ✅ faithfulness={faith:.2f} relevancy={relev:.2f} precision={prec:.2f} recall={rec:.2f}")

        rows.append({
            "question": question,
            "answer": answer,
            "faithfulness": faith,
            "answer_relevancy": relev,
            "context_precision": prec,
            "context_recall": rec
        })

    df = pd.DataFrame(rows)

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(df[["question", "faithfulness", "answer_relevancy", "context_precision", "context_recall"]].to_string(index=False))
    print("\nAggregate Scores:")
    print(f"  Faithfulness:      {df['faithfulness'].mean():.3f}")
    print(f"  Answer Relevancy:  {df['answer_relevancy'].mean():.3f}")
    print(f"  Context Precision: {df['context_precision'].mean():.3f}")
    print(f"  Context Recall: {df['context_recall'].mean():.3f}")
    print("=" * 60)

    df.to_csv("eval_results.csv", index=False)
    print("\n✅ Results saved to eval_results.csv")


if __name__ == "__main__":
    run_evaluation()