import os
import chromadb
import ollama
from dotenv import load_dotenv

load_dotenv()

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection("university_docs")

def retrieve(question, top_k=3):
    query_embedding = ollama.embeddings(model="nomic-embed-text", prompt=question)["embedding"]
    results = collection.query(query_embeddings=[query_embedding], n_results=top_k)
    docs = results["documents"][0]
    sources = [m["source"] for m in results["metadatas"][0]]
    return docs, sources

def build_prompt(question, chunks):
    context = "\n\n---\n\n".join(chunks)
    return f"""Answer the question using only the context below. If the answer isn't in the context, say you don't know.

Context:
{context}

Question: {question}
Answer:"""

def generate_local(prompt):
    response = ollama.generate(model="llama3.2", prompt=prompt)
    return response["response"]

def generate_cloud(prompt):
    from groq import Groq
    groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    response = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

def answer_question(question, mode="local"):
    chunks, sources = retrieve(question)
    prompt = build_prompt(question, chunks)

    if mode == "local":
        answer = generate_local(prompt)
    else:
        answer = generate_cloud(prompt)

    return answer, sources

# Quick manual test - remove or comment out later
if __name__ == "__main__":
    print("Ask questions about your handbook. Type 'exit' to quit.\n")
    while True:
        q = input("Your question: ")
        if q.lower() == "exit":
            break
        answer, sources = answer_question(q, mode="local")
        print("\nAnswer:", answer)
        print("Sources:", sources)
        print("-" * 50)