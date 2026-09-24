import os
import chromadb
import ollama
import fitz  # pymupdf

DATA_DIR = "data"
CHUNK_SIZE = 500      # words per chunk
CHUNK_OVERLAP = 50

def load_pdf_text(filepath):
    doc = fitz.open(filepath)
    text = ""
    for page in doc:
        text += page.get_text() + "\n"
    doc.close()
    return text

def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

def main():
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_or_create_collection("university_docs")

    chunk_id = 0
    for filename in os.listdir(DATA_DIR):
        if not filename.endswith(".pdf"):
            continue
        filepath = os.path.join(DATA_DIR, filename)
        print(f"Processing {filename}...")

        text = load_pdf_text(filepath)
        chunks = chunk_text(text)

        for chunk in chunks:
            if not chunk.strip():
                continue
            embedding = ollama.embeddings(model="nomic-embed-text", prompt=chunk)["embedding"]
            collection.add(
                embeddings=[embedding],
                documents=[chunk],
                metadatas=[{"source": filename}],
                ids=[f"chunk_{chunk_id}"]
            )
            chunk_id += 1
            print(f"  Stored chunk {chunk_id}")

    print(f"Done. Stored {chunk_id} chunks total.")

if __name__ == "__main__":
    main()