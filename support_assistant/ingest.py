import os
import glob
from sentence_transformers import SentenceTransformer
import chromadb

DOCS_DIR = "docs"
CHROMA_DIR = "chroma_db"

model = SentenceTransformer("all-MiniLM-L6-v2")

doc_paths = sorted(glob.glob(os.path.join(DOCS_DIR, "*.txt")))

chunks = []
chunk_ids = []
metadatas = []

for path in doc_paths:
    doc_id = os.path.splitext(os.path.basename(path))[0]
    with open(path, "r", encoding="utf-8") as f:
        text = f.read().strip()
    chunks.append(text)
    chunk_ids.append(doc_id)
    metadatas.append({"doc_id": doc_id, "source": path})

print(f"Loaded {len(chunks)} chunks from {len(doc_paths)} documents.")

print("Embedding chunks...")
embeddings = model.encode(chunks).tolist()

print("Storing in ChromaDB...")
client = chromadb.PersistentClient(path=CHROMA_DIR)
try:
    client.delete_collection("zepto_policies")
except Exception:
    pass
collection = client.create_collection("zepto_policies")

collection.add(
    ids=chunk_ids,
    embeddings=embeddings,
    documents=chunks,
    metadatas=metadatas,
)

print(f"Stored {collection.count()} chunks in ChromaDB collection 'zepto_policies'.")

if __name__ == "__main__":
    test_query = "How do I cancel my order?"
    query_emb = model.encode([test_query]).tolist()
    results = collection.query(query_embeddings=query_emb, n_results=3)
    print(f"\nSanity check query: '{test_query}'")
    print("Top matches:", results["ids"][0])
    print("Top chunk snippet:", results["documents"][0][0][:150])