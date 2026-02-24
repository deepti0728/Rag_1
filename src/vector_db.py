# src/vector_db.py
import chromadb
import os

# Folder where DB will be stored
persist_dir = "./vector_store"

# Make sure folder exists
os.makedirs(persist_dir, exist_ok=True)

# ✅ Use PersistentClient (THIS IS THE FIX)
client = chromadb.PersistentClient(path=persist_dir)


def create_vector_store(chunks, embeddings):
    docs = [str(c) for c in chunks]
    embs = embeddings.embed_documents(docs)

    ids = [str(i) for i in range(len(docs))]

    collection = client.get_or_create_collection(
        name="chatbot_collection"
    )

    collection.add(
        ids=ids,
        documents=docs,
        embeddings=embs,
        metadatas=[{"source": "doc"} for _ in docs]
    )

    print("✅ ChromaDB vector store created at:", persist_dir)


def load_or_create_vector_store(chunks=None, embeddings=None):
    try:
        collection = client.get_collection("chatbot_collection")
        print("✅ ChromaDB vector store loaded from:", persist_dir)
        return collection
    except Exception:
        if chunks is not None and embeddings is not None:
            print("⚡ Vector store not found. Creating new one...")
            create_vector_store(chunks, embeddings)
            return client.get_collection("chatbot_collection")
        else:
            return None