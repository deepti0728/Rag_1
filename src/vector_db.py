import csv
import json
import os
from hashlib import sha256

import chromadb

# Folder where DB will be stored
persist_dir = "./vector_store"
collection_name = "chatbot_collection"
state_file = os.path.join(persist_dir, "index_state.json")

# Make sure folder exists
os.makedirs(persist_dir, exist_ok=True)

client = chromadb.PersistentClient(path=persist_dir)


def _build_index_signature(chunks):
    payload = []
    for chunk in chunks:
        payload.append(
            {
                "text": chunk.page_content,
                "source": chunk.metadata.get("source", ""),
                "page": chunk.metadata.get("page", ""),
            }
        )
    encoded = json.dumps(payload, ensure_ascii=True, sort_keys=True).encode("utf-8")
    return sha256(encoded).hexdigest()


def _build_file_signature(file_path):
    if not os.path.exists(file_path):
        return None

    hasher = sha256()
    with open(file_path, "rb") as file:
        for block in iter(lambda: file.read(8192), b""):
            hasher.update(block)
    return hasher.hexdigest()


def _load_saved_signature():
    if not os.path.exists(state_file):
        return None

    try:
        with open(state_file, "r", encoding="utf-8") as file:
            data = json.load(file)
        return data.get("signature")
    except Exception:
        return None


def _save_signature(signature):
    with open(state_file, "w", encoding="utf-8") as file:
        json.dump({"signature": signature}, file)


def _collection_exists():
    try:
        client.get_collection(collection_name)
        return True
    except Exception:
        return False


def create_vector_store(chunks, embeddings):
    docs = [chunk.page_content for chunk in chunks]
    embs = embeddings.embed_documents(docs)
    ids = [str(i) for i in range(len(docs))]

    metadatas = []
    for chunk in chunks:
        metadatas.append(
            {
                "source": str(chunk.metadata.get("source", "unknown")),
                "page": int(chunk.metadata.get("page", 0)),
            }
        )

    if _collection_exists():
        client.delete_collection(collection_name)

    collection = client.get_or_create_collection(name=collection_name)
    collection.add(ids=ids, documents=docs, embeddings=embs, metadatas=metadatas)

    _save_signature(_build_index_signature(chunks))
    print("ChromaDB vector store created at:", persist_dir)
    return collection


def create_vector_store_from_embeddings_csv(embeddings_csv_path):
    if not os.path.exists(embeddings_csv_path):
        return None

    with open(embeddings_csv_path, "r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))

    if not rows:
        return None

    ids = []
    docs = []
    embs = []
    metadatas = []

    for index, row in enumerate(rows):
        chunk_id = str(row.get("chunk_id", index))
        chunk_text = row.get("chunk_text", "")
        source = str(row.get("source", "unknown"))
        page_raw = row.get("page", "")
        page = int(page_raw) if str(page_raw).strip().isdigit() else 0
        embedding = json.loads(row.get("embedding", "[]"))

        ids.append(chunk_id)
        docs.append(chunk_text)
        embs.append(embedding)
        metadatas.append({"source": source, "page": page})

    if _collection_exists():
        client.delete_collection(collection_name)

    collection = client.get_or_create_collection(name=collection_name)
    collection.add(ids=ids, documents=docs, embeddings=embs, metadatas=metadatas)

    signature = _build_file_signature(embeddings_csv_path)
    _save_signature(f"csv:{signature}")
    print("ChromaDB vector store created from CSV embeddings:", persist_dir)
    return collection


def load_or_create_vector_store(chunks=None, embeddings=None):
    should_reindex = False
    collection = None

    if _collection_exists():
        collection = client.get_collection(collection_name)
    elif chunks is not None and embeddings is not None:
        should_reindex = True
    else:
        return None

    if chunks is not None and embeddings is not None:
        current_signature = _build_index_signature(chunks)
        saved_signature = _load_saved_signature()
        if saved_signature != current_signature:
            should_reindex = True

    if should_reindex:
        print("Rebuilding vector store from current PDFs...")
        create_vector_store(chunks, embeddings)
        collection = client.get_collection(collection_name)
    else:
        print("ChromaDB vector store loaded from:", persist_dir)

    return collection


def load_or_create_vector_store_from_embeddings_csv(embeddings_csv_path):
    if not os.path.exists(embeddings_csv_path):
        return None

    current_signature = _build_file_signature(embeddings_csv_path)
    saved_signature = _load_saved_signature()
    expected_signature = f"csv:{current_signature}"

    should_reindex = (not _collection_exists()) or (saved_signature != expected_signature)

    if should_reindex:
        print("Rebuilding vector store from embeddings CSV...")
        return create_vector_store_from_embeddings_csv(embeddings_csv_path)

    print("ChromaDB vector store loaded from:", persist_dir)
    return client.get_collection(collection_name)
