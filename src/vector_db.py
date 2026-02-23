from langchain_community.vectorstores import FAISS
import os

def create_vector_store(chunks, embeddings):
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local("vector_store")

def load_vector_store(embeddings):
    if os.path.exists("vector_store"):
        return FAISS.load_local(
            "vector_store",
            embeddings,
            allow_dangerous_deserialization=True
        )
    else:
        return None

