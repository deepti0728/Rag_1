from loader import load_pdfs
from chunker import chunk_documents
from embeddings import get_embeddings
from vector_db import create_vector_store, load_vector_store
from retriever import retrieve_documents
from llm import generate_answer
import os

#  Load PDFs
documents = load_pdfs("data")

#  Chunk
chunks = chunk_documents(documents)

#  Embeddings
embeddings = get_embeddings()

#  If vector_store folder doesn't exist → create it
if not os.path.exists("vector_store/index.faiss"):
    print("Creating vector store...")
    create_vector_store(chunks, embeddings)

#  Load vector store
vectorstore = load_vector_store(embeddings)

#  Ask question
query = input("Ask your question: ")

results = retrieve_documents(vectorstore, query)

# Combine retrieved context
context = "\n".join([doc.page_content for doc in results])

answer = generate_answer(context, query)

print("\nAnswer:\n")
print(answer)


    
