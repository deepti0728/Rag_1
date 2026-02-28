from chunker import chunk_documents
from embeddings import get_embeddings
from llm import generate_answer
from loader import load_pdfs
from retriever import retrieve_documents
from vector_db import load_or_create_vector_store

# Load PDFs
documents = load_pdfs("data")

# Chunk
chunks = chunk_documents(documents)

# Embeddings
embeddings = get_embeddings()

# Load or create vector store
vectorstore = load_or_create_vector_store(chunks, embeddings)

# Ask question
query = input("Ask your question: ")

results = retrieve_documents(vectorstore, query, embeddings, k=5)

# Combine retrieved context
context = "\n".join([doc.page_content for doc in results])

answer = generate_answer(context, query)

print("\nAnswer:\n")
print(answer)
