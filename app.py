import streamlit as st
from src.loader import load_pdfs
from src.chunker import chunk_documents
from src.embeddings import get_embeddings
from src.vector_db import create_vector_store, load_vector_store
from src.retriever import retrieve_documents
from src.llm import generate_answer
import os

st.set_page_config(page_title="WhizBot", layout="centered")

# -------------------- SIDEBAR --------------------
st.sidebar.image("robot3.png", use_container_width=True)
st.sidebar.title("WhizBot Chatbot")
st.sidebar.markdown("---")
st.sidebar.write("A robot designed to support learning, interaction, engagement, and assisted guidance in indoor environments.")

st.title("🤖 WhizBot Chatbot")
st.write("Ask any question about whizbot.")

# Load embeddings
embeddings = get_embeddings()

# Create vector store if not exists
if not os.path.exists("vector_store/index.faiss"):
    documents = load_pdfs("data")
    chunks = chunk_documents(documents)
    create_vector_store(chunks, embeddings)

vectorstore = load_vector_store(embeddings)

# User input
query = st.text_input("Enter your question:")


if query:
  with st.spinner("Generating answer..."):
     results = retrieve_documents(vectorstore, query)
     context = "\n".join([doc.page_content for doc in results])
     answer = generate_answer(context, query)

  st.subheader("Answer:")
  st.write(answer)
else:
        st.warning("Please enter a question.")
