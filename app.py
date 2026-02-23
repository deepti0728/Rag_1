import streamlit as st
from src.loader import load_pdfs
from src.chunker import chunk_documents
from src.embeddings import get_embeddings
from src.vector_db import create_vector_store, load_vector_store
from src.retriever import retrieve_documents
from src.llm import generate_answer
import os

st.set_page_config(page_title="WhizBot", layout="centered")

# Absolute paths
BASE_DIR = os.path.dirname(__file__)
DATA_FOLDER = os.path.join(BASE_DIR, "data")
VECTOR_STORE_FOLDER = os.path.join(BASE_DIR, "vector_store")
VECTOR_STORE_FILE = os.path.join(VECTOR_STORE_FOLDER, "index.faiss")

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
if not os.path.exists(VECTOR_STORE_FILE):
    documents = load_pdfs(DATA_FOLDER)
    chunks = chunk_documents(documents)
    
    # Make sure folder exists
    os.makedirs(VECTOR_STORE_FOLDER, exist_ok=True)
    
    create_vector_store(chunks, embeddings)

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
