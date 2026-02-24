# app.py
import streamlit as st
from src.loader import load_pdfs
from src.chunker import chunk_documents
from src.embeddings import get_embeddings
from src.vector_db import load_or_create_vector_store
from src.retriever import retrieve_documents
from src.llm import generate_answer

st.set_page_config(page_title="WhizBot", layout="centered")

# -------------------- SIDEBAR --------------------
st.sidebar.image("robot3.png", use_container_width=True)
st.sidebar.title("WhizBot Chatbot")
st.sidebar.markdown("---")
st.sidebar.write(
    "A robot designed to support learning, interaction, engagement, and assisted guidance in indoor environments."
)

st.title("🤖 WhizBot Chatbot")
st.write("Ask any question about WhizBot.")

# -------------------- LOAD DATA --------------------
# Load PDFs
documents = load_pdfs("data")

# Chunk documents
chunks = chunk_documents(documents)

# Load embeddings
embeddings = get_embeddings()

# Load or create vector store automatically
vectorstore = load_or_create_vector_store(chunks, embeddings)

# -------------------- USER INPUT --------------------
query = st.text_input("Enter your question:")

if query:
    with st.spinner("Generating answer..."):
        results = retrieve_documents(vectorstore, query, embeddings, k=5
                                     )
        context = "\n".join([doc.page_content for doc in results])
        answer = generate_answer(context, query)

    st.subheader("Answer:")
    st.write(answer)
else:
    st.warning("Please enter a question.")