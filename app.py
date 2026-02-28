from pathlib import Path

import streamlit as st

from src.embeddings import get_embeddings
from src.llm import generate_answer
from src.retriever import retrieve_documents
from src.vector_db import load_or_create_vector_store_from_embeddings_csv

st.set_page_config(page_title="WhizBot", layout="centered")

# Sidebar
st.sidebar.image("robot3.png", use_container_width=True)
st.sidebar.title("WhizBot Chatbot")
st.sidebar.markdown("---")
st.sidebar.write(
    "A robot designed to support learning, interaction, engagement, and assisted guidance in indoor environments."
)

st.title("WhizBot Chatbot")
st.write("Ask any question about WhizBot from the indexed PDF chunks.")

EMBEDDINGS_CSV_PATH = "data/chunks_embeddings.csv"


@st.cache_data(show_spinner=False)
def get_file_signature(file_path):
    path = Path(file_path)
    if not path.exists():
        return "missing"
    stat = path.stat()
    return f"{path.name}:{stat.st_mtime_ns}:{stat.st_size}"


@st.cache_resource(show_spinner=False)
def get_cached_embeddings():
    return get_embeddings()


@st.cache_resource(show_spinner=False)
def get_cached_vectorstore(embeddings_csv_signature):
    return load_or_create_vector_store_from_embeddings_csv(EMBEDDINGS_CSV_PATH)


embeddings_csv_signature = get_file_signature(EMBEDDINGS_CSV_PATH)
embeddings = get_cached_embeddings()
vectorstore = get_cached_vectorstore(embeddings_csv_signature)

if vectorstore is None:
    st.error("Missing data/chunks_embeddings.csv. Run: python src/export_chunk_embeddings.py")
    st.stop()

query = st.text_input("Enter your question:")

if query:
    with st.spinner("Generating answer..."):
        results = retrieve_documents(vectorstore, query, embeddings, k=5)
        context = "\n".join([doc.page_content for doc in results])
        answer = generate_answer(context, query)

    st.subheader("Answer:")
    st.write(answer)

    sources = []
    for doc in results:
        source = doc.metadata.get("source")
        page = doc.metadata.get("page")
        if source and page:
            sources.append(f"{source} (page {page})")
        elif source:
            sources.append(source)

    if sources:
        st.caption("Sources: " + " | ".join(sorted(set(sources))))
else:
    st.warning("Please enter a question.")
