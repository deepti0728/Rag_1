import os
from langchain_community.document_loaders import PyPDFLoader

def clean_text(text):
    text = text.replace("\n", " ")
    text = " ".join(text.split())
    return text

def load_pdfs(folder_path):
    documents = []

    for file in os.listdir(folder_path):
        if file.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(folder_path, file))
            docs = loader.load()

            # Clean each page
            for doc in docs:
                doc.page_content = clean_text(doc.page_content)

            documents.extend(docs)

    return documents
