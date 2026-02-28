import os
from langchain_community.document_loaders import PyPDFLoader

def clean_text(text):
    text = text.replace("\n", " ")
    text = " ".join(text.split())
    return text

def load_pdfs(folder_path):
    documents = []

    if not os.path.isdir(folder_path):
        return documents

    for file in sorted(os.listdir(folder_path)):
        if file.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(folder_path, file))
            docs = loader.load()

            # Clean each page
            for doc in docs:
                doc.page_content = clean_text(doc.page_content)
                doc.metadata["source"] = file
                if "page" in doc.metadata:
                    doc.metadata["page"] = int(doc.metadata["page"]) + 1

            documents.extend(docs)

    return documents
