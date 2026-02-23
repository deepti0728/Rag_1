from langchain_text_splitters import RecursiveCharacterTextSplitter

def chunk_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50
    )

    chunks = splitter.split_documents(documents)

     # Remove very short useless chunks
    cleaned_chunks = [
        chunk for chunk in chunks
        if len(chunk.page_content) > 100
    ]

    return chunks

