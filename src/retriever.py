def retrieve_documents(collection, query, embeddings, k=5):
    query_emb = embeddings.embed_query(query)
    results = collection.query(
        query_embeddings=[query_emb],
        n_results=k
    )
    # results['documents'][0] contains top chunks
    top_chunks = results['documents'][0]
    
    # Convert them to objects with page_content to match old code
    class Doc:
        def __init__(self, content):
            self.page_content = content
    
    top_docs = [Doc(text) for text in top_chunks]
    return top_docs
