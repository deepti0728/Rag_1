def retrieve_documents(collection, query, embeddings, k=5):
    query_emb = embeddings.embed_query(query)
    results = collection.query(
        query_embeddings=[query_emb],
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )

    top_chunks = results.get("documents", [[]])[0]
    top_metadata = results.get("metadatas", [[]])[0]
    top_distances = results.get("distances", [[]])[0]

    class Doc:
        def __init__(self, content, metadata=None, score=None):
            self.page_content = content
            self.metadata = metadata or {}
            self.score = score

    top_docs = []
    for idx, text in enumerate(top_chunks):
        metadata = top_metadata[idx] if idx < len(top_metadata) else {}
        score = top_distances[idx] if idx < len(top_distances) else None
        top_docs.append(Doc(text, metadata, score))

    return top_docs
