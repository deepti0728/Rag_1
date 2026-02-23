def retrieve_documents(vectorstore, query):
    results = vectorstore.similarity_search_with_score(query, k=5)

    # Sort by best similarity
    results = sorted(results, key=lambda x: x[1])

    # Take top 3 best matches
    top_docs = [doc for doc, score in results[:3]]

    return top_docs
