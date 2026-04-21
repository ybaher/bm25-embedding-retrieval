def embedding_search(model, index, products, query, top_k=3):
    """Retrieve the top-k products by semantic similarity using FAISS vector search."""
    
    query_embedding = model.encode([query]).astype("float32")
    distances, indices = index.search(query_embedding, top_k)
    return [(products[i], distances[0][rank]) for rank, i in enumerate(indices[0])]