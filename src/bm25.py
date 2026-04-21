from src.simple_tokenize import simple_tokenize

def bm25_search(bm25, products, query, top_k=3):
    """Return the top-k products ranked by BM25 relevance score for a given query."""
    
    tokenized_query = simple_tokenize(query)
    scores = bm25.get_scores(tokenized_query)
    ranked_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    return [(products[i], scores[i]) for i in ranked_idx]