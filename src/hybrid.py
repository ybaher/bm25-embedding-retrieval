import pandas as pd
from sentence_transformers import SentenceTransformer
from langchain_groq import ChatGroq
import faiss
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document


# Load data
meta = pd.read_json("data/raw/meta_Toys_and_Games.jsonl", lines=True, nrows=50000)
review = pd.read_json("data/raw/Toys_and_Games.jsonl", lines=True, nrows=50000)

cleaned_meta = meta.drop(columns=['videos', 'price', 'images', 'bought_together', 'subtitle', 'author'], errors='ignore')

reviews = review[review['verified_purchase'] == True]
cleaned_reviews = reviews.drop(columns=['images', 'timestamp', 'user_id', 'verified_purchase'], errors='ignore')


# Clean text columns
cleaned_meta['description'] = cleaned_meta['description'].apply(
    lambda x: " ".join(x) if isinstance(x, list) else (x if isinstance(x, str) else "")
).str.lower()

cleaned_meta['details'] = cleaned_meta['details'].apply(
    lambda x: " ".join([f"{k} {v}" for k, v in x.items()]) if isinstance(x, dict) else ""
).str.lower()

cleaned_meta['features'] = cleaned_meta['features'].apply(
    lambda x: " ".join(x) if isinstance(x, list) else ""
).str.lower()

cleaned_meta['categories'] = cleaned_meta['categories'].apply(
    lambda x: " ".join(x) if isinstance(x, list) else ""
).str.lower()

cleaned_meta['title'] = cleaned_meta['title'].str.lower()

cleaned_meta = cleaned_meta[
    (cleaned_meta['title'].str.strip() != '') &
    (cleaned_meta['description'].str.strip() != '') &
    (cleaned_meta['features'].str.strip() != '') &
    (cleaned_meta['categories'].str.strip() != '')
].reset_index(drop=True)


# Prepare review text per product
cleaned_reviews = cleaned_reviews.copy()
review_text_cols = [col for col in ['title', 'text'] if col in cleaned_reviews.columns]
cleaned_reviews['combined_review_text'] = cleaned_reviews[review_text_cols].fillna('').agg(' '.join, axis=1)
cleaned_reviews['combined_review_text'] = cleaned_reviews['combined_review_text'].str.lower()

grouped_reviews = (
    cleaned_reviews.groupby('parent_asin')['combined_review_text']
    .apply(lambda x: " ".join(x.astype(str)))
    .reset_index()
)

rag_df = cleaned_meta.merge(grouped_reviews, on='parent_asin', how='left')
rag_df['combined_review_text'] = rag_df['combined_review_text'].fillna('')


# Build product text strings
products = (
    rag_df['title'] + ' ' +
    rag_df['description'] + ' ' +
    rag_df['features'] + ' ' +
    rag_df['categories'] + ' ' +
    rag_df['combined_review_text']
).tolist()


# Embeddings and FAISS vector store
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
product_embeddings = embed_model.encode(products).astype("float32")

index = faiss.IndexFlatL2(product_embeddings.shape[1])
index.add(product_embeddings)


# Build LangChain Documents for BM25
# (each Document carries the row index in metadata so we can look up rag_df)
lc_docs = [
    Document(page_content=text, metadata={"row_index": i})
    for i, text in enumerate(products)
]


# BM25 Retriever
# LangChain's BM25Retriever handles tokenisation internally
bm25_retriever = BM25Retriever.from_documents(lc_docs, k=5)


# Semantic Retriever (FAISS-based, returns LangChain Documents)
def semantic_retrieve(query: str, top_k: int = 5) -> list[Document]:
    query_embedding = embed_model.encode([query]).astype("float32")
    distances, indices = index.search(query_embedding, top_k)
    results = []
    for idx in indices[0]:
        results.append(
            Document(
                page_content=products[idx],
                metadata={"row_index": int(idx)}
            )
        )
    return results


# Hybrid Retriever: merge BM25 + semantic, deduplicate by row_index
def hybrid_retriever(query: str, top_k: int = 5) -> list[Document]:
    bm25_results = bm25_retriever.invoke(query)
    semantic_results = semantic_retrieve(query, top_k=top_k)

    seen_indices = set()
    merged = []

    # Interleave results from both retrievers to preserve ranking signal
    for bm25_doc, sem_doc in zip(bm25_results, semantic_results):
        for doc in (bm25_doc, sem_doc):
            row_idx = doc.metadata.get("row_index")
            if row_idx not in seen_indices:
                seen_indices.add(row_idx)
                merged.append(doc)

    # Handle any leftover docs if one list is longer than the other
    for doc in bm25_results + semantic_results:
        row_idx = doc.metadata.get("row_index")
        if row_idx not in seen_indices:
            seen_indices.add(row_idx)
            merged.append(doc)

    return merged[:top_k]


# Context builder (accepts list[Document])
def build_context(docs: list[Document]) -> str:
    blocks = []
    for doc in docs:
        row_idx = doc.metadata.get("row_index")
        if row_idx is None:
            continue
        row = rag_df.iloc[row_idx]
        review_snippet = row.get('combined_review_text', '')[:500]
        block = (
            f"Product ASIN: {row.get('parent_asin', 'N/A')}\n"
            f"Title: {row.get('title', '')}\n"
            f"Description: {row.get('description', '')}\n"
            f"Features: {row.get('features', '')}\n"
            f"Categories: {row.get('categories', '')}\n"
            f"Review Evidence: {review_snippet}\n"
        )
        blocks.append(block)
    return "\n\n".join(blocks)


# Prompt variants
prompt1 = ChatPromptTemplate.from_template(
"""
You must answer using ONLY the information in the context.

- If the answer is present, extract and summarize it clearly.
- Do NOT say "I don't know" if the answer exists in the context.
- Only say "I don't know" if the context truly does not contain the answer.

Context:
{context}

Question:
{question}

Answer:
"""
)

prompt2 = ChatPromptTemplate.from_template(
"""
You must answer using ONLY the information in the context.

- Keep the answer shorter than 3 sentences.
- Make sure nothing is repeated, and only necessary details are written.
- If the answer is not in the context, say: "The context does not provide enough information."

Context:
{context}

Input:
{question}

Answer:
"""
)

prompt3 = ChatPromptTemplate.from_template(
"""
You must answer using ONLY the information in the context.

- Be clear and helpful
- Give specific statements instead of general ones.
- If something is missing, say "not enough context to answer your question"

Context:
{context}

Question:
{question}

Answer:
"""
)


# Hybrid RAG Pipeline
llm = ChatGroq(model="llama-3.1-8b-instant")

rag_chain = (
    {
        "context": RunnableLambda(hybrid_retriever) | RunnableLambda(build_context),
        "question": RunnablePassthrough()
    }
    | prompt1
    | llm
    | StrOutputParser()
)


# Run example queries
if __name__ == "__main__":
    queries = [
        "A good board game for kids age 8 and up",
        "A toy for toddlers",
        "Educational toys for kids",
        "A good gift for a child who likes building toys",
        "A fun indoor activity toy for kids"
    ]

    for q in queries:
        print(f"\nQUERY: {q}")
        print(rag_chain.invoke(q))

    # Prompt comparison on a single query
    prompts = {
        "prompt1": prompt1,
        "prompt2": prompt2,
        "prompt3": prompt3,
    }

    query = "A good board game for kids age 8 and up"

    for name, prompt in prompts.items():
        test_chain = (
            {
                "context": RunnableLambda(hybrid_retriever) | RunnableLambda(build_context),
                "question": RunnablePassthrough()
            }
            | prompt
            | llm
            | StrOutputParser()
        )
        print(f"\n===== {name} =====")
        print(test_chain.invoke(query))