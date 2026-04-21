import pandas as pd
from sentence_transformers import SentenceTransformer
from langchain_groq import ChatGroq
import faiss
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from pathlib import Path
import sys

# -------------------------
# Load data
# -------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
META_FILE   = PROJECT_ROOT / "data" / "raw" / "meta_Toys_and_Games.jsonl"
REVIEW_FILE = PROJECT_ROOT / "data" / "raw" / "Toys_and_Games.jsonl"

meta = pd.read_json(META_FILE, lines=True, nrows=50000)
review = pd.read_json(REVIEW_FILE, lines=True, nrows=50000)

cleaned_meta = meta.drop(columns=['videos', 'price', 'images', 'bought_together', 'subtitle', 'author'], errors='ignore')
cleaned_meta.head()

reviews = review[review['verified_purchase'] == True]
cleaned_reviews = reviews.drop(columns=['images', 'timestamp', 'user_id', 'verified_purchase'], errors='ignore')
cleaned_reviews.head()

# -------------------------
# Clean text columns
# -------------------------
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

# -------------------------
# Prepare review text per product
# -------------------------
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

# -------------------------
# Build product documents
# -------------------------
products = (
    rag_df['title'] + ' ' +
    rag_df['description'] + ' ' +
    rag_df['features'] + ' ' +
    rag_df['categories'] + ' ' +
    rag_df['combined_review_text']
).tolist()

# -------------------------
# Embeddings and vector store
# -------------------------
model = SentenceTransformer("all-MiniLM-L6-v2")
product_embeddings = model.encode(products).astype("float32")

index = faiss.IndexFlatL2(product_embeddings.shape[1])
index.add(product_embeddings)

# -------------------------
# Retriever
# -------------------------

def retrieve(query, top_k=5):
    """Retrieve the top-k most semantically similar products 
    from the RAG dataframe for a given query."""
    query_embedding = model.encode([query]).astype("float32")
    distances, indices = index.search(query_embedding, top_k)
    return rag_df.iloc[indices[0]]

# -------------------------
# Building context
# -------------------------
# Implemented with the help of chatGPT
def build_context(docs):
    """Format retrieved product rows into a structured text context string for the LLM."""
    blocks = []
    for _, row in docs.iterrows():
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


# -------------------------
# Wrapper function
# -------------------------
def retrieve_and_build_context(query):
    """Call the previous retrieve function and build_context function sequentially"""
    docs = retrieve(query, top_k=5)
    return build_context(docs)

# -------------------------
# Prompt variants
# -------------------------
prompt1 = ChatPromptTemplate.from_template(

"""
You must answer using ONLY the information in the context.

- If the answer is present, extract and summarize it clearly.
- Do NOT say "I don't know" if the answer exists in the context.
- Only say "I don't know" if the context truly does not contain the answer.

Context:
{context}

Question:
{input}

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
{input}

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
{input}

Answer:
"""
)

# -------------------------
# Rag Pipeline
# -------------------------
# Implemented with the help of chatGPT
llm = ChatGroq(model="llama-3.1-8b-instant")

rag_chain = (
    {
        "context": RunnableLambda(retrieve_and_build_context),
        "input": RunnablePassthrough()
    }
    | prompt1
    | llm
    | StrOutputParser()
)

queries = [
    "A good board game for kids age 8 and up",
    "A toy for toddlers",
    "Educational toys for kids"
]

for q in queries:
    print(f"\nQUERY: {q}")
    print(rag_chain.invoke(q))

prompts = {
    "prompt1": prompt1,
    "prompt2": prompt2,
    "prompt3": prompt3
}

query = "A good board game for kids age 8 and up"

for name, prompt in prompts.items():
    test_chain = (
        {
            "context": RunnableLambda(retrieve_and_build_context),
            "input": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    print(f"\n===== {name} =====")
    print(test_chain.invoke(query))