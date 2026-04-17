# import libraries
import pandas as pd
import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
import re
import faiss
import dash
from dash import dcc, html, Input, Output, State, Dash
import dash_bootstrap_components as dbc
import os
import logging
from transformers import logging as hf_logging
from src.bm25 import bm25_search
from src.semantic import embedding_search
from src.simple_tokenize import simple_tokenize


# suppress warnings
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
hf_logging.set_verbosity_error()
logging.getLogger("werkzeug").setLevel(logging.ERROR)

# Data loading & cleaning
meta = pd.read_json("data/raw/meta_Toys_and_Games.jsonl", lines=True, nrows=10000)
review = pd.read_json("data/raw/Toys_and_Games.jsonl", lines=True, nrows=10000)

cleaned_meta = meta.drop(
    columns=["videos", "price", "images", "bought_together", "subtitle", "author"],
    errors="ignore",
)
reviews = review[review["verified_purchase"] == True]
cleaned_reviews = reviews.drop(
    columns=["images", "timestamp", "user_id", "verified_purchase"], errors="ignore"
)

parent_asin = list(
    pd.Series(
        np.intersect1d(cleaned_reviews["parent_asin"], cleaned_meta["parent_asin"])
    )
)
cleaned_reviews = cleaned_reviews[cleaned_reviews["parent_asin"].isin(parent_asin)]
cleaned_meta = cleaned_meta[cleaned_meta["parent_asin"].isin(parent_asin)]

for col, fn in [
    ("description", lambda x: " ".join(x) if isinstance(x, list) else (x if isinstance(x, str) else "")),
    ("details",     lambda x: " ".join([f"{k} {v}" for k, v in x.items()]) if isinstance(x, dict) else ""),
    ("features",    lambda x: " ".join(x) if isinstance(x, list) else ""),
    ("categories",  lambda x: " ".join(x) if isinstance(x, list) else ""),
]:
    cleaned_meta[col] = cleaned_meta[col].apply(fn)

# Product strings (Milestone 1 – BM25 / Semantic)
products = (
    "TITLE: " + cleaned_meta["title"].fillna("") + " | " +
    "RATING: " + cleaned_meta["average_rating"].fillna(0).astype(str) + " | " +
    "TEXT: " +
    cleaned_meta["description"].fillna("") + " " +
    cleaned_meta["features"].fillna("") + " " +
    cleaned_meta["categories"].fillna("")
).tolist()

tokenized_products = [simple_tokenize(p) for p in products]
bm25 = BM25Okapi(tokenized_products)

model = SentenceTransformer("all-MiniLM-L6-v2")
product_embeddings = model.encode(products).astype("float32")

index_m1 = faiss.IndexFlatL2(product_embeddings.shape[1])
index_m1.add(product_embeddings)

# RAG data (Milestone 2)
rag_meta = cleaned_meta.copy()
for col in ["description", "features", "categories", "title"]:
    rag_meta[col] = rag_meta[col].str.lower()

review_text_cols = [c for c in ["title", "text"] if c in cleaned_reviews.columns]
cleaned_reviews = cleaned_reviews.copy()
cleaned_reviews["combined_review_text"] = (
    cleaned_reviews[review_text_cols].fillna("").agg(" ".join, axis=1).str.lower()
)
grouped_reviews = (
    cleaned_reviews.groupby("parent_asin")["combined_review_text"]
    .apply(lambda x: " ".join(x.astype(str)))
    .reset_index()
)
rag_df = rag_meta.merge(grouped_reviews, on="parent_asin", how="left")
rag_df["combined_review_text"] = rag_df["combined_review_text"].fillna("")

rag_products = (
    rag_df["title"] + " " +
    rag_df["description"] + " " +
    rag_df["features"] + " " +
    rag_df["categories"] + " " +
    rag_df["combined_review_text"]
).tolist()

rag_embeddings = model.encode(rag_products).astype("float32")
rag_index = faiss.IndexFlatL2(rag_embeddings.shape[1])
rag_index.add(rag_embeddings)

lc_docs = [
    Document(page_content=text, metadata={"row_index": i})
    for i, text in enumerate(rag_products)
]
bm25_retriever = BM25Retriever.from_documents(lc_docs, k=5)


def semantic_retrieve(query: str, top_k: int = 5) -> list:
    q_emb = model.encode([query]).astype("float32")
    _, indices = rag_index.search(q_emb, top_k)
    return [
        Document(page_content=rag_products[i], metadata={"row_index": int(i)})
        for i in indices[0]
    ]


def hybrid_retriever(query: str, top_k: int = 5) -> list:
    bm25_res = bm25_retriever.invoke(query)
    sem_res = semantic_retrieve(query, top_k=top_k)
    seen, merged = set(), []
    for b, s in zip(bm25_res, sem_res):
        for doc in (b, s):
            ri = doc.metadata.get("row_index")
            if ri not in seen:
                seen.add(ri)
                merged.append(doc)
    for doc in bm25_res + sem_res:
        ri = doc.metadata.get("row_index")
        if ri not in seen:
            seen.add(ri)
            merged.append(doc)
    return merged[:top_k]


def build_context(docs: list) -> str:
    blocks = []
    for doc in docs:
        ri = doc.metadata.get("row_index")
        if ri is None:
            continue
        row = rag_df.iloc[ri]
        blocks.append(
            f"Product ASIN: {row.get('parent_asin', 'N/A')}\n"
            f"Title: {row.get('title', '')}\n"
            f"Description: {row.get('description', '')}\n"
            f"Features: {row.get('features', '')}\n"
            f"Categories: {row.get('categories', '')}\n"
            f"Review Evidence: {row.get('combined_review_text', '')[:500]}\n"
        )
    return "\n\n".join(blocks)


rag_prompt = ChatPromptTemplate.from_template(
    """You must answer using ONLY the information in the context.
- If the answer is present, extract and summarize it clearly.
- Do NOT say "I don't know" if the answer exists in the context.
- Only say "I don't know" if the context truly does not contain the answer.

Context:
{context}

Question:
{question}

Answer:"""
)

llm = ChatGroq(model="llama-3.1-8b-instant")

rag_chain = (
    {
        "context": RunnableLambda(hybrid_retriever) | RunnableLambda(build_context),
        "question": RunnablePassthrough(),
    }
    | rag_prompt
    | llm
    | StrOutputParser()
)

# App layout
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = dbc.Container(
    [
        html.H2("Toys & Games Retrieval Models", className="mt-4 mb-3 text-center"),

        # Mode tabs
        dbc.Tabs(
            id="mode-tabs",
            active_tab="search",
            children=[
                dbc.Tab(label="Search Only (Milestone 1)", tab_id="search"),
                dbc.Tab(label="RAG Mode (Milestone 2)", tab_id="rag"),
            ],
            className="mb-4",
        ),

        # Search bar
        dbc.Row(
            [
                dbc.Col(
                    [
                        # Search-only model selector (hidden in RAG mode)
                        html.Div(
                            id="search-model-selector",
                            children=[
                                dbc.Label("Search Model"),
                                dbc.RadioItems(
                                    id="search-mode",
                                    options=[
                                        {"label": "BM25",            "value": "bm25"},
                                        {"label": "Embedding Search","value": "semantic"},
                                    ],
                                    value="bm25",
                                    inline=True,
                                    className="mb-2",
                                ),
                            ],
                        ),
                        dbc.InputGroup(
                            [
                                dbc.Input(
                                    id="query-input",
                                    type="text",
                                    placeholder="Enter your search query…",
                                    debounce=False,
                                ),
                                dbc.Button(
                                    "Retrieve",
                                    id="search-btn",
                                    color="primary",
                                    n_clicks=0,
                                ),
                            ]
                        ),
                    ],
                    md=8,
                    className="mx-auto",
                )
            ],
            className="mb-4",
        ),

        # Results
        dbc.Row(
            [dbc.Col(html.Div(id="results-container"), md=10, className="mx-auto")]
        ),
    ],
    fluid=True,
)

# Hide / show search-model selector based on active tab
@app.callback(
    Output("search-model-selector", "style"),
    Input("mode-tabs", "active_tab"),
)
def toggle_model_selector(tab):
    return {"display": "none"} if tab == "rag" else {}


@app.callback(
    Output("results-container", "children"),
    Input("search-btn", "n_clicks"),
    Input("query-input", "n_submit"),
    State("query-input", "value"),
    State("search-mode", "value"),
    State("mode-tabs", "active_tab"),
    prevent_initial_call=True,
)
def retrieve(n_clicks, n_submit, query, search_mode, active_tab):
    if not query or not query.strip():
        return dbc.Alert("Please enter a search query.", color="warning")

    # SEARCH ONLY (Milestone 1)
    if active_tab == "search":
        results = (
            bm25_search(bm25, products, query)
            if search_mode == "bm25"
            else embedding_search(model, index_m1, products, query)
        )

        cards = []
        for rank, (product, _score) in enumerate(results, start=1):
            title_m = re.search(r"TITLE:\s*(.*?)\s*\|", product)
            title   = title_m.group(1) if title_m else "—"
            rating_m = re.search(r"RATING:\s*(.*?)\s*\|", product)
            rating   = rating_m.group(1) if rating_m else "—"

            cards.append(
                dbc.Card(
                    dbc.CardBody(
                        [
                            html.H6(f"{rank}. {title}", className="card-title mb-1"),
                            html.P(
                                f"Average Rating: {rating}",
                                className="card-text text-muted mb-0",
                            ),
                        ]
                    ),
                    className="mb-2",
                )
            )

        header = html.P(
            f'Top {len(results)} results for "{query}" '
            f'using {"BM25" if search_mode == "bm25" else "Embedding"} Search',
            className="text-muted mb-3",
        )
        return [header] + cards

    # RAG MODE (Milestone 2)
    else:
        # Retrieve docs for source attribution
        retrieved_docs = hybrid_retriever(query, top_k=5)

        # Generate answer
        context_str = build_context(retrieved_docs)
        answer_full = rag_chain.invoke(query)

        # Truncate if very long (keep ≤ 600 chars, add ellipsis)
        MAX_ANSWER = 600
        answer_display = (
            answer_full[:MAX_ANSWER] + "…"
            if len(answer_full) > MAX_ANSWER
            else answer_full
        )

        # Answer panel
        answer_panel = dbc.Card(
            [
                dbc.CardHeader(
                    html.Strong("🤖 RAG Answer"),
                    style={"background": "#f0f4ff"},
                ),
                dbc.CardBody(html.P(answer_display, className="mb-0")),
            ],
            className="mb-4 border-primary",
        )

        # Source attribution cards
        source_cards = []
        for i, doc in enumerate(retrieved_docs, start=1):
            ri = doc.metadata.get("row_index")
            if ri is None:
                continue
            row   = rag_df.iloc[ri]
            title = row.get("title", "—").title()
            rating = row.get("average_rating", "—")

            source_cards.append(
                dbc.Card(
                    dbc.CardBody(
                        [
                            html.H6(
                                f"[{i}] {title}",
                                className="card-title mb-1",
                            ),
                            html.P(
                                f"Average Rating: {rating}",
                                className="card-text text-muted mb-0",
                            ),
                        ]
                    ),
                    className="mb-2",
                )
            )

        sources_section = html.Div(
            [
                html.P(
                    "Retrieved Sources (Hybrid RAG):",
                    className="text-muted fw-semibold mb-2",
                ),
                *source_cards,
            ]
        )

        header = html.P(
            f'RAG results for "{query}"',
            className="text-muted mb-3",
        )

        return [header, answer_panel, sources_section]


if __name__ == "__main__":
    app.run(jupyter_mode="external")