# import libraries
import pandas as pd
import numpy as np
from rank_bm25 import BM25Okapi
from langchain_community.retrievers import BM25Retriever
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import re
import pickle
from langchain_openai import OpenAIEmbeddings
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

# read and clean data
meta = pd.read_json("data/raw/meta_Toys_and_Games.jsonl", lines = True, nrows=10000)
review = pd.read_json("data/raw/Toys_and_Games.jsonl", lines = True, nrows=10000)

cleaned_meta = meta.drop(columns = ['videos', 'price', 'images', 'bought_together', 'subtitle', 'author'])
reviews = review[review['verified_purchase'] == True]
cleaned_reviews = reviews.drop(columns = ['images', 'timestamp', 'user_id', 'verified_purchase'])
parent_asin = list(pd.Series(np.intersect1d(cleaned_reviews['parent_asin'], cleaned_meta['parent_asin'])))
cleaned_reviews = cleaned_reviews[cleaned_reviews['parent_asin'].isin(parent_asin)]
cleaned_meta = cleaned_meta[cleaned_meta['parent_asin'].isin(parent_asin)]
cleaned_meta['description'] = cleaned_meta['description'].apply(
    lambda x: " ".join(x) if isinstance(x, list) else (x if isinstance(x, str) else "")
)
cleaned_meta['details'] = cleaned_meta['details'].apply(
    lambda x: " ".join([f"{k} {v}" for k, v in x.items()]) if isinstance(x, dict) else ""
)
cleaned_meta['features'] = cleaned_meta['features'].apply(
    lambda x: " ".join(x) if isinstance(x, list) else ""
)
cleaned_meta['categories'] = cleaned_meta['categories'].apply(
    lambda x: " ".join(x) if isinstance(x, list) else ""
)

# code adapted from lecture 5

# combining all the important and useful columns from the meta dataset
products = (
    "TITLE: " + cleaned_meta['title'].fillna('') + " | " +
    "RATING: " + cleaned_meta['average_rating'].fillna(0).astype(str) + " | " +
    "TEXT: " +
    cleaned_meta['description'].fillna('') + ' ' +
    cleaned_meta['features'].fillna('') + ' ' +
    cleaned_meta['categories'].fillna('')
)
products = products.tolist()
queries = cleaned_reviews['title'].tolist()

tokenized_products = [simple_tokenize(p) for p in products]
bm25 = BM25Okapi(tokenized_products)

model = SentenceTransformer("all-MiniLM-L6-v2")
product_embeddings = model.encode(products).astype("float32")

# using faiss to index product embeddings for semantic search
index = faiss.IndexFlatL2(product_embeddings.shape[1])
index.add(product_embeddings)

# building the app
app = Dash(__name__)
 
app.layout = dbc.Container([
 
    html.H2("Toys & Games Retrieval Models", className="mt-4 mb-1 text-center"),
 
    dbc.Row([
        dbc.Col([
            dbc.Label("Search Model"),
            dbc.RadioItems(
                id="search-mode",
                options=[
                    {"label": "BM25", "value": "bm25"},
                    {"label": "Embedding Search", "value": "semantic"},
                ],
                value="bm25",
                inline=True,
                className="mb-2",
            ),
            dbc.InputGroup([
                dbc.Input(
                    id="query-input",
                    type="text",
                    debounce=False,
                ),
                dbc.Button("Retrieve", id="search-btn", color="primary", n_clicks=0),
            ]),
        ], md=8, className="mx-auto"),
    ], className="mb-4"),
 
    dbc.Row([
        dbc.Col(html.Div(id="results-container"), md=10, className="mx-auto"),
    ]),
 
], fluid=True)
 

@app.callback(
    Output("results-container", "children"),
    Input("search-btn", "n_clicks"),
    Input("query-input", "n_submit"),
    State("query-input", "value"),
    State("search-mode", "value"),
    prevent_initial_call=True,
)
def retrieve(n_clicks, n_submit, query, mode):
    if not query or not query.strip():
        return dbc.Alert("Please enter a search query.", color="warning")
 
    results = bm25_search(bm25, products, query) if mode == "bm25" else embedding_search(model, index, products, query)
    reviews = []
    for rank, (product, score) in enumerate(results, start=1):
        title = re.search(r"TITLE:\s*(.*?)\s*\|", product)
        title = title.group(1) if title else ""
        rating = re.search(r"RATING:\s*(.*?)\s*\|", product)
        rating = rating.group(1) if rating else ""
        text = re.search(r"TEXT:\s*(.*)", product)
        text = text.group(1) if text else ""
        
        reviews.extend([
            f"Product title = {title}",
            html.Br(),
            f"Average rating = {rating}",
            html.Br(),
            f"Retrieval Score = {score:.3f}",
            html.Br(),
            f"Text: {text[:200]}......",
            html.Br(),
            html.Br(),
            html.Br()
        ])

    cards = dbc.Card([
        dbc.CardBody(reviews)
    ])
 
    header = html.P(
        f'Top 3 results for "{query}" '
        f'using {"BM25" if mode == "bm25" else "Embedding"} Search',
        className="text-muted mb-3"
    )
    return [header, cards]
 

if __name__ == "__main__":
    app.run(jupyter_mode="external")