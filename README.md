# Information Retrieval with BM25 and Embeddings

Authors: Yasaman Baher, Jessie Liang

## Environment setup

-   Please clone this project locally on your computer
-   Change the working directory to the root project directory
-   Run the following command in terminal:

```{bash}
conda env create -f environment.yml
```

## Dataset description

This project uses a large-scale Amazon Reviews dataset collected by McAuley Lab in 2023. Among all 33 product categories, the category selected for this project was `Toys_and_Games`. The size of the data files can be as large as 800M+. There are 2 main component of the dataset: `review` and `meta`: 

- `review`: This consists of user reviews such as rating, helpfulness votes, etc.

- `meta`: This is the item metadata file including the description, price, etc of the item itself.

## Data Processing description

-   Limit the size of raw data files so that they only include 100,000 rows
    -   Note that our team tried to convert both `meta_Toys_and_Games.jsonl` and `Toys_and_Games.jsonl` from jsonl to parquet, however, due to the large size of the files, our local machines were not able to handle loading the full dataset into memory without slowing down or crashing.
    -   To address this issue, we limited the data loading process by reading a subset of the data. This would allow us to continue the development while staying within the memory constraint.
-   Filtering: only keep the reviews where `verified_purchase == True`, and only the products that appear on both the `review` and `meta` dataset
-   Column selection: drop 'videos', 'price', 'images', 'bought_together', 'subtitle', 'author' columns from `meta`, and drop 'images', 'timestamp', 'user_id', 'verified_purchase' from `review`
-   Combine useful columns from `meta` into one single string for each product, including information of 'title', 'average_rating', 'description', 'features', 'categories', etc
-   For nested fields: join lists into strings, concatenate key-value pairs of dictionary, handle missing strings by replacing with the empty string, etc

## Retrieval workflows

- `BM25`
1. Preprocess the input query by tokenizing it
2. Calculate BM25 scores for every tokenized product
3. Order products by BM25 score from highest to lowest
4. Return top k results

- `Embedding Search`
1. Preprocess the input query using `SentenceTransformer`
2. Search FAISS index of embedded products by leveraging L2 distance
3. Order products using L2 distances from lowest to highest
4. Return top k results

## How to run the app locally

-   Change the working directory to the root project directory
-   Run the following command in terminal:

```{bash}
python app/app.py
```

-   Then there will be some outputs in the terminal similar to this:

```{bash}
Loading weights: 100%|█████████████████████| 103/103 [00:00<00:00, 10030.26it/s]
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
Dash is running on http://127.0.0.1:8050/

 * Serving Flask app 'app'
 * Debug mode: off
```

-   Copy the URL from the above output and open it in a browser. For example, load this URL `http://127.0.0.1:8050/`, and the app will be shown in the browser.
-   Use the radio button to select a retrieval method, and input a query in the text box
-   Click `Retrieve`, done!
