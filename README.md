# Information Retrieval with BM25 and Embeddings

Authors: Yasaman Baher, Jessie Liang

## Environment setup

-   Please clone this project locally on your computer
-   Change the working directory to the root project directory
-   Run the following command in terminal:

```{bash}
conda env create -f environments.yml
```

## Project Structure

```         
├── app/
│   └── app.py               # Dash web app with Search and RAG modes
├── notebooks/
│   └── milestone2_rag.ipynb # Exploratory notebook for RAG development
├── results/
│   └── milestone2_discussion.md  # Qualitative evaluation and findings
├── src/
│   ├── rag_pipeline.py      # Full RAG pipeline 
│   ├── hybrid.py            # Hybrid retriever combining BM25 and FAISS
│   └── prompts.py           # Prompt templates for the LLM
├── data/                    # Processed Amazon Toys & Games dataset
├── environments.yml         # Conda environment specification
└── README.md
```

## Dataset description

This project uses a large-scale Amazon Reviews dataset collected by McAuley Lab in 2023. Among all 33 product categories, the category selected for this project was `Toys_and_Games`. The size of the data files can be as large as 800M+. There are 2 main component of the dataset: `review` and `meta`:

-   `review`: This consists of user reviews such as rating, helpfulness votes, etc.

-   `meta`: This is the item metadata file including the description, price, etc of the item itself.

## Data Processing description

-   **Size limit:** We limited the size of raw data files so that they only include 50,000 rows
    -   Note that our team tried to convert both `meta_Toys_and_Games.jsonl` and `Toys_and_Games.jsonl` from jsonl to parquet, however, due to the large size of the files, our local machines were not able to handle loading the full dataset into memory without slowing down or crashing.
        -   After preprocessing, filtering verified purchases, and keeping overlapping products, the final pipeline processed 28,947 products.
    -   To address this issue, we limited the data loading process by reading a subset of the data. This would allow us to continue the development while staying within the memory constraint.
-   **Filtering:** only keep the reviews where `verified_purchase == True`, and only the products that appear on both the `review` and `meta` dataset
-   **Column selection:** drop 'videos', 'price', 'images', 'bought_together', 'subtitle', 'author' columns from `meta`, and drop 'images', 'timestamp', 'user_id', 'verified_purchase' from `review`
-   **Column combination:** we combined useful columns from `meta` into one single string for each product, including information of 'title', 'average_rating', 'description', 'features', 'categories', etc
-   **For nested fields:** join lists into strings, concatenate key-value pairs of dictionary, handle missing strings by replacing with the empty string, etc.
-   **Review aggregation:** The reviewed texts were grouped by `parent_asin` and merged into a single string per product. The new aggregated review information was later merged with the metadata so that both product details and user feedback are included in each document used for retrieval.
-   **Retrieval:** We implemented a custom retriever using `SentenceTransformer` and `FAISS` rather than using LangChain's `HuggingFaceEmbeddings` and `vectorstore.as_retriever()`. This was permitted by the spec for custom Python retriever implementations.

## Retrieval workflows

-   `BM25`

1.  Preprocess the input query by tokenizing it
2.  Calculate BM25 scores for every tokenized product
3.  Order products by BM25 score from highest to lowest
4.  Return top k results

-   `Embedding Search`

1.  Preprocess the input query using `SentenceTransformer`
2.  Search FAISS index of embedded products by leveraging L2 distance
3.  Order products using L2 distances from lowest to highest
4.  Return top k results

-   `Rag Pipeline` (Semantic)

1.  User Query
2.  Retriever (`FAISS` + sentence embeddings)
3.  Top-k relevant products retrieved
4.  Context builder formats metadata and review evidence
5.  Prompt template adds instructions
6.  LLM (Groq Llama 3.1 8B)
7.  Final answer

## RAG Workflow Diagram

``` mermaid
flowchart TD
    A[User Query] --> B[Retriever]
    B --> C[Semantic Search FAISS]
    B --> D[BM25 Search]
    C --> E[Hybrid Retriever]
    D --> E
    E --> F[Top-k Retrieved Products]
    F --> G[Context Builder]
    G --> H[Prompt Template]
    H --> I[LLM via Groq]
    I --> J[Final Answer]
```

## App Features

### Search Mode (Milestone 1)

-   Select a retrieval method via radio button: **BM25**, **Semantic Search**, or **Hybrid**
-   Enter a natural language query in the text box and click **Retrieve**
-   View the top 5 results, where each shows:
    -   Product title
    -   Average rating
    -   Truncated review texts

### RAG Mode (Milestone 2)

-   Switch to RAG mode using the tab at the top of the platform
-   Enter a query to run the full **Hybrid RAG pipeline**
-   A generated answer is shown above the given documents
-   Given source products are shown below the answer, each showing:
    -   Product title
    -   Average rating
    -   Truncated review text (\~200 chars)

## API Setup

This project uses Groq for the LLM.

1.  Create an API key at: https://console.groq.com/keys

2.  Set your API key as an environment variable:

Mac/Linux: `export GROQ_API_KEY=<your given gsk key>`

Windows: `setx GROQ_API_KEY <your given gsk key>`

3.  (Optional) verify: `echo $GROQ_API_KEY`

## How to run the RAG pipeline

-   Change the working directory to the root project directory
-   Make sure your `GROQ_API_KEY` is set (see API Setup above)
-   Run the following command in terminal to test Milestone 2 Step 2:

``` bash
python src/rag_pipeline.py
```

-   Run the following command in terminal to test Milestone 2 Step 3:

``` bash
python src/hybrid.py
```

-   This will run the RAG pipeline with a set of sample queries and print the answers to the terminal.

**Important note**: If you created API here `https://console.groq.com/keys` and used it, you might encounter an error if you send too many requests too fast, hitting the free tier limit of 6,000 tokens/minute. The error is a Groq rate limit — not a code bug. To resolve this error, just wait for a while and send requests again afterwards, giving it a cooldown period.

## How to run the app locally

-   Change the working directory to the root project directory
-   Run from the project root directory:

``` bash
python app/app.py 
```

-   Then there will be some outputs in the terminal similar to this:

``` bash
Loading weights: 100%|█████████████████████| 103/103 [00:00<00:00, 10030.26it/s]
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
Dash is running on http://127.0.0.1:8050/

 * Serving Flask app 'app'
 * Debug mode: off
```

-   Copy the URL from the above output and open it in a browser. For example, load this URL `http://127.0.0.1:8050/`, and the app will be shown in the browser.
-   Use the tab on the top bar to select a mode. In the first tab, use the radio button to select a retrieval method, and input a query in the text box
-   Click `Retrieve`, done!