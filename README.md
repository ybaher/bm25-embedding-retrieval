# Information Retrieval with BM25 and Embeddings

Authors: Yasaman Baher, Jessie Liang

## Environment setup

- Please clone this project locally on your computer
- Change the working directory to the root project directory
- Run the following command in terminal:
```{bash}
conda env create -f environment.yml
```

## Dataset description
This project uses a large-scale Amazon Reviews dataset collected by McAuley Lab in 2023. Among all 33 product categories, the category selected for this project was `Toys_and_Games`. The size of the data files can be as large as 800M+. There are 2 main component of the dataset: `review` and `meta`:
- `review`: This consists of user reviews such as rating, helpfulness votes, etc.
- `meta`: This is the item metadata file including the description, price, etc of the item itself.

## How to run the app locally
- Change the working directory to the root project directory
- Run the following command in terminal:
```{bash}
python app/app.py
```
- Then there will be some outputs in the terminal similar to this:
```{bash}
Loading weights: 100%|█████████████████████| 103/103 [00:00<00:00, 10030.26it/s]
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
Dash is running on http://127.0.0.1:8050/

 * Serving Flask app 'app'
 * Debug mode: off
```
- Copy the URL from the above output and open it in a browser. For example, load this URL `http://127.0.0.1:8050/`, and the app will be shown in the browser.
- Use the radio button to select a retrieval method, and input a query in the text box
- Click `Retrieve`, done!
