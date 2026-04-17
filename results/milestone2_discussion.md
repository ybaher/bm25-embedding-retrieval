## Step 1: Why we chose `Groq` (`llama-3.1-8b-instant`)

We decided to use `llama-3.1-8b-instant` from Groq because it has a fast inference and it does not require local GPU resources. This made it more practical when running and testing the RAG pipeline on a laptop while also giving good performance that follow the instructions. Compared to the other models, such as `Mistral-7B` and `Llama-3.2-3B`, Groq's hosted API would remove all the needs to download and run the large models weight locally on the computer, which is a constrain that we constantly ran into given our computer hardware limitations. The 8B parameter size is a good balance between response quality and speed, which is large enough to understand the queries and generate answers that are coherent. We chose this HuggingFace Inference API because Groq's free tier gave a faster and more reliable response during testing.

## Step 2.3: System prompt variant findings

We evaluated the three prompts for the RAG pipeline. The first prompt was the most complete and informative response, as it gave multiple products and provided several reasons based on the context, however, it was more wordy. Prompt 2 was concise but sometimes skipped relevant information and details which led to a less informative response.. Prompt 3 was overly cautious and returned "not enough context" a lot of times, even when useful information was available. Overall, prompt 1 performed the best as it balanced accuracy and completeness while also staying relevant to the context provided.

### ===== prompt1 =====

Considering the context, a good board game for kids age 8 and up that is mentioned is the Sorry! Board Game (Product ASIN: B00000IWD0). The description states that children under age 8 may find it hard to handle the frustration of losing pawns, but it's suitable for kids as young as 6. However, it's likely that most kids would enjoy it by the age of 8.

### ===== prompt2 =====

The Sorry! game is known to be hard for children under age 8 to handle due to the frustration caused by sending each other's pawns back to the starting line.

### ===== prompt3 =====

Based on the description of the Sorry! Board Game, it's recommended for kids ages 6 and up. However, it's mentioned that children under 8 may have a hard time handling the frustration of having their pawns sent back to the starting line. If you're looking for a board game suitable for kids age 8 and up, the Chinese Checkers game (Product ASIN: B08K7ST2ZT) might be a good option, as it's recommended for ages 7 and up.

## Step 5: Qualitative Evaluation of Hybrid RAG

| Query | Accuracy | Completeness | Fluency | Notes |
|---------------|---------------|---------------|---------------|---------------|
| A good board game for kids age 8 and up | Yes | Yes | Yes | The answer matched the retrieved context and gave a good and relevant recommendation with supporting reasons. |
| A toy for toddlers | Yes | Yes | Yes | The answer had several toddler toys which was supported by the context, but it was more like a list than a single recommendation. |
| Educational toys for kids | Yes | Yes | Yes | the response was supported by the retrieved context and provided a bunch of relevant educational toy options. |
| A good gift for a child who likes building toys | Yes | Yes | Yes | The answer was able to find multiple building toy options from the context and explained why they fit the query. |
| A fun indoor activity toy for kids | Yes | Yes | Yes | The answer that was chosen shows a clear relevance to the context and supports its with the details about imaginative play and indoor use. |