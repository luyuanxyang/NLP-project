
from sentence_transformers import SentenceTransformer
import pandas as pd
import torch
from huggingface_hub import login
from sentence_transformers import SentenceTransformer
import ast

# ⚠️ Put YOUR real token string here (keep it secret, don’t share it)
login(token="hf_xxxxxxx")

# Download from the 🤗 Hub
model = SentenceTransformer("google/embeddinggemma-300m")

# load data
df = pd.read_csv("C:\\Users\\Luyuan Yang\\Downloads\\evaluation_llm_gpt4.csv")

score_list=[]

for i in range(len(df)):
    # Run inference with queries and documents
    query =df['pred_answer'][i]
    documents =ast.literal_eval(df['true_answer'][i])

    if len(documents) == 0 or not isinstance(query, str) or query.strip() == "":
        print("(EMPTY) → score = 0")
        score_list.append(0)
        continue

    query_embeddings = model.encode_query(query)
    document_embeddings = model.encode_document(documents)
    print(query_embeddings.shape, document_embeddings.shape)
    # (768,) (4, 768)

    # Compute similarities to determine a ranking
    similarities = model.similarity(query_embeddings, document_embeddings)
    score=torch.max(similarities).item()
    print(similarities,score)
    score_list.append(score)

df["cos score"] = score_list
df.to_csv("C:\\Users\\Luyuan Yang\\Downloads\\evaluation_llm_gpt4_cos_scored.csv", index=False)