# recommend_utils.py
import numpy as np
import pandas as pd


def find_similar_sparse(index_query, corpus, tfidf_model, index_sim, df, top_k=5):
    vec_query = tfidf_model[corpus[index_query]]
    sims = index_sim[vec_query]

    top_indices = np.argsort(sims)[::-1][1 : top_k + 1]
    top_scores = sims[top_indices]

    result = df.iloc[top_indices][["product_id", "product_name"]].copy()
    result["similarity_score"] = top_scores
    return result
