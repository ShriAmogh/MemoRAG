import os
import json
import numpy as np
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


cache_file = "semantic_cache.json"
if os.path.exists(cache_file):
    with open(cache_file, "r") as f:
        semantic_cache = json.load(f)
else:
    semantic_cache = []  


def save_cache():
    """Save semantic cache to disk."""
    with open(cache_file, "w") as f:
        json.dump(semantic_cache, f, indent=2)


def cosine_similarity(vec1, vec2):
    """Compute cosine similarity between two vectors."""
    v1, v2 = np.array(vec1), np.array(vec2)
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

#semantic searching between new query and saved ones
def cached_node(query: str, tool: str, call_fn, threshold: float = 0.85):
    """
    Semantic cache wrapper for nodes (retriever or LLM).
    query: input string
    tool: 'retriever' or 'llm'
    call_fn: function to call if cache miss
    threshold: similarity threshold
    """
    query_vec = embeddings.embed_query(query)

    for entry in semantic_cache:
        if entry["tool"] == tool:
            sim = cosine_similarity(query_vec, entry["embedding"])
            if sim >= threshold:
                print(f"[Cache Hit] [{tool}] reusing cached result (similarity={sim:.2f})")
                return entry["output"]


    result = call_fn(query)

    semantic_cache.append({
        "query": query,
        "embedding": query_vec,
        "tool": tool,
        "output": result
    })
    save_cache()

    return result
