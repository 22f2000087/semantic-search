import json
import time
import numpy as np
import faiss
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

app = FastAPI()

# Load documents
with open("documents.json", "r") as f:
    documents = json.load(f)

# Load model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Load FAISS index
index = faiss.read_index("vector.index")

class SearchRequest(BaseModel):
    query: str
    k: int = 8
    rerank: bool = True
    rerankK: int = 5

@app.post("/search")
def search(request: SearchRequest):
    start = time.time()

    query_embedding = model.encode([request.query], convert_to_numpy=True)
    faiss.normalize_L2(query_embedding)

    scores, indices = index.search(query_embedding, request.k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        results.append({
            "id": int(idx),
            "score": float((score + 1) / 2),  # normalize 0–1
            "content": documents[idx]["content"],
            "metadata": {"source": documents[idx]["id"]}
        })

    # Sort descending
    results = sorted(results, key=lambda x: x["score"], reverse=True)

    if request.rerank:
        results = results[:request.rerankK]

    latency = int((time.time() - start) * 1000)

    return {
        "results": results,
        "reranked": request.rerank,
        "metrics": {
            "latency": latency,
            "totalDocs": len(documents)
        }
    }

