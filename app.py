from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI
import numpy as np
import os

app = FastAPI()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class SimilarityRequest(BaseModel):
    docs: list[str]
    query: str

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

@app.post("/similarity")
def similarity(req: SimilarityRequest):
    # Embed documents
    doc_embeddings = []
    for doc in req.docs:
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=doc
        )
        doc_embeddings.append(response.data[0].embedding)

    # Embed query
    query_embedding = client.embeddings.create(
        model="text-embedding-3-small",
        input=req.query
    ).data[0].embedding

    # Compute similarities
    scores = []
    for emb in doc_embeddings:
        score = cosine_similarity(query_embedding, emb)
        scores.append(score)

    # Get top 3
    top_indices = np.argsort(scores)[-3:][::-1]

    return {
        "matches": [req.docs[i] for i in top_indices]
    }


