

import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

print("Loading documents...")
with open("documents.json", "r") as f:
    documents = json.load(f)

texts = [doc["content"] for doc in documents]

print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

print("Generating embeddings...")
embeddings = model.encode(texts, convert_to_numpy=True)

# Normalize for cosine similarity
faiss.normalize_L2(embeddings)

dimension = embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)  # cosine similarity
index.add(embeddings)

faiss.write_index(index, "vector.index")

print("Index built and saved as vector.index")


