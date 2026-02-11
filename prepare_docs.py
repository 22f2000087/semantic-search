from datasets import load_dataset
import json

print("Downloading dataset...")

dataset = load_dataset("ccdv/arxiv-classification", split="train[:127]")

documents = []

for i, item in enumerate(dataset):
    documents.append({
        "id": i,
        "content": item["text"],   # correct field
        "metadata": {
            "label": item["label"]
        }
    })

with open("documents.json", "w") as f:
    json.dump(documents, f)

print("Saved 127 documents to documents.json")



