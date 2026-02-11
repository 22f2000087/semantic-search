import os
import numpy as np
import faiss
from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
import time
import json

app = Flask(__name__)

# Add your documents HERE
SCIENTIFIC_ABSTRACTS = [
    {
        "id": 0,
        "content": "Deep learning for protein structure prediction using attention mechanisms. Our model achieves state-of-the-art accuracy on CASP14 benchmark for folding protein sequences.",
        "metadata": {"source": "Nature Biotechnology", "year": 2023}
    },
    {
        "id": 1,
        "content": "Quantum machine learning for drug discovery. Hybrid quantum-classical algorithms demonstrate quadratic speedup for molecular property prediction and optimization.",
        "metadata": {"source": "Nature Chemistry", "year": 2023}
    },
    {
        "id": 2,
        "content": "Machine learning applications in climate science. Neural networks and random forests for predicting temperature anomalies and extreme weather events from historical data.",
        "metadata": {"source": "Environmental Research Letters", "year": 2022}
    },
    {
        "id": 3,
        "content": "Reinforcement learning for autonomous driving. Deep Q-networks with attention mechanisms for safe navigation in complex urban traffic scenarios.",
        "metadata": {"source": "IEEE Transactions on Intelligent Vehicles", "year": 2024}
    },
    {
        "id": 4,
        "content": "Transfer learning for medical imaging. Pretraining on natural images enables accurate pneumonia detection from chest X-rays with only 100 labeled examples.",
        "metadata": {"source": "Medical Image Analysis", "year": 2023}
    },
    {
        "id": 5,
        "content": "Federated learning for privacy-preserving NLP. Secure aggregation protocols enable training large language models on decentralized data without exposing sensitive information.",
        "metadata": {"source": "ACL Conference", "year": 2023}
    },
    {
        "id": 6,
        "content": "Graph neural networks for social network analysis. Capturing structural and temporal dynamics for community detection and influence prediction.",
        "metadata": {"source": "Social Networks Journal", "year": 2022}
    },
    {
        "id": 7,
        "content": "Explainable AI for credit scoring. Comparative analysis of LIME, SHAP, and counterfactual explanations for loan approval decisions.",
        "metadata": {"source": "Journal of Banking & Finance", "year": 2023}
    },
    {
        "id": 8,
        "content": "Transformer models for long document summarization. Hierarchical attention mechanisms process 10,000+ tokens efficiently for arXiv and PubMed papers.",
        "metadata": {"source": "ACL Conference", "year": 2023}
    },
    {
        "id": 9,
        "content": "Deep learning for drug discovery. Generative models and molecular property prediction for identifying novel compounds with high binding affinity.",
        "metadata": {"source": "Journal of Medicinal Chemistry", "year": 2023}
    }
]

class SemanticSearchEngine:
    
    def __init__(self, documents):
        self.documents = documents
        print("Loading lightweight embedding model...")
        # Use SMALLER model that fits in 512MB RAM
        self.model = SentenceTransformer('paraphrase-albert-small-v2', device='cpu')
        
        # Create embeddings in batches to save memory
        print("Creating document embeddings (this may take a moment)...")
        doc_texts = [doc['content'] for doc in self.documents]
        
        # Encode in smaller batches to prevent memory issues
        self.embeddings = self.model.encode(doc_texts, 
                                           normalize_embeddings=True,
                                           batch_size=32,  # Process 32 at a time
                                           show_progress_bar=True)
        
        # Create FAISS index
        self.index = faiss.IndexFlatIP(self.embeddings.shape[1])
        self.index.add(self.embeddings.astype(np.float32))
        print(f"✓ Index ready with {len(self.documents)} documents")
    
    def search(self, query, k=8):
        # Get query embedding
        query_embedding = self.model.encode([query], normalize_embeddings=True)
        
        # Search
        scores, indices = self.index.search(query_embedding.astype(np.float32), min(k, len(self.documents)))
        
        # Format results
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1:
                doc = self.documents[idx]
                # Normalize cosine similarity to 0-1
                norm_score = (float(score) + 1) / 2
                results.append({
                    'id': doc['id'],
                    'score': round(norm_score, 4),
                    'content': doc['content'],
                    'metadata': doc['metadata']
                })
        
        return results


# Initialize search engine
print("🚀 Starting Semantic Search API...")
try:
    search_engine = SemanticSearchEngine(SCIENTIFIC_ABSTRACTS)
    print("✓ Ready to accept requests!")
except Exception as e:
    print(f"❌ ERROR INITIALIZING SEARCH ENGINE: {e}")
    import traceback
    traceback.print_exc()
    # Create a dummy search engine as fallback
    class DummySearchEngine:
        def __init__(self):
            self.documents = SCIENTIFIC_ABSTRACTS
        def search(self, query, k=8):
            return []
    search_engine = DummySearchEngine()
    print("⚠️ Using dummy search engine - ML features disabled")

@app.route('/search', methods=['POST'])
def search():
    start_time = time.time()
    
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON payload"}), 400

        query = data.get('query', '')
        k = data.get('k', 8)
        rerank = data.get('rerank', False)
        rerank_k = data.get('rerankK', 5)  # Fixed: capital K

        if not query:
            return jsonify({"error": "Query is required"}), 400
            
        # Rest of your code...
        results = search_engine.search(query, k)
        results = results[:rerank_k]
        
        return jsonify({
            "results": results,
            "reranked": rerank,
            "metrics": {
                "latency": round((time.time() - start_time) * 1000, 2),
                "totalDocs": len(search_engine.documents)
            }
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

        
        # Perform search
        results = search_engine.search(query, k)
        
        # Limit to rerank_k results
        results = results[:rerank_k]
        
        return jsonify({
            "results": results,
            "reranked": rerank,
            "metrics": {
                "latency": round((time.time() - start_time) * 1000, 2),
                "totalDocs": len(search_engine.documents)
            }
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        "status": "✅ Semantic Search API is running!",
        "endpoints": {
            "search": "/search (POST)",
            "health": "/health (GET)",
            "home": "/ (GET)"
        },
        "documents": len(search_engine.documents),
        "version": "1.0.0"
    })

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "healthy",
        "documents": len(search_engine.documents),
        "model": "all-MiniLM-L6-v2"
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"🔌 Attempting to bind to port: {port}")
    app.run(host='0.0.0.0', port=port, debug=False)
