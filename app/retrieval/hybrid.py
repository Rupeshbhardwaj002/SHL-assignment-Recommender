# FAISS + BM25 search logic
import json
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import os

# --- Configurations ---
FAISS_INDEX_PATH = "data/index.faiss"
BM25_INDEX_PATH = "data/bm25_index.pkl"
DOC_MAPPING_PATH = "data/doc_mapping.json"
MODEL_NAME = "BAAI/bge-small-en-v1.5"

# --- Global State for FastAPI ---
# We load these ONCE when the server starts, not on every request!
# This is critical to beat the 30-second timeout limit.
print("Loading Hybrid Retrieval Engine...")
try:
    embedder = SentenceTransformer(MODEL_NAME)
    faiss_index = faiss.read_index(FAISS_INDEX_PATH)
    
    with open(BM25_INDEX_PATH, 'rb') as f:
        bm25_index = pickle.load(f)
        
    with open(DOC_MAPPING_PATH, 'r', encoding='utf-8') as f:
        # JSON keys are strings, convert them back to integers for mapping
        doc_mapping = {int(k): v for k, v in json.load(f).items()}
        
    print("Hybrid Retrieval Engine Ready!")
except Exception as e:
    print(f"Warning: Indices not found or failed to load. Run build_index.py first. Error: {e}")

def normalize_scores(scores):
    """Min-Max normalization to put scores on a 0.0 to 1.0 scale."""
    if len(scores) == 0:
        return scores
    min_val = np.min(scores)
    max_val = np.max(scores)
    if max_val == min_val:
        return np.ones_like(scores)
    return (scores - min_val) / (max_val - min_val)

def hybrid_search(query: str, top_k: int = 15):
    """
    Executes a hybrid search combining FAISS (dense) and BM25 (sparse).
    Returns a list of the top catalog items.
    """
    # 1. FAISS (Dense/Semantic Search)
    query_vector = embedder.encode([query])
    query_vector = np.array(query_vector).astype('float32')
    faiss.normalize_L2(query_vector)
    
    # Get top 30 to allow room for re-ranking
    dense_scores, dense_indices = faiss_index.search(query_vector, 30)
    dense_scores = dense_scores[0]
    dense_indices = dense_indices[0]
    
    # 2. BM25 (Sparse/Keyword Search)
    tokenized_query = query.lower().split(" ")
    sparse_scores = bm25_index.get_scores(tokenized_query)
    
    # 3. Combine Scores (The Secret to High Recall@10)
    # We only care about the documents retrieved by FAISS for this fast re-ranking
    combined_results = []
    
    # Normalize dense scores
    norm_dense = normalize_scores(dense_scores)
    
    # Extract and normalize corresponding sparse scores
    selected_sparse = np.array([sparse_scores[idx] for idx in dense_indices])
    norm_sparse = normalize_scores(selected_sparse)
    
    # Weighting: 60% Semantic meaning, 40% Exact Keyword match
    for i, doc_idx in enumerate(dense_indices):
        final_score = (0.60 * norm_dense[i]) + (0.40 * norm_sparse[i])
        combined_results.append((final_score, doc_idx))
        
    # Sort by final hybrid score descending
    combined_results.sort(key=lambda x: x[0], reverse=True)
    
    # 4. Format Output
    final_docs = []
    for score, doc_idx in combined_results[:top_k]:
        item = doc_mapping[doc_idx]
        final_docs.append(item)
        
    return final_docs

# Quick test if you run this file directly
if __name__ == "__main__":
    test_query = "Java backend developer with stakeholder skills"
    print(f"\nTesting Query: '{test_query}'")
    results = hybrid_search(test_query, top_k=3)
    for i, res in enumerate(results):
        print(f"{i+1}. {res.get('name')} (Type: {res.get('test_type')})")