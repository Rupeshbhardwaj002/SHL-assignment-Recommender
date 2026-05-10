# Converts catalog.json into FAISS and BM25 indices
import json
import os
import pickle
import numpy as np
import faiss
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

CATALOG_PATH = "data/catalog.json"
FAISS_INDEX_PATH = "data/index.faiss"
BM25_INDEX_PATH = "data/bm25_index.pkl"
DOC_MAPPING_PATH = "data/doc_mapping.json"

# Using the highly-rated, lightweight BAAI embedding model
MODEL_NAME = "BAAI/bge-small-en-v1.5"

def prepare_searchable_text(item):
    """
    Extracts all useful text from a catalog item to make it searchable.
    Since we don't know the exact keys SHL uses, this safely grabs values.
    """
    text_parts = []
    
    # Common keys we expect to see
    for key in ['name', 'test_type', 'description', 'category', 'family', 'competencies', 'skills']:
        val = item.get(key, "")
        if isinstance(val, str) and val:
            text_parts.append(val)
        elif isinstance(val, list):
            text_parts.extend([str(v) for v in val if v])
            
    # Fallback if the keys are weird: just dump the whole dict as a string
    if not text_parts:
        return str(item)
        
    return " ".join(text_parts).lower()

def build_indices():
    print(f"Loading {CATALOG_PATH}...")
    with open(CATALOG_PATH, 'r', encoding='utf-8') as f:
        catalog = json.load(f)
        
    print(f"Loaded {len(catalog)} assessments. Preparing documents...")
    
    documents = []
    doc_mapping = {}
    
    for i, item in enumerate(catalog):
        doc_text = prepare_searchable_text(item)
        documents.append(doc_text)
        # Save a reference so we can link the search result back to the exact URL/Name
        doc_mapping[i] = item 

    # 1. BUILD BM25 INDEX (Keyword Search)
    print("Building BM25 sparse index...")
    tokenized_corpus = [doc.split(" ") for doc in documents]
    bm25 = BM25Okapi(tokenized_corpus)
    
    with open(BM25_INDEX_PATH, 'wb') as f:
        pickle.dump(bm25, f)
    print("BM25 index saved.")

    # 2. BUILD FAISS INDEX (Semantic Search)
    print(f"Downloading/Loading embedding model '{MODEL_NAME}'... (This may take a minute on the first run)")
    model = SentenceTransformer(MODEL_NAME)
    
    print("Encoding documents into vectors...")
    embeddings = model.encode(documents, show_progress_bar=True)
    embeddings = np.array(embeddings).astype('float32')
    
    # Normalize vectors for cosine similarity
    faiss.normalize_L2(embeddings)
    
    # Create the FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension) # Inner Product == Cosine Similarity for normalized vectors
    index.add(embeddings)
    
    faiss.write_index(index, FAISS_INDEX_PATH)
    print("FAISS index saved.")
    
    # 3. SAVE DOCUMENT MAPPING
    with open(DOC_MAPPING_PATH, 'w', encoding='utf-8') as f:
        json.dump(doc_mapping, f, indent=4)
    print("Document mapping saved.")
    
    print("\n✅ All search indices successfully built!")

if __name__ == "__main__":
    build_indices()