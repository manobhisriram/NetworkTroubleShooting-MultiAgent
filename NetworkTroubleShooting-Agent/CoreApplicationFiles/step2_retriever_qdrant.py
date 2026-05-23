# step2_retriever_qdrant.py
"""
Agent 1 — Retriever using Qdrant Cloud (with SentenceTransformer embeddings)
Env vars:
 - QDRANT_URL
 - QDRANT_API_KEY
 - QDRANT_COLLECTION (optional, default 'network_issues')
"""

import os

from dotenv import load_dotenv
load_dotenv()

import pandas as pd
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models
from sentence_transformers import SentenceTransformer

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "network_issues")

if not QDRANT_URL or not QDRANT_API_KEY:
    print("⚠️ Qdrant credentials not found in environment. Set QDRANT_URL and QDRANT_API_KEY.")
    client = None
else:
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

print("🧠 Loading SentenceTransformer model...")
model = SentenceTransformer("all-MiniLM-L6-v2")
VECTOR_SIZE = model.get_sentence_embedding_dimension()

# Data files
DATA_FILES = [
    "data/cleaned/cleaned_src_tech_records.csv",
    "data/cleaned/cleaned_src_incident_records.csv",
    "data/cleaned/cleaned_metadata_tech_records.csv",
    "data/cleaned/cleaned_metadata_incident_records.csv",
]

def load_data():
    """Load and merge all data files with proper field mapping."""
    dfs = []
    
    # Try to read cleaned files
    for p in DATA_FILES:
        if os.path.exists(p):
            df_temp = pd.read_csv(p)
            print(f"✅ Loaded {p} with {len(df_temp)} records")
            dfs.append(df_temp)
    
    if not dfs:
        # fallback: attempt to read original dataset files
        candidates = [
            "data/src_tech_records.csv",
            "data/src_incident_records.csv",
            "data/metadata_tech_records.csv",
            "data/metadata_incident_records.csv",
        ]
        dfs = [pd.read_csv(p) for p in candidates if os.path.exists(p)]
    
    if not dfs:
        raise FileNotFoundError("No data files found in data/ or data/cleaned/.")
    
    # Process each dataframe based on its structure
    processed_dfs = []
    
    for df in dfs:
        df_copy = df.copy()
        
        # Identify file type and map fields accordingly
        if 'ProblemDescription' in df_copy.columns and 'SolutionDetails' not in df_copy.columns:
            # This is src_incident_records - has problem but no solution
            df_copy['problem_text'] = df_copy['ProblemDescription'].fillna("")
            df_copy['solution_text'] = ""
            df_copy['source'] = 'incident_record'
            
        elif 'step_description' in df_copy.columns:
            # This is src_tech_records - has solution steps but no problem
            df_copy['problem_text'] = ""
            df_copy['solution_text'] = df_copy['step_description'].fillna("")
            df_copy['source'] = 'tech_record'
            
        elif 'SolutionDetails' in df_copy.columns:
            # This is metadata_incident_records - has both problem and solution
            if 'ProblemDescription' in df_copy.columns:
                df_copy['problem_text'] = df_copy['ProblemDescription'].fillna("")
            elif 'ProductInformation' in df_copy.columns:
                df_copy['problem_text'] = df_copy['ProductInformation'].fillna("")
            else:
                df_copy['problem_text'] = ""
            df_copy['solution_text'] = df_copy['SolutionDetails'].fillna("")
            df_copy['source'] = 'metadata_incident'
            
        elif 'SolutionSteps' in df_copy.columns:
            # This is metadata_tech_records - has solution steps
            if 'ProductInformation' in df_copy.columns:
                df_copy['problem_text'] = df_copy['ProductInformation'].fillna("")
            else:
                df_copy['problem_text'] = ""
            df_copy['solution_text'] = df_copy['SolutionSteps'].fillna("")
            df_copy['source'] = 'metadata_tech'
            
        else:
            # Fallback - try to infer from column names
            problem_cols = [c for c in df_copy.columns if any(x in c.lower() for x in ['description', 'problem', 'issue'])]
            solution_cols = [c for c in df_copy.columns if any(x in c.lower() for x in ['solution', 'steps', 'detail'])]
            
            if problem_cols:
                df_copy['problem_text'] = df_copy[problem_cols].fillna("").astype(str).agg(" ".join, axis=1).str.strip()
            else:
                df_copy['problem_text'] = ""
                
            if solution_cols:
                df_copy['solution_text'] = df_copy[solution_cols].fillna("").astype(str).agg(" ".join, axis=1).str.strip()
            else:
                df_copy['solution_text'] = ""
                
            df_copy['source'] = 'unknown'
        
        # Standardize product_id and doc_id fields
        if 'ProductID' in df_copy.columns:
            df_copy['product_id'] = df_copy['ProductID'].fillna("").astype(str)
        elif 'productid' in df_copy.columns:
            df_copy['product_id'] = df_copy['productid'].fillna("").astype(str)
        else:
            df_copy['product_id'] = ""
        
        if 'DocID' in df_copy.columns:
            df_copy['doc_id'] = df_copy['DocID'].fillna("").astype(str)
        elif 'docid' in df_copy.columns:
            df_copy['doc_id'] = df_copy['docid'].fillna("").astype(str)
        else:
            df_copy['doc_id'] = ""
        
        processed_dfs.append(df_copy)
    
    # Merge all dataframes
    df = pd.concat(processed_dfs, ignore_index=True, sort=False)
    
    # Clean up text fields
    df['problem_text'] = df['problem_text'].fillna("").astype(str).str.strip()
    df['solution_text'] = df['solution_text'].fillna("").astype(str).str.strip()
    
    # Remove rows where both problem and solution are empty
    df = df[(df['problem_text'] != "") | (df['solution_text'] != "")]
    
    df = df.reset_index(drop=True)
    df['__id'] = df.index
    
    print(f"✅ Loaded and merged {len(df)} total records")
    print(f"   - Records with problem_text: {(df['problem_text'] != '').sum()}")
    print(f"   - Records with solution_text: {(df['solution_text'] != '').sum()}")
    print(f"   - Records with both: {((df['problem_text'] != '') & (df['solution_text'] != '')).sum()}")
    
    return df

def setup_qdrant_collection():
    if client is None:
        print("⚠️ Qdrant client not initialized; skipping collection setup.")
        return
    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME not in existing:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(size=VECTOR_SIZE, distance=models.Distance.COSINE),
        )
        print(f"✅ Created collection: {COLLECTION_NAME}")
    else:
        print(f"✅ Collection '{COLLECTION_NAME}' exists.")

def index_data():
    df = load_data()
    # Combine problem and solution for embedding
    texts = (df["problem_text"].fillna("") + " " + df["solution_text"].fillna("")).tolist()
    print("⚙️ Generating embeddings locally...")
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    
    payloads = []
    for _, row in df.iterrows():
        payloads.append({
            "problem_text": row.get("problem_text", ""),
            "solution_text": row.get("solution_text", ""),
            "source": row.get("source", ""),
            "product_id": row.get("product_id", ""),
            "doc_id": row.get("doc_id", "")
        })

    if client is None:
        # Save locally for offline testing
        os.makedirs("models", exist_ok=True)
        np.save("models/embeddings.npy", embeddings)
        df.to_pickle("models/retriever_payloads.pkl")
        print("⚠️ Qdrant disabled — embeddings saved to models/ for offline testing.")
        return

    print("🚀 Uploading to Qdrant Cloud...")
    client.upsert(
        collection_name=COLLECTION_NAME,
        points=models.Batch(
            ids=df["__id"].tolist(),
            vectors=embeddings.tolist(),
            payloads=payloads,
        ),
    )
    print(f"✅ Indexed {len(df)} records into Qdrant Cloud: {COLLECTION_NAME}")

def retrieve_qdrant(query_text: str, limit: int = 5):
    """
    Return list of dicts: rank, score, problem, solution, source, product_id, doc_id
    """
    try:
        if client is None:
            # offline mode: load embeddings & do cosine search locally
            import numpy as np
            emb = model.encode([query_text])[0]
            stored = np.load("models/embeddings.npy")
            # cosine similarity
            sims = (stored @ emb) / (np.linalg.norm(stored, axis=1) * np.linalg.norm(emb) + 1e-12)
            idxs = sims.argsort()[::-1][:limit]
            results = []
            payload_df = pd.read_pickle("models/retriever_payloads.pkl")
            for rank, i in enumerate(idxs, start=1):
                results.append({
                    "rank": rank,
                    "score": float(sims[i]),
                    "problem": payload_df.iloc[i].get("problem_text", ""),
                    "solution": payload_df.iloc[i].get("solution_text", ""),
                    "source": payload_df.iloc[i].get("source", ""),
                    "product_id": payload_df.iloc[i].get("product_id", ""),
                    "doc_id": payload_df.iloc[i].get("doc_id", "")
                })
            return results

        # normal Qdrant query
        q_vec = model.encode([query_text])[0].tolist()
        resp = client.search(collection_name=COLLECTION_NAME, query_vector=q_vec, limit=limit)
        results = []
        for rank, r in enumerate(resp, start=1):
            payload = r.payload or {}
            score = getattr(r, "score", None)
            score = float(score) if score is not None else 0.0
            results.append({
                "rank": rank,
                "score": score,
                "problem": payload.get("problem_text", ""),
                "solution": payload.get("solution_text", ""),
                "source": payload.get("source", ""),
                "product_id": payload.get("product_id", ""),
                "doc_id": payload.get("doc_id", "")
            })
        
        if not results:
            results.append({
                "rank": 0, "score": 0.0,
                "problem": "No similar records found",
                "solution": "Please try rephrasing your query or contact support.",
                "source": "N/A",
                "product_id": "",
                "doc_id": ""
            })
        return results
    except Exception as e:
        print("❌ Retrieval error:", e)
        return [{
            "rank": 0, "score": 0.0,
            "problem": "Error during retrieval",
            "solution": str(e),
            "source": "Error",
            "product_id": "",
            "doc_id": ""
        }]

# small alias for imports
retrieve_from_qdrant = retrieve_qdrant

if __name__ == "__main__":
    # index then test
    setup_qdrant_collection()
    index_data()
    print("\n" + "="*60)
    print("Testing retrieval with sample query...")
    print("="*60)
    results = retrieve_qdrant("router not assigning IP address")
    for r in results:
        print(f"\nRank {r['rank']} | Score: {r['score']:.3f}")
        print(f"Problem: {r['problem'][:100]}...")
        print(f"Solution: {r['solution'][:100]}...")
        print(f"Source: {r['source']} | Product: {r['product_id']}")