# step1_index_pro.py
"""
Build cleaned datasets and a TF-IDF knowledge base for the Network Troubleshooter.
Improvements:
 - Drop duplicates, fillna, utf-8 normalize, whitespace cleaning
 - Save cleaned CSVs to data/cleaned_*.csv
 - Build TF-IDF index and save models/network_index.pkl
"""

import pandas as pd
import os
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

DATA_PATH = "data"
CLEAN_PATH = os.path.join(DATA_PATH, "cleaned")
os.makedirs(CLEAN_PATH, exist_ok=True)
os.makedirs("models", exist_ok=True)

src_tech_path = os.path.join(DATA_PATH, "src_tech_records.csv")
src_incident_path = os.path.join(DATA_PATH, "src_incident_records.csv")
meta_tech_path = os.path.join(DATA_PATH, "metadata_tech_records.csv")
meta_incident_path = os.path.join(DATA_PATH, "metadata_incident_records.csv")

for p in [src_tech_path, src_incident_path, meta_tech_path, meta_incident_path]:
    if not os.path.exists(p):
        raise FileNotFoundError(f"Missing required file: {p}")

def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.drop_duplicates(inplace=True)
    df.fillna("", inplace=True)
    # normalize text columns
    for col in df.select_dtypes(include=["object"]).columns:
        # ensure string, remove control characters, normalize whitespace
        df[col] = (
            df[col]
            .astype(str)
            .apply(lambda s: s.encode("utf-8", "ignore").decode("utf-8", errors="ignore"))
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
        )
    return df

print("🔄 Loading CSVs...")
src_tech = basic_clean(pd.read_csv(src_tech_path))
src_incident = basic_clean(pd.read_csv(src_incident_path))
meta_tech = basic_clean(pd.read_csv(meta_tech_path))
meta_incident = basic_clean(pd.read_csv(meta_incident_path))

# Save cleaned copies for traceability
src_tech.to_csv(os.path.join(CLEAN_PATH, "cleaned_src_tech_records.csv"), index=False)
src_incident.to_csv(os.path.join(CLEAN_PATH, "cleaned_src_incident_records.csv"), index=False)
meta_tech.to_csv(os.path.join(CLEAN_PATH, "cleaned_metadata_tech_records.csv"), index=False)
meta_incident.to_csv(os.path.join(CLEAN_PATH, "cleaned_metadata_incident_records.csv"), index=False)
print("✅ Cleaned CSVs saved to", CLEAN_PATH)

# Compose combined records for TF-IDF indexing and optional QA tests
records = []
def segment_text(text, max_len=1000):
    # naive segmentation: split text into chunks if longer than max_len chars
    text = text or ""
    if len(text) <= max_len:
        return [text]
    chunks = []
    i = 0
    while i < len(text):
        chunks.append(text[i:i+max_len])
        i += max_len
    return chunks

# src_incident: use ProblemDescription if present
for _, row in src_incident.iterrows():
    problem = str(row.get("ProblemDescription", "")).strip()
    if not problem:
        continue
    for seg in segment_text(problem):
        records.append({
            "source": "Incident Ticket",
            "problem": seg,
            "solution": "",
            "tags": row.get("Tags", "")
        })

# src_tech: step_description or StepDescription
for _, row in src_tech.iterrows():
    step_desc = str(row.get("step_description", row.get("StepDescription", ""))).strip()
    if not step_desc:
        continue
    for seg in segment_text(step_desc):
        records.append({
            "source": "Technical Steps",
            "problem": "",
            "solution": seg,
            "tags": row.get("TechnicalTags", "")
        })

# metadata tech
for _, row in meta_tech.iterrows():
    product = str(row.get("ProductInformation", "")).strip()
    solution = str(row.get("SolutionSteps", "")).strip()
    tags = str(row.get("TechnicalTags", "")).strip()
    combined = f"Product Info: {product}\nSolution: {solution}\nTags: {tags}"
    for seg in segment_text(combined):
        records.append({
            "source": "Tech Metadata",
            "problem": product,
            "solution": solution,
            "tags": tags
        })

# metadata incident
for _, row in meta_incident.iterrows():
    product = str(row.get("ProductInformation", "")).strip()
    solution = str(row.get("SolutionDetails", "")).strip()
    tags = str(row.get("Tags", "")).strip()
    combined = f"Product Info: {product}\nSolution: {solution}\nTags: {tags}"
    for seg in segment_text(combined):
        records.append({
            "source": "Incident Metadata",
            "problem": product,
            "solution": solution,
            "tags": tags
        })

df = pd.DataFrame(records)
print(f"✅ Prepared {len(df)} combined records for indexing.")

# Build TF-IDF (keeps for local diagnostics / fallback)
vectorizer = TfidfVectorizer(stop_words="english", max_features=5000, max_df=0.85, min_df=1)
X = vectorizer.fit_transform((df["problem"].fillna("") + " " + df["solution"].fillna("") + " " + df["tags"].fillna("")).values.astype(str))

with open("models/network_index.pkl", "wb") as f:
    pickle.dump({"tfidf": X, "vectorizer": vectorizer, "data": df}, f)

print("✅ TF-IDF index saved to models/network_index.pkl")
print("\nSample records:")
print(df.sample(min(5, len(df)))[["source", "problem", "solution"]])
