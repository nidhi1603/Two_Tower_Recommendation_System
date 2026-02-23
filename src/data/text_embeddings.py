import json, numpy as np, pandas as pd
from sentence_transformers import SentenceTransformer

def generate_text_embeddings(raw_dir, processed_dir):
    meta_df = pd.read_parquet(f"{raw_dir}/amazon_metadata.parquet")
    with open(f"{processed_dir}/item2idx.json") as f:
        item2idx = json.load(f)
    lookup = dict(zip(meta_df["parent_asin"], meta_df["title"].fillna("Unknown")))
    titles = [lookup.get(a, "Unknown") for a, _ in sorted(item2idx.items(), key=lambda x: x[1])]
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embs = model.encode(titles, batch_size=512, show_progress_bar=True, normalize_embeddings=True)
    np.save(f"{processed_dir}/item_text_embeddings.npy", embs)
    return embs
