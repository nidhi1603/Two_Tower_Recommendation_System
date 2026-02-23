import os, json
import pandas as pd
import numpy as np

DRIVE_RAW       = "/content/drive/MyDrive/two_tower_data/raw"
DRIVE_PROCESSED = "/content/drive/MyDrive/two_tower_data/processed"
os.makedirs(DRIVE_PROCESSED, exist_ok=True)

def k_core_filter(df, user_col, item_col, k=5):
    while True:
        item_counts = df[item_col].value_counts()
        df = df[df[item_col].isin(item_counts[item_counts >= k].index)]
        user_counts = df[user_col].value_counts()
        df = df[df[user_col].isin(user_counts[user_counts >= k].index)]
        if df[item_col].value_counts().min() >= k and df[user_col].value_counts().min() >= k:
            break
    return df.reset_index(drop=True)

def run():
    reviews = pd.read_parquet(f"{DRIVE_RAW}/amazon_reviews.parquet")
    meta    = pd.read_parquet(f"{DRIVE_RAW}/amazon_metadata.parquet")
    reviews = reviews.rename(columns={"parent_asin": "item_id"})
    filtered = k_core_filter(reviews, "user_id", "item_id", k=5)
    users = sorted(filtered["user_id"].unique())
    items = sorted(filtered["item_id"].unique())
    user2idx = {u: i for i, u in enumerate(users)}
    item2idx = {it: i for i, it in enumerate(items)}
    filtered["user_idx"] = filtered["user_id"].map(user2idx)
    filtered["item_idx"] = filtered["item_id"].map(item2idx)
    filtered = filtered.sort_values(["user_idx", "timestamp"]).reset_index(drop=True)
    train_rows, val_rows, test_rows = [], [], []
    for _, group in filtered.groupby("user_idx"):
        rows = group.to_dict("records")
        if len(rows) < 3:
            train_rows.extend(rows)
        else:
            train_rows.extend(rows[:-2])
            val_rows.append(rows[-2])
            test_rows.append(rows[-1])
    pd.DataFrame(train_rows).to_parquet(f"{DRIVE_PROCESSED}/train.parquet", index=False)
    pd.DataFrame(val_rows).to_parquet(f"{DRIVE_PROCESSED}/val.parquet", index=False)
    pd.DataFrame(test_rows).to_parquet(f"{DRIVE_PROCESSED}/test.parquet", index=False)
    for name, obj in [("user2idx", user2idx), ("item2idx", item2idx)]:
        with open(f"{DRIVE_PROCESSED}/{name}.json", "w") as f:
            json.dump(obj, f)

if __name__ == "__main__":
    run()
