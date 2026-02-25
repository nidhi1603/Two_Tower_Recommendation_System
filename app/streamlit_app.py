import streamlit as st
st.set_page_config(page_title="Two-Tower Recommender", page_icon="🎮", layout="wide")

import numpy as np
import json, pickle, os
import faiss
import gdown
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence

# ============================================================
# FILL IN YOUR GOOGLE DRIVE FILE IDs BELOW
# How to get an ID: Drive → right-click file → Share →
# "Anyone with link" → copy URL → extract the long ID string
# e.g. https://drive.google.com/file/d/1BxiMVs0XRA5nFMd.../view
#                                        ^^^^^^^^^^^^^^^^ this part
# ============================================================
DRIVE_FILES = {
    "tt_user_embs.npy":         "YOUR_FILE_ID_HERE",
    "tt_item_embs.npy":         "YOUR_FILE_ID_HERE",
    "mf_user_embs.npy":         "YOUR_FILE_ID_HERE",
    "mf_item_embs.npy":         "YOUR_FILE_ID_HERE",
    "text_embs.npy":            "YOUR_FILE_ID_HERE",
    "item_info.json":           "YOUR_FILE_ID_HERE",
    "user_history.pkl":         "YOUR_FILE_ID_HERE",
    "tt_cold_start_weights.pt": "YOUR_FILE_ID_HERE",
    "stats.json":               "YOUR_FILE_ID_HERE",
}

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)


# ── Minimal encoder — only used for cold-start ────────────
class ColdStartEncoder(nn.Module):
    def __init__(self, text_dim, emb_dim, gru_hidden, n_user_feats, hidden):
        super().__init__()
        self.text_proj  = nn.Linear(text_dim, emb_dim)
        self.user_gru   = nn.GRU(input_size=emb_dim, hidden_size=gru_hidden,
                                  num_layers=1, batch_first=True)
        self.user_tower = nn.Sequential(
            nn.Linear(gru_hidden + emb_dim + n_user_feats, hidden),
            nn.ReLU(), nn.LayerNorm(hidden),
            nn.Linear(hidden, emb_dim),
        )

    def encode_cold_user(self, item_indices, text_embs_np):
        text_t  = torch.tensor(text_embs_np[item_indices], dtype=torch.float32)
        proj    = self.text_proj(text_t).unsqueeze(0)
        lengths = torch.tensor([len(item_indices)])
        packed  = pack_padded_sequence(proj, lengths, batch_first=True,
                                       enforce_sorted=False)
        _, hidden = self.user_gru(packed)
        gru_out   = hidden.squeeze(0)
        id_emb    = torch.zeros(1, 64)
        feats     = torch.zeros(1, self.user_tower[0].in_features - 64 - gru_out.shape[1])
        x         = torch.cat([gru_out, id_emb, feats], dim=1)
        return torch.nn.functional.normalize(self.user_tower(x), dim=1).detach().numpy()


# ── Load everything (cached — runs once per session) ──────
@st.cache_resource(show_spinner="⬇️ Downloading files from Drive (first run only)...")
def load_all():
    for fname, fid in DRIVE_FILES.items():
        fpath = f"{DATA_DIR}/{fname}"
        if not os.path.exists(fpath):
            gdown.download(f"https://drive.google.com/uc?id={fid}",
                           fpath, quiet=False)

    with open(f"{DATA_DIR}/stats.json") as f:
        stats = json.load(f)

    tt_user  = np.load(f"{DATA_DIR}/tt_user_embs.npy").astype(np.float32)
    tt_item  = np.load(f"{DATA_DIR}/tt_item_embs.npy").astype(np.float32)
    mf_user  = np.load(f"{DATA_DIR}/mf_user_embs.npy").astype(np.float32)
    mf_item  = np.load(f"{DATA_DIR}/mf_item_embs.npy").astype(np.float32)
    text_embs = np.load(f"{DATA_DIR}/text_embs.npy").astype(np.float32)

    # Normalize for cosine similarity via inner product
    def norm(x): return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-9)
    tt_user, tt_item = norm(tt_user), norm(tt_item)
    mf_user, mf_item = norm(mf_user), norm(mf_item)

    # Build FAISS indices
    dim = tt_item.shape[1]
    tt_index = faiss.IndexFlatIP(dim); tt_index.add(tt_item)
    mf_index = faiss.IndexFlatIP(dim); mf_index.add(mf_item)

    with open(f"{DATA_DIR}/item_info.json") as f:
        item_info = {int(k): v for k, v in json.load(f).items()}

    with open(f"{DATA_DIR}/user_history.pkl", "rb") as f:
        user_history = pickle.load(f)

    weights = torch.load(f"{DATA_DIR}/tt_cold_start_weights.pt",
                         map_location="cpu", weights_only=False)
    cold_encoder = ColdStartEncoder(
        text_dim=stats["text_dim"], emb_dim=64, gru_hidden=64,
        n_user_feats=stats["n_user_feats"], hidden=128
    )
    cold_encoder.text_proj.load_state_dict(weights["text_proj"])
    cold_encoder.user_gru.load_state_dict(weights["user_gru"])
    cold_encoder.user_tower.load_state_dict(weights["user_tower"])
    cold_encoder.eval()

    return dict(stats=stats, tt_user=tt_user, tt_item=tt_item,
                mf_user=mf_user, mf_item=mf_item, text_embs=text_embs,
                tt_index=tt_index, mf_index=mf_index,
                item_info=item_info, user_history=user_history,
                cold_encoder=cold_encoder)


# ── Helpers ───────────────────────────────────────────────
def get_item_display(idx, item_info):
    info   = item_info.get(idx, {})
    title  = info.get("title", "Unknown")[:55]
    cat    = info.get("category", "")
    rating = f"⭐ {info['rating']:.1f}" if info.get("rating") else ""
    price  = f"${info['price']:.2f}"   if info.get("price")  else ""
    return title, cat, rating, price

def recommend(query_emb, index, exclude_set, k=10):
    scores, indices = index.search(query_emb.reshape(1, -1), k + len(exclude_set) + 50)
    results = []
    for idx, score in zip(indices[0], scores[0]):
        if int(idx) not in exclude_set and len(results) < k:
            results.append((int(idx), float(score)))
    return results


# ── Load data ─────────────────────────────────────────────
try:
    data = load_all()
except Exception as e:
    st.error(f"Failed to load data: {e}")
    st.info("Check that all Google Drive file IDs are set correctly in DRIVE_FILES.")
    st.stop()


# ── Sidebar ───────────────────────────────────────────────
st.sidebar.title("🎮 Two-Tower RecSys")
st.sidebar.markdown("Amazon Video Games 2023")
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigate", [
    "🏠 Overview",
    "👤 User Recommendations",
    "🆕 Cold-Start Demo",
    "📊 Model Comparison",
])
st.sidebar.markdown("---")
st.sidebar.markdown("Built by [Nidhi](https://github.com/nidhi1603)")
st.sidebar.markdown("[GitHub](https://github.com/nidhi1603/Two_Tower_Recommendation_System)")


# ============================================================
# PAGE 1: OVERVIEW
# ============================================================
if page == "🏠 Overview":
    st.title("🎮 Two-Tower Recommendation System")
    st.markdown("### Deep Learning Recommender with 12-Variant Ablation Study")

    s = data["stats"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Users",    f"{s['n_users']:,}")
    c2.metric("Items",    f"{s['n_items']:,}")
    c3.metric("Sparsity", "99.97%")

    st.markdown("---")
    st.markdown("### Model Performance (Full Ranking vs All Items)")
    r = s["results"]
    st.dataframe({
        "Model":            ["MF (BPR)",       "Two-Tower v5",       "LightGCN"],
        "HR@10":            [r["mf_hr10_full"], r["tt_hr10_full"],    r["lgcn_hr10_full"]],
        "Type":             ["Collaborative",   "Content+Sequential", "Graph Neural Net"],
        "Cold-Start":       ["❌ No",            "✅ Yes",              "❌ No"],
        "FAISS Deployable": ["⚠️ Partial",       "✅ Yes (<1ms)",       "❌ No"],
    }, hide_index=True, use_container_width=True)

    st.markdown("---")
    st.markdown("### Deployment Pipeline")
    st.code("""
Two-Tower (retrieval) → FAISS Index (<1ms) → Top 1000 candidates
                                                      │
                                               Re-ranking layer
                                                      │
                                              Top 10 shown to user
    """)

    st.markdown("---")
    st.markdown("### 12-Variant Ablation Study")
    st.dataframe({
        "Version": ["v1","v2","v3","v4","v4b","v4-BPR","v5","v5b","v5c","v6","v7","v8"],
        "Change":  ["Baseline","MSE distill","Cosine distill","+ Title text",
                    "Batch 1024","BPR+hard neg","+ GRU sequence","+ Rich text",
                    "LightGCN init","Curriculum neg","+ CLIP images","FM-style"],
        "HR@10":   [0.6195,0.6210,0.6195,0.6355,0.6280,0.1520,
                    0.6395,0.6385,0.6330,0.6355,0.6355,0.6305],
        "Verdict": ["baseline","❌ Gradient drowned","❌ Structural blind",
                    "✅ +2.6%","❌ Too many negs","❌ Collapsed","✅ Best",
                    "❌ Noisy text","❌ Scrambled","❌ Hurt perf",
                    "❌ No help","📊 Interpretable"],
    }, hide_index=True, use_container_width=True)


# ============================================================
# PAGE 2: USER RECOMMENDATIONS
# ============================================================
elif page == "👤 User Recommendations":
    st.title("👤 Recommendations for Existing Users")

    c1, c2 = st.columns([1, 2])
    with c1:
        user_id = st.number_input("User ID", min_value=0,
                                   max_value=data["stats"]["n_users"] - 1, value=100)
        model_choice = st.selectbox("Model", ["Two-Tower v5", "Matrix Factorization", "Both"])

    history     = data["user_history"].get(user_id, [])
    history_set = set(history)

    st.markdown(f"### Purchase History ({len(history)} items)")
    if history:
        cols = st.columns(min(5, len(history[:10])))
        for i, item_idx in enumerate(history[:10]):
            with cols[i % len(cols)]:
                title, cat, rating, price = get_item_display(item_idx, data["item_info"])
                st.markdown(f"**{title}**")
                st.caption(f"{cat}  {rating}  {price}")
    else:
        st.info("No history found for this user ID.")

    st.markdown("---")

    if model_choice in ["Two-Tower v5", "Both"]:
        st.markdown("### 🤖 Two-Tower v5 Recommendations")
        recs = recommend(data["tt_user"][user_id], data["tt_index"], history_set)
        for rank, (idx, score) in enumerate(recs, 1):
            title, cat, rating, price = get_item_display(idx, data["item_info"])
            st.markdown(f"**{rank}. {title}** — {cat} {rating} {price} `{score:.3f}`")

    if model_choice in ["Matrix Factorization", "Both"]:
        st.markdown("### 📐 Matrix Factorization Recommendations")
        recs = recommend(data["mf_user"][user_id], data["mf_index"], history_set)
        for rank, (idx, score) in enumerate(recs, 1):
            title, cat, rating, price = get_item_display(idx, data["item_info"])
            st.markdown(f"**{rank}. {title}** — {cat} {rating} {price} `{score:.3f}`")


# ============================================================
# PAGE 3: COLD-START DEMO
# ============================================================
elif page == "🆕 Cold-Start Demo":
    st.title("🆕 Cold-Start: New User Recommendations")
    st.markdown("""
    **This is where Two-Tower shines.** Select games you like and Two-Tower
    recommends more — even though you are a **brand new user with zero purchase history**.
    MF and LightGCN cannot do this at all.
    """)

    all_titles = {idx: data["item_info"].get(idx, {}).get("title", "Unknown")
                  for idx in range(data["stats"]["n_items"])}

    scenario = st.selectbox("Choose a scenario or pick custom games below", [
        "Custom",
        "Souls-like Gamer (Dark Souls, Elden Ring, Sekiro)",
        "Nintendo Fan (Mario, Zelda, Pokemon)",
        "FPS Player (Call of Duty, Battlefield, Halo)",
    ])

    scenario_keywords = {
        "Souls-like Gamer (Dark Souls, Elden Ring, Sekiro)": ["Dark Souls", "Elden Ring", "Sekiro"],
        "Nintendo Fan (Mario, Zelda, Pokemon)":              ["Mario", "Zelda", "Pokemon"],
        "FPS Player (Call of Duty, Battlefield, Halo)":      ["Call of Duty", "Battlefield", "Halo"],
    }

    if scenario == "Custom":
        search_term = st.text_input("Search for a game title:")
        if search_term:
            matches = [(idx, t) for idx, t in all_titles.items()
                       if search_term.lower() in t.lower()][:20]
            if matches:
                selected = st.multiselect("Select games you like:",
                    options=[idx for idx, _ in matches],
                    format_func=lambda x: all_titles[x][:60])
            else:
                st.warning("No games found. Try a different search term.")
                selected = []
        else:
            selected = []
    else:
        selected = []
        for kw in scenario_keywords[scenario]:
            for idx, t in all_titles.items():
                if kw.lower() in t.lower():
                    selected.append(idx)
                    break

    if len(selected) >= 2:
        st.markdown("### Your browsing history:")
        for idx in selected:
            title, cat, rating, price = get_item_display(idx, data["item_info"])
            st.markdown(f"- **{title}** ({cat})  {rating}  {price}")

        st.markdown("---")

        # Encode cold user via GRU over browsed item text embeddings
        cold_emb = data["cold_encoder"].encode_cold_user(selected, data["text_embs"])
        recs = recommend(cold_emb, data["tt_index"], set(selected), k=10)

        st.markdown("### 🤖 Two-Tower Recommendations (Cold-Start via GRU)")
        for rank, (idx, score) in enumerate(recs, 1):
            title, cat, rating, price = get_item_display(idx, data["item_info"])
            st.markdown(f"**{rank}. {title}** — {cat} {rating} {price} `{score:.3f}`")

        st.markdown("---")
        c1, c2 = st.columns(2)
        c1.error("❌ **MF**: No embedding for unseen user — cannot recommend")
        c2.error("❌ **LightGCN**: User not in training graph — cannot recommend")
        st.success("✅ **Two-Tower**: Encodes user from browsed item text via GRU → works instantly")
    else:
        st.info("Select at least 2 games to get recommendations.")


# ============================================================
# PAGE 4: MODEL COMPARISON
# ============================================================
elif page == "📊 Model Comparison":
    st.title("📊 Model Comparison & Analysis")

    tab1, tab2, tab3 = st.tabs(["📈 Performance", "⚡ Latency", "⚖️ Trade-offs"])

    with tab1:
        st.markdown("### Sampled Evaluation (100 negatives)")
        st.markdown("""
| Model | HR@10 | NDCG@10 |
|---|---|---|
| MF (BPR) | 0.6755 | 0.4516 |
| Two-Tower v5 | 0.6395 | 0.4148 |
| LightGCN | **0.7285** | **0.4940** |
        """)
        st.markdown("### Full Ranking (all 26,354 items)")
        st.markdown("""
| Model | HR@5 | HR@10 | HR@20 | NDCG@10 |
|---|---|---|---|---|
| MF | 0.0270 | 0.0420 | 0.0650 | 0.0228 |
| Two-Tower | 0.0190 | 0.0270 | 0.0410 | 0.0125 |
| LightGCN | **0.0300** | **0.0440** | **0.0740** | **0.0227** |
        """)
        st.markdown("### Performance by User Activity")
        st.markdown("""
| User Bucket | MF HR@10 | TT HR@10 | Winner |
|---|---|---|---|
| 5–7 purchases | 0.6780 | 0.6700 | MF |
| 8–15 purchases | 0.6680 | 0.6360 | MF |
| 16–30 purchases | 0.6500 | 0.6220 | MF |
| 31+ purchases | 0.5780 | 0.5340 | MF |
| **New user (0 purchases)** | **IMPOSSIBLE** | **✅ Works** | **Two-Tower** |
        """)

    with tab2:
        st.markdown("### FAISS Retrieval Latency (26,354 items)")
        st.markdown("""
| Index Type | Latency / Query | Queries / Second |
|---|---|---|
| HNSW | **29 μs** | 34,483 |
| IVF (approximate) | 35 μs | 28,571 |
| Flat (exact) | 310 μs | 3,226 |
| Brute-force GPU | 894 μs | 1,119 |
        """)
        st.info("Two-Tower serves **34,000 users/second** on a single CPU core via HNSW.")

    with tab3:
        st.markdown("### The Retrieval-Ranking Trade-off")
        st.markdown("""
| Aspect | Two-Tower | MF | LightGCN |
|---|---|---|---|
| Accuracy (known users) | 3rd | 2nd | **1st** |
| Cold-start (new users) | ✅ Yes | ❌ No | ❌ No |
| Cold-start (new items) | ✅ Yes | ❌ No | ❌ No |
| FAISS deployable | ✅ <1ms | ⚠️ Needs norm | ❌ Needs graph |
| Content-aware | ✅ Text+features | ❌ IDs only | ❌ IDs only |
| Industry role | **Retrieval** (top 1000) | Baseline | **Re-ranking** (top 10) |
        """)

        st.markdown("### FM Gate Weights: What the Model Learned (v8)")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**User representation**")
            st.progress(0.62, text="ID embedding — 62%")
            st.progress(0.28, text="GRU sequence — 28%")
            st.progress(0.10, text="User features — 10%")
        with c2:
            st.markdown("**Item representation**")
            st.progress(0.54, text="ID embedding — 54%")
            st.progress(0.23, text="Text — 23%")
            st.progress(0.22, text="Item features — 22%")
        st.markdown("> On 99.97% sparse data, collaborative ID signals carry **2-3x more weight** than content features. Content helps most at cold-start.")
