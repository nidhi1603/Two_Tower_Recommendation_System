import streamlit as st
st.set_page_config(
    page_title="Deep Learning Recommendation System",
    layout="wide",
    initial_sidebar_state="expanded"
)

import numpy as np
import json, pickle, os
import faiss
import gdown
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
import plotly.graph_objects as go

# ── Custom CSS ────────────────────────────────────────────
st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
    .metric-card {
        background: #1e1e2e;
        border: 1px solid #313244;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
    }
    .metric-value { font-size: 2rem; font-weight: 700; color: #cba6f7; }
    .metric-label { font-size: 0.8rem; color: #a6adc8; margin-top: 4px; }
    .section-header {
        font-size: 1.1rem;
        font-weight: 600;
        color: #cdd6f4;
        border-left: 3px solid #cba6f7;
        padding-left: 10px;
        margin: 1.5rem 0 1rem 0;
    }
    .insight-box {
        background: #1e1e2e;
        border: 1px solid #a6e3a1;
        border-radius: 8px;
        padding: 12px 16px;
        margin: 8px 0;
    }
    /* ── Architecture diagram styles ── */
    .arch-container { display:flex; flex-direction:column; align-items:center; gap:8px; padding:16px 0; }
    .arch-row { display:flex; justify-content:center; gap:24px; width:100%; }
    .arch-box {
        background: #313244;
        border: 2px solid #6c7086;
        border-radius: 10px;
        padding: 14px 22px;
        text-align: center;
        transition: all 0.3s ease;
        cursor: default;
        position: relative;
    }
    .arch-box:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.4);
    }
    .arch-box .box-title {
        font-size: 1.05rem;
        font-weight: 700;
        margin-bottom: 4px;
    }
    .arch-box .box-detail {
        font-size: 0.82rem;
        color: #a6adc8;
        max-height: 0;
        overflow: hidden;
        transition: max-height 0.3s ease, opacity 0.3s ease;
        opacity: 0;
    }
    .arch-box:hover .box-detail {
        max-height: 80px;
        opacity: 1;
    }
    .arch-box.purple { border-color: #cba6f7; }
    .arch-box.purple .box-title { color: #cba6f7; }
    .arch-box.blue { border-color: #89b4fa; }
    .arch-box.blue .box-title { color: #89b4fa; }
    .arch-box.green { border-color: #a6e3a1; }
    .arch-box.green .box-title { color: #a6e3a1; }
    .arch-box.yellow { border-color: #f9e2af; }
    .arch-box.yellow .box-title { color: #f9e2af; }
    .arch-box.red { border-color: #f38ba8; }
    .arch-box.red .box-title { color: #f38ba8; }
    .arch-box.peach { border-color: #fab387; }
    .arch-box.peach .box-title { color: #fab387; }
    .arch-arrow {
        font-size: 1.4rem;
        color: #6c7086;
        text-align: center;
        line-height: 1;
    }
    .arch-arrow.side { writing-mode: horizontal-tb; }
    .arch-label {
        font-size: 0.78rem;
        color: #585b70;
        text-align: center;
        font-style: italic;
    }
    .tower-container {
        display: flex;
        justify-content: center;
        gap: 40px;
        width: 100%;
    }
    .tower {
        border: 2px solid #6c7086;
        border-radius: 14px;
        padding: 18px 14px;
        display: flex;
        flex-direction: column;
        align-items: center;
        gap: 8px;
        min-width: 220px;
        background: rgba(49,50,68,0.3);
    }
    .tower.user { border-color: #cba6f7; }
    .tower.item { border-color: #89b4fa; }
    .tower-title {
        font-size: 1.1rem;
        font-weight: 700;
        margin-bottom: 4px;
    }
    .tower.user .tower-title { color: #cba6f7; }
    .tower.item .tower-title { color: #89b4fa; }
    .tower .arch-box { width: 100%; }
</style>
""", unsafe_allow_html=True)

# ── Drive file IDs ────────────────────────────────────────
DRIVE_FILES = {
    "tt_user_embs.npy":         "1NcZJZNI3JNd4gmpNsOEaM8NkmaLWsk3N",
    "tt_item_embs.npy":         "1uHA2UoTkDxijR8lKgw8ACTWFXayc2jSx",
    "mf_user_embs.npy":         "1ilhAQnLmH65N0DKjCSIG1f2xO7wAX8SZ",
    "mf_item_embs.npy":         "1yy1xOrvBlf7gzeRpy6wd7QZj7hM7KulM",
    "text_embs.npy":            "14_dyDwcV40J8ZaVB6ogZXysnXSW72smM",
    "item_info.json":           "1MtRFoznpLMPZV7DWTmNtKWQ4znbUSMOD",
    "user_history.pkl":         "14z-VcI0r7ZcU9WqJY7co11MY3pVseziu",
    "tt_cold_start_weights.pt": "1PgUoB8U1d9K05OdCsAiEO1TCEjObfdFH",
    "stats.json":               "17To2CT7k2wINECILMqd7O-QWIQEPZT49",
}

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)


# ── Cold-start encoder ────────────────────────────────────
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
        gru_out = hidden.squeeze(0)
        id_emb  = torch.zeros(1, 64)
        feats   = torch.zeros(1, self.user_tower[0].in_features - 64 - gru_out.shape[1])
        x       = torch.cat([gru_out, id_emb, feats], dim=1)
        return F.normalize(self.user_tower(x), dim=1).detach().numpy()


# ── Load all data ─────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model files...")
def load_all():
    for fname, fid in DRIVE_FILES.items():
        fpath = f"{DATA_DIR}/{fname}"
        if not os.path.exists(fpath):
            gdown.download(f"https://drive.google.com/uc?id={fid}",
                           fpath, quiet=False)

    with open(f"{DATA_DIR}/stats.json") as f:
        stats = json.load(f)

    tt_user   = np.load(f"{DATA_DIR}/tt_user_embs.npy").astype(np.float32)
    tt_item   = np.load(f"{DATA_DIR}/tt_item_embs.npy").astype(np.float32)
    mf_user   = np.load(f"{DATA_DIR}/mf_user_embs.npy").astype(np.float32)
    mf_item   = np.load(f"{DATA_DIR}/mf_item_embs.npy").astype(np.float32)
    text_embs = np.load(f"{DATA_DIR}/text_embs.npy").astype(np.float32)

    def norm(x): return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-9)
    tt_user, tt_item = norm(tt_user), norm(tt_item)
    mf_user, mf_item = norm(mf_user), norm(mf_item)

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
    rating = f"  {info['rating']:.1f} stars" if info.get("rating") else ""
    price  = f"  ${info['price']:.2f}"       if info.get("price")  else ""
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
    st.stop()

s = data["stats"]


# ── Sidebar navigation ──────────────────────────────────
st.sidebar.title("Presentation Flow")
page = st.sidebar.radio("Navigate to:", [
    "1. Overview & Dataset",
    "2. Model Architectures",
    "3. Live Demo",
    "4. Results & Analysis",
    "5. Key Findings",
], label_visibility="collapsed")


# ============================================================
# PAGE 1: OVERVIEW & DATASET
# ============================================================
if page == "1. Overview & Dataset":
    st.markdown("## Deep Learning Recommendation System")
    st.markdown("*Four models, 12 ablation variants — what actually improves retrieval on sparse data?*")
    st.markdown("---")

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.markdown('<div class="metric-card"><div class="metric-value">98,906</div><div class="metric-label">Users</div></div>', unsafe_allow_html=True)
    c2.markdown('<div class="metric-card"><div class="metric-value">26,354</div><div class="metric-label">Items</div></div>', unsafe_allow_html=True)
    c3.markdown('<div class="metric-card"><div class="metric-value">659K</div><div class="metric-label">Interactions</div></div>', unsafe_allow_html=True)
    c4.markdown('<div class="metric-card"><div class="metric-value">99.97%</div><div class="metric-label">Sparsity</div></div>', unsafe_allow_html=True)
    c5.markdown('<div class="metric-card"><div class="metric-value">4</div><div class="metric-label">Models</div></div>', unsafe_allow_html=True)
    c6.markdown('<div class="metric-card"><div class="metric-value">12</div><div class="metric-label">Ablation Variants</div></div>', unsafe_allow_html=True)

    st.markdown("")
    left, right = st.columns([1, 1], gap="large")

    with left:
        st.markdown('<div class="section-header">The Problem</div>', unsafe_allow_html=True)
        st.markdown("""
        **Goal:** Recommend video games to users on Amazon.

        **Challenge:** 99.97% of the user-item matrix is empty.
        Most users have only 5-7 purchases. Most items have few reviews.

        **Questions we answer:**
        - Does graph structure (who bought what) beat content features (text, price)?
        - Can we serve brand-new users with zero purchase history?
        - Can we serve recommendations in microseconds?
        """)

        st.markdown('<div class="section-header">Data Pipeline</div>', unsafe_allow_html=True)
        st.markdown("""
        1. **Download** Amazon Video Games 2023 from HuggingFace
        2. **K-core filter** (k=5) — keep users & items with ≥5 interactions
        3. **Feature engineering** — 8 user features, 15 item features
        4. **Text embeddings** — item titles via SentenceTransformer (384d)
        5. **Leave-last-2-out** split — chronological per user
        """)

    with right:
        st.markdown('<div class="section-header">Four Models at a Glance</div>', unsafe_allow_html=True)
        st.dataframe({
            "Model":      ["MF (BPR)", "LightGCN", "Two-Tower v5", "FG-LightGCN"],
            "HR@10":      [0.6825, 0.7290, 0.6395, 0.7190],
            "Type":       ["Collaborative", "Graph NN", "Dual Encoder", "Graph + Features"],
            "Cold-Start": ["No", "No", "Yes", "No"],
            "FAISS":      ["Partial", "No", "Yes (<1ms)", "No"],
            "Role":       ["Baseline", "Re-ranking", "Retrieval", "Research"],
        }, hide_index=True, use_container_width=True)

        st.markdown('<div class="section-header">Why These Four?</div>', unsafe_allow_html=True)
        st.markdown("""
        | Model | Purpose |
        |---|---|
        | **MF** | Baseline — if you can't beat this, complexity isn't justified |
        | **LightGCN** | Best accuracy — multi-hop graph captures collaborative patterns |
        | **Two-Tower** | Production — cold-start + FAISS serving at 29μs |
        | **FG-LightGCN** | My contribution — tests if features improve graph models |
        """)


# ============================================================
# PAGE 2: MODEL ARCHITECTURES
# ============================================================
elif page == "2. Model Architectures":
    st.markdown("## Model Architectures")
    st.markdown("---")

    model_tab = st.tabs(["Matrix Factorization", "LightGCN", "Two-Tower", "Feature-Gated LightGCN (New)", "FAISS Serving"])

    # ── MF ──
    with model_tab[0]:
        a1, a2 = st.columns([1, 1], gap="large")
        with a1:
            st.markdown("#### Matrix Factorization — The Baseline")
            st.markdown("""
            Each user and item gets a **64-dimensional embedding**.
            Score = dot product of the two vectors.

            Trained with **BPR loss** (Bayesian Personalized Ranking):
            push the score of a purchased item above a random negative.

            ```
            score(u, i) = dot(e_user, e_item)
            loss = -log sigmoid(score(u, pos) - score(u, neg))
            ```

            **No features. No graph. No text.** Just ID co-occurrence.
            If a fancier model can't beat this, the added complexity isn't worth it.
            """)
        with a2:
            st.markdown("""
            <div class="arch-container">
                <div class="arch-row">
                    <div class="arch-box purple" style="min-width:180px;">
                        <div class="box-title">User Embedding</div>
                        <div style="color:#cdd6f4;font-size:0.9rem;">64 dimensions</div>
                        <div class="box-detail">98,906 learnable vectors — one per user</div>
                    </div>
                    <div class="arch-box blue" style="min-width:180px;">
                        <div class="box-title">Item Embedding</div>
                        <div style="color:#cdd6f4;font-size:0.9rem;">64 dimensions</div>
                        <div class="box-detail">26,354 learnable vectors — one per item</div>
                    </div>
                </div>
                <div class="arch-arrow">&#8595; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; &#8595;</div>
                <div class="arch-row">
                    <div class="arch-box green" style="min-width:200px;">
                        <div class="box-title">Dot Product &rarr; Score</div>
                        <div class="box-detail">Higher score = more likely to purchase</div>
                    </div>
                </div>
                <div class="arch-arrow">&#8595;</div>
                <div class="arch-row">
                    <div class="arch-box peach" style="min-width:280px;">
                        <div class="box-title">BPR Loss</div>
                        <div style="color:#cdd6f4;font-size:0.85rem;">-log &sigma;(s<sub>pos</sub> - s<sub>neg</sub>)</div>
                        <div class="box-detail">Push purchased item score above random negative</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            st.metric("HR@10", "0.6825", help="Sampled evaluation with 100 negatives")

    # ── LightGCN ──
    with model_tab[1]:
        b1, b2 = st.columns([1, 1], gap="large")
        with b1:
            st.markdown("#### LightGCN — Graph Neural Network")
            st.markdown("""
            Users and items form a **bipartite interaction graph**.
            LightGCN propagates embeddings through 3 layers of neighborhood averaging.

            ```
            E^(k+1) = D^(-½) A D^(-½) · E^(k)    ← sparse matrix multiply
            E_final = mean(E⁰, E¹, E², E³)        ← average all layers
            ```

            **No feature transforms, no nonlinearities.**
            Just pure neighborhood averaging — deliberately simple.

            **Why it wins:** A user's 3-hop neighborhood captures
            "users who liked similar items liked THIS too" — transitively.
            On sparse data, this structural signal is extremely powerful.
            """)
        with b2:
            st.markdown("""
            <div class="arch-container">
                <div class="arch-row">
                    <div class="arch-box blue" style="min-width:320px;">
                        <div class="box-title">Layer 0 &mdash; Raw ID Embeddings</div>
                        <div style="color:#cdd6f4;font-size:0.9rem;">98K users + 26K items &rarr; 64d each</div>
                        <div class="box-detail">Initial embeddings before any propagation</div>
                    </div>
                </div>
                <div class="arch-arrow">&#8595; <span style="font-size:0.75rem;color:#585b70;">D<sup>-&frac12;</sup>AD<sup>-&frac12;</sup> &middot; E</span></div>
                <div class="arch-row">
                    <div class="arch-box blue" style="min-width:320px;">
                        <div class="box-title">Layer 1 &mdash; Direct Neighbors</div>
                        <div style="color:#cdd6f4;font-size:0.9rem;">Average of 1-hop connections</div>
                        <div class="box-detail">"What did this user buy?" / "Who bought this item?"</div>
                    </div>
                </div>
                <div class="arch-arrow">&#8595; <span style="font-size:0.75rem;color:#585b70;">sparse matrix multiply</span></div>
                <div class="arch-row">
                    <div class="arch-box blue" style="min-width:320px;">
                        <div class="box-title">Layer 2 &mdash; 2-Hop Neighbors</div>
                        <div style="color:#cdd6f4;font-size:0.9rem;">Friends-of-friends patterns</div>
                        <div class="box-detail">"Users who bought similar items also bought&hellip;"</div>
                    </div>
                </div>
                <div class="arch-arrow">&#8595;</div>
                <div class="arch-row">
                    <div class="arch-box blue" style="min-width:320px;">
                        <div class="box-title">Layer 3 &mdash; 3-Hop Neighborhood</div>
                        <div style="color:#cdd6f4;font-size:0.9rem;">Deep transitive collaborative signal</div>
                        <div class="box-detail">Captures community-level taste clusters</div>
                    </div>
                </div>
                <div class="arch-arrow">&#8595;</div>
                <div class="arch-row">
                    <div class="arch-box green" style="min-width:320px;">
                        <div class="box-title">Mean Pool &rarr; Final Embedding</div>
                        <div style="color:#cdd6f4;font-size:0.9rem;">E<sub>final</sub> = mean(E<sup>0</sup>, E<sup>1</sup>, E<sup>2</sup>, E<sup>3</sup>)</div>
                        <div class="box-detail">Multi-scale: combines local + global structure</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            st.metric("HR@10", "0.7290", delta="+6.8% vs MF", help="Best overall accuracy")

    # ── Two-Tower ──
    with model_tab[2]:
        t1, t2 = st.columns([1, 1], gap="large")
        with t1:
            st.markdown("#### Two-Tower — Production Retrieval")
            st.markdown("""
            Two independent neural networks encode users and items separately:

            **User Tower:** User ID emb + GRU over last 20 items + user features → 64d
            **Item Tower:** Item ID emb + text embedding (384→64) + item features → 64d

            Both outputs are **L2-normalized**. Score = dot product.
            Trained with **InfoNCE loss** (τ=0.2).

            **Why it matters:**
            - **Cold-start**: GRU encodes browsing history — no ID needed
            - **FAISS**: Pre-compute item vectors, search in 29μs
            - **Scale**: Used by YouTube, Pinterest, DoorDash
            """)
        with t2:
            st.markdown("""
            <div class="arch-container">
                <div class="tower-container">
                    <div class="tower user">
                        <div class="tower-title">USER TOWER</div>
                        <div class="arch-box purple" style="width:100%;">
                            <div class="box-title">User ID Embedding</div>
                            <div style="color:#cdd6f4;font-size:0.85rem;">64 dimensions</div>
                            <div class="box-detail">Learnable vector per user in training set</div>
                        </div>
                        <div class="arch-box purple" style="width:100%;">
                            <div class="box-title">GRU Sequence Encoder</div>
                            <div style="color:#cdd6f4;font-size:0.85rem;">Last 20 items &rarr; 64d</div>
                            <div class="box-detail">Captures temporal purchase patterns</div>
                        </div>
                        <div class="arch-box purple" style="width:100%;">
                            <div class="box-title">User Features</div>
                            <div style="color:#cdd6f4;font-size:0.85rem;">8 features &rarr; Linear &rarr; 64d</div>
                            <div class="box-detail">Avg rating, activity count, recency&hellip;</div>
                        </div>
                        <div class="arch-arrow">&#8595; Concat + MLP</div>
                        <div class="arch-box green" style="width:100%;">
                            <div class="box-title">L2 Normalize &rarr; 64d</div>
                        </div>
                    </div>
                    <div class="tower item">
                        <div class="tower-title">ITEM TOWER</div>
                        <div class="arch-box blue" style="width:100%;">
                            <div class="box-title">Item ID Embedding</div>
                            <div style="color:#cdd6f4;font-size:0.85rem;">64 dimensions</div>
                            <div class="box-detail">Learnable vector per item</div>
                        </div>
                        <div class="arch-box blue" style="width:100%;">
                            <div class="box-title">Text Embedding</div>
                            <div style="color:#cdd6f4;font-size:0.85rem;">SentenceTransformer 384d &rarr; 64d</div>
                            <div class="box-detail">Item title encoded by all-MiniLM-L6-v2</div>
                        </div>
                        <div class="arch-box blue" style="width:100%;">
                            <div class="box-title">Item Features</div>
                            <div style="color:#cdd6f4;font-size:0.85rem;">15 features &rarr; Linear &rarr; 64d</div>
                            <div class="box-detail">Price, avg rating, rating count, category&hellip;</div>
                        </div>
                        <div class="arch-arrow">&#8595; Concat + MLP</div>
                        <div class="arch-box green" style="width:100%;">
                            <div class="box-title">L2 Normalize &rarr; 64d</div>
                        </div>
                    </div>
                </div>
                <div class="arch-arrow" style="font-size:1.1rem;">&#8600; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; &#8601;</div>
                <div class="arch-row">
                    <div class="arch-box green" style="min-width:260px;">
                        <div class="box-title">Dot Product &rarr; InfoNCE Loss</div>
                        <div style="color:#cdd6f4;font-size:0.85rem;">&tau; = 0.2 &nbsp;|&nbsp; in-batch negatives</div>
                        <div class="box-detail">Contrastive loss over batch of 256 pairs</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            m1, m2 = st.columns(2)
            m1.metric("HR@10", "0.6395")
            m2.metric("FAISS Latency", "29 μs")

    # ── Feature-Gated LightGCN ──
    with model_tab[3]:
        fg1, fg2 = st.columns([1, 1], gap="large")
        with fg1:
            st.markdown("#### Feature-Gated LightGCN — My Contribution")
            st.markdown("**Question:** Can side features improve graph-based recommendations?")
            st.markdown("""
            <div class="arch-container" style="gap:6px;">
                <div class="arch-row" style="gap:16px;">
                    <div class="arch-box blue" style="flex:1;">
                        <div class="box-title">LightGCN (3 layers)</div>
                        <div style="color:#cdd6f4;font-size:0.85rem;">Graph propagation &rarr; 64d</div>
                        <div class="box-detail">Pure neighborhood averaging on interaction graph</div>
                    </div>
                    <div class="arch-box yellow" style="flex:1;">
                        <div class="box-title">Feature Projections</div>
                        <div style="color:#cdd6f4;font-size:0.85rem;">User + Item + Text &rarr; 64d</div>
                        <div class="box-detail">Linear projections of side features</div>
                    </div>
                </div>
                <div class="arch-arrow" style="font-size:1rem;">&#8600; &nbsp; (1 - gate) &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; gate &nbsp; &#8601;</div>
                <div class="arch-row">
                    <div class="arch-box red" style="min-width:280px;">
                        <div class="box-title">Learnable Sigmoid Gate</div>
                        <div style="color:#cdd6f4;font-size:0.95rem;font-weight:600;">
                            82% graph &nbsp;&bull;&nbsp; 18% features
                        </div>
                        <div class="box-detail">Single learnable param &mdash; model decides the blend!</div>
                    </div>
                </div>
                <div class="arch-arrow">&#8595;</div>
                <div class="arch-row">
                    <div class="arch-box green" style="min-width:280px;">
                        <div class="box-title">Final Embedding &rarr; BPR Loss</div>
                        <div class="box-detail">E = (1-g)&middot;E_graph + g&middot;E_features</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            st.info("**Finding:** On 99.97% sparse data, collaborative graph structure dominates. "
                    "Side features help modestly (+5.4% over MF) but can't match pure LightGCN.")
        with fg2:
            st.markdown("#### Training Curve — Gate Convergence")
            epochs = list(range(1, 45))
            hr_curve = [0.5605,0.5700,0.5920,0.6020,0.6160,0.6300,0.6380,0.6405,
                         0.6500,0.6555,0.6625,0.6650,0.6695,0.6755,0.6805,0.6825,
                         0.6860,0.6860,0.6910,0.6885,0.6920,0.6955,0.6995,0.7055,
                         0.7055,0.7055,0.7090,0.7105,0.7130,0.7145,0.7135,0.7165,
                         0.7140,0.7180,0.7180,0.7190,0.7170,0.7175,0.7175,0.7175,
                         0.7185,0.7185,0.7180,0.7185]
            gate_curve = [0.62,0.62,0.59,0.53,0.48,0.44,0.40,0.37,0.34,0.32,0.30,
                          0.29,0.27,0.26,0.25,0.24,0.23,0.23,0.22,0.22,0.21,0.21,
                          0.20,0.20,0.20,0.20,0.19,0.19,0.19,0.19,0.19,0.19,0.18,
                          0.18,0.18,0.18,0.18,0.18,0.18,0.18,0.18,0.18,0.18,0.18]
            fig_fg = go.Figure()
            fig_fg.add_trace(go.Scatter(x=epochs, y=hr_curve, name="HR@10",
                                         line=dict(color="#a6e3a1", width=2.5)))
            fig_fg.add_trace(go.Scatter(x=epochs, y=gate_curve, name="Feature Gate",
                                         line=dict(color="#cba6f7", width=2.5, dash="dot"), yaxis="y2"))
            fig_fg.add_hline(y=0.7290, line_dash="dash", line_color="#f38ba8",
                             annotation_text="LightGCN (0.729)", annotation_position="top left",
                             annotation_font=dict(color="#f38ba8", size=11))
            fig_fg.update_layout(
                height=340, margin=dict(l=0, r=40, t=10, b=0),
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#cdd6f4"),
                xaxis=dict(title="Epoch", gridcolor="#313244"),
                yaxis=dict(title="HR@10", gridcolor="#313244", range=[0.5, 0.76]),
                yaxis2=dict(title="Gate Value", overlaying="y", side="right", range=[0, 0.7], gridcolor="#313244"),
                legend=dict(orientation="h", y=1.12),
            )
            st.plotly_chart(fig_fg, use_container_width=True)
            m1, m2 = st.columns(2)
            m1.metric("HR@10", "0.7190", delta="-1.4% vs LightGCN")
            m2.metric("Gate", "0.18", delta="82% graph / 18% features")

    # ── FAISS ──
    with model_tab[4]:
        f1, f2 = st.columns([1, 1], gap="large")
        with f1:
            st.markdown("#### FAISS — Serving at Scale")
            st.markdown("""
            <div class="arch-container" style="gap:6px;">
                <div class="arch-label">OFFLINE (once)</div>
                <div class="arch-row">
                    <div class="arch-box blue" style="min-width:280px;">
                        <div class="box-title">Pre-compute 26,354 Item Vectors</div>
                        <div style="color:#cdd6f4;font-size:0.85rem;">One forward pass through Item Tower</div>
                        <div class="box-detail">Done once, stored for all future queries</div>
                    </div>
                </div>
                <div class="arch-arrow">&#8595;</div>
                <div class="arch-row">
                    <div class="arch-box blue" style="min-width:280px;">
                        <div class="box-title">Build FAISS HNSW Index</div>
                        <div style="color:#cdd6f4;font-size:0.85rem;">Hierarchical navigable small world graph</div>
                        <div class="box-detail">~99% recall, sublinear search time</div>
                    </div>
                </div>
                <div style="border-top:1px dashed #585b70;width:80%;margin:12px auto;"></div>
                <div class="arch-label">ONLINE (per request)</div>
                <div class="arch-row">
                    <div class="arch-box purple" style="min-width:280px;">
                        <div class="box-title">User Tower Forward Pass</div>
                        <div style="color:#cdd6f4;font-size:0.85rem;">~0.5 ms</div>
                        <div class="box-detail">Encode user into 64d vector</div>
                    </div>
                </div>
                <div class="arch-arrow">&#8595;</div>
                <div class="arch-row">
                    <div class="arch-box green" style="min-width:280px;">
                        <div class="box-title">FAISS HNSW Search</div>
                        <div style="color:#a6e3a1;font-size:1.1rem;font-weight:700;">29 &micro;s</div>
                        <div class="box-detail">Top-1,000 nearest items from 26K candidates</div>
                    </div>
                </div>
                <div class="arch-arrow">&#8595;</div>
                <div class="arch-row">
                    <div class="arch-box peach" style="min-width:280px;">
                        <div class="box-title">Top 10 Recommendations</div>
                        <div class="box-detail">Re-rank candidates with business rules</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        with f2:
            st.markdown("#### Latency Comparison")
            methods   = ["Brute Force GPU", "FAISS Flat", "FAISS IVF", "FAISS HNSW"]
            latencies = [894, 310, 35, 29]
            bar_colors = ["#f38ba8", "#fab387", "#89b4fa", "#a6e3a1"]
            fig3 = go.Figure(go.Bar(
                x=latencies, y=methods, orientation="h", marker_color=bar_colors,
                text=[f"{l} μs" for l in latencies], textposition="outside",
                textfont=dict(size=13)
            ))
            fig3.update_layout(height=280, margin=dict(l=0, r=80, t=10, b=0),
                               plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                               font=dict(color="#cdd6f4", size=13),
                               xaxis=dict(title="Latency (microseconds)", gridcolor="#313244"))
            st.plotly_chart(fig3, use_container_width=True)
            st.success("HNSW: 34,000 queries/second on a single CPU core")
            st.markdown("""
            **Why Two-Tower + FAISS wins for production:**
            - LightGCN needs the **full adjacency matrix** at inference
            - MF can pre-compute embeddings but can't handle new users
            - Two-Tower: pre-compute items, real-time user encoding, FAISS search
            """)


# ============================================================
# PAGE 3: LIVE DEMO
# ============================================================
elif page == "3. Live Demo":
    st.markdown("## Live Demo")
    st.markdown("---")

    demo_tab1, demo_tab2 = st.tabs(["Existing User — MF vs Two-Tower", "Cold-Start — New User"])

    with demo_tab1:
        st.markdown('<div class="section-header">Recommend for an Existing User</div>', unsafe_allow_html=True)
        st.caption("Same user, two different models — see how recommendations differ.")

        user_id = st.number_input("User ID", min_value=0, max_value=s["n_users"] - 1, value=100)
        history     = data["user_history"].get(user_id, [])
        history_set = set(history)

        if history:
            st.caption(f"Purchase history: {len(history)} items — showing last 5")
            for item_idx in history[-5:]:
                title, cat, rating, price = get_item_display(item_idx, data["item_info"])
                st.markdown(f"- {title}{rating}{price}")
        else:
            st.info("No history for this user ID.")

        st.markdown("")
        tt_recs = recommend(data["tt_user"][user_id], data["tt_index"], history_set)
        mf_recs = recommend(data["mf_user"][user_id], data["mf_index"], history_set)
        tt_items = [idx for idx, _ in tt_recs]
        mf_items = [idx for idx, _ in mf_recs]
        overlap  = set(tt_items) & set(mf_items)
        n_overlap = len(overlap)

        tt_col, mf_col = st.columns(2, gap="medium")
        with tt_col:
            st.markdown(f"**Two-Tower v5** — {n_overlap}/10 overlap with MF")
            for rank, (idx, score) in enumerate(tt_recs, 1):
                title, cat, rating, price = get_item_display(idx, data["item_info"])
                match_pct = int(score * 100)
                tag = "  `both`" if idx in overlap else ""
                st.markdown(f"**{rank}.** {title}{rating}{price} — `{match_pct}%`{tag}")

        with mf_col:
            st.markdown(f"**Matrix Factorization** — {n_overlap}/10 overlap with TT")
            for rank, (idx, score) in enumerate(mf_recs, 1):
                title, cat, rating, price = get_item_display(idx, data["item_info"])
                match_pct = int(score * 100)
                tag = "  `both`" if idx in overlap else ""
                st.markdown(f"**{rank}.** {title}{rating}{price} — `{match_pct}%`{tag}")

        with st.expander("Why do they recommend different items?"):
            st.markdown(f"""
**Two-Tower** uses ID + GRU + text + features → finds semantically similar items.
**MF** uses only ID co-occurrence → finds items that co-appear in purchase histories.

**{n_overlap} items overlap** — high-confidence recs both models agree on.
**{10 - n_overlap} items differ** — Two-Tower finds genre-similar items; MF finds co-purchased items.

On this dataset, MF beats Two-Tower by 6.7% HR@10 for known users.
But a new user arrives → MF outputs random. Two-Tower still works.
            """)

    with demo_tab2:
        st.markdown('<div class="section-header">Cold-Start — Brand New User</div>', unsafe_allow_html=True)
        st.caption("Pick some games you like. Two-Tower recommends from just browsing history — no account needed.")
        st.warning("MF and LightGCN **cannot** serve new users. Only Two-Tower handles cold-start.")

        scenario = st.selectbox("Quick scenario:", [
            "Custom search",
            "Souls-like (Dark Souls, Elden Ring, Sekiro)",
            "Nintendo (Mario, Zelda, Pokemon)",
            "FPS (Call of Duty, Battlefield, Halo)",
        ])

        scenario_keywords = {
            "Souls-like (Dark Souls, Elden Ring, Sekiro)": ["Dark Souls", "Elden Ring", "Sekiro"],
            "Nintendo (Mario, Zelda, Pokemon)":            ["Mario", "Zelda", "Pokemon"],
            "FPS (Call of Duty, Battlefield, Halo)":       ["Call of Duty", "Battlefield", "Halo"],
        }

        all_titles = {idx: data["item_info"].get(idx, {}).get("title", "Unknown")
                      for idx in range(s["n_items"])}

        if scenario == "Custom search":
            search_term = st.text_input("Search game title:")
            if search_term:
                matches = [(idx, t) for idx, t in all_titles.items()
                           if search_term.lower() in t.lower()][:20]
                selected = st.multiselect("Select games you like:",
                    options=[idx for idx, _ in matches],
                    format_func=lambda x: all_titles[x][:55]) if matches else []
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
            st.caption("Your browsing history (cold-start input):")
            for idx in selected:
                title, _, rating, price = get_item_display(idx, data["item_info"])
                st.markdown(f"- {title}{rating}{price}")

            cold_emb = data["cold_encoder"].encode_cold_user(selected, data["text_embs"])
            recs = recommend(cold_emb, data["tt_index"], set(selected), k=10)

            st.markdown("**Two-Tower recommendations via GRU:**")
            for rank, (idx, score) in enumerate(recs, 1):
                title, cat, rating, price = get_item_display(idx, data["item_info"])
                match_pct = int(score * 100)
                st.markdown(f"**{rank}.** {title}{rating}{price} — `{match_pct}% match`")

            st.markdown("""
            > **MF**: No embedding for this user — cannot recommend
            > **LightGCN**: User not in training graph — cannot recommend
            > **Two-Tower**: Encodes from item text via GRU in real time
            """)
        else:
            st.info("Select at least 2 games to see recommendations.")


# ============================================================
# PAGE 4: RESULTS & ANALYSIS
# ============================================================
elif page == "4. Results & Analysis":
    st.markdown("## Results & Analysis")
    st.markdown("---")

    # ── Model comparison chart ──
    st.markdown('<div class="section-header">Model Comparison (Sampled Eval, 100 negatives)</div>', unsafe_allow_html=True)
    left, right = st.columns([1, 1], gap="large")

    with left:
        models_bar = ["Two-Tower", "MF", "FG-LightGCN", "LightGCN"]
        hr_bar     = [0.6395, 0.6825, 0.7190, 0.7290]
        bar_colors = ["#cba6f7", "#89b4fa", "#f9e2af", "#a6e3a1"]
        fig = go.Figure(go.Bar(
            x=models_bar, y=hr_bar, marker_color=bar_colors,
            text=[f"{v:.4f}" for v in hr_bar], textposition="outside", textfont=dict(size=11)
        ))
        fig.update_layout(height=320, margin=dict(l=0,r=0,t=30,b=0), title="HR@10",
                          plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                          font=dict(color="#cdd6f4"), yaxis=dict(gridcolor="#313244", range=[0.55, 0.78]))
        st.plotly_chart(fig, use_container_width=True)

    with right:
        ndcg_bar = [0.4148, 0.4550, 0.4824, 0.4940]
        fig2 = go.Figure(go.Bar(
            x=models_bar, y=ndcg_bar, marker_color=bar_colors,
            text=[f"{v:.4f}" for v in ndcg_bar], textposition="outside", textfont=dict(size=11)
        ))
        fig2.update_layout(height=320, margin=dict(l=0,r=0,t=30,b=0), title="NDCG@10",
                           plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                           font=dict(color="#cdd6f4"), yaxis=dict(gridcolor="#313244", range=[0.35, 0.55]))
        st.plotly_chart(fig2, use_container_width=True)

    # ── Ablation study ──
    st.markdown('<div class="section-header">12-Variant Ablation Study</div>', unsafe_allow_html=True)
    st.caption("Each variant changes one thing from the previous. Isolate each component's contribution.")

    versions = ["v1","v2","v3","v4","v4b","v4-BPR","v5","v5b","v5c","v6","v7","v8"]
    hr_vals  = [0.6195,0.6210,0.6195,0.6355,0.6280,0.1520,
                0.6395,0.6385,0.6330,0.6355,0.6355,0.6305]
    ablation_colors = ["#89b4fa"] * 12
    ablation_colors[3] = "#a6e3a1"
    ablation_colors[5] = "#f38ba8"
    ablation_colors[6] = "#a6e3a1"

    fig3 = go.Figure(go.Bar(
        x=versions, y=hr_vals, marker_color=ablation_colors,
        text=[f"{v:.3f}" for v in hr_vals], textposition="outside", textfont=dict(size=9)
    ))
    fig3.add_hline(y=0.6395, line_dash="dot", line_color="#a6e3a1",
                   annotation_text="Best: v5 (GRU)", annotation_position="top right")
    fig3.update_layout(height=300, margin=dict(l=0, r=0, t=30, b=0),
                       plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                       font=dict(color="#cdd6f4"), yaxis=dict(gridcolor="#313244", range=[0, 0.72]),
                       yaxis_title="HR@10")
    st.plotly_chart(fig3, use_container_width=True)

    with st.expander("Ablation details — what each variant tested"):
        st.markdown("""
        | Version | Change | HR@10 | Finding |
        |---|---|---|---|
        | v1 | Baseline (InfoNCE, batch=256) | 0.6195 | Starting point |
        | v2 | + MSE distillation (α=0.5) | 0.6210 | Marginal |
        | v3 | + Cosine distillation (α=0.9) | 0.6195 | No gain |
        | **v4** | **+ Title text embeddings** | **0.6355** | **+2.6% — text helps** |
        | v4b | + Larger batch (1024) | 0.6280 | Too many negatives hurt |
        | v4-BPR | BPR + hard negatives | 0.1520 | **Collapsed** |
        | **v5** | **+ GRU sequential encoding** | **0.6395** | **Best variant** |
        | v5b | + Rich text (desc+features) | 0.6385 | Noisy text |
        | v5c | + LightGCN init | 0.6330 | Init scrambled |
        | v6 | Curriculum negatives | 0.6355 | No gain |
        | v7 | + CLIP image embeddings | 0.6355 | Marginal |
        | v8 | FM-style additive fusion | 0.6305 | Interpretable gates |
        """)

    # ── Gate analysis ──
    st.markdown('<div class="section-header">Learned Gate Weights — What Signals Matter?</div>', unsafe_allow_html=True)
    g1, g2, g3 = st.columns(3)
    with g1:
        st.markdown("**FM Two-Tower: User side**")
        st.progress(0.62, text="ID embedding — 62%")
        st.progress(0.28, text="GRU sequence — 28%")
        st.progress(0.10, text="User features — 10%")
    with g2:
        st.markdown("**FM Two-Tower: Item side**")
        st.progress(0.54, text="ID embedding — 54%")
        st.progress(0.23, text="Text — 23%")
        st.progress(0.22, text="Item features — 22%")
    with g3:
        st.markdown("**Feature-Gated LightGCN**")
        st.progress(0.82, text="Graph signal — 82%")
        st.progress(0.18, text="Projected features — 18%")
        st.info("Both models independently confirm: collaborative ID/graph signal >> content features on sparse data")


# ============================================================
# PAGE 5: KEY FINDINGS
# ============================================================
elif page == "5. Key Findings":
    st.markdown("## Key Findings")
    st.markdown("---")

    f1, f2 = st.columns(2, gap="large")
    with f1:
        st.markdown("""
        #### 1. Graph structure wins on sparse data
        LightGCN (0.729) beats all feature-enriched models.
        Feature-Gated LightGCN's gate confirms: **82% graph, 18% features**.
        Multi-hop neighborhood averaging captures "users who liked similar items"
        — a signal that content features cannot replicate.
        """)

        st.markdown("""
        #### 2. Text embeddings help modestly (+2.6%)
        Adding item titles improved Two-Tower from 0.619 → 0.636.
        But rich text (descriptions + features) actually **hurt** —
        noise overwhelms signal on sparse data.
        """)

    with f2:
        st.markdown("""
        #### 3. Cold-start needs a different architecture
        Best-accuracy model (LightGCN) **cannot serve new users**.
        Two-Tower trades 12% accuracy for the ability to recommend
        from just 3 browsed items. In production, this tradeoff is always worth it.
        """)

        st.markdown("""
        #### 4. Loss function > features
        BPR collapsed the Two-Tower (0.23 HR@10).
        Same model, wrong loss = 3× worse.
        InfoNCE with temperature scaling is essential for dual-encoder training.
        """)

    st.markdown("---")
    st.markdown("""
    #### Bottom Line
    > **There is no single best model.** LightGCN for accuracy, Two-Tower for production,
    > and the 12-variant ablation study tells you *why* each component helps or hurts.
    > Feature-Gated LightGCN (my contribution) proves that on extremely sparse data,
    > collaborative structure dominates — the model itself learns to ignore features.
    """)

    st.markdown("")
    st.markdown("---")
    st.caption("Nidhi Rajani  |  EAS 509  |  Amazon Video Games 2023  |  github.com/nidhi1603/Two_Tower_Recommendation_System")
