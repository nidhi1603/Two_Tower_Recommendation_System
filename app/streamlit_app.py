import streamlit as st
st.set_page_config(
    page_title="Two-Tower Recommendation System",
    layout="wide",
    initial_sidebar_state="collapsed"
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
    .block-container { padding-top: 2rem; padding-bottom: 2rem; }
    .metric-card {
        background: #1e1e2e;
        border: 1px solid #313244;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
    }
    .metric-value { font-size: 2.2rem; font-weight: 700; color: #cba6f7; }
    .metric-label { font-size: 0.85rem; color: #a6adc8; margin-top: 4px; }
    .section-header {
        font-size: 1.1rem;
        font-weight: 600;
        color: #cdd6f4;
        border-left: 3px solid #cba6f7;
        padding-left: 10px;
        margin: 1.5rem 0 1rem 0;
    }
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
@st.cache_resource(show_spinner="Downloading model files (first run only)...")
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


# ============================================================
# HEADER
# ============================================================
st.markdown("## Two-Tower Recommendation System")
st.markdown("Deep learning retrieval system trained on Amazon Video Games 2023 — 12-variant ablation study")
st.markdown("---")

# ── Top metric tiles ──────────────────────────────────────
c1, c2, c3, c4, c5 = st.columns(5)
c1.markdown('''<div class="metric-card"><div class="metric-value">98,906</div><div class="metric-label">Users</div></div>''', unsafe_allow_html=True)
c2.markdown('''<div class="metric-card"><div class="metric-value">26,354</div><div class="metric-label">Items</div></div>''', unsafe_allow_html=True)
c3.markdown('''<div class="metric-card"><div class="metric-value">99.97%</div><div class="metric-label">Sparsity</div></div>''', unsafe_allow_html=True)
c4.markdown('''<div class="metric-card"><div class="metric-value">29 μs</div><div class="metric-label">FAISS Latency</div></div>''', unsafe_allow_html=True)
c5.markdown('''<div class="metric-card"><div class="metric-value">12</div><div class="metric-label">Model Variants</div></div>''', unsafe_allow_html=True)

st.markdown("")


# ============================================================
# ROW 1: Model comparison + Ablation
# ============================================================
col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.markdown('<div class="section-header">Model Comparison (Full Ranking vs All Items)</div>', unsafe_allow_html=True)
    st.dataframe({
        "Model":         ["MF (BPR)",  "Two-Tower v5",  "LightGCN"],
        "HR@10":         [0.0420,       0.0270,           0.0440],
        "Cold-Start":    ["No",         "Yes",            "No"],
        "FAISS Ready":   ["Partial",    "Yes  <1ms",      "No"],
        "Industry Role": ["Baseline",   "Retrieval",      "Re-ranking"],
    }, hide_index=True, use_container_width=True)

    st.markdown('<div class="section-header">HR@10 by User Activity Level</div>', unsafe_allow_html=True)
    fig = go.Figure()
    buckets   = ["5-7 purchases", "8-15", "16-30", "31+", "New user"]
    mf_scores = [0.6780, 0.6680, 0.6500, 0.5780, 0.0]
    tt_scores = [0.6700, 0.6360, 0.6220, 0.5340, 0.35]
    fig.add_trace(go.Bar(name="MF",        x=buckets, y=mf_scores, marker_color="#89b4fa"))
    fig.add_trace(go.Bar(name="Two-Tower", x=buckets, y=tt_scores, marker_color="#cba6f7"))
    fig.add_annotation(x="New user", y=0.37, text="Two-Tower only option",
                        showarrow=True, arrowhead=2, font=dict(color="#a6e3a1", size=11))
    fig.update_layout(
        barmode="group", height=280,
        margin=dict(l=0, r=0, t=10, b=0),
        legend=dict(orientation="h", y=1.1),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#cdd6f4"),
        yaxis=dict(gridcolor="#313244"),
    )
    st.plotly_chart(fig, use_container_width=True)

with col_right:
    st.markdown('<div class="section-header">12-Variant Ablation Study (HR@10)</div>', unsafe_allow_html=True)
    versions = ["v1","v2","v3","v4","v4b","v4-BPR","v5","v5b","v5c","v6","v7","v8"]
    hr_vals  = [0.6195,0.6210,0.6195,0.6355,0.6280,0.1520,
                0.6395,0.6385,0.6330,0.6355,0.6355,0.6305]
    colors   = ["#89b4fa"] * 12
    colors[3] = "#a6e3a1"   # v4 best content
    colors[5] = "#f38ba8"   # v4-BPR collapsed
    colors[6] = "#a6e3a1"   # v5 best overall

    fig2 = go.Figure(go.Bar(
        x=versions, y=hr_vals, marker_color=colors,
        text=[f"{v:.3f}" for v in hr_vals],
        textposition="outside", textfont=dict(size=9)
    ))
    fig2.add_hline(y=0.6395, line_dash="dot", line_color="#a6e3a1",
                   annotation_text="Best: v5", annotation_position="top right")
    fig2.update_layout(
        height=300, margin=dict(l=0, r=0, t=30, b=0),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#cdd6f4"),
        yaxis=dict(gridcolor="#313244", range=[0, 0.72]),
        yaxis_title="HR@10",
    )
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown('<div class="section-header">FM Gate Weights — What the Model Learned</div>', unsafe_allow_html=True)
    g1, g2 = st.columns(2)
    with g1:
        st.caption("User representation")
        st.progress(0.62, text="ID embedding — 62%")
        st.progress(0.28, text="GRU sequence — 28%")
        st.progress(0.10, text="User features — 10%")
    with g2:
        st.caption("Item representation")
        st.progress(0.54, text="ID embedding — 54%")
        st.progress(0.23, text="Text — 23%")
        st.progress(0.22, text="Item features — 22%")


st.markdown("---")


# ============================================================
# ROW 2: Live demos side by side
# ============================================================
demo_left, demo_right = st.columns([1, 1], gap="large")

with demo_left:
    st.markdown('<div class="section-header">Live Demo — Existing User</div>', unsafe_allow_html=True)

    user_id = st.number_input("User ID", min_value=0,
                               max_value=s["n_users"] - 1, value=100)
    model_choice = st.selectbox("Model", ["Two-Tower v5", "Matrix Factorization", "Both"])

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

    if model_choice in ["Two-Tower v5", "Both"]:
        st.caption("Two-Tower v5 recommendations")
        recs = recommend(data["tt_user"][user_id], data["tt_index"], history_set)
        for rank, (idx, score) in enumerate(recs, 1):
            title, cat, rating, price = get_item_display(idx, data["item_info"])
            match_pct = int(score * 100)
            st.markdown(f"**{rank}.** {title}{rating}{price} — `{match_pct}% match`")

    if model_choice in ["Matrix Factorization", "Both"]:
        st.caption("Matrix Factorization recommendations")
        recs = recommend(data["mf_user"][user_id], data["mf_index"], history_set)
        for rank, (idx, score) in enumerate(recs, 1):
            title, cat, rating, price = get_item_display(idx, data["item_info"])
            match_pct = int(score * 100)
            st.markdown(f"**{rank}.** {title}{rating}{price} — `{match_pct}% match`")

with demo_right:
    st.markdown('<div class="section-header">Live Demo — Cold-Start (New User)</div>', unsafe_allow_html=True)
    st.caption("Two-Tower works instantly. MF and LightGCN cannot handle new users at all.")

    scenario = st.selectbox("Scenario", [
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

        st.caption("Two-Tower recommendations via GRU over browsed item texts:")
        for rank, (idx, score) in enumerate(recs, 1):
            title, cat, rating, price = get_item_display(idx, data["item_info"])
            match_pct = int(score * 100)
            st.markdown(f"**{rank}.** {title}{rating}{price} — `{match_pct}% match`")

        st.markdown("""
        > **MF**: No embedding for this user — cannot recommend  
        > **LightGCN**: User not in training graph — cannot recommend  
        > **Two-Tower**: Encodes user from item text via GRU in real time
        """)
    else:
        st.info("Select at least 2 games to see recommendations.")


st.markdown("---")


# ============================================================
# ROW 3: Interactive Algorithm Explainer
# ============================================================
st.markdown('<div class="section-header">How It Works — Interactive Algorithm Walkthrough</div>', unsafe_allow_html=True)

algo_tab1, algo_tab2, algo_tab3, algo_tab4 = st.tabs([
    "Step 1: Two-Tower Architecture",
    "Step 2: FAISS Retrieval",
    "Step 3: Cold-Start Problem",
    "Step 4: Why Not Just MF?",
])

with algo_tab1:
    a1, a2 = st.columns([1, 1], gap="large")
    with a1:
        st.markdown("#### The Two-Tower Architecture")
        st.markdown("""
        Two separate neural networks run independently:

        **User Tower** takes:
        - User ID embedding (learned collaborative signal)
        - GRU over last 20 interacted items
        - User features (activity level, avg rating given, etc.)

        **Item Tower** takes:
        - Item ID embedding
        - Text embedding from item title (384-dim via sentence-transformers)
        - Item features (price, rating, category)

        Both towers output a **64-dimensional vector**.  
        Relevance = dot product between user and item vectors.

        Training: **InfoNCE loss** — correct (user, item) pair
        must score higher than all other items in the batch.
        """)
    with a2:
        fig_arch = go.Figure()
        # User tower
        fig_arch.add_shape(type="rect", x0=0, y0=3, x1=2, y1=7,
                           fillcolor="#313244", line=dict(color="#cba6f7", width=2))
        fig_arch.add_annotation(x=1, y=7.3, text="USER TOWER",
                                 font=dict(color="#cba6f7", size=12), showarrow=False)
        for i, label in enumerate(["User ID Emb", "GRU (sequence)", "User features"]):
            fig_arch.add_shape(type="rect", x0=0.1, y0=3.3+i*1.1, x1=1.9, y1=4.0+i*1.1,
                               fillcolor="#45475a", line=dict(color="#6c7086"))
            fig_arch.add_annotation(x=1, y=3.65+i*1.1, text=label,
                                     font=dict(color="#cdd6f4", size=10), showarrow=False)
        # Item tower
        fig_arch.add_shape(type="rect", x0=4, y0=3, x1=6, y1=7,
                           fillcolor="#313244", line=dict(color="#89b4fa", width=2))
        fig_arch.add_annotation(x=5, y=7.3, text="ITEM TOWER",
                                 font=dict(color="#89b4fa", size=12), showarrow=False)
        for i, label in enumerate(["Item ID Emb", "Text Emb (384d)", "Item features"]):
            fig_arch.add_shape(type="rect", x0=4.1, y0=3.3+i*1.1, x1=5.9, y1=4.0+i*1.1,
                               fillcolor="#45475a", line=dict(color="#6c7086"))
            fig_arch.add_annotation(x=5, y=3.65+i*1.1, text=label,
                                     font=dict(color="#cdd6f4", size=10), showarrow=False)
        # Output vectors
        for x, col in [(1, "#cba6f7"), (5, "#89b4fa")]:
            fig_arch.add_shape(type="rect", x0=x-0.5, y0=1.5, x1=x+0.5, y1=2.5,
                               fillcolor=col, line=dict(color=col))
            fig_arch.add_annotation(x=x, y=2.0, text="64-dim",
                                     font=dict(color="#1e1e2e", size=10), showarrow=False)
            fig_arch.add_annotation(x=x, y=3.0, ax=x, ay=2.5,
                                     arrowhead=2, arrowcolor="#a6adc8",
                                     arrowwidth=2, showarrow=True, text="")
        # Dot product
        fig_arch.add_shape(type="rect", x0=2.3, y0=0.5, x1=3.7, y1=1.5,
                           fillcolor="#a6e3a1", line=dict(color="#a6e3a1"))
        fig_arch.add_annotation(x=3, y=1.0, text="dot product",
                                 font=dict(color="#1e1e2e", size=10), showarrow=False)
        fig_arch.add_annotation(x=1.7, y=1.0, ax=2.3, ay=1.0,
                                 arrowhead=2, arrowcolor="#a6adc8", showarrow=True, text="")
        fig_arch.add_annotation(x=4.3, y=1.0, ax=3.7, ay=1.0,
                                 arrowhead=2, arrowcolor="#a6adc8", showarrow=True, text="")
        fig_arch.update_layout(
            height=380, showlegend=False,
            margin=dict(l=10, r=10, t=30, b=10),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(visible=False, range=[-0.5, 6.5]),
            yaxis=dict(visible=False, range=[0, 8]),
        )
        st.plotly_chart(fig_arch, use_container_width=True)

with algo_tab2:
    b1, b2 = st.columns([1, 1], gap="large")
    with b1:
        st.markdown("#### Why FAISS? The Serving Problem")
        st.markdown("""
        At inference: 98,906 users and 26,354 items.

        **Naive brute force:**
        - 98,906 x 26,354 = 2.6 billion dot products per request
        - Completely undeployable at scale

        **Two-Tower + FAISS:**
        - Pre-compute all 26,354 item vectors **once offline**
        - Index in FAISS
        - At serving: encode user (1 forward pass) → FAISS search
        - HNSW retrieves top-1,000 in **29 microseconds**

        This is why Two-Tower is the industry standard for retrieval.
        LightGCN requires the full graph at runtime — cannot do this.
        """)
    with b2:
        methods   = ["Brute Force GPU", "FAISS Flat", "FAISS IVF", "FAISS HNSW"]
        latencies = [894, 310, 35, 29]
        bar_colors = ["#f38ba8", "#fab387", "#89b4fa", "#a6e3a1"]
        fig3 = go.Figure(go.Bar(
            x=latencies, y=methods, orientation="h",
            marker_color=bar_colors,
            text=[f"{l} μs" for l in latencies],
            textposition="outside"
        ))
        fig3.update_layout(
            height=260, margin=dict(l=0, r=70, t=20, b=0),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#cdd6f4"),
            xaxis=dict(title="Latency (microseconds)", gridcolor="#313244"),
        )
        st.plotly_chart(fig3, use_container_width=True)
        st.success("HNSW: 34,000 users/second on a single CPU core")

with algo_tab3:
    c1, c2 = st.columns([1, 1], gap="large")
    with c1:
        st.markdown("#### The Cold-Start Problem")
        st.markdown("""
        A new user signs up with zero purchase history.

        **Why MF fails:**  
        MF is a lookup table — user ID → embedding.
        A new user has no entry. Result: random vector → random recommendations → user churns.

        **Why Two-Tower solves it:**  
        The user tower takes text as input, not just an ID.
        We run 3 browsed items through the GRU.
        The GRU reads item descriptions and infers taste.
        No training needed for the new user.

        **Measured result on this project:**
        """)
        m1, m2 = st.columns(2)
        m1.metric("MF (new user)", "~9.9% HR@10", "= random chance")
        m2.metric("Two-Tower (3 items)", "~35% HR@10", "3.5x better")
    with c2:
        steps = [
            ("New user browses 3 games", "#cba6f7"),
            ("Text embeddings extracted", "#89b4fa"),
            ("GRU reads text sequence", "#89b4fa"),
            ("64-dim user vector produced", "#a6e3a1"),
            ("FAISS: top 1,000 candidates", "#a6e3a1"),
            ("Top 10 shown to user", "#a6e3a1"),
        ]
        fig4 = go.Figure()
        for i, (label, color) in enumerate(steps):
            y = len(steps) - i
            fig4.add_shape(type="rect", x0=0.5, y0=y-0.35, x1=3.5, y1=y+0.35,
                           fillcolor="#313244", line=dict(color=color, width=2))
            fig4.add_annotation(x=2, y=y, text=label,
                                 font=dict(color="#cdd6f4", size=11), showarrow=False)
            if i < len(steps) - 1:
                fig4.add_annotation(x=2, y=y-0.35, ax=2, ay=y-0.65,
                                     arrowhead=2, arrowcolor="#a6adc8",
                                     arrowwidth=2, showarrow=True, text="")
        fig4.update_layout(
            height=380, showlegend=False,
            margin=dict(l=0, r=0, t=10, b=0),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(visible=False, range=[0, 4]),
            yaxis=dict(visible=False, range=[0, len(steps)+0.5]),
        )
        st.plotly_chart(fig4, use_container_width=True)

with algo_tab4:
    d1, d2 = st.columns([1, 1], gap="large")
    with d1:
        st.markdown("#### MF vs Two-Tower — The Real Tradeoff")
        st.markdown("""
        *"If LightGCN beats Two-Tower on accuracy, why use Two-Tower?"*

        **LightGCN wins on accuracy** for known users —
        graph propagation captures higher-order collaborative patterns.

        **LightGCN cannot:**
        - Serve new users (not in graph)
        - Serve new items (not in graph)
        - Be indexed in FAISS (needs full graph at runtime)
        - Scale to 100M+ items without full retraining

        **Production pipeline at companies like YouTube, Pinterest:**

        Two-Tower retrieves top 1,000 candidates in microseconds.  
        A ranking model (like LightGCN) re-scores those 1,000.  
        Top 10 shown to user.

        You get both: the scale of Two-Tower AND the accuracy of graph models.
        """)
    with d2:
        st.markdown("#### FM v8 Gate Weights — Interpretability")
        categories = ["User: ID Emb", "User: GRU Seq", "User: Features",
                      "Item: ID Emb", "Item: Text", "Item: Features"]
        weights    = [62, 28, 10, 54, 23, 22]
        bar_colors = ["#cba6f7","#cba6f7","#cba6f7",
                      "#89b4fa","#89b4fa","#89b4fa"]
        fig5 = go.Figure(go.Bar(
            y=categories, x=weights, orientation="h",
            marker_color=bar_colors,
            text=[f"{w}%" for w in weights],
            textposition="outside"
        ))
        fig5.update_layout(
            height=300, margin=dict(l=0, r=50, t=10, b=0),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#cdd6f4"),
            xaxis=dict(title="Contribution (%)", gridcolor="#313244", range=[0, 75]),
        )
        st.plotly_chart(fig5, use_container_width=True)
        st.caption("On 99.97% sparse data, collaborative ID signals carry 2-3x more weight than content. Content features matter most for cold-start.")

st.markdown("---")
st.caption("Built by Nidhi  |  github.com/nidhi1603/Two_Tower_Recommendation_System  |  Dataset: Amazon Video Games 2023")
