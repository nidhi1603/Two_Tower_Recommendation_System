# Two-Tower Recommendation System

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit-ff4b4b?style=for-the-badge&logo=streamlit)](https://sdnsjdfd.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.10-blue?style=flat-square&logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0-orange?style=flat-square&logo=pytorch)](https://pytorch.org)
[![Dataset](https://img.shields.io/badge/Dataset-Amazon%20Video%20Games%202023-yellow?style=flat-square)](https://amazon-reviews-2023.github.io)

> **Live demo →** [your-app-url.streamlit.app](https://YOUR_APP_URL.streamlit.app)  
> Try the cold-start demo: pick 3 games you like and get personalized recommendations instantly — no account needed.

---

## Overview

End-to-end deep learning recommendation system trained on Amazon Video Games 2023 (98,906 users, 26,354 items, 99.97% sparsity). Built three production-grade models — Matrix Factorization, LightGCN, and Two-Tower — and conducted a systematic 12-variant ablation study to understand what actually improves retrieval quality.

The Two-Tower model serves recommendations via FAISS at **29 μs per query** and handles cold-start users through GRU encoding over browsed item text — something neither MF nor LightGCN can do.

---

## Results

### Sampled Evaluation (100 negatives per user)

| Model | HR@5 | HR@10 | NDCG@10 | Cold-Start | FAISS Ready |
|---|---|---|---|---|---|
| MF (BPR) | — | 0.6825 | 0.4550 | No | Partial |
| Two-Tower v5 | — | 0.6395 | 0.4148 | **Yes** | **Yes (<1ms)** |
| LightGCN | — | **0.7290** | **0.4940** | No | No |

### Full Ranking (all 26,354 items — publication standard)

| Model | HR@5 | HR@10 | HR@20 | NDCG@10 | NDCG@20 |
|---|---|---|---|---|---|
| MF (BPR) | 0.0280 | 0.0420 | 0.0610 | 0.0222 | 0.0271 |
| Two-Tower v5 | 0.0140 | 0.0240 | 0.0430 | 0.0113 | 0.0161 |
| LightGCN | **0.0280** | **0.0430** | **0.0720** | **0.0247** | **0.0320** |

### Simulated Cold-Start (HR@10 by history size)

| History Size | MF | LightGCN | Two-Tower |
|---|---|---|---|
| 3 interactions | 0.4960 | 0.4000 | **0.4900** |
| 5 interactions | 0.4960 | 0.3980 | **0.4900** |
| 10 interactions | 0.4960 | 0.4120 | **0.4920** |
| 20 interactions | 0.4980 | 0.4440 | **0.4940** |
| Full history | 0.4980 | **0.5000** | 0.4940 |
| **Brand new user** | **Cannot serve** | **Cannot serve** | **Works** |

MF maintains constant performance across history sizes because it ignores sequence entirely. Two-Tower degrades gracefully and is the only option for truly new users.

---

## 12-Variant Ablation Study

Systematic ablation to isolate the contribution of each component:

| Version | Change | HR@10 | NDCG@10 | Finding |
|---|---|---|---|---|
| v1 | Baseline (InfoNCE, b=256) | 0.6195 | 0.4000 | Baseline |
| v2 | + MSE distillation (α=0.5) | 0.6210 | — | Marginal — LightGCN gradient drowned InfoNCE |
| v3 | + Cosine distillation (α=0.9) | 0.6195 | — | No gain — structural info lost in cosine |
| v4 | + Title text embeddings | 0.6355 | 0.4147 | **+2.6% — text helps** |
| v4b | + Larger batch (1024) | 0.6280 | 0.4042 | Too many in-batch negatives hurt |
| v4-BPR | BPR + hard negatives | 0.2295 | — | **Collapsed** — BPR incompatible with InfoNCE scale |
| **v5** | **+ GRU sequential encoding** | **0.6395** | **0.4148** | **Best MLP variant** |
| v5b | + Rich text (desc+features) | 0.6295 | 0.4078 | Noisy text hurt slightly |
| v5c | + LightGCN init | 0.6320 | 0.4176 | Init scrambled by InfoNCE loss |
| v6 | Curriculum negatives | 0.6375 | 0.4133 | No gain over random negs |
| v7 | + CLIP image embeddings | 0.6405 | 0.4143 | Marginal — images add noise on sparse data |
| v8 | FM-style additive fusion | 0.6400 | 0.4264 | Interpretable gate weights |

**Key finding:** On 99.97% sparse data, ID collaborative signal dominates. Text embeddings help (+2.6%) but are most valuable for cold-start, not warm users.

### FM Gate Weights — What the Model Learned (v8)

The FM-style model (v8) learned explicit gate weights revealing signal contribution:

| Component | User Side | Item Side |
|---|---|---|
| ID embedding | **63%** | **54%** |
| Sequential (GRU) | 27% | — |
| Text embedding | — | 23% |
| Features | 10% | 22% |

Collaborative ID signals carry 2-3x more weight than content features on this dataset. Content features become critical only for cold-start and new item scenarios.

---

## Architecture

### Two-Tower (Retrieval)

```
User Tower                          Item Tower
──────────────────────              ──────────────────────
User ID  →  Embedding (64d)         Item ID  →  Embedding (64d)
History  →  GRU (20 steps, 64d)     Title    →  SentenceTransformer (384d)
Features →  Linear (8 → 64d)                    → Projection (64d)
                                    Features →  Linear (15 → 64d)
     ↓                                   ↓
  Concat → MLP → LayerNorm → 64d    Concat → MLP → LayerNorm → 64d
     ↓                                   ↓
  L2 normalize                       L2 normalize
                    ↓
              Dot product
                    ↓
            InfoNCE loss (τ=0.2)
```

### FAISS Serving Pipeline

```
Offline: Pre-compute 26,354 item vectors → FAISS HNSW index

Online:  User request
              ↓
         User Tower (1 forward pass, ~0.5ms)
              ↓
         FAISS HNSW search (29 μs)
              ↓
         Top 1,000 candidates
              ↓
         Re-ranking layer (LightGCN scores)
              ↓
         Top 10 shown to user
```

### FAISS Index Comparison

| Index Type | Latency | Queries/Second | Accuracy |
|---|---|---|---|
| HNSW | **29 μs** | 34,483 | ~99% recall |
| IVF (approximate) | 35 μs | 28,571 | ~97% recall |
| Flat (exact) | 310 μs | 3,226 | 100% |
| Brute force GPU | 894 μs | 1,119 | 100% |

---

## Cold-Start Solution

New users have no ID embedding. Two-Tower handles this via GRU encoding over browsed item text:

1. User browses 3+ items (no purchase needed)
2. Sentence-Transformers encodes each item title (384d)
3. GRU reads the sequence → 64d user representation
4. FAISS retrieves top-1,000 relevant items
5. Recommendations served in real time

MF and LightGCN require the user to exist in the training set — they cannot serve new users at all.

---

## Dataset

**Amazon Video Games 2023** — [McAuley Lab](https://amazon-reviews-2023.github.io)

| Stat | Value |
|---|---|
| Users | 98,906 |
| Items | 26,354 |
| Interactions | 659,693 (train) + 98,906 (val) |
| Sparsity | 99.97% |
| User features | 8 (activity, avg rating given, recency, etc.) |
| Item features | 15 (price, avg rating, rating count, category, etc.) |
| Text | Item titles via `all-MiniLM-L6-v2` (384d) |
| Images | Item cover images via CLIP ViT-B/32 (512d) |

---

## Project Structure

```
Two_Tower_Recommendation_System/
├── app/
│   └── streamlit_app.py        # Live demo app
├── src/
│   └── models/
│       ├── mf.py               # Matrix Factorization (BPR)
│       ├── lightgcn.py         # LightGCN (3-layer)
│       └── two_tower.py        # Two-Tower v5 (GRU + text + features)
├── notebooks/
│   └── training.ipynb          # Full training + ablation notebook
├── requirements.txt
└── README.md
```

---

## Training Details

| Hyperparameter | MF | LightGCN | Two-Tower v5 |
|---|---|---|---|
| Embedding dim | 64 | 64 | 64 |
| Batch size | 2048 | 2048 | 256 |
| Loss | BPR | BPR | InfoNCE (τ=0.2) |
| Optimizer | Adam | Adam | Adam |
| LR | 1e-3 | 1e-3 | 1e-3 |
| Epochs (best) | 20 | 27 | 8 |
| Parameters | ~8M | ~8M | ~8.1M |
| Device | A100 (Colab) | A100 (Colab) | A100 (Colab) |

---

## Why These Three Models?

| | MF | LightGCN | Two-Tower |
|---|---|---|---|
| Methodology | Collaborative filtering | Graph neural network | Dual-encoder + content |
| Accuracy (known users) | 2nd | **1st** | 3rd |
| Cold-start | No | No | **Yes** |
| New item support | No | No | **Yes** |
| FAISS deployable | Partial | No | **Yes** |
| Industry role | Baseline | Re-ranking | **Retrieval** |
| Scale (100M+ items) | No | No | **Yes** |

LightGCN wins accuracy but cannot deploy at scale or handle new users. Two-Tower is the production retrieval standard — used by YouTube, Pinterest, and DoorDash — precisely because it solves cold-start and serves via FAISS at microsecond latency.

---

## Running Locally

```bash
git clone https://github.com/nidhi1603/Two_Tower_Recommendation_System
cd Two_Tower_Recommendation_System
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

The app downloads model files from Google Drive on first run (~110 MB, cached after).

---

## Built With

- **PyTorch** — model training
- **FAISS** — approximate nearest neighbor retrieval
- **Sentence-Transformers** — text embeddings (`all-MiniLM-L6-v2`)
- **CLIP** (OpenCLIP ViT-B/32) — image embeddings
- **Streamlit** — live demo
- **Plotly** — interactive charts

---

*Amazon Video Games 2023 dataset — [McAuley Lab, UCSD](https://amazon-reviews-2023.github.io)*
