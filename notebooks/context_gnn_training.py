# ============================================================
# Context-GNN Training Cell
# Paste this into your Colab notebook after the re-setup block.
# Requires: preprocessed data + text embeddings on Drive
# ============================================================

import os, json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader

DRIVE_PROCESSED = '/content/drive/MyDrive/two_tower_data/processed'
DRIVE_CKPT      = '/content/drive/MyDrive/two_tower_data/checkpoints'
os.makedirs(DRIVE_CKPT, exist_ok=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ── Load data ────────────────────────────────────────────
with open(f'{DRIVE_PROCESSED}/stats.json') as f:
    stats = json.load(f)
N_USERS      = stats['n_users']
N_ITEMS      = stats['n_items']
N_USER_FEATS = stats['n_user_features']
N_ITEM_FEATS = stats['n_item_features']

train_df     = pd.read_parquet(f'{DRIVE_PROCESSED}/train.parquet')
val_df       = pd.read_parquet(f'{DRIVE_PROCESSED}/val.parquet')
user_feats   = pd.read_parquet(f'{DRIVE_PROCESSED}/user_features.parquet').values.astype(np.float32)
item_feats   = pd.read_parquet(f'{DRIVE_PROCESSED}/item_features.parquet').values.astype(np.float32)
text_embs    = np.load(f'{DRIVE_PROCESSED}/item_text_embeddings.npy').astype(np.float32)
user_history = train_df.groupby('user_idx')['item_idx'].apply(set).to_dict()

user_feats_t = torch.tensor(user_feats, device=device)
item_feats_t = torch.tensor(item_feats, device=device)
text_embs_t  = torch.tensor(text_embs, device=device)
N_TEXT_DIM   = text_embs.shape[1]

print(f"Users: {N_USERS:,} | Items: {N_ITEMS:,}")
print(f"User feats: {N_USER_FEATS} | Item feats: {N_ITEM_FEATS} | Text dim: {N_TEXT_DIM}")

# ── Build adjacency matrix ───────────────────────────────
def build_adj(train_df, n_users, n_items):
    users = train_df['user_idx'].values
    items = train_df['item_idx'].values + n_users
    row   = np.concatenate([users, items])
    col   = np.concatenate([items, users])
    N     = n_users + n_items
    deg   = np.bincount(row, minlength=N).astype(np.float32)
    d_inv = np.where(deg > 0, 1.0 / np.sqrt(deg), 0.0)
    vals  = d_inv[row] * d_inv[col]
    idx   = torch.tensor(np.stack([row, col]), dtype=torch.long)
    v     = torch.tensor(vals, dtype=torch.float32)
    return torch.sparse_coo_tensor(idx, v, (N, N)).coalesce().to(device)

adj = build_adj(train_df, N_USERS, N_ITEMS)
print(f"Adj matrix: {adj.shape[0]:,} x {adj.shape[1]:,} | edges: {adj._nnz():,}")

# ── Context-GNN Model ───────────────────────────────────
class ContextGNN(nn.Module):
    def __init__(self, n_users, n_items, n_user_feats, n_item_feats,
                 text_dim=384, dim=64, n_layers=3, dropout=0.1):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.n_layers = n_layers
        self.dim = dim

        self.user_emb = nn.Embedding(n_users, dim)
        self.item_emb = nn.Embedding(n_items, dim)
        nn.init.normal_(self.user_emb.weight, std=0.01)
        nn.init.normal_(self.item_emb.weight, std=0.01)

        self.user_feat_proj = nn.Linear(n_user_feats, dim, bias=False)
        self.item_feat_proj = nn.Linear(n_item_feats, dim, bias=False)
        self.text_proj = nn.Linear(text_dim, dim, bias=False)

        self.attn_layers = nn.ModuleList([
            nn.Linear(2 * dim, 1, bias=False) for _ in range(n_layers)
        ])

        self.feat_gate = nn.Parameter(torch.tensor(0.3))
        self.dropout = nn.Dropout(dropout)

    def _build_node_features(self):
        u_feat = self.user_feat_proj(user_feats_t)
        i_feat = self.item_feat_proj(item_feats_t) + self.text_proj(text_embs_t)
        return torch.cat([u_feat, i_feat], dim=0)

    def propagate(self, adj):
        E0 = torch.cat([self.user_emb.weight, self.item_emb.weight], dim=0)
        node_ctx = self._build_node_features()

        embs = [E0]
        E = E0

        indices = adj.coalesce().indices()
        src, dst = indices[0], indices[1]
        base_vals = adj.coalesce().values()

        for layer_idx in range(self.n_layers):
            src_feat = node_ctx[src]
            dst_feat = node_ctx[dst]
            attn_input = torch.cat([src_feat, dst_feat], dim=1)
            attn_raw = self.attn_layers[layer_idx](attn_input).squeeze(-1)
            attn_raw = attn_raw * base_vals

            # Scatter softmax (vectorized, no Python loop)
            attn_max = torch.zeros(adj.shape[0], device=device)
            attn_max.scatter_reduce_(0, dst, attn_raw, reduce='amax',
                                     include_self=False)
            attn_shifted = attn_raw - attn_max[dst]
            attn_exp = torch.exp(attn_shifted)
            attn_sum = torch.zeros(adj.shape[0], device=device)
            attn_sum.scatter_add_(0, dst, attn_exp)
            attn_weights = attn_exp / (attn_sum[dst] + 1e-8)

            attn_adj = torch.sparse_coo_tensor(indices, attn_weights,
                                                adj.shape, device=device)
            E = torch.sparse.mm(attn_adj, E)
            E = self.dropout(E)
            embs.append(E)

        E_graph = torch.stack(embs, dim=1).mean(dim=1)
        gate = torch.sigmoid(self.feat_gate)
        E_final = (1 - gate) * E_graph + gate * node_ctx

        return E_final[:self.n_users], E_final[self.n_users:]

    def bpr_loss(self, users, pos_items, neg_items, adj):
        user_embs, item_embs = self.propagate(adj)
        u  = user_embs[users]
        pi = item_embs[pos_items]
        ni = item_embs[neg_items]
        pos_scores = (u * pi).sum(dim=1)
        neg_scores = (u * ni).sum(dim=1)
        loss = -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-8).mean()
        reg = (self.user_emb(users).norm(2).pow(2) +
               self.item_emb(pos_items).norm(2).pow(2) +
               self.item_emb(neg_items).norm(2).pow(2)) / len(users)
        return loss + 1e-4 * reg

# ── Dataset (same BPR sampling as your other models) ─────
class BPRDataset(Dataset):
    def __init__(self, df, n_items, user_history):
        self.users     = df['user_idx'].values
        self.pos_items = df['item_idx'].values
        self.n_items   = n_items
        self.user_history = user_history
    def __len__(self): return len(self.users)
    def __getitem__(self, idx):
        u   = self.users[idx]
        pos = self.pos_items[idx]
        while True:
            neg = np.random.randint(0, self.n_items)
            if neg not in self.user_history.get(u, set()):
                break
        return u, pos, neg

# ── Evaluation (same as your notebook) ───────────────────
def evaluate(model, adj, val_df, user_history, n_items, K=10, n_neg=100, n_eval=2000):
    model.eval()
    rng = np.random.default_rng(42)
    hr_list, ndcg_list = [], []
    sample_df = val_df.sample(min(n_eval, len(val_df)), random_state=42)
    with torch.no_grad():
        user_embs, item_embs = model.propagate(adj)
        for _, row in sample_df.iterrows():
            u, pos = int(row['user_idx']), int(row['item_idx'])
            seen   = user_history.get(u, set()) | {pos}
            negs   = []
            while len(negs) < n_neg:
                cands = rng.integers(0, n_items, n_neg * 2).tolist()
                negs.extend([c for c in cands if c not in seen])
            negs      = negs[:n_neg]
            candidates = [pos] + negs
            u_emb  = user_embs[u].unsqueeze(0)
            i_embs = item_embs[candidates]
            scores = (u_emb * i_embs).sum(dim=1).cpu().numpy()
            rank   = int(np.where(np.argsort(-scores) == 0)[0][0]) + 1
            hr_list.append(1.0 if rank <= K else 0.0)
            ndcg_list.append(1.0 / np.log2(rank + 1) if rank <= K else 0.0)
    return np.mean(hr_list), np.mean(ndcg_list)

# ── Training ─────────────────────────────────────────────
BATCH_SIZE = 2048
EPOCHS     = 30
PATIENCE   = 5
LR         = 1e-3

model     = ContextGNN(N_USERS, N_ITEMS, N_USER_FEATS, N_ITEM_FEATS,
                        text_dim=N_TEXT_DIM, dim=64, n_layers=3, dropout=0.1).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
dataset   = BPRDataset(train_df, N_ITEMS, user_history)
loader    = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

best_hr, best_epoch, patience_counter = 0.0, 0, 0

print(f"\nTraining Context-GNN (3 layers, dim=64, feature-attention) for {EPOCHS} epochs...")
print(f"Batch: {BATCH_SIZE} | LR: {LR} | Dropout: 0.1")
print(f"\n{'Epoch':>5} {'Loss':>10} {'HR@10':>8} {'NDCG@10':>10} {'Gate':>6}")
print("-" * 46)

for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0

    for users, pos_items, neg_items in loader:
        users     = users.to(device)
        pos_items = pos_items.to(device)
        neg_items = neg_items.to(device)

        loss = model.bpr_loss(users, pos_items, neg_items, adj)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    hr, ndcg = evaluate(model, adj, val_df, user_history, N_ITEMS)
    gate_val = torch.sigmoid(model.feat_gate).item()
    print(f"{epoch:>5} {avg_loss:>10.4f} {hr:>8.4f} {ndcg:>10.4f} {gate_val:>6.2f}")

    if hr > best_hr:
        best_hr, best_epoch, patience_counter = hr, epoch, 0
        torch.save({
            'epoch': epoch, 'hr': hr, 'ndcg': ndcg,
            'gate': gate_val,
            'model_state': model.state_dict(),
            'config': {
                'arch': 'Context-GNN (feature-attention message passing)',
                'n_layers': 3, 'dim': 64, 'dropout': 0.1,
                'text_dim': N_TEXT_DIM,
            }
        }, f'{DRIVE_CKPT}/context_gnn_best.pt')
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f"\nEarly stopping at epoch {epoch}")
            break

# ── Results ──────────────────────────────────────────────
ckpt = torch.load(f'{DRIVE_CKPT}/context_gnn_best.pt', weights_only=False)
gate = ckpt['gate']

print(f"\n{'='*55}")
print(f"  CONTEXT-GNN RESULTS")
print(f"{'='*55}")
print(f"  Best Epoch:     {ckpt['epoch']}")
print(f"  HR@10:          {ckpt['hr']:.4f}")
print(f"  NDCG@10:        {ckpt['ndcg']:.4f}")
print(f"  Feature Gate:   {gate:.2f} (graph={1-gate:.0%}, features={gate:.0%})")
print(f"\n  LEADERBOARD:")
print(f"  MF:             HR@10=0.6825")
print(f"  Two-Tower v5:   HR@10=0.6395")
print(f"  LightGCN:       HR@10=0.7290")
print(f"  Context-GNN:    HR@10={ckpt['hr']:.4f}  <-- NEW")
print(f"\n  Key insight: Feature gate = {gate:.2f}")
print(f"  → Graph signal contributes {1-gate:.0%}, features contribute {gate:.0%}")
print(f"  Compare with FM Two-Tower: ID=63%, GRU=27%, Features=10%")
