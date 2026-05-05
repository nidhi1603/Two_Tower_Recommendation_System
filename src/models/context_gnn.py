"""Feature-Gated LightGCN: LightGCN + side feature projections + learnable gate.

Extends LightGCN by projecting user features, item features, and text embeddings
into the embedding space, then blending graph-propagated embeddings with projected
features via a learnable sigmoid gate. The gate learns how much contextual signal
to mix with collaborative signal.

Trained result: gate = 0.18 → graph contributes 82%, features 18%.
"""

import torch
import torch.nn as nn


class ContextGNN(nn.Module):
    def __init__(self, n_users, n_items, n_user_feats, n_item_feats,
                 text_dim=384, dim=64, n_layers=3, dropout=0.05):
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

        self.feat_gate = nn.Parameter(torch.tensor(0.3))
        self.dropout = nn.Dropout(dropout)

    def propagate(self, adj, user_feats, item_feats, text_embs):
        E0 = torch.cat([self.user_emb.weight, self.item_emb.weight], dim=0)

        embs = [E0]
        E = E0
        for _ in range(self.n_layers):
            E = torch.sparse.mm(adj, E)
            E = self.dropout(E)
            embs.append(E)

        E_graph = torch.stack(embs, dim=1).mean(dim=1)

        u_feat = self.user_feat_proj(user_feats)
        i_feat = self.item_feat_proj(item_feats) + self.text_proj(text_embs)
        node_ctx = torch.cat([u_feat, i_feat], dim=0)

        gate = torch.sigmoid(self.feat_gate)
        E_final = (1 - gate) * E_graph + gate * node_ctx

        return E_final[:self.n_users], E_final[self.n_users:]

    def forward(self, users, items, adj, user_feats, item_feats, text_embs):
        user_embs, item_embs = self.propagate(adj, user_feats, item_feats, text_embs)
        return (user_embs[users] * item_embs[items]).sum(dim=1)

    def bpr_loss(self, users, pos_items, neg_items, adj,
                 user_feats, item_feats, text_embs):
        user_embs, item_embs = self.propagate(adj, user_feats, item_feats, text_embs)
        u = user_embs[users]
        pi = item_embs[pos_items]
        ni = item_embs[neg_items]
        pos_scores = (u * pi).sum(dim=1)
        neg_scores = (u * ni).sum(dim=1)
        loss = -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-8).mean()
        reg = (self.user_emb(users).norm(2).pow(2) +
               self.item_emb(pos_items).norm(2).pow(2) +
               self.item_emb(neg_items).norm(2).pow(2)) / len(users)
        return loss + 1e-4 * reg
