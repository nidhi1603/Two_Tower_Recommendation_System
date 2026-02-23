import torch
import torch.nn as nn
import numpy as np

class LightGCN(nn.Module):
    def __init__(self, n_users, n_items, dim=64, n_layers=3):
        super().__init__()
        self.n_users  = n_users
        self.n_items  = n_items
        self.n_layers = n_layers
        self.user_emb = nn.Embedding(n_users, dim)
        self.item_emb = nn.Embedding(n_items, dim)
        nn.init.normal_(self.user_emb.weight, std=0.01)
        nn.init.normal_(self.item_emb.weight, std=0.01)

    def propagate(self, adj):
        E0 = torch.cat([self.user_emb.weight, self.item_emb.weight], dim=0)
        embs = [E0]
        E = E0
        for _ in range(self.n_layers):
            E = torch.sparse.mm(adj, E)
            embs.append(E)
        E_final = torch.stack(embs, dim=1).mean(dim=1)
        return E_final[:self.n_users], E_final[self.n_users:]

    def forward(self, users, items, adj):
        user_embs, item_embs = self.propagate(adj)
        return (user_embs[users] * item_embs[items]).sum(dim=1)

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
