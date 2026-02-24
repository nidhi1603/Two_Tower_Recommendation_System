import torch
import torch.nn as nn
import torch.nn.functional as F

class TwoTower(nn.Module):
    def __init__(self, n_users, n_items, n_user_feats, n_item_feats, emb_dim=64, hidden=128):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, emb_dim)
        self.item_emb = nn.Embedding(n_items, emb_dim)
        nn.init.normal_(self.user_emb.weight, std=0.01)
        nn.init.normal_(self.item_emb.weight, std=0.01)
        self.user_tower = nn.Sequential(
            nn.Linear(emb_dim + n_user_feats, hidden),
            nn.ReLU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, emb_dim)
        )
        self.item_tower = nn.Sequential(
            nn.Linear(emb_dim + n_item_feats, hidden),
            nn.ReLU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, emb_dim)
        )

    def encode_users(self, user_ids, user_feats):
        x = torch.cat([self.user_emb(user_ids), user_feats[user_ids]], dim=1)
        return F.normalize(self.user_tower(x), dim=1)

    def encode_items(self, item_ids, item_feats):
        x = torch.cat([self.item_emb(item_ids), item_feats[item_ids]], dim=1)
        return F.normalize(self.item_tower(x), dim=1)
