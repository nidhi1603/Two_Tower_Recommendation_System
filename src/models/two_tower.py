"""Two-Tower v5: GRU + Text Embeddings."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence

class TwoTower(nn.Module):
    def __init__(self, n_users, n_items, n_user_feats, n_item_feats,
                 text_dim=384, emb_dim=64, gru_hidden=64, hidden=128):
        super().__init__()
        self.user_id_emb = nn.Embedding(n_users, emb_dim)
        self.item_id_emb = nn.Embedding(n_items, emb_dim)
        nn.init.normal_(self.user_id_emb.weight, std=0.01)
        nn.init.normal_(self.item_id_emb.weight, std=0.01)
        self.text_proj = nn.Linear(text_dim, emb_dim)
        self.user_gru = nn.GRU(input_size=emb_dim, hidden_size=gru_hidden,
                                num_layers=1, batch_first=True)
        self.user_tower = nn.Sequential(
            nn.Linear(gru_hidden + emb_dim + n_user_feats, hidden),
            nn.ReLU(), nn.LayerNorm(hidden),
            nn.Linear(hidden, emb_dim))
        self.item_tower = nn.Sequential(
            nn.Linear(emb_dim + n_item_feats + emb_dim, hidden),
            nn.ReLU(), nn.LayerNorm(hidden),
            nn.Linear(hidden, emb_dim))

    def encode_users(self, user_ids, user_feats, user_seqs, seq_lengths, text_embs):
        seqs = user_seqs[user_ids]
        lengths = seq_lengths[user_ids]
        proj = self.text_proj(text_embs[seqs.clamp(min=0)])
        proj = proj * (seqs >= 0).unsqueeze(-1).float()
        packed = pack_padded_sequence(proj, lengths.cpu().clamp(min=1),
                                       batch_first=True, enforce_sorted=False)
        _, hidden = self.user_gru(packed)
        x = torch.cat([hidden.squeeze(0), self.user_id_emb(user_ids), user_feats[user_ids]], dim=1)
        return F.normalize(self.user_tower(x), dim=1)

    def encode_items(self, item_ids, item_feats, text_embs):
        x = torch.cat([self.item_id_emb(item_ids), item_feats[item_ids],
                        self.text_proj(text_embs[item_ids])], dim=1)
        return F.normalize(self.item_tower(x), dim=1)
