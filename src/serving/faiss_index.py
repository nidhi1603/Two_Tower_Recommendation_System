import numpy as np
import faiss
import torch


class FAISSRecommender:
    def __init__(self, model, n_items, device, index_type='flat'):
        self.model = model
        self.device = device
        self.n_items = n_items
        self._build_index(index_type)

    def _build_index(self, index_type):
        self.model.eval()
        with torch.no_grad():
            items = torch.arange(self.n_items, device=self.device)
            embs = []
            for i in range(0, self.n_items, 4096):
                embs.append(self.model.encode_items(items[i:i+4096]))
            self.item_embs = torch.cat(embs).cpu().numpy()
        dim = self.item_embs.shape[1]
        if index_type == 'ivf':
            quantizer = faiss.IndexFlatIP(dim)
            self.index = faiss.IndexIVFFlat(quantizer, dim, 256, faiss.METRIC_INNER_PRODUCT)
            self.index.train(self.item_embs)
            self.index.nprobe = 16
        else:
            self.index = faiss.IndexFlatIP(dim)
        self.index.add(self.item_embs)

    def recommend(self, user_ids, k=10, exclude_history=None):
        self.model.eval()
        with torch.no_grad():
            u_embs = self.model.encode_users(user_ids).cpu().numpy()
        scores, indices = self.index.search(u_embs, k + 100)
        results = []
        for i, uid in enumerate(user_ids.cpu().tolist()):
            seen = exclude_history.get(uid, set()) if exclude_history else set()
            recs = [(idx, sc) for idx, sc in zip(indices[i], scores[i]) if idx not in seen][:k]
            results.append(recs)
        return results
