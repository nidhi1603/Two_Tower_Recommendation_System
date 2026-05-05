# What's Good vs What Can Be Improved

## What Was Done Well

1. **Systematic ablation study (12 variants)** — Most projects build one model and stop. You isolated variables one at a time and measured their contribution. This is actual research methodology.

2. **Three complementary models** — MF (baseline), LightGCN (accuracy), Two-Tower (production). You covered the full spectrum from simple to deployable.

3. **Cold-start solution** — The GRU-based fallback for new users is a real differentiator. Most academic projects ignore cold-start entirely.

4. **FM gate weight analysis** — Showing that ID signal = 63% vs features = 10% is a genuinely useful finding backed by learned parameters, not speculation.

5. **Full-ranking evaluation** — You did both sampled (100 neg) and full-ranking (all 26K items). Full ranking is publication-standard and most student projects skip it.

6. **FAISS serving pipeline** — Showed you understand the retrieval-then-ranking production pattern, not just offline metrics.

---

## What Can Be Improved

### High Priority (would meaningfully strengthen the project)

- [ ] **No training notebook or scripts in repo.** The `notebooks/` directory listed in the README doesn't exist. Anyone cloning this repo cannot reproduce your results. Add the Colab notebook or convert it to a standalone training script.

- [ ] **No evaluation code.** There's no eval script for HR@10/NDCG@10. The metrics exist only as numbers in results/ text files. Add a reusable `evaluate.py` that takes a model + test set and computes metrics.

- [ ] **Preprocessing is hardcoded to Google Drive paths.** `preprocess.py` uses `/content/drive/MyDrive/...` — this only works in your specific Colab session. Make paths configurable via arguments or a config file.

- [ ] **No adjacency matrix construction code.** LightGCN needs a normalized sparse adjacency matrix, but there's no code that builds it from the interaction data. This is a critical missing piece for reproducibility.

- [x] **Context-GNN integration.** Model in `src/models/context_gnn.py`, training notebook in `notebooks/Context_GNN_Colab.ipynb`. Results: HR@10=0.7190, NDCG@10=0.4824. Feature gate=0.18 (graph 82%, features 18%). Added to README and results/.

### Medium Priority (polish and depth)

- [ ] **No hyperparameter tuning documentation.** You settled on lr=1e-3, dim=64, batch=256 — but was this searched or copied from a paper? Document what you tried and why these were chosen.

- [ ] **Missing data download instructions.** README says "downloads from Google Drive on first run" but the training data pipeline has no public download link. New users can run the Streamlit demo but can't retrain.

- [ ] **No statistical significance testing.** Is the 2.6% improvement from text embeddings (v4) real or noise? Run multiple seeds and report mean +/- std. Even 3 seeds would help.

- [ ] **LightGCN cannot be used for FAISS.** You mention "re-ranking with LightGCN" but LightGCN needs the full adjacency matrix at inference — it can't pre-compute standalone embeddings. The re-ranking pipeline as described doesn't quite work. Clarify or implement it properly.

- [ ] **No negative sampling strategy comparison.** You tested BPR vs InfoNCE but didn't try hard negative mining (popularity-weighted, in-batch hard negatives) which often gives significant gains.

### Lower Priority (nice-to-haves)

- [ ] **Streamlit app doesn't show Context-GNN.** Add it as a fourth model option in the comparison view.

- [ ] **No model checkpointing code visible.** If training crashes at epoch 25 of 27, do you restart from scratch? Add checkpoint saving/loading.

- [ ] **`__init__.py` files are empty.** Consider adding explicit imports so users can do `from src.models import LightGCN`.

- [ ] **No unit tests for models.** `test_smoke.py` exists but likely only checks imports. Add tests that verify forward pass shapes, loss computation, and gradient flow.

- [ ] **Results files are plain text.** The `results/` directory has unstructured text files. Convert to structured format (JSON or CSV) so results can be programmatically compared.

- [ ] **No learning rate scheduling.** All models use fixed lr=1e-3. Cosine annealing or ReduceLROnPlateau could squeeze out a few more points, especially for LightGCN.

- [ ] **README claims HNSW index but code only implements Flat and IVF.** `faiss_index.py` has `flat` and `ivf` options but no HNSW. Either add HNSW or correct the README.
