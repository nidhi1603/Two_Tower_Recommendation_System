# Two-Tower Recommendation System

Deep learning recommendation system on Amazon Video Games 2023.
98,906 users | 26,354 items | 99.97% sparsity

## Models

| Model | HR@10 (sampled) | HR@10 (full-rank) | Role |
|---|---|---|---|
| MF (BPR) | 0.6755 | 0.0420 | Collaborative baseline |
| **Two-Tower v5** | **0.6395** | **0.0270** | **FAISS-deployable retriever** |
| LightGCN | 0.7285 | 0.0440 | Graph-based ranker |

## Key Finding

12-variant ablation study proves that on ultra-sparse data,
**model complexity must match data density**. MLP bottlenecks
lose more collaborative signal than content features add.
FM-style learned gates show ID embeddings carry 62% of user
signal vs 28% GRU and 10% features.

## Architecture

```
Two-Tower (retrieval) --> FAISS Index (<1ms) --> Top 1000
                                                    |
                                              LightGCN (re-rank)
                                                    |
                                              Top 10 shown
```

## FAISS Latency

| Index | Latency/query |
|---|---|
| HNSW | 29 us |
| IVF | 35 us |
| Flat | 310 us |
| Brute-force | 894 us |

## Ablation (12 Variants)

What worked: title text (+2.6%), GRU sequence (+0.6%)

What failed: knowledge distillation, hard negatives, larger batch,
rich text, CLIP images, LightGCN init, curriculum negatives, FM-style

## Project Structure

```
src/models/       - MF, LightGCN, Two-Tower, Two-Tower FM
src/data/         - Preprocessing, text embeddings
src/serving/      - FAISS index for production serving
results/          - Full ablation + publication-ready metrics
```
