# Presentation Prep — Complete Technical Guide
### EAS 509 | Two-Tower Recommendation System | Nidhi Rajani

---

## PART 1: 4-MINUTE PRESENTATION SCRIPT

> **Structure:** Hook (30s) → Dataset + Problem (30s) → Architecture walk (90s) → Live Demo (45s) → Results (30s) → My contribution (15s)

---

### [0:00 – 0:30] HOOK

> *"Imagine you open Netflix and it recommends exactly what you want — with no account. Or you open Amazon and the front page already knows your taste. That's not magic. It's a recommendation system. I built one from scratch, trained four different deep learning models on 98,000 real Amazon users, and figured out what actually works — and what doesn't."*

**Why this lands:** You're connecting to something everyone has experienced. Don't open with "my project is about." Open with the human problem.

---

### [0:30 – 1:00] DATASET & PROBLEM

> *"The dataset is Amazon Video Games 2023 — 98,906 users, 26,354 items, 659,000 purchases. The challenge? 99.97% of the user-item matrix is empty. Most users only bought 5-7 games. Most items have very few reviews. That's called the sparsity problem — and it's the hardest thing in recommendation systems."*

> *"I asked three questions: Does graph structure beat content features? Can we serve brand-new users who have zero purchase history? And can we do this in microseconds at scale?"*

---

### [1:00 – 2:30] ARCHITECTURE WALK (show Streamlit → Model Architectures)

**Matrix Factorization — the baseline:**
> *"First, I built the simplest possible model — Matrix Factorization. Every user gets a 64-number vector, every item gets a 64-number vector. The recommendation score is just the dot product — think of it as measuring how aligned those two vectors are. Trained with BPR loss — Bayesian Personalized Ranking — which just says: the score for something you bought should be higher than a random item you didn't. No features, no graph, no text. Just ID co-occurrence. HR@10 of 0.68."*

**LightGCN — the accuracy king:**
> *"Next, LightGCN. Users and items form a graph — you bought a game, there's an edge between you and that item. LightGCN runs 3 layers of neighborhood averaging. Layer 1 says 'who else bought what this user bought.' Layer 2 says 'what do friends-of-friends buy.' Layer 3 goes even deeper. Then it averages all layers. No fancy nonlinearities — just sparse matrix multiplies. Deliberately simple. And it won — HR@10 of 0.729, best accuracy."*

**Two-Tower — the production model:**
> *"But LightGCN has a fatal flaw. It needs the full graph at every inference. You can't pre-compute and cache anything. And a brand-new user? It can't serve them at all. That's where Two-Tower comes in — this is what YouTube, Pinterest, and DoorDash actually use in production.*

> *Two separate neural networks — one for users, one for items. The user tower takes: your ID embedding, a GRU running over your last 20 purchases to capture your taste sequence, and 8 user features like activity level and recency. The item tower takes: item ID, text embedding of the title from SentenceTransformer, and 15 item features like price and rating count. Both towers output a 64-dimensional vector, L2-normalized. Score = dot product.*

> *The key insight: I can pre-compute all 26,354 item vectors once, load them into FAISS — that's Facebook's approximate nearest neighbor library — and then at serving time, I just encode the user and search. 29 microseconds. That's 34,000 recommendations per second on a single CPU."*

**Feature-Gated LightGCN — my contribution:**
> *"My novel contribution: I asked — what if we combine graph structure with side features? I kept LightGCN's graph propagation, but added projections for user features, item features, and text embeddings. Then I added a single learnable parameter — a sigmoid gate — that lets the model itself decide how much to weight graph vs features. It started at 0.62 and converged to 0.18. The model independently learned: 82% graph signal, 18% features. This confirmed our whole ablation study."*

---

### [2:30 – 3:15] LIVE DEMO (switch to Streamlit → Live Demo)

> *"Let me show it live."*

**Tab 1 — Existing User:**
> *"I pick User 100. You can see their purchase history — they bought mostly action games. Two-Tower recommends genre-similar items using text. MF recommends items that co-appear in other users' histories. They overlap on 4-5 items — those are high-confidence recommendations both models agree on. The ones that differ show the fundamental difference: Two-Tower understands content, MF understands behavior."*

**Tab 2 — Cold-Start:**
> *"Now the real party trick. I pick a scenario: Souls-like games — Dark Souls, Elden Ring, Sekiro. This user has never made a purchase. Watch: the GRU encodes the text embeddings of those three items, creates a user representation from scratch, and FAISS finds the nearest items. Real recommendations in milliseconds — no account needed. MF outputs nothing. LightGCN outputs nothing. Only Two-Tower works."*

---

### [3:15 – 3:45] RESULTS

> *"Results. Sampled evaluation against 100 negatives per user: LightGCN wins accuracy at 0.729, Feature-Gated LightGCN gets 0.719, MF gets 0.68, Two-Tower gets 0.64. But on the full-ranking evaluation — all 26,354 items, no shortcuts — LightGCN and MF are comparable at 0.042-0.044 HR@10. These are publication-standard metrics. Most student projects only do sampled eval.*

> *The 12-variant ablation study showed: text embeddings give +2.6%, GRU gives another +0.4%, but the wrong loss function — BPR instead of InfoNCE — collapsed the model to 0.23. Loss function choice matters more than architecture."*

---

### [3:45 – 4:00] CLOSE

> *"Bottom line: there's no single best model. LightGCN for accuracy, Two-Tower for production. My Feature-Gated LightGCN proved that on 99.97% sparse data, graph structure dominates — the model itself learned to discount features 4.5 to 1. The system I built handles 34,000 queries per second and can recommend to users who don't even have an account yet. Thank you."*

---
---

## PART 2: TECHNICAL DEEP-DIVE — Q&A PREP

---

### SECTION A: THE MODELS

---

#### Matrix Factorization

**What is it, in one sentence?**
Learn a vector for every user and every item; score = dot product; train to rank purchased items above random ones.

**What is BPR loss?**
Bayesian Personalized Ranking. For each training triple (user, positive item, negative item), it maximizes the sigmoid of (positive score − negative score). Formally:
`L = -log(σ(s(u,i+) - s(u,i-))) + λ||Θ||²`
It doesn't need explicit negative labels — any item the user didn't interact with is a valid negative.

**What's the regularization term?**
L2 regularization on the embeddings of the users and items in each batch, with coefficient λ=1e-4. Prevents embeddings from growing arbitrarily large and overfitting.

**Why 64 dimensions?**
Standard sweet spot for this dataset size. Too small → can't represent user taste. Too large → overfits. Embedding size was not tuned — it was held constant across all models to isolate other variables.

**Why is MF the baseline?**
If a more complex model doesn't beat MF, the extra complexity doesn't justify the cost. MF uses only purchase co-occurrence, no features, no graph. It's the minimum viable model.

**MF HR@10 = 0.6825 — how is that calculated?**
For each test user: their held-out item is the target. Sample 100 random items the user never bought. Rank all 101. If the target appears in the top 10, that's a hit. HR@10 = fraction of users who got a hit. Full-ranking HR@10 (0.042) uses all 26,354 items as candidates.

---

#### LightGCN

**What is the bipartite graph?**
Two types of nodes: users and items. An edge exists between user u and item i if u bought i. No user-user or item-item edges. The adjacency matrix A is (n_users + n_items) × (n_users + n_items).

**What is the normalized adjacency?**
`Â = D^(-½) A D^(-½)` where D is the degree matrix. This symmetric normalization prevents high-degree nodes from dominating. A user who bought 100 items shouldn't dominate a user who bought 5.

**What happens in each layer?**
`E^(k+1) = Â · E^(k)`
Each node's embedding becomes the degree-normalized weighted average of its neighbors' embeddings. A user embedding becomes an average of the item embeddings they interacted with. An item embedding becomes an average of the user embeddings that interacted with it.

**Why no nonlinearity (ReLU, etc.)?**
The original LightGCN paper (He et al., 2020) showed that removing feature transforms and nonlinearities improves performance. They are unnecessary for collaborative filtering — the linear propagation already captures the structural signal. Nonlinearities add parameters and can hurt.

**What is mean pooling over layers?**
After computing E^0, E^1, E^2, E^3, take their elementwise mean:
`E_final = (E^0 + E^1 + E^2 + E^3) / 4`
This gives multi-scale representations — local (layer 1) and global (layer 3) neighborhood info both contribute.

**Why can't LightGCN do FAISS?**
LightGCN embeddings are computed by running `Â · E` — a graph operation that requires the full adjacency matrix. You can't get an item's embedding without the graph. So item vectors change every time the model is retrained. More importantly, at inference you need the live graph, meaning you can't pre-compute a static index.

**Why can't LightGCN handle new users?**
A new user has no edges in the graph. Their embedding is just the random initialization — no information propagated. LightGCN needs the user to exist in the training graph.

**LightGCN HR@10 = 0.729 — why does it win?**
Multi-hop neighborhood averaging captures "users who liked similar items also liked X" — transitively, across 3 hops. On 99.97% sparse data, this graph-structural collaborative signal is the strongest available signal. Content features can't replicate it.

---

#### Two-Tower

**Why two separate towers?**
The key insight: if you separate user and item encoding, you can pre-compute all item vectors offline. At serving time, you only need to encode the user (one forward pass) then search. This enables FAISS. A single network that takes (user, item) together — like early YouTube models — can't be pre-computed.

**What does the GRU do?**
GRU (Gated Recurrent Unit) processes the sequence of the user's last 20 item text embeddings in order. Text embeddings: 384-dimensional vectors from SentenceTransformer `all-MiniLM-L6-v2`, projected to 64d first. The GRU outputs a 64d hidden state that captures: what kind of items did this user browse recently, and in what order? This is temporal collaborative signal.

**Why is the GRU key for cold-start?**
If a user is brand new, their ID embedding is zero (not in training set). But the GRU can still run on the text embeddings of items they've browsed, producing a meaningful 64d representation without needing a trained ID. That's the cold-start solution.

**What is InfoNCE loss?**
Information Noise-Contrastive Estimation. For a batch of 256 user-item pairs, treat the 255 other items in the batch as negatives for each user. Loss:
`L = -log( exp(sim(u,i+)/τ) / Σ_j exp(sim(u,j)/τ) )`
where τ=0.2 is the temperature. Lower τ makes the distribution sharper — harder negatives, stronger gradient signal. This is why InfoNCE with in-batch negatives beats BPR for dual-encoder training.

**Why did BPR collapse the Two-Tower? (v4-BPR, HR@10=0.23)**
BPR uses one sampled negative per positive. InfoNCE uses 255 in-batch negatives per positive in a batch of 256. BPR's gradient signal is too weak for the multi-signal user tower — the model can't learn useful representations from one negative at a time. The towers collapse: both output near-zero vectors that score similarly for everything.

**What is L2 normalization?**
Divide each output vector by its magnitude so it lies on a unit sphere. Then dot product equals cosine similarity (ranges -1 to 1). This is important for FAISS — IndexFlatIP (inner product) on unit vectors is equivalent to cosine similarity, which is what InfoNCE trains on.

**What is LayerNorm?**
Layer Normalization normalizes the activations across the feature dimension (not the batch). Applied after the first linear layer in the MLP. Stabilizes training, especially with multi-source inputs (ID + GRU + features all concatenated). Helps gradients flow consistently.

**What is FAISS?**
Facebook AI Similarity Search. A library for efficient similarity search over vectors. Three index types used:
- **Flat** (exact): brute-force dot product, 310μs, 100% recall
- **IVF** (approximate): clusters vectors into Voronoi cells, searches only nearby clusters, 35μs, ~97% recall
- **HNSW** (approximate): hierarchical navigable small world graph, 29μs, ~99% recall

**Why HNSW over IVF?**
HNSW is faster (29 vs 35μs) and has better recall (~99% vs ~97%). It uses a multi-layer graph where each node connects to its nearest neighbors at multiple granularities, enabling logarithmic-time search.

**Which companies use Two-Tower?**
- **YouTube** (2016 paper by Covington et al.) — retrieval stage: Two-Tower to get top-1000 candidates, then ranking model on those 1000
- **Pinterest** — PinSage, graph-based but same dual-encoder concept for FAISS retrieval
- **DoorDash** — restaurant and item recommendations
- **Airbnb** — listing recommendations using embedding similarity
- **Spotify** — podcast recommendations
- **Twitter/X** — tweet ranking
- **Meta** — Facebook feed and marketplace

The pattern is always the same: Two-Tower → FAISS candidates → heavy ranking model → business filters → user.

**Is it scalable?**
Yes. This is the specific reason Two-Tower is used at scale:
- Item vectors pre-computed once, hosted in FAISS — independent of user load
- User encoding: one MLP forward pass, ~0.5ms, stateless
- FAISS search: 29μs regardless of how many items
- Horizontal scaling: just add more user-tower instances; the FAISS index is read-only and shared
- At 26K items: 29μs. At 100M items (YouTube scale): HNSW still sub-millisecond

---

#### Feature-Gated LightGCN (Your Contribution)

**What was the original idea?**
The research question: does adding side features (user stats, item stats, text embeddings) to LightGCN's graph embeddings improve recommendations? Prior work (NGCF, LightGCN+) adds features but mostly through concatenation or fixed weighting. My approach: let the model learn the blend via a single learnable gate.

**What did you change from baseline LightGCN?**
Three additions:
1. `user_feat_proj = nn.Linear(n_user_feats, dim)` — projects 8 user features to 64d
2. `item_feat_proj = nn.Linear(n_item_feats, dim)` + `text_proj = nn.Linear(384, dim)` — projects item features + text to 64d
3. `feat_gate = nn.Parameter(torch.tensor(0.3))` — single learnable scalar

Final embedding: `E_final = (1 - sigmoid(gate)) * E_graph + sigmoid(gate) * E_features`

**What was the gate initialized to?**
0.3 → sigmoid(0.3) ≈ 0.574. So initially ~57% features, ~43% graph. The model then freely adjusts this during training.

**What did the gate converge to?**
sigmoid → 0.18. So 82% graph, 18% features. The model discovered that graph signal is 4.5× more valuable than features on this dataset.

**Why did features not help more?**
The dataset is 99.97% sparse. There's very little explicit feature signal that isn't already captured by the graph structure. Also, the 8 user features (activity, avg rating, recency) and 15 item features (price, rating count, category) are relatively weak signals compared to who-bought-what patterns.

**Why is this a novel contribution?**
Most prior work either ignores features (pure LightGCN) or uses a fixed concatenation. The learnable gate lets the model self-determine the optimal blend without human-tuned weighting. And the gate value (0.18) is an interpretable, data-driven finding about signal dominance on sparse collaborative data.

**Why is it worse than pure LightGCN by 1.4%?**
Two reasons: (1) The feature projections add noise — features contain some irrelevant information. (2) The gate regularizes toward a blend, which means even at gate=0.18, 18% of the signal is from features, which adds a small amount of noise to the otherwise-clean graph signal. A hard gate (0 or 1) would recover LightGCN exactly.

**What LR schedule did you use and why?**
Cosine annealing from 1e-3 to 1e-5 over 50 epochs. Standard for graph neural networks — high LR for fast initial convergence, decay to fine-tune embeddings without overshooting the minimum. MF and Two-Tower used fixed 1e-3 — cosine was added for FG-LightGCN specifically because it trained for 50 epochs.

---

### SECTION B: TRAINING & EVALUATION

**What is the train/val/test split?**
Leave-last-2-out per user, chronological. Each user's second-to-last interaction = validation, last = test. All earlier interactions = training. This simulates real deployment: model trained on historical data, evaluated on most recent purchases.

**What is k-core filtering?**
Remove all users with fewer than k=5 interactions and all items with fewer than k=5 interactions, iteratively, until the dataset is stable. This ensures every user and item has enough signal to learn from. The raw Amazon dataset has many users with 1-2 purchases — k-core removes them.

**What is sampled evaluation vs full ranking?**
- **Sampled**: For each test user, rank their held-out item against 100 random negatives. HR@10 = hit rate in top 10 of 101. Fast to compute. Inflated scores (not competing against full catalog).
- **Full ranking**: Rank against all 26,354 items. More realistic. HR@10 ≈ 0.042 for LightGCN (vs 0.729 sampled). The drop is expected — harder task. Full ranking is publication standard.

**What is NDCG@10?**
Normalized Discounted Cumulative Gain. Unlike HR@10 which just says hit/no-hit, NDCG rewards ranking the correct item higher. If you rank the true item #1, NDCG is higher than if you rank it #9. Formula: `NDCG = (1/log2(rank+1)) / (1/log2(2))` normalized by the ideal (rank=1). Values 0-1.

**What is HR@5, HR@10, HR@20?**
Hit Rate at different cutoffs. HR@5 is stricter — correct item must be in top 5. HR@20 is more lenient. The delta between them tells you about ranking distribution.

---

### SECTION C: DATASET & FEATURES

**Amazon Video Games 2023 — where does it come from?**
McAuley Lab at UCSD (Julian McAuley's group). They regularly release Amazon review datasets for research. The 2023 version has rich metadata including prices, categories, descriptions, and images. Available at amazon-reviews-2023.github.io.

**What are the 8 user features?**
Engineered from interaction history:
1. Total interaction count
2. Average rating given
3. Rating variance (consistency)
4. Days since first interaction (tenure)
5. Days since last interaction (recency)
6. Average price of items purchased
7. Interaction count in last 30 days (activity)
8. Unique category count (breadth of taste)

**What are the 15 item features?**
From Amazon metadata:
1. Price
2. Average rating received
3. Rating count (popularity)
4. Rating variance
5. Category (encoded)
6–15: Various metadata flags (has description, has images, etc.)

**What are the text embeddings?**
Item titles encoded by `sentence-transformers/all-MiniLM-L6-v2` — a 22M parameter transformer distilled from BERT. Outputs 384-dimensional vectors. Fast enough to run on CPU for 26K items. The output captures semantic meaning: "Dark Souls III" and "Elden Ring" will have similar vectors because the model knows they're related.

---

### SECTION D: ARCHITECTURE COMPARISONS

**Original Two-Tower (Covington et al., YouTube 2016) vs your Two-Tower:**

| | YouTube 2016 | Your Implementation |
|---|---|---|
| User signal | Watch history (avg pooling) | GRU over last 20 items (sequential) |
| Text | None | SentenceTransformer (384d → 64d) |
| Loss | Softmax over all items | InfoNCE with in-batch negatives |
| Serving | Nearest neighbor lookup | FAISS HNSW (29μs) |
| Cold-start | Not addressed | GRU over browsed item text |

Your version improves on the original in three ways: sequential encoding (GRU) instead of averaging, modern contrastive loss (InfoNCE), and explicit cold-start handling.

**Original LightGCN (He et al., 2020) vs your Feature-Gated LightGCN:**

| | LightGCN | Feature-Gated LightGCN |
|---|---|---|
| Inputs | ID embeddings only | ID + user features + item features + text |
| Graph propagation | Same | Same (3-layer sparse mm) |
| Feature blending | None | Learnable sigmoid gate |
| Parameters | ~8M | ~8.1M |
| HR@10 | 0.7290 | 0.7190 |

You added features without breaking graph propagation, and let the model self-determine the blend.

---

### SECTION E: POTENTIAL HARD QUESTIONS

**Q: Why is Two-Tower worse than MF by 6.7%?**
A: MF uses only ID co-occurrence, which on this dataset is the strongest signal. Two-Tower dilutes that with GRU and text features, which add value for cold-start but add noise for warm users. The InfoNCE loss with in-batch negatives also trains a different objective — similarity in embedding space — vs BPR which directly optimizes ranking. MF's simplicity wins on warm users. Two-Tower is chosen for production despite lower accuracy because it's the only model that handles cold-start and can serve at scale.

**Q: Is 0.6395 HR@10 for Two-Tower good enough for production?**
A: Yes, for two reasons. First, it's a retrieval model — it produces a candidate set of 1,000, which is then re-ranked by a heavier model (like LightGCN). The retrieval model needs high recall, not perfect precision. Second, cold-start capability is non-negotiable at production scale. A user's first visit cannot be met with "no recommendations."

**Q: How do you handle the OOM problem in training?**
A: The original design had per-edge attention (creating n_edges × dim × n_layers tensors). With 1.3M edges, that's ~58GB for attention weights. Replaced with standard LightGCN sparse.mm — O(edges × dim) memory instead of O(edges × dim × layers). Added 5% dropout on propagated embeddings to regularize without attention overhead.

**Q: Why did larger batch size hurt in v4b?**
A: InfoNCE uses in-batch negatives. With batch=256, each positive pair has 255 negatives. With batch=1024, there are 1023 negatives. More negatives sounds better, but many of those "negatives" may actually be relevant items for that user — false negatives. The model then learns to push away items the user actually likes, degrading quality. This is known as the "false negative" problem in contrastive learning.

**Q: Why did CLIP images not help (v7)?**
A: CLIP embeddings (512d) encode visual appearance. For video games on Amazon, most users make purchase decisions based on genre and gameplay, not box art. The visual signal adds noise without useful collaborative signal. Also, images are expensive to encode and the 512d → 64d projection may lose useful information.

**Q: What is the cold-start evaluation?**
A: Simulated by restricting the user history to [3, 5, 10, 20, full] interactions at test time. For brand-new users (0 interactions), only Two-Tower works. MF outputs a random embedding. LightGCN can't propagate. Two-Tower's GRU still produces a meaningful representation from the browsed items.

**Q: Can you explain the 12-variant ablation study?**
A: Ablation study = systematically remove or add one component at a time, starting from a baseline. v1 = InfoNCE baseline (0.619). v4 adds text → +2.6%. v5 adds GRU → another +0.4%. This isolates each component's contribution. If you add everything at once and get a 3% improvement, you don't know which piece caused it. Ablation studies are the standard methodology for understanding model components.

**Q: Why did LightGCN init (v5c) hurt Two-Tower?**
A: The idea was to warm-start the ID embeddings with LightGCN's pre-trained embeddings. But LightGCN embeddings are trained with BPR to be close in Euclidean space. InfoNCE trains embeddings to be close in cosine similarity on the unit sphere. The two objectives have different geometry — LightGCN init is good for BPR space but gets immediately distorted by InfoNCE loss. Net effect: slight degradation.

**Q: Why not just use LightGCN for everything?**
A: Three production problems:
1. **Cold-start**: New users have no edges → random embeddings
2. **FAISS**: Can't pre-compute static item vectors — they depend on the live graph
3. **Scale**: 100M users × 100M items graph doesn't fit in memory. Two-Tower sidesteps the graph entirely.

**Q: How would this scale to Netflix/YouTube scale (100M+ users)?**
A:
- Two-Tower: user tower is stateless, scales horizontally. FAISS HNSW works on billions of vectors with quantization (PQ = product quantization). Netflix and YouTube do exactly this.
- LightGCN: doesn't scale. Even sampling-based mini-batch graph training struggles past ~10M nodes.
- Solution at scale: Two-Tower for retrieval (FAISS), LightGCN-style graph model for re-ranking on the retrieved candidates only.

**Q: What would you do if you had more time?**
A: Three things:
1. Hard negative mining — sample negatives that are similar to the query but wrong (popularity-weighted), which produces stronger gradient signal than random negatives.
2. Statistical significance testing — run 3 seeds and report mean ± std to confirm the 2.6% text improvement is real, not noise.
3. Knowledge distillation — have LightGCN (teacher) train the Two-Tower (student). LightGCN knows the graph structure; Two-Tower needs to learn it without the graph. This is known to close the accuracy gap.

**Q: What is sparsity and why does it matter?**
A: The user-item matrix has 98,906 × 26,354 = ~2.6 billion possible entries. Only 659,000 are filled (interactions). That's 659K / 2.6B = 0.025% filled → 99.97% empty. For a user with 5 purchases, you only have 5 data points to learn their entire taste. This is why collaborative filtering (who bought what) works better than content features — the few interactions you have are the strongest signal about what the user actually likes.

---

### SECTION F: ONE-LINE DEFINITIONS (for fast recall)

| Term | What it means in plain English |
|---|---|
| Embedding | A list of numbers (vector) that represents something — a user, an item, a word |
| Dot product | Multiply two vectors element-wise and sum — measures alignment |
| BPR loss | Train to rank purchased items above random ones |
| InfoNCE loss | Train to rank the correct item above 255 other items in the same batch |
| Temperature τ | Sharpens the InfoNCE distribution — lower τ = harder negatives |
| GRU | Recurrent network that reads a sequence and outputs a summary vector |
| L2 normalize | Scale a vector to have magnitude 1 (unit sphere) |
| LayerNorm | Normalize across features within a single example — stabilizes training |
| FAISS | Library for fast vector search — finds nearest neighbors at scale |
| HNSW | Graph-based approximate nearest neighbor index — 29μs at 26K items |
| Bipartite graph | Two types of nodes (users + items), edges = purchases |
| Sparse matrix multiply | Multiply a very sparse adjacency matrix with embeddings — LightGCN's core op |
| Graph propagation | Average your neighbors' embeddings → encode structural similarity |
| Mean pooling | Average the outputs of all layers — multi-scale representation |
| Cold-start | Serving recommendations to a user with no purchase history |
| Ablation study | Systematically remove/add one component to measure its contribution |
| HR@10 | Fraction of users where the correct item appears in top 10 recommendations |
| NDCG@10 | HR@10 with a bonus for ranking the correct item higher |
| Full ranking | Evaluate against all 26K items, not just 100 sampled negatives |
| K-core filter | Remove users/items with fewer than k interactions, iteratively |
| Cosine annealing | LR schedule that decays from max to min following a cosine curve |
| Feature gate | Learnable scalar that blends two signals — lets model self-tune the mix |

---

*Nidhi Rajani | EAS 509 | Spring 2026*
