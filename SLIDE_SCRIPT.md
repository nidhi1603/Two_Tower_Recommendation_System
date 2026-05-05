# Slide-by-Slide Speaking Script
### Two-Tower Recommendation System | EAS 509 | Nidhi Rajani
### Total time: ~4 minutes | 15 slides

---

> **Before you start:** Open Streamlit at http://localhost:8501 in a browser tab. Open the PPT in full-screen. Take one breath. Make eye contact with the room first — then speak.

---

## SLIDE 1 — HOOK
**⏱ Time: ~25 seconds**

> "You open Netflix. It already knows what you want to watch.
> You open Amazon. The front page is already personalized.
> That's not magic — it's a recommendation system running in milliseconds."

*(pause one beat)*

> "I built one from scratch. Four deep learning models, 98,000 real Amazon users, 12 experiments to figure out what actually works — and what doesn't."

**Delivery tip:** Start slow. Let "that's not magic" land before continuing. Don't rush the hook.

---

## SLIDE 2 — THE PROBLEM
**⏱ Time: ~30 seconds**

*(point to the grid on the left)*

> "Here's the dataset. 98,906 users, 26,354 items. This grid represents the user-item matrix — every cell where a user bought an item."

> "99.97% of it is empty. Most users bought only 5 to 7 games their entire life on Amazon. That's the sparsity problem."

*(gesture to the three cards on the right)*

> "This creates three hard problems. First — sparsity, barely any signal to learn from. Second — cold-start: what do you show a brand-new user who has never bought anything? Third — scale: 98K users times 26K items is 2.6 billion comparisons per request. It needs to run in microseconds."

**Delivery tip:** Point physically to the cards as you name them. It anchors the audience.

---

## SLIDE 3 — DATASET
**⏱ Time: ~20 seconds**

> "The data is Amazon Video Games 2023, from McAuley Lab at UCSD — a standard academic benchmark."

> "After k-core filtering — removing users and items with fewer than 5 interactions — we have 659,000 training interactions. Each user has 8 engineered features like activity level and recency. Each item has 15 features including price and average rating, plus a 384-dimensional text embedding from a SentenceTransformer on the item title."

> "The pipeline goes: download, filter, engineer features, embed text, split chronologically, train four models."

**Delivery tip:** Keep this fast — it's setup, not the interesting part. Get through it in under 20 seconds.

---

## SLIDE 4 — RESEARCH PAPERS
**⏱ Time: ~25 seconds**

> "Four papers are the foundation of this work."

*(gesture left to right)*

> "Matrix Factorization from 2009 — that's the idea of giving every user and item a vector. Neural Collaborative Filtering from WWW 2017 — showed MLPs can model non-linear interactions. LightGCN from SIGIR 2020 — the key insight: remove all the complexity from graph networks, just do pure neighborhood averaging, and it actually performs better."

> "And the YouTube Two-Tower paper from RecSys 2016 — that's the production architecture I built on and improved. I added sequential encoding with a GRU, replaced their loss with InfoNCE, and added FAISS HNSW serving."

**Delivery tip:** Don't read the paper names robotically. Say "the YouTube paper" and "the LightGCN paper" — it sounds more natural.

---

## SLIDE 5 — MATRIX FACTORIZATION
**⏱ Time: ~20 seconds**

*(point to the two boxes)*

> "Model one — Matrix Factorization. The simplest possible approach. Every user gets a 64-dimensional embedding vector, every item gets one too. The recommendation score is just the dot product — how aligned are these two vectors."

*(point to BPR box)*

> "Trained with BPR loss — Bayesian Personalized Ranking. It says: the score for something you actually bought should be higher than the score for a random item you didn't. That's it. No features, no graph, no text."

> "HR@10 of 0.68. This is the baseline. Anything fancier has to beat this number to justify the added complexity."

**Delivery tip:** Emphasize "that's it" — the simplicity is the point.

---

## SLIDE 6 — LIGHTGCN
**⏱ Time: ~30 seconds**

*(point to the graph on the left)*

> "Model two — LightGCN. Users and items form a bipartite graph. An edge exists between a user and an item if that user bought that item. 659,000 edges in total."

*(trace the layers on the right)*

> "LightGCN runs 3 layers of neighborhood averaging. Layer 1: a user's embedding becomes the average of all the item embeddings they purchased. Layer 2: you go one hop further — now you're capturing 'users who bought similar items also bought this.' Layer 3: even deeper, community-level patterns."

> "No nonlinearities. No feature transforms. Just sparse matrix multiplies. Deliberately simple. And it wins — HR@10 of 0.729, best accuracy in the study."

*(point to the red warning)*

> "But it has a fatal flaw — it cannot handle new users, and it cannot serve via FAISS. I'll come back to that."

**Delivery tip:** When you say "deliberately simple" — pause slightly. It's a counterintuitive insight.

---

## SLIDE 7 — TWO-TOWER
**⏱ Time: ~40 seconds**

*(point to the left tower)*

> "Model three — Two-Tower. This is what YouTube, Pinterest, and DoorDash actually use in production."

> "The user tower takes three inputs: a 64-dimensional ID embedding, a GRU running over the user's last 20 purchased items — capturing sequential taste — and 8 user features. All concatenated, passed through an MLP with LayerNorm, L2-normalized to 64 dimensions."

*(point to the right tower)*

> "The item tower: item ID embedding, the text embedding of the title projected from 384 to 64 dimensions, and 15 item features. Same MLP, same normalization."

*(point to the dot product in the middle)*

> "Score = dot product. Trained with InfoNCE loss — for every user in the batch, rank their item above 255 other items in the same batch. Temperature 0.2 makes the distribution sharp, giving strong gradient signal."

*(point to the green cold-start box)*

> "The key production advantage: I pre-compute all 26,354 item vectors once and load them into a FAISS HNSW index. At serving time — one user forward pass, then FAISS search. 29 microseconds. 34,000 queries per second on a single CPU."

> "And the GRU means: even a brand-new user with zero purchase history can get recommendations — just from browsing 3 items."

**Delivery tip:** This is your most important slide. Speak clearly. Don't rush the FAISS numbers.

---

## SLIDE 8 — WHAT I IMPROVED OVER YOUTUBE ORIGINAL
**⏱ Time: ~20 seconds**

*(gesture across the table)*

> "Here's the direct comparison between the 2016 YouTube paper and my implementation."

> "They averaged watch history. I used a GRU — sequential order matters, because what you watched last week tells you more than what you watched two years ago."

> "They didn't use text. I added SentenceTransformer title embeddings — that gave a 2.6% improvement."

> "Their loss was softmax over all items — slow and requires the full catalog in memory. I used InfoNCE with in-batch negatives — 10 times faster and stronger gradients."

> "And I added FAISS HNSW serving and a proper cold-start solution — neither of which the original paper addressed."

**Delivery tip:** Keep your finger moving across the table row by row. Visual anchoring helps.

---

## SLIDE 9 — FEATURE-GATED LIGHTGCN
**⏱ Time: ~35 seconds**

> "Model four — this is my original contribution."

*(point to the research question box)*

> "The question I wanted to answer: can side features improve LightGCN? Prior work either ignores features entirely, or concatenates them with fixed weights. My approach: add a single learnable parameter — a sigmoid gate — and let the model decide the blend."

*(point to the two boxes at top)*

> "I kept LightGCN's graph propagation unchanged. I added three linear projections — for user features, item features, and text embeddings — projecting everything into 64 dimensions."

*(point to the gate box)*

> "The final embedding is: (1 minus gate) times the graph embedding, plus gate times the feature embedding. The model learns gate during training."

*(point to the convergence result on the right)*

> "It started at 0.57 — basically 50/50. By epoch 36, it converged to 0.18. The model independently learned that graph signal is 4.5 times more valuable than features on this dataset. 82% graph, 18% features."

> "That's not something I told it. The data told it. And that finding validates the entire ablation study."

**Delivery tip:** "The data told it" is your punchline. Pause after it.

---

## SLIDE 10 — LIVE DEMO
**⏱ Time: ~45 seconds**

*(click the link or switch to browser tab)*

> "Let me show it running live."

**[Switch to Streamlit — go to Live Demo tab]**

**Demo 1 — Existing User:**
> "I'll pick User 100. You can see they bought mostly action and adventure games — that's their history. Now watch — Two-Tower recommends genre-similar games using the text embeddings. Matrix Factorization recommends games that co-appear in other users' purchase histories. They agree on about 4 or 5 items — those are the high-confidence recommendations. The ones they disagree on show the difference: Two-Tower understands content, MF understands behavior."

**Demo 2 — Cold-Start:**
> "Now the real test. I'll pick the Souls-like scenario — Dark Souls, Elden Ring, Sekiro. This user has made zero purchases. Watch what happens."

*(click and show results)*

> "The GRU encodes the text embeddings of those three items, builds a 64-dimensional user representation from scratch, and FAISS retrieves the nearest items. Real recommendations. MF: cannot serve. LightGCN: cannot serve. Only Two-Tower works."

**[Switch back to PPT]**

**Delivery tip:** Practice this demo twice before the presentation. Know exactly which buttons to click. Keep narrating while it loads.

---

## SLIDE 11 — RESULTS
**⏱ Time: ~25 seconds**

*(point to the bar chart)*

> "Results. On sampled evaluation — ranking against 100 random negatives per user: LightGCN wins at 0.729, Feature-Gated LightGCN at 0.719, MF at 0.68, Two-Tower at 0.64."

*(point to the full ranking table)*

> "On full ranking — all 26,354 items as candidates — the numbers drop significantly, as expected. LightGCN and MF are comparable at about 0.042-0.044 HR@10. This is publication-standard evaluation. Most projects only report sampled eval."

*(point to cold-start table)*

> "And for cold-start: Two-Tower is the only model that can serve a brand-new user. MF and LightGCN output nothing — they simply can't serve a user who doesn't exist in the training set."

**Delivery tip:** Don't apologize for Two-Tower's lower number. Explain why it doesn't matter for production.

---

## SLIDE 12 — ABLATION STUDY
**⏱ Time: ~25 seconds**

*(gesture at the bar chart)*

> "The ablation study is where the real learning happened. 12 variants, each changing exactly one thing."

*(point to the green bars)*

> "Version 4 adds title text embeddings — that's a 2.6% improvement. Version 5 adds GRU sequential encoding — another 0.4%. These are real, isolated, measurable contributions."

*(point to the collapsed bar)*

> "Version 4-BPR is the most important finding. Same model, same features — but I swapped InfoNCE for BPR loss. HR@10 collapsed to 0.23. The model completely broke. Same architecture, wrong loss, 3 times worse performance."

> "The lesson: loss function matters more than architecture. You can have the most sophisticated model in the world, but with the wrong training objective it falls apart."

**Delivery tip:** The collapse bar is dramatic. Let people see it. Give it a moment.

---

## SLIDE 13 — INDUSTRY & SCALABILITY
**⏱ Time: ~20 seconds**

*(sweep gesture across company cards)*

> "This isn't academic. YouTube uses this exact pattern — Two-Tower retrieval to get top-1000 candidates, then a heavy ranking model. Pinterest, DoorDash, Airbnb, Spotify, Twitter — all the same architecture."

*(point to the scalability section)*

> "The reason it scales: item vectors are pre-computed once. The user tower is stateless — you can run it on a thousand servers in parallel. FAISS with product quantization works on billions of vectors. At YouTube's scale — 2 billion users, 800 million videos — this is still the architecture."

> "LightGCN doesn't scale past about 10 million nodes — you can't fit the full adjacency matrix in memory. That's why it stays in re-ranking on small candidate sets."

**Delivery tip:** Name-dropping real companies makes this feel serious and relevant. Don't skip it.

---

## SLIDE 14 — KEY FINDINGS
**⏱ Time: ~20 seconds**

*(point to each card)*

> "Four takeaways."

> "One: graph structure beats features on sparse data. The gate I trained proved it — the model itself assigned 82% to graph signal."

> "Two: text embeddings help, but rich text hurts. Title alone adds 2.6%. Full descriptions added noise and hurt performance. Less is more on sparse data."

> "Three: loss function beats architecture. BPR collapsed the Two-Tower. InfoNCE saved it. Same model."

> "Four: there's no single best model. LightGCN for accuracy, Two-Tower for production. The right choice depends on the deployment constraint — can you afford cold-start failures? Can you afford the graph at inference?"

**Delivery tip:** Number them out loud — "one, two, three, four." It gives structure and makes you sound organized.

---

## SLIDE 15 — CLOSE
**⏱ Time: ~15 seconds**

*(let the slide breathe for a moment)*

> "Bottom line — there is no single best model. LightGCN wins accuracy. Two-Tower wins production. My Feature-Gated LightGCN showed that on 99.97% sparse data, the model itself learns to discount features."

> "The system serves 34,000 queries per second and recommends to users who don't even have an account yet."

*(pause — look up from the screen)*

> "Thank you. Happy to take questions."

**Delivery tip:** End on "thank you" and stop talking. Don't trail off. Silence after is fine.

---

## ANTICIPATED QUESTIONS — QUICK ANSWERS

| Question | Answer in 2 sentences |
|---|---|
| Why is Two-Tower worse than MF? | MF uses only ID co-occurrence — the strongest signal on sparse data. Two-Tower trades some accuracy for cold-start ability and FAISS scalability — a tradeoff that's always worth it in production. |
| What is BPR loss? | For each (user, purchased item, random item) triple, maximize sigmoid(score_pos − score_neg). It directly trains the model to rank what you bought above what you didn't. |
| What is InfoNCE? | For a batch of 256 pairs, rank the correct item above the other 255 in the batch. Temperature 0.2 makes the distribution sharp, giving strong gradients. |
| What does the GRU do? | It reads the sequence of item text embeddings the user bought — in order — and outputs a 64-dimensional summary of their sequential taste. This is what enables cold-start. |
| Why can't LightGCN scale? | It needs the full user-item adjacency matrix at inference. At 100M users × 100M items that doesn't fit in memory, and you can't pre-compute static item vectors. |
| What is FAISS HNSW? | Facebook's approximate nearest neighbor library. HNSW builds a hierarchical graph of vectors enabling logarithmic-time search. 29μs at 26K items, ~99% recall. |
| What would you do differently? | Hard negative mining for stronger gradient signal, 3-seed significance testing on the text improvement, and knowledge distillation from LightGCN into Two-Tower. |
| What is the feature gate? | A single learnable scalar parameter. sigmoid(θ) blends graph and feature embeddings. Started at 0.57, converged to 0.18 — model self-learned the optimal blend. |
| What is the ablation study? | Change exactly one thing at a time and measure the impact. 12 variants total. Without it, you don't know which of your improvements actually helped. |
| Is 0.042 full-ranking HR@10 good? | It's expected — you're competing against all 26,354 items, not 100. LightGCN at 0.044 is competitive with published results on similar Amazon datasets. |

---

*Nidhi Rajani | EAS 509 | Spring 2026*
