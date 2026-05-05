"""Convert PRESENTATION_PREP.md to a styled PDF using reportlab."""
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, PageBreak
)
from reportlab.lib.enums import TA_LEFT, TA_CENTER
import re

# ── Colours ──────────────────────────────────────────────────
PURPLE   = colors.HexColor("#7c3aed")
DARKBG   = colors.HexColor("#1e1e2e")
BLUE     = colors.HexColor("#1d4ed8")
GREEN    = colors.HexColor("#15803d")
CODEBG   = colors.HexColor("#f1f5f9")
TEXTGRAY = colors.HexColor("#374151")
LIGHTGRAY= colors.HexColor("#e5e7eb")
WHITE    = colors.white
BLACK    = colors.black

W, H = A4

# ── Styles ───────────────────────────────────────────────────
base = getSampleStyleSheet()

def S(name, **kw):
    s = ParagraphStyle(name, **kw)
    return s

STYLES = {
    "cover_title": S("cover_title",
        fontName="Helvetica-Bold", fontSize=28, textColor=WHITE,
        alignment=TA_CENTER, spaceAfter=8),
    "cover_sub": S("cover_sub",
        fontName="Helvetica", fontSize=13, textColor=colors.HexColor("#c4b5fd"),
        alignment=TA_CENTER, spaceAfter=4),
    "cover_line": S("cover_line",
        fontName="Helvetica", fontSize=11, textColor=colors.HexColor("#a5b4fc"),
        alignment=TA_CENTER, spaceAfter=2),

    "h1": S("h1",
        fontName="Helvetica-Bold", fontSize=18, textColor=PURPLE,
        spaceBefore=18, spaceAfter=6,
        borderPadding=(0,0,4,0)),
    "h2": S("h2",
        fontName="Helvetica-Bold", fontSize=14, textColor=BLUE,
        spaceBefore=14, spaceAfter=4),
    "h3": S("h3",
        fontName="Helvetica-Bold", fontSize=12, textColor=GREEN,
        spaceBefore=10, spaceAfter=3),
    "h4": S("h4",
        fontName="Helvetica-BoldOblique", fontSize=11, textColor=TEXTGRAY,
        spaceBefore=8, spaceAfter=2),

    "body": S("body",
        fontName="Helvetica", fontSize=10, textColor=TEXTGRAY,
        leading=15, spaceBefore=2, spaceAfter=4),
    "bold_body": S("bold_body",
        fontName="Helvetica-Bold", fontSize=10, textColor=BLACK,
        leading=15, spaceBefore=2, spaceAfter=4),
    "quote": S("quote",
        fontName="Helvetica-Oblique", fontSize=10.5,
        textColor=colors.HexColor("#1e3a5f"),
        leading=16, spaceBefore=4, spaceAfter=4,
        leftIndent=16, rightIndent=8,
        borderPadding=(6,10,6,10),
        backColor=colors.HexColor("#eff6ff"),
        borderColor=BLUE, borderWidth=0),
    "code": S("code",
        fontName="Courier", fontSize=9, textColor=colors.HexColor("#1e293b"),
        leading=13, spaceBefore=2, spaceAfter=2,
        leftIndent=12, backColor=CODEBG,
        borderPadding=(6,8,6,8)),
    "bullet": S("bullet",
        fontName="Helvetica", fontSize=10, textColor=TEXTGRAY,
        leading=14, spaceBefore=1, spaceAfter=1,
        leftIndent=16, bulletIndent=4),
    "table_hdr": S("table_hdr",
        fontName="Helvetica-Bold", fontSize=9, textColor=WHITE,
        alignment=TA_CENTER),
    "table_cell": S("table_cell",
        fontName="Helvetica", fontSize=9, textColor=TEXTGRAY,
        leading=12),
    "table_cell_bold": S("table_cell_bold",
        fontName="Helvetica-Bold", fontSize=9, textColor=BLACK,
        leading=12),
    "timing": S("timing",
        fontName="Helvetica-Bold", fontSize=11, textColor=WHITE,
        backColor=PURPLE, spaceBefore=10, spaceAfter=2,
        leftIndent=0, borderPadding=(4,8,4,8)),
}


def make_table(rows, col_widths=None, header=True):
    """Build a styled table from a list of row lists."""
    data = []
    for i, row in enumerate(rows):
        if i == 0 and header:
            data.append([Paragraph(str(c), STYLES["table_hdr"]) for c in row])
        else:
            data.append([Paragraph(str(c), STYLES["table_cell"]) for c in row])

    if col_widths is None:
        n = len(rows[0])
        col_widths = [(W - 4*cm) / n] * n

    t = Table(data, colWidths=col_widths, repeatRows=1 if header else 0)
    style = [
        ("BACKGROUND", (0,0), (-1,0), PURPLE),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [WHITE, colors.HexColor("#f5f3ff")]),
        ("GRID", (0,0), (-1,-1), 0.4, LIGHTGRAY),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
        ("TOPPADDING", (0,0), (-1,-1), 5),
        ("BOTTOMPADDING", (0,0), (-1,-1), 5),
        ("LEFTPADDING", (0,0), (-1,-1), 6),
        ("RIGHTPADDING", (0,0), (-1,-1), 6),
        ("ROUNDEDCORNERS", [3]),
    ]
    if header:
        style.append(("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"))
    t.setStyle(TableStyle(style))
    return t


def hr(color=LIGHTGRAY, thickness=1):
    return HRFlowable(width="100%", thickness=thickness, color=color, spaceAfter=6, spaceBefore=6)


def inline_bold(text):
    """Replace **text** with bold spans."""
    text = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', text)
    text = re.sub(r'`(.+?)`', r'<font name="Courier" size="9" color="#7c3aed">\1</font>', text)
    text = re.sub(r'\*(.+?)\*', r'<i>\1</i>', text)
    # escape chars that break reportlab
    text = text.replace('&', '&amp;').replace('<b>', '<b>').replace('</b>', '</b>')
    # fix double-escape of already replaced tags
    return text


def build():
    out = "/Users/nidhirajani/Desktop/Two_Tower_Recommendation_System/Presentation_Prep.pdf"
    doc = SimpleDocTemplate(
        out, pagesize=A4,
        leftMargin=2*cm, rightMargin=2*cm,
        topMargin=2*cm, bottomMargin=2*cm,
        title="Presentation Prep — Two-Tower Recommendation System",
        author="Nidhi Rajani",
    )

    story = []

    # ── COVER PAGE ──────────────────────────────────────────
    cover_table = Table(
        [[Paragraph("Recommendation System", STYLES["cover_title"]),],
         [Paragraph("Presentation Prep &amp; Technical Deep-Dive", STYLES["cover_sub"]),],
         [Paragraph("EAS 509  |  Nidhi Rajani  |  Spring 2026", STYLES["cover_line"]),],
         [Paragraph("4 Models · 12 Ablation Variants · 29μs Serving · Cold-Start", STYLES["cover_line"]),],
        ],
        colWidths=[W - 4*cm],
    )
    cover_table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,-1), DARKBG),
        ("TOPPADDING", (0,0), (-1,-1), 18),
        ("BOTTOMPADDING", (0,0), (-1,-1), 18),
        ("LEFTPADDING", (0,0), (-1,-1), 24),
        ("RIGHTPADDING", (0,0), (-1,-1), 24),
        ("ROUNDEDCORNERS", [8]),
    ]))
    story += [Spacer(1, 3*cm), cover_table, Spacer(1, 1.5*cm)]

    metrics = [
        ["98,906", "26,354", "659K", "99.97%", "4", "12"],
        ["Users",  "Items",  "Interactions", "Sparsity", "Models", "Ablations"],
    ]
    mt = Table(metrics, colWidths=[(W-4*cm)/6]*6)
    mt.setStyle(TableStyle([
        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE", (0,0), (-1,0), 16),
        ("FONTNAME", (0,1), (-1,1), "Helvetica"),
        ("FONTSIZE", (0,1), (-1,1), 9),
        ("TEXTCOLOR", (0,0), (-1,0), PURPLE),
        ("TEXTCOLOR", (0,1), (-1,1), TEXTGRAY),
        ("ALIGN", (0,0), (-1,-1), "CENTER"),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
        ("TOPPADDING", (0,0), (-1,-1), 6),
        ("BOTTOMPADDING", (0,0), (-1,-1), 6),
        ("LINEBELOW", (0,0), (-1,0), 1, PURPLE),
    ]))
    story += [mt, PageBreak()]

    # ── PART 1: SCRIPT ──────────────────────────────────────
    story += [
        Paragraph("PART 1 — 4-Minute Presentation Script", STYLES["h1"]),
        hr(PURPLE, 2),
        Spacer(1, 0.2*cm),
    ]

    sections = [
        ("[0:00 – 0:30]", "HOOK", [
            ('quote', '"Imagine you open Netflix and it recommends exactly what you want — with no account. Or you open Amazon and the front page already knows your taste. That\'s not magic. It\'s a recommendation system. I built one from scratch, trained four different deep learning models on 98,000 real Amazon users, and figured out what actually works — and what doesn\'t."'),
            ('body', '**Why this lands:** Connect to something everyone has experienced. Don\'t open with "my project is about." Open with the human problem.'),
        ]),
        ("[0:30 – 1:00]", "DATASET & PROBLEM", [
            ('quote', '"The dataset is Amazon Video Games 2023 — 98,906 users, 26,354 items, 659,000 purchases. The challenge? 99.97% of the user-item matrix is empty. Most users only bought 5-7 games. Most items have very few reviews. That\'s called the sparsity problem — and it\'s the hardest thing in recommendation systems."'),
            ('quote', '"I asked three questions: Does graph structure beat content features? Can we serve brand-new users who have zero purchase history? And can we do this in microseconds at scale?"'),
        ]),
        ("[1:00 – 2:30]", "ARCHITECTURE WALK", [
            ('h4', 'Matrix Factorization — the baseline'),
            ('quote', '"Every user gets a 64-number vector, every item gets a 64-number vector. The score is the dot product. Trained with BPR loss — rank purchased items above random ones. No features, no graph, no text. HR@10 of 0.68. This is the floor — anything fancier needs to beat it."'),
            ('h4', 'LightGCN — the accuracy king'),
            ('quote', '"Users and items form a graph. LightGCN runs 3 layers of neighborhood averaging. Layer 1: who else bought what this user bought. Layer 2: friends-of-friends. Layer 3: deeper transitive patterns. No nonlinearities — just sparse matrix multiplies. And it won — HR@10 of 0.729."'),
            ('h4', 'Two-Tower — the production model (YouTube, Pinterest, DoorDash)'),
            ('quote', '"Two separate neural networks. User tower: ID embedding + GRU over last 20 purchases + 8 user features → 64d. Item tower: ID + text embedding + 15 item features → 64d. Both L2-normalized. Score = dot product, trained with InfoNCE. The key: pre-compute all 26K item vectors once, load into FAISS. At serving: encode user, FAISS search. 29 microseconds. 34,000 recommendations per second."'),
            ('h4', 'Feature-Gated LightGCN — my contribution'),
            ('quote', '"I asked: what if we combine graph structure with side features? I kept LightGCN\'s propagation, added projections for user features, item features, and text, then added one learnable gate — a sigmoid parameter that lets the model decide the blend. It converged to 0.18. The model independently learned: 82% graph, 18% features."'),
        ]),
        ("[2:30 – 3:15]", "LIVE DEMO", [
            ('body', '**Tab 1 — Existing User:**'),
            ('quote', '"User 100 bought mostly action games. Two-Tower recommends genre-similar items via text. MF recommends co-purchased items. They overlap on 4-5 items — high-confidence. The differences show the fundamental contrast: Two-Tower understands content, MF understands behavior."'),
            ('body', '**Tab 2 — Cold-Start:**'),
            ('quote', '"Souls-like scenario: Dark Souls, Elden Ring, Sekiro. Zero purchases. The GRU encodes the text embeddings of those three items, creates a user representation from scratch, FAISS retrieves the nearest items. Real recommendations in milliseconds. MF: nothing. LightGCN: nothing. Only Two-Tower works."'),
        ]),
        ("[3:15 – 3:45]", "RESULTS", [
            ('quote', '"Sampled eval: LightGCN 0.729, FG-LightGCN 0.719, MF 0.68, Two-Tower 0.64. Full ranking (all 26K items): LightGCN and MF comparable at 0.042-0.044. The 12-variant ablation: text embeddings +2.6%, GRU +0.4%. Wrong loss function — BPR instead of InfoNCE — collapsed the model to 0.23. Loss function matters more than architecture."'),
        ]),
        ("[3:45 – 4:00]", "CLOSE", [
            ('quote', '"Bottom line: there\'s no single best model. LightGCN for accuracy, Two-Tower for production. My Feature-Gated LightGCN proved that on 99.97% sparse data, graph structure dominates — the model learned to discount features 4.5 to 1. The system handles 34,000 queries per second and serves users who don\'t even have an account. Thank you."'),
        ]),
    ]

    for timing, title, items in sections:
        # Timing pill
        timing_table = Table([[Paragraph(f"{timing}  {title}", STYLES["timing"])]],
                              colWidths=[W - 4*cm])
        timing_table.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,-1), PURPLE),
            ("LEFTPADDING", (0,0), (-1,-1), 10),
            ("RIGHTPADDING", (0,0), (-1,-1), 10),
            ("TOPPADDING", (0,0), (-1,-1), 6),
            ("BOTTOMPADDING", (0,0), (-1,-1), 6),
            ("ROUNDEDCORNERS", [4]),
        ]))
        story += [Spacer(1, 0.3*cm), timing_table, Spacer(1, 0.2*cm)]

        for kind, text in items:
            text_clean = text.replace('&', '&amp;').replace('  ', ' ')
            text_clean = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', text_clean)
            text_clean = re.sub(r'`(.+?)`', r'<font name="Courier" size="9" color="#7c3aed">\1</font>', text_clean)
            if kind == 'quote':
                story.append(Paragraph(text_clean, STYLES["quote"]))
            elif kind == 'h4':
                story.append(Paragraph(text_clean, STYLES["h4"]))
            elif kind == 'body':
                story.append(Paragraph(text_clean, STYLES["body"]))
            story.append(Spacer(1, 0.15*cm))

    story.append(PageBreak())

    # ── PART 2: TECHNICAL Q&A ───────────────────────────────
    story += [
        Paragraph("PART 2 — Technical Q&amp;A Prep", STYLES["h1"]),
        hr(PURPLE, 2),
    ]

    # Section A: Models
    story.append(Paragraph("Section A: The Models", STYLES["h2"]))
    story.append(hr())

    qa_sections = [
        ("Matrix Factorization", [
            ("What is BPR loss?", "Bayesian Personalized Ranking. For each triple (user, positive item, negative item), maximize sigmoid(s_pos − s_neg). Formally: L = −log σ(s(u,i+) − s(u,i−)) + λ‖Θ‖². Doesn't need explicit negatives — any un-purchased item is a valid negative."),
            ("What is the regularization term?", "L2 regularization on batch embeddings with coefficient λ=1e-4. Prevents embeddings from growing large and overfitting the training interactions."),
            ("Why 64 dimensions?", "Standard sweet spot for this dataset size. Too small → can't represent taste. Too large → overfits. Held constant across all models to isolate other variables."),
            ("Why is MF the baseline?", "If a more complex model doesn't beat MF, extra complexity isn't justified. MF uses only purchase co-occurrence — minimum viable model."),
            ("How is HR@10 calculated?", "For each test user: rank their held-out item against 100 random negatives (101 total). HR@10 = fraction of users where held-out item appears in top 10. Full-ranking uses all 26,354 items — harder, publication-standard."),
        ]),
        ("LightGCN", [
            ("What is the bipartite graph?", "Two node types: users and items. Edge between u and i if u purchased i. The adjacency matrix A is (n_users + n_items) × (n_users + n_items) — sparse with ~1.3M non-zero entries."),
            ("What is the normalized adjacency?", "Â = D^(−½) A D^(−½) where D is the degree matrix. Symmetric normalization prevents high-degree nodes from dominating — a user with 100 purchases doesn't swamp a user with 5."),
            ("What happens in each layer?", "E^(k+1) = Â · E^(k). Each node's embedding becomes a degree-normalized average of its neighbors' embeddings. User embeddings average over bought items. Item embeddings average over purchasers."),
            ("Why no nonlinearity?", "He et al. (2020) showed removing feature transforms and nonlinearities improves performance on collaborative filtering. Linear propagation already captures structural signal. Nonlinearities add parameters and can hurt."),
            ("What is mean pooling over layers?", "E_final = mean(E^0, E^1, E^2, E^3). Multi-scale: combines local (layer 1) and global (layer 3) neighborhood info. Avoids over-smoothing from deep propagation."),
            ("Why can't LightGCN do FAISS or cold-start?", "Embeddings require running Â·E — needs the full live graph. Can't pre-compute static item vectors. New users have no edges → random embeddings, no propagation."),
        ]),
        ("Two-Tower", [
            ("Why two separate towers?", "Separate encoding allows pre-computing all item vectors offline. At serving: encode user (one forward pass) → FAISS search. A single joint network can't be pre-computed, killing scalability."),
            ("What does the GRU do?", "Processes the user's last 20 item text embeddings (384d → 64d projected) in sequence order. Outputs 64d hidden state capturing temporal purchase patterns. Essential for cold-start: no user ID needed, GRU runs on browsed items."),
            ("What is InfoNCE loss?", "For batch of 256 pairs, treat the 255 other items as negatives. L = −log(exp(sim(u,i+)/τ) / Σ_j exp(sim(u,j)/τ)), τ=0.2. Lower τ = sharper distribution = harder negatives = stronger gradients."),
            ("Why did BPR collapse Two-Tower? (v4-BPR: 0.23)", "BPR uses 1 sampled negative. InfoNCE uses 255 in-batch negatives. BPR's gradient is too weak for a multi-signal tower — the model collapses to near-zero vectors. Wrong loss for dual-encoder training."),
            ("Why did larger batch hurt? (v4b)", "More in-batch negatives sounds better, but many are false negatives — items the user actually likes. Model learns to push away relevant items. Known as the 'false negative' problem in contrastive learning."),
            ("What is FAISS HNSW?", "Hierarchical Navigable Small World graph. Multi-layer graph where each node connects to nearest neighbors at multiple granularities → logarithmic search time. 29μs at 26K items. ~99% recall. Outperforms IVF (35μs, ~97% recall)."),
            ("Which companies use Two-Tower?", "YouTube (2016, Covington et al.) — retrieval stage to get top-1000 candidates. Pinterest (PinSage). DoorDash. Airbnb. Spotify. Twitter/X. Meta. Pattern: Two-Tower → FAISS → heavy ranker → business filters → user."),
            ("Is it scalable to 100M+ items?", "Yes. User tower: stateless, scales horizontally. FAISS HNSW with product quantization works on billions of vectors (YouTube scale). Item pre-computation: one-time offline job. At 26K: 29μs. At 100M: still sub-millisecond with PQ compression."),
        ]),
        ("Feature-Gated LightGCN — Your Contribution", [
            ("What did you add to LightGCN?", "Three additions: (1) user_feat_proj = Linear(8, 64). (2) item_feat_proj = Linear(15, 64) + text_proj = Linear(384, 64). (3) feat_gate = nn.Parameter(tensor(0.3)) — a single learnable scalar. Final: E = (1−gate)×E_graph + gate×E_features."),
            ("What did the gate converge to, and what does it mean?", "sigmoid(0.3) ≈ 0.57 initially → converged to 0.18. The model learned graph contributes 82%, features 18%. On 99.97% sparse data, collaborative structure is 4.5× more valuable than side features."),
            ("Why is it novel?", "Prior work uses fixed concatenation or hand-tuned weighting. The learnable gate lets the model self-determine the optimal blend — no human tuning. Gate value (0.18) is an interpretable, data-driven finding about signal dominance."),
            ("Why is it 1.4% worse than pure LightGCN?", "Feature projections add noise — features contain irrelevant information. Even at gate=0.18, 18% of signal is from features, introducing small noise to clean graph signal. A hard gate (0 or 1) would recover LightGCN exactly."),
            ("What LR schedule and why?", "Cosine annealing 1e-3 → 1e-5 over 50 epochs. High LR for fast initial convergence, decays to fine-tune without overshooting. Standard for GNNs training for many epochs."),
        ]),
    ]

    for model_name, qas in qa_sections:
        story.append(Paragraph(model_name, STYLES["h3"]))
        for q, a in qas:
            q_clean = q.replace('&', '&amp;')
            a_clean = a.replace('&', '&amp;')
            a_clean = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', a_clean)
            a_clean = re.sub(r'`(.+?)`', r'<font name="Courier" size="9" color="#7c3aed">\1</font>', a_clean)
            story.append(Paragraph(f"<b>Q: {q_clean}</b>", STYLES["bold_body"]))
            story.append(Paragraph(f"A: {a_clean}", STYLES["body"]))
            story.append(Spacer(1, 0.1*cm))
        story.append(Spacer(1, 0.3*cm))

    # Section B: Evaluation
    story += [Paragraph("Section B: Training &amp; Evaluation", STYLES["h2"]), hr()]
    eval_qa = [
        ("What is the train/val/test split?", "Leave-last-2-out per user, chronological. Second-to-last interaction = validation. Last = test. All earlier = training. Simulates real deployment: model trained on history, evaluated on most recent purchase."),
        ("What is k-core filtering?", "Remove users with <5 interactions and items with <5 interactions, iteratively until stable. Ensures every node has enough signal to learn from. Eliminates low-signal users from raw Amazon data."),
        ("Sampled vs full ranking evaluation?", "Sampled: rank held-out item vs 100 random negatives (101 total). Fast, inflated scores. Full ranking: rank vs all 26,354 items. Realistic, publication-standard. HR@10 drops from ~0.73 to ~0.044 — expected, much harder task."),
        ("What is NDCG@10?", "Normalized Discounted Cumulative Gain. Unlike HR@10 (hit/no-hit), NDCG rewards ranking the correct item higher. Item at rank 1 scores log2(2)/log2(2)=1.0. Item at rank 9 scores log2(2)/log2(10)=0.301. Values 0–1."),
    ]
    for q, a in eval_qa:
        q_c = q.replace('&', '&amp;')
        a_c = a.replace('&', '&amp;')
        story.append(Paragraph(f"<b>Q: {q_c}</b>", STYLES["bold_body"]))
        story.append(Paragraph(f"A: {a_c}", STYLES["body"]))
        story.append(Spacer(1, 0.1*cm))

    story.append(PageBreak())

    # Section C: Architecture comparisons table
    story += [Paragraph("Section C: Architecture Comparisons", STYLES["h2"]), hr()]

    story.append(Paragraph("YouTube 2016 Two-Tower vs Your Two-Tower", STYLES["h3"]))
    yt_rows = [
        ["Feature", "YouTube 2016", "Your Implementation"],
        ["User signal", "Watch history (avg pooling)", "GRU over last 20 items (sequential)"],
        ["Text", "None", "SentenceTransformer 384d → 64d"],
        ["Loss", "Softmax over all items", "InfoNCE, in-batch negatives, τ=0.2"],
        ["Serving", "Nearest neighbor lookup", "FAISS HNSW — 29μs"],
        ["Cold-start", "Not addressed", "GRU over browsed item text embeddings"],
    ]
    story.append(make_table(yt_rows, col_widths=[4*cm, 5.5*cm, 6.5*cm]))
    story.append(Spacer(1, 0.4*cm))

    story.append(Paragraph("Original LightGCN vs Feature-Gated LightGCN", STYLES["h3"]))
    lg_rows = [
        ["Feature", "LightGCN", "Feature-Gated LightGCN"],
        ["Inputs", "ID embeddings only", "ID + 8 user feats + 15 item feats + 384d text"],
        ["Graph propagation", "3-layer sparse mm", "Same — unchanged"],
        ["Feature blending", "None", "Learnable sigmoid gate (converged: 0.18)"],
        ["Parameters", "~8M", "~8.1M"],
        ["HR@10", "0.7290", "0.7190 (−1.4%)"],
        ["Cold-start", "No", "No"],
    ]
    story.append(make_table(lg_rows, col_widths=[4.5*cm, 5*cm, 6.5*cm]))
    story.append(Spacer(1, 0.4*cm))

    story.append(Paragraph("All Four Models at a Glance", STYLES["h3"]))
    all_rows = [
        ["Model", "HR@10", "NDCG@10", "Cold-Start", "FAISS", "Role"],
        ["MF (BPR)", "0.6825", "0.4550", "No", "Partial", "Baseline"],
        ["LightGCN", "0.7290", "0.4940", "No", "No", "Re-ranking"],
        ["Two-Tower v5", "0.6395", "0.4148", "Yes", "Yes (29μs)", "Retrieval"],
        ["FG-LightGCN", "0.7190", "0.4824", "No", "No", "Research"],
    ]
    story.append(make_table(all_rows, col_widths=[3.5*cm, 2*cm, 2.5*cm, 2.5*cm, 3*cm, 2.5*cm]))

    story.append(PageBreak())

    # Section D: Hard questions
    story += [Paragraph("Section D: Hard Questions", STYLES["h2"]), hr()]
    hard_qa = [
        ("Why is Two-Tower worse than MF by 6.7%?",
         "MF uses only ID co-occurrence — the strongest signal on sparse data. Two-Tower dilutes it with GRU and text features, which add value for cold-start but add noise for warm users. Also, InfoNCE optimizes cosine similarity space while BPR directly optimizes ranking — different objectives. MF wins on warm users. Two-Tower is chosen for production despite lower accuracy because it's the only model that handles cold-start and scales via FAISS."),
        ("Is 0.64 HR@10 good enough for production?",
         "Yes. Two-Tower is a retrieval model — it generates a candidate set of 1,000 items which is then re-ranked by a heavier model (like LightGCN). Retrieval needs high recall, not perfect precision. And cold-start capability is non-negotiable at scale — a new user's first visit cannot be met with no recommendations."),
        ("Why did CLIP images not help? (v7)",
         "CLIP encodes visual appearance. For video games, purchase decisions are driven by genre and gameplay, not box art. Visual signal adds noise without useful collaborative information. Also 512d → 64d projection may compress out useful detail."),
        ("How would this scale to Netflix/YouTube scale?",
         "Two-Tower: user tower is stateless, scales horizontally. FAISS HNSW with product quantization (PQ) works on billions of vectors — YouTube does exactly this. LightGCN doesn't scale past ~10M nodes even with mini-batch sampling. Production solution: Two-Tower for retrieval, LightGCN-style re-ranker on the retrieved candidates only."),
        ("What would you do with more time?",
         "Three things: (1) Hard negative mining — sample negatives similar to the query but wrong (popularity-weighted or in-batch hard), stronger gradient signal. (2) Statistical significance — run 3 seeds, report mean ± std to confirm text improvement (+2.6%) is real. (3) Knowledge distillation — LightGCN as teacher, Two-Tower as student. LightGCN knows graph structure; Two-Tower learns it without needing the graph at inference."),
        ("Why not just use LightGCN for everything?",
         "Three production blockers: (1) Cold-start: new users have no edges → random embeddings. (2) FAISS: needs full live graph at inference, can't pre-compute static vectors. (3) Scale: 100M users × 100M items graph doesn't fit in memory. Two-Tower sidesteps the graph entirely at serving time."),
    ]
    for q, a in hard_qa:
        q_c = q.replace('&', '&amp;')
        a_c = a.replace('&', '&amp;')
        a_c = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', a_c)
        story.append(Paragraph(f"<b>Q: {q_c}</b>", STYLES["bold_body"]))
        story.append(Paragraph(f"A: {a_c}", STYLES["body"]))
        story.append(Spacer(1, 0.2*cm))

    story.append(PageBreak())

    # Section E: One-line definitions
    story += [Paragraph("Section E: One-Line Definitions (Quick Recall)", STYLES["h2"]), hr()]
    defs = [
        ["Term", "Plain English Definition"],
        ["Embedding", "A list of numbers (vector) representing a user or item"],
        ["Dot product", "Multiply two vectors element-wise and sum — measures alignment"],
        ["BPR loss", "Train so purchased item scores higher than a random un-purchased item"],
        ["InfoNCE loss", "Train so correct item scores higher than 255 other items in the batch"],
        ["Temperature τ", "Sharpens InfoNCE distribution — lower τ = harder negatives"],
        ["GRU", "Recurrent network that reads a sequence and outputs a summary vector"],
        ["L2 normalize", "Scale a vector to magnitude 1 (puts it on unit sphere)"],
        ["LayerNorm", "Normalize across features within one example — stabilizes training"],
        ["FAISS", "Facebook's library for fast similarity search over millions of vectors"],
        ["HNSW", "Graph-based approx. nearest neighbor index — 29μs at 26K items"],
        ["Bipartite graph", "Two node types (users + items), edges = purchase events"],
        ["Sparse matrix multiply", "LightGCN's core op — multiplies sparse adjacency with embeddings"],
        ["Graph propagation", "Average neighbors' embeddings → encode structural similarity"],
        ["Mean pooling", "Average outputs of all layers — captures local + global structure"],
        ["Cold-start", "Serving recommendations to a user with zero purchase history"],
        ["Ablation study", "Remove/add one component at a time to measure its contribution"],
        ["HR@10", "Fraction of users where the correct item appears in top 10 recs"],
        ["NDCG@10", "HR@10 with bonus for ranking the correct item higher in the list"],
        ["Full ranking", "Evaluate against all 26,354 items — publication standard"],
        ["K-core filter", "Remove users/items with fewer than k interactions, iteratively"],
        ["Cosine annealing", "LR schedule decaying from max to min following a cosine curve"],
        ["Feature gate", "Learnable scalar blending two signals — model self-tunes the mix"],
        ["Sparsity 99.97%", "Only 0.03% of user-item pairs have any interaction — extreme scarcity"],
        ["In-batch negatives", "Other items in the same training batch treated as negatives for InfoNCE"],
        ["False negatives", "Items labelled as negatives but actually relevant — problem at large batch"],
    ]
    story.append(make_table(defs, col_widths=[4.5*cm, 11.5*cm]))

    # ── Footer ───────────────────────────────────────────────
    story += [
        Spacer(1, 1*cm),
        hr(PURPLE),
        Paragraph(
            "Nidhi Rajani  |  EAS 509  |  Amazon Video Games 2023  |  "
            "github.com/nidhi1603/Two_Tower_Recommendation_System",
            ParagraphStyle("footer", fontName="Helvetica", fontSize=8,
                           textColor=TEXTGRAY, alignment=TA_CENTER)
        ),
    ]

    doc.build(story)
    print(f"PDF saved → {out}")


if __name__ == "__main__":
    build()
