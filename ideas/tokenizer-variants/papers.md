# Tokenizer construction & inference: literature read

Notes from a literature exploration on 2026-05-28 covering four papers on tokenizer construction algorithms (BPE alternatives, optimality bounds, inference-time segmentation). Recorded for future reference so the same ground doesn't have to be re-covered.

**Scope**: tokenizer-side algorithms (construction, segmentation). Not LM architecture, not training dynamics.

**Source quality**: papers were read end-to-end (full PDF for PathPiece's Appendix B; HTML for others). GreedTok's source code was reviewed at `github.com/PreferredAI/pcatt`. Web-search summaries alone were rejected as a basis for claims after one round of corrected mischaracterizations.

## Papers

| # | Citation | What it asks | Scale tested |
|---|---|---|---|
| 1 | [Zouhar et al. 2023](https://arxiv.org/abs/2306.16837) — *A Formal Perspective on Byte-Pair Encoding* (ACL 2023) | Is greedy BPE provably suboptimal? By how much? | Theoretical only; toy strings for empirical bounds |
| 2 | [Schmidt et al. 2024](https://arxiv.org/html/2402.18376v2) — *Tokenization Is More Than Compression* (PathPiece) | Does compression predict downstream LM performance? | 350M / 1.3B / 2.4B LMs, 200B tokens, MPT + ALiBi + AdamW |
| 3 | [Lim et al. 2025](https://arxiv.org/abs/2501.06246) — *A Partition Cover Approach to Tokenization* (GreedTok) | Can we beat greedy BPE without using pairwise merges? | 1B LM, 629B tokens, vocab 65K |
| 4 | [Sawada & Goyal 2025](https://arxiv.org/abs/2508.06621) — *Train It and Forget It: Merge Lists are Unnecessary for BPE Inference* | Can BPE inference work without the merge list (privacy angle)? | Qwen-2-7B-Instruct, post-hoc inference swap (no retraining) |

## 1. Zouhar et al. 2023 — Formal Perspective on BPE

**Setup**: pure theory + small empirical bound estimates on toy strings.

**Key claims**:
- Formalizes BPE training as maximizing **compression utility** `κ(μ) = |x| − |Apply(μ, x)|` over length-`M` merge sequences (equivalently: minimize the length of the merge-applied string, since `|x|` is fixed).
- Proves greedy BPE is a `(1/σ)(1 − e^{−σ})`-approximation of the optimal merge sequence, where `σ` is "total backward curvature of κ with respect to an optimal merge sequence."
- Empirical bound on toy alphabet (|Σ|=5, |x|≤15, |μ*|<5): σ ≈ 2.5 → approximation ≈ **0.37**.
- Tighter bound on English (small sample): σ ≈ 2.0 → approximation ≈ **0.43**.
- With greedy-prefix conditioning: ≈ 0.52.
- Provides an O(N log M) implementation (improving Kudo & Richardson 2018's O(N log N)).
- Provides a memoized exponential exact-optimal algorithm; not run at scale.

**Doesn't prove**: NP-hardness (that claim belongs to GreedTok). No large-scale empirical comparison of greedy vs optimal BPE.

**Critical disconnect from production BPE**: The objective they prove the bound for is *compression utility* (symbols eliminated). Real BPE picks merges by *per-step pair frequency*, which is a different (myopic) objective. The paper assumes they align via submodularity but doesn't prove the practical gap matters.

**Relevance to our project**: Sets a theoretical floor for "how much could you beat greedy BPE on compression." Floor is ~0.37–0.52 — i.e., greedy is guaranteed to achieve at least 37–52% of optimal compression utility in the worst case (the bound implies a worst-case *ceiling* of ~2.3–2.7× greedy, but that's bound-implied, not an observed or achievable gap). The bound is over total compression of a length-M merge sequence, not per-step. The actual practical greedy-vs-optimal gap on real corpora is unmeasured at scale. **But** they make no claim about downstream LM quality, only compression. Useful as a "yes greedy is provably suboptimal in this specific sense" reference; less useful for picking a next experiment.

## 2. Schmidt et al. 2024 — PathPiece / *Tokenization Is More Than Compression*

**Setup**:
- Architecture: **MPT (MosaicML decoder-only Transformer)** via `llm-foundry`.
- 350M model: `d_model=1024, n_heads=16, n_layers=24, expansion_ratio=4, max_seq_len=2048, attn_impl=triton, alibi=true, clip_qkv=6`.
- 1.3B and 2.4B variants scaled accordingly (Appendix B has the exact config).
- Optimizer: **`decoupled_adamw`**, lr=3e-4, betas=(0.9, 0.95), eps=1e-8, wd=1e-4, cosine schedule, 5% warmup, amp_bf16.
- Batch: 2M tokens, 100K steps → 200B tokens.
- 54 models at 350M, 6 at 1.3B, 4 at 2.4B.
- Eval: 10 lm-eval-harness multi-choice tasks (5-shot): `arc_easy, copa, hendrycksTest-marketing, mathqa, piqa, qa4mre_2013, race, sciq, hendrycksTest-sociology, wsc273`. Random baseline 32%. Tasks pre-filtered to "BPE at vocab=40,960 does well above random."

**Algorithm contributions**:
- *Constructor*: top-down vocab pruning from a 262,144-token initial vocab, removing the token whose removal raises corpus token count least, until target size reached. O(nL²) per iteration.
- *Inference (PathPieceL / PathPieceR)*: shortest-path segmentation through a DAG over byte positions. O(nL). PathPieceL takes longest-token tie; PathPieceR random tie. Applicable to any vocabulary.

**Headline findings**:
- **Pearson(corpus token count, avg accuracy) = 0.241**. Weak positive — i.e., compression barely predicts downstream.
- Rényi efficiency correlation ≈ −0.151 at α=2.5.
- Top 6 tokenizers (350M, 32K vocab) are statistically indistinguishable (all pairwise p > 0.05). At 32K avg accuracies: **BPE+Merge 48.8, Unigram+Likelihood 49.2, WordPiece+Greedy 48.5, BPE+Greedy 48.3, PathPieceL+BPE-init 45.6, etc.**
- At 40K vocab, top results: BPE+Merge 50.0, PathPieceL+BPE-init+PathPieceL 49.4.
- **Vocab size 32K–49K is "not a crucial decision over this range of sizes"** (paper's exact phrasing; R² 0.75–0.83 across sizes).
- **Pre-tokenization matters more than constructor** — pre-tokenization swings span ~2-4pp depending on the comparison (None→FirstSpDigit ~2.3pp, None→SpaceDigit ~4.0pp per Figure 4).
- BPE+Merge is consistently competitive across all configurations.

**Caveats** (paper-acknowledged):
- 138K GPU-hours total; the 100K-step 1.3B checkpoint underperformed (paper attributes to noise: "could be an issue").
- 10-task benchmark is cherry-picked for measurable signal at 350M scale.
- English-only; pre-tokenization findings don't generalize to unsegmented languages.
- "Confounding factors" acknowledged in the CTC-vs-accuracy analysis.

**Relevance to our project**:
- 350M / 200B tokens is the closest scale analog to nanochat d24 (~560M params / ~22B tokens). We're more token-starved, but params are within 2×.
- Architectural differences are real: nanochat uses **RoPE** (not ALiBi), **Muon** for hidden-layer weights (not decoupled_adamw), sliding-window attention. Tokenizer effects might transfer; might not.
- Their result that BPE+Merge stays competitive even when alternative constructions don't beat it on compression is **directly relevant** to our Unigram-vs-BPE experiment (see [`unigram-vs-bpe.md`](unigram-vs-bpe.md)).
- The Unigram+Likelihood 49.2 vs BPE+Merge 48.8 result at 32K is a positive data point for Unigram at downstream, even when our offline compression shows Unigram trailing by 12%. **This is the primary literature reason not to write off Unigram on compression alone.**

## 3. Lim et al. 2025 — GreedTok / *A Partition Cover Approach*

**Setup**:
- Reformulates tokenization as **partition cover** over the corpus, not pairwise BPE merges.
- Objective: select S ⊆ T with |S| ≤ k minimizing `Σ_w count(w) · partition(w, S∪B)`.
- Proves the decision problem **NP-hard** via vertex-cover reduction.
- Objective is neither submodular nor supermodular; relaxes to weighted maximum coverage (which is (1−1/e)-approximable).

**Algorithm**: GreedTok
- Inputs: words W with counts, candidate substring set T (default = all substrings of length ≥2 in W), budget k.
- Greedy: iteratively add the substring with the largest marginal cover gain, until |S|=k.
- Public C++ impl with Python bindings; supports HF Tokenizers format.

**Empirical scale**:
- Compression: on UN/arXiv/Wikipedia/PubMed corpora, vocab 5K–10K.
- LM training: **two 1B-parameter transformers**, vocab=65K, ~629B training tokens of DCLM-Dedup. ~75% token-set overlap between BPE and GreedTok vocabs.

**Headline numbers**:
- Compression: GreedTok beats BPE by **2.88%** and Unigram by **3.43%** on large corpora (tokens/word).
- 1B LM bits/byte (Wikitext): BPEM 0.7066 → GTEP 0.7028 → GTET 0.6989 (GTET ≈1.1% improvement over BPEM).
- 1B LM benchmark average accuracy: BPEM 62.0 → GTEP 62.5 → GTET **63.2** (GTET = +1.2pp over BPEM).
- "GreedTok has achieved better compression while still maintaining model performance" — paper's framing.

**Acknowledged limitations**:
- 1B param scale only; "more empirical confirmation at scale needed."
- Compared only BPE and Unigram baselines.
- "Mainly English / common languages" — low-resource not evaluated.
- No formal proof of approximation guarantee (empirical ≈0.9(1−1/e)).
- Memory cost: 160GB RAM for the largest run (14.3M words → 251M candidate substrings).

**Code review** (`github.com/PreferredAI/pcatt`, spot-checked — **not pursued**, kept as caveats): a likely index-mixing bug in the reference `max_cover.cpp` (`get_score_helper`; the production `pco_tokenizer.cpp` is clean); the production lib's `candidate_tokens` filter appears **non-functional** (all substrings are materialized regardless, so candidate-set pre-filtering needs a C++ fork); and lazy-greedy is used despite the objective being non-submodular (no exhaustive-greedy fallback to check divergence). The practical blocker is memory — no substring-level frequency pruning drives the 160GB-RAM run. The paper also under-specifies `max_token_length` / `min_word_count` per experiment (defaults are too small to reproduce the published vocabs).

**Relevance to our project**:
- Closest paper to the "learn a non-greedy construction" pivot we considered.
- The 1B-param LM result (~1.1% bits/byte, +1.2pp benchmark) is at adjacent scale to d24 (560M / 22B tokens).
- The 160GB RAM requirement at full scale rules out reproduction on M1 (16GB). Could replicate at arXiv scale (~5GB RAM) for the compression-only checkpoint.
- Their downstream gain (+1.2pp on a 32% random baseline) is modest in absolute terms — combined with PathPiece's "compression doesn't track downstream" finding, the prior probability that GreedTok transfers to nanochat d24 is moderate at best.

## 4. Sawada & Goyal 2025 — *Train It and Forget It* (Merge-list-free BPE inference)

**Setup**:
- Authors: Tomohiro Sawada, Kartik Goyal (Georgia Tech). Arxiv 2508.06621, August 2025.
- Motivation is **security**, not compression. A separate paper (Hayase et al. 2024) showed merge lists leak training-data signals; Sawada & Goyal cite that work and ask whether BPE inference can function without the merge list as a defense.
- Two replacement algorithms:
  - **Left-to-right greedy**: at each position, emit the longest vocab-matching prefix. Paper describes this as "simple and efficient" but doesn't give an explicit complexity bound (longest-prefix matching is O(n) with a trie).
  - **Maximal compression DP**: find the segmentation that minimizes total token count via dynamic programming. Paper states "dynamic programming reduces this to quadratic time" — O(n²) per pretoken.
- Both use **only the vocabulary** (the set of tokens), not the merge sequence.

**Evaluation**:
- Model: **Qwen-2-7B-Instruct** (151,645 vocab; HF tokenizer backend). Pretrained; **no retraining** — the inference algorithm is swapped post-hoc.
- Tasks: MMLU, ARC-Easy/Challenge (multi-choice), WMT15 De→En + WMT16 Cz→En (machine translation), S2ORC open-ended generation (5,899 abstracts).
- Metrics: exact accuracy (QA), COMET (MT primary), BLEU/METEOR (MT secondary), MAUVE (generation).

**Headline numbers**:

| Task | Standard merge | Left-to-right | Maximal compression |
|---|---:|---:|---:|
| ARC | 0.869 | **0.903** ↑ | 0.863 |
| MMLU | 0.656 | **0.705** ↑ | 0.678 ↑ |
| MAUVE (open-ended) | 0.904 | **0.985** ↑ | 0.927 ↑ |
| De→En COMET | 0.502 | 0.495 | 0.494 |
| Cz→En COMET | 0.685 | 0.632 ↓ | 0.633 ↓ |

- **MMLU jumps +4.9pp** on left-to-right greedy with no retraining. Same vocab, different segmentation.
- Cz→En COMET drops **5.3pp absolute** (0.685 → 0.632; ≈7.7% relative) — described as "potentially meaningful."
- Targeted corruption baselines (random deletion, merge shuffle, character-level) catastrophically degrade — so it's *not* "any inference algorithm works." Compression-optimizing algorithms specifically work.

**Caveats**:
- One model, English-dominant tasks. "Findings might not hold for low-resource languages or non-monotonic writing orders."
- Cz→En regression unexplained; blamed on morphological richness.
- Perplexity inflates (Table 3: left-to-right 95.891 vs standard 83.798) despite downstream improvements — suggests undertrained tokens fire more often.
- Doesn't defend against tokenization attacks generally; only against merge-list leakage.

**Relevance to our project**:
- Demonstrates that **segmentation-side swaps on a pretrained model can move downstream metrics by 5pp** without retraining. Most relevant to a post-hoc inference experiment.
- Less relevant to nanochat d24 specifically because their setup is "evaluate a pretrained big model" not "train a small model from scratch." The pretrained model has already learned what it learned; swapping segmentation changes which tokenization the model sees at inference but doesn't change the parameters.
- For "train from scratch with greedy inference instead of merge replay" — this paper doesn't speak to that directly. Training distribution differs from inference distribution would be the open question.

## Cross-paper synthesis

**On compression vs downstream**:

PathPiece is the load-bearing paper here. At 350M / 200B tokens (closest scale to nanochat d24), they find compression weakly predicts downstream (Pearson 0.241), and the top tokenizers at 32K vocab span ~0.9pp (49.2 → 48.3) yet are statistically indistinguishable (top-6 all pairwise p > 0.05; see §2). **Conclusion**: compression delta alone shouldn't drive tokenizer selection at our scale. (These are imported priors from a different architecture + optimizer — unverified in our setup; they gate which experiments are worth a GPU A/B, not the selection itself.)

Zouhar and GreedTok both focus on the compression objective. Zouhar proves greedy BPE is suboptimal *at compression*; GreedTok reports +2.88% compression vs BPE and a modest downstream gain at 1B params (+1.2pp benchmark average accuracy, GTET 63.2 vs BPEM 62.0). PathPiece's framing predicts the compression delta would weakly transfer to downstream — and GreedTok's modest downstream gain is consistent with that prediction.

**On segmentation-side vs construction-side**:

PathPiece's data shows the **inference algorithm matters as much as the vocab constructor** at fixed vocab. Native merge replay (BPE+Merge) is competitive; PathPieceL shortest-path on a BPE vocab is *worse* (overall rank 13 vs BPE+Greedy's rank 4 and BPE+Merge's rank 3; pairwise p ≈ 4.4e-5 against rank 3 and 8.8e-6 against rank 4, per the paper's Appendix E.1). Train-It appears to cut the other way on a different setup (left-to-right greedy on Qwen-2 BPE-vocab beats merge replay by 4.9pp on MMLU).

The reconciliation: PathPiece evaluates **train-from-scratch** with each segmentation; Train-It evaluates **post-hoc swap on a pretrained model**. Different experimental designs measure different things. For our nanochat d24 use case (train from scratch), PathPiece is the more relevant precedent — and it suggests native BPE inference is hard to beat by changing segmentation alone.

**On hyperparameters not investigated**:

UnigramTrainer's `max_piece_length` (default 16) demonstrably truncates competitive long tokens in our experiment (see [`unigram-vs-bpe.md`](unigram-vs-bpe.md)). Neither PathPiece nor GreedTok sweeps this. Worth a sweep before drawing strong "Unigram loses" conclusions.

**On scale gaps from our setup**:

| Paper | Closest scale match to nanochat d24 | Confidence transfer |
|---|---|---|
| PathPiece 350M / 200B tokens | High — same model-size order, same vocab range | Medium-high |
| GreedTok 1B / 629B tokens | Adjacent — 2× more params, ~30× more tokens | Medium |
| Zouhar (toy theory) | None — no LM | Low |
| Train-It pretrained-7B / post-hoc swap | Low — different experimental design | Low |

## Recommended next reads (not read — backlog)

- **SuperBPE** ([2503.13423](https://arxiv.org/html/2503.13423v1)) — relaxes whitespace pre-tokenization; most adjacent to our `force_merges` work.
- **"Greed is All You Need"** ([2403.01289](https://arxiv.org/html/2403.01289v1)) — evaluation of tokenizer inference methods; complements PathPiece.
- Lower priority: **MANTa** ([2212.07284](https://arxiv.org/abs/2212.07284)) and **Charformer / GBST** ([2106.12672](https://arxiv.org/abs/2106.12672)) — learned / byte-level tokenization; **BlockBPE** ([2507.11941](https://arxiv.org/html/2507.11941v1)) — parallel BPE inference, only if inference becomes a bottleneck.

(Dropped a search-summary-only "Frequency-Ordered Tokenization" hit — unvalidated, arXiv id unconfirmed.)
