# Token-level loss weighting — proposed setup

Status: **proposal / not yet implemented**
Target repo: `nanochat` (Karpathy) — kept **pristine**, never edited.

## Goal

Reweight per-token cross-entropy so the optimizer spends more gradient on **informative** tokens and less on **easy / low-information** ones. Standard pretraining gives every target token equal weight; in practice most tokens carry almost no signal (whitespace, function words, predictable continuations of common n-grams), and a small minority carry most of the learnable information. The hypothesis is that flat weighting wastes sample efficiency.

## Brain-inspired motivation

Loose mapping to **attention-gated learning**: biological credit assignment is not uniform across experience — surprising, novel, or behaviorally relevant stimuli get gated up by neuromodulatory signals (dopamine, acetylcholine) and drive disproportionate plasticity, while routine inputs are largely ignored. Token-level loss weighting is the analog at the supervision side: not every token deserves equal "learning attention."

## What it is

Replace flat cross-entropy

```
L = mean_i [ CE(logits_i, target_i) ]
```

with a weighted version

```
L = sum_i [ w_i · CE(logits_i, target_i) ] / sum_i [ w_i ]
```

where `w_i ≥ 0` is computed per target token. The interesting design space is in **how `w_i` is computed**.

## Weighting schemes — choice points

Roughly ordered cheap-to-rich:

1. **Entropy-based** — `w_i ∝ H(p_i)` (or `1 - max(p_i)`), where `p_i` is the model's own predicted distribution at position `i`. Down-weights tokens the model is already confident on. Free (logits already computed); fully online. **Recommend as default first variant.**
2. **Loss-based (focal-style)** — `w_i ∝ CE(logits_i, target_i) ** γ`, with `γ ∈ [0.5, 2]`. The natural-language version of focal loss (Lin et al. 2017). Same per-step cost as flat CE; emphasizes hard tokens automatically.
3. **Reducible-loss-based (RHO-Loss)** — `w_i ∝ CE_model(i) − CE_ref(i)`, where `CE_ref` is a small pretrained "irreducibility" reference model run alongside (or precomputed and cached per-token). Higher-quality signal than raw loss because tokens that are *intrinsically* unpredictable (random names, IDs, future randomness) don't get up-weighted just for being hard. Mendlowitz et al. 2022 / Lin et al. 2024 ("Not All Tokens Are What You Need").
4. **Position / role-based** — fixed multipliers (e.g., down-weight whitespace token IDs, up-weight rare-vocab tokens). Trivial, but blunt; useful as an ablation against the learned schemes.

All four can be applied with a **temperature / floor** so no token's weight goes to zero (preserves coverage and avoids degenerate selection).

## Why it fits the overlay cleanly

**Pure model-side change.** Touches only `GPT.forward()` — replaces the single `F.cross_entropy(..., reduction='mean')` call with a per-token loss + weighting + reduction. The training loop in `scripts/base_train.py` runs **completely unmodified**, same patch + `runpy` pattern as z-loss and MTP.

Note the **`loss_reduction != 'mean'` eval-path branch** is mandatory here — `loss_eval.py` calls the model with `loss_reduction='none'` to compute `val/bpb`, and that path must return **unweighted** per-token CE so bpb stays comparable to baseline. Same lesson as MTP — see [`../../overlay/README.md`](../../overlay/README.md) "Aux losses are a training-time regularizer".

For scheme **(3)** (reducible loss with a reference model) the wrapper additionally needs to load the reference model into device memory before `runpy`ing `base_train`. That's still zero edits to upstream — the reference is just an extra model held by the overlay subclass and consulted in `forward`.

## Design

- Subclass `GPT` → `WeightedCEGPT`. Override `forward()` only.
- Scheme selected via env var, e.g. `TOKEN_WEIGHT_SCHEME=entropy|focal|rholoss|fixed` at `__init__`, stored as an instance attribute (**not** a `GPTConfig` field — see [zloss README](../zloss/README.md) "Implementation note").
- Scheme-specific knobs also via env (e.g. `FOCAL_GAMMA=1.0`, `TOKEN_WEIGHT_FLOOR=0.1`, `RHO_REF_CKPT=/path/to/ref.pt`).
- Implement weighting with `F.cross_entropy(..., reduction='none')` → multiply by `w_i` → divide by `w_i.sum()`. No custom autograd Function needed; per-op cost is one extra elementwise mul/div over `(B, T)`.
- **Exclude `ignore_index = -1` positions from both numerator and denominator** — nanochat uses `-1` for padded/skip tokens (`gpt.py:477`), and `F.cross_entropy` already zeros their loss, so just mask them out of `w_i` before normalizing.
- Eval-path branch (`loss_reduction != 'mean'`) returns the unweighted per-token CE that `loss_eval.py` expects.

### Reference model (scheme 3 only)

- Reference model is a *small* baseline (e.g. d4 or d6 nanochat) trained once on a subset of the same corpus and frozen. The point isn't a powerful teacher — it's a cheap irreducibility estimator.
- Held by the overlay subclass on the same device, run in `inference_mode()` inside `forward()`. Memory cost is the ref model's params + a single forward's activations (no grads, no optimizer state).
- Trade-off vs schemes 1–2: extra compute per step (one extra forward through the ref) and engineering (training + loading the ref), in exchange for higher-quality weights.

## Overlay implementation

`overlay/token_loss_weight.py` subclassing `GPT`, plus `wrappers/train_token_loss_weight.py` (patch + `runpy`) and `wrappers/smoke_token_loss_weight.py`. Smoke must cover:

- (a) Weighted loss math matches a hand-computed reference on a tiny example for each scheme.
- (b) Config-portability — no overlay-only field leaks into the saved checkpoint (verified the same way as zloss `(b)`).
- (c) Monkeypatch substitution.
- (d) Eval-path (`loss_reduction='none'`) returns **flat** unweighted per-token CE — bit-identical to baseline. This is the critical invariant; weighted bpb would be incomparable to the baseline.

## Combines well with

- **z-loss** ([`../zloss/README.md`](../zloss/README.md)) — orthogonal; z-loss penalizes logit-magnitude drift, weighting reshapes the per-token loss term itself. Trivial to combine.
- **MTP** ([`../mtp/README.md`](../mtp/README.md)) — token weighting on the `t+1` head transfers naturally; whether to also weight `t+2…t+k` heads is an open question (recommend: same weights, same mask).
- **Deep supervision** ([`../deep-supervision/README.md`](../deep-supervision/README.md)) — apply the same per-token weights to each aux head's loss.
- **Online data selection** ([`../online-data-selection/README.md`](../online-data-selection/README.md)) — same intuition (focus capacity on informative signal) at a different granularity (sample vs. token). The two are complementary, not substitutes; data selection chooses *which sequences* to train on, token weighting reshapes *within* each sequence. Worth ablating both alone and together.

## Expected cost

**Compute (schemes 1, 2, 4)**: minimal — one extra `(B, T)` weight tensor, one elementwise multiply, one sum. Same FLOP order as z-loss. Expect <2% throughput hit at any scale.

**Compute (scheme 3, reducible)**: one extra forward pass through a *small* reference model per training step. If the reference is ~1/10 the trunk size, expect ~10% throughput hit; if same size, ~50%. Caching reference-model per-token CE on disk per shard avoids the runtime cost entirely at the price of ~`shard_size × log(V)` bits of storage (cheap).

**Memory**: scheme 1/2/4 — negligible (one `(B, T)` weight tensor). Scheme 3 — adds the reference model's parameters and per-step activations.

**Inference cost**: unchanged — weighting is a training-time loss shape only.

## Why this is worth doing here

The catalog leans toward **densifying** the supervision signal (MTP across time, deep supervision across depth). Token weighting is the third axis — **reshape** the existing signal so the optimizer spends gradient where it matters. It's pure model-side, cheap, and the literature shows non-trivial gains at small scale specifically (RHO-Loss; "Not All Tokens Are What You Need" reported substantial sample-efficiency improvements at the 1B-and-below regime — exactly where the nanochat-sandbox project operates).

The variant ladder also gives a clean ablation story: entropy / focal are essentially free and isolate "does any non-flat weighting help"; RHO-Loss tests "is a *learned* notion of informativeness needed."

## Testing / A-B

- Same pinned nanochat commit + same seed as baseline.
- Primary metrics: `val/bpb`, CORE.
- Per the project's `ideas/README.md` "Lessons learned" — d6/runcpu is mechanism verification only; real comparison needs Lambda d24 speedrun.
- Smoke-test wiring first (`--depth=4 --num-iterations=20`); confirm eval bpb matches baseline bit-for-bit (invariant (d) above).
- Recommended order: scheme 1 (entropy) first — cheapest, no extra model — then scheme 2 (focal), then scheme 3 (RHO-Loss) only if 1/2 show signal.

## Open questions

- Which weighting scheme dominates at nanochat scale — entropy, focal, or reducible loss.
- Whether to apply weights to **all** tokens or only to positions past a warmup point (cold-start models have garbage entropy estimates).
- For RHO-Loss: what reference-model size gives the best gain-per-extra-FLOP trade-off; how often the reference needs retraining vs. a once-frozen baseline.
- Interaction with z-loss: does heavy down-weighting of confident tokens reduce the logit-drift problem z-loss targets, making z-loss less necessary (or vice versa)?
- Whether token weights should also rescale the optimizer's effective LR per-token (probably no — keep weights inside the loss).

## References

- Lin et al. 2017 — "Focal Loss for Dense Object Detection." Original focal-loss formulation (vision).
- Mendlowitz et al. 2022 — "Prioritized Training on Points that are Learnable, Worth Learning, and Not Yet Learnt" (RHO-Loss).
- Lin et al. 2024 — "Not All Tokens Are What You Need" (selective-LM, token-level reducible-loss filtering at LM pretraining scale).
