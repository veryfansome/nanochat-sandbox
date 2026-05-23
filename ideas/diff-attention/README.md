# Differential attention (Diff Transformer) — proposed setup

Status: **proposal / not yet implemented**
Target repo: `nanochat` (Karpathy) — kept **pristine**, never edited.

## Goal

Replace standard softmax attention with **differential attention** to cancel attention noise — yielding sparser, cleaner attention maps, better long-context retrieval, less hallucination, and improved robustness. A capability-positive architecture change at roughly parameter parity (no scale, no data change).

## What it is

Ye et al. 2024, "Differential Transformer" (Microsoft). Each attention layer computes **two** softmax attention maps and **subtracts** them:

```
DiffAttn(X) = ( softmax(Q1 K1ᵀ/√d) − λ · softmax(Q2 K2ᵀ/√d) ) V
```

- `λ` is a learnable scalar, **re-parameterized** for stable optimization: `λ = exp(λ_q1·λ_k1) − exp(λ_q2·λ_k2) + λ_init`.
- `λ_init` is **depth-dependent**, e.g. `λ_init = 0.8 − 0.6·exp(−0.3·(layer−1))`.
- A **headwise normalization** (GroupNorm / RMSNorm per head) is applied to each head's output before the output projection, then scaled by `(1 − λ_init)` — important for training stability.
- Head count / head_dim are chosen so total params and FLOPs match the baseline.

The common-mode (noise) component of the two attention maps cancels, leaving signal.

## Why it fits the overlay

The change is confined to the attention module. `Block` instantiates `CausalSelfAttention` directly, so the overlay patches `nanochat.gpt.CausalSelfAttention` before `runpy` — the same trick used to swap `GPT` in the MTP plan (`../mtp/README.md`). **Zero harness edits.**

## Design — integration with nanochat specifics

Subclass `CausalSelfAttention → DiffCausalSelfAttention`. Reconcile with what the repo already does in attention:

- **QK-norm**: the repo does `q, k = norm(q), norm(k)` then `* 1.2` (`gpt.py`) — apply to both the `Q1/K1` and `Q2/K2` groups.
- **RoPE**: `apply_rotary_emb` — apply to both groups.
- **FlashAttention-3**: the repo calls `flash_attn.flash_attn_func`. Differential attention = **two FA3 calls** (one per group) then subtract the outputs.
- **Value embeddings / value residual**: `V` is shared across the two groups; the existing `v = v + gate * ve` still applies to the shared `V`.
- **Headwise norm**: add a per-head RMSNorm/GroupNorm on the attention output.
- **FA3 constraint**: head_dim must be divisible by 8. The repo targets `head_dim = 128` (`--head-dim`); splitting into two groups must preserve divisibility — factor into the head-count / head_dim sizing.

## Overlay scope — training vs. inference

- **Training** (`kv_cache is None`) is a clean overlay — the forward path is self-contained.
- **Inference**: the `KVCache` in `engine.py` assumes a single K/V per layer. Differential attention has **two K groups**, so the cache layout changes. For base-model capability A/B (`val_bpb`, CORE — likelihood evals over the forward path) the cache is **not** needed. Generation / chat would require a `KVCache` adaptation (subclass or copied engine) — **defer that** until the capability A/B justifies it.

## Expected cost

- **Paper-faithful sizing** (each group half-width) → roughly parameter and FLOP **parity** with the baseline.
- **Naive sizing** (two full-width attention calls) → attention compute ~2×; since attention is only part of a layer, total training time roughly **+15–30%**. Prefer the paper sizing.

## Testing / A-B

- Same pinned nanochat commit + same seed as baseline.
- Metrics: `val_bpb`, CORE.
- **Caveat**: Diff Transformer's biggest wins are long-context retrieval, robustness, and reduced hallucination. At nanochat's 2048-token context the visible gain on CORE / `val_bpb` may be modest — set expectations, and add a long-context retrieval probe if feasible.
- Smoke-test the wiring first with the repo's tiny config (`--depth=4 --num-iterations=20`).

## Open questions

- `λ_init` depth schedule.
- Headwise norm choice (GroupNorm vs. RMSNorm) and placement relative to the repo's existing norms.
- Head count / head_dim sizing for true parameter parity (and FA3 divisibility).
- Reconciling with the existing QK-norm + `1.2` scaling.
- Whether the win is visible at 2048 context, or needs longer context to show.

## References

- Ye et al. 2024 — "Differential Transformer" (Microsoft).
