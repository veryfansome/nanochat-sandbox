# Multi-Token Prediction (MTP)

Status: **validated at d24/8xA100 — deprioritized.** val/bpb regression got *worse* at scale (+4.17% at d6 → +8.78% at d24), CORE "win" is essentially driven by a single task. Three rescue paths exist (scale, DeepSeek-MTP architecture, lower α + k=1) but none are next-up.
Target repo: `nanochat` (Karpathy) — kept **pristine**, never edited.

## Current state — d24 results (8xA100 40GB)

| metric | baseline | mtp (k=3, α=0.3) | Δ |
|---|---:|---:|---:|
| val/bpb (final) | 0.71609 | 0.77898 | **+0.06288 (+8.78%)** |
| CORE (`base_eval`, full) | 0.250488 | 0.258635 | +0.00815 (+3.25%) — but see caveat |
| CORE (wandb, mid-train sub) | 0.260098 | 0.262987 | +0.00289 (+1.11%) |
| train tok/sec | 291,210 | 260,517 | −10.54% |
| total time | 5h33m | 6h13m | +12% |

**MTP needed `DEVICE_BATCH_SIZE=4` on 40GB HBM** (vs baseline's 8); the extra k=3 saved softmaxes pushed OOM at batch=8.

### The CORE "win" is one task

Per-task sign test: mtp wins 13/22 tasks, loses 8, ties 1. P(≥13/21) ≈ 0.19 — **not statistically significant**. The mean +0.008 CORE delta is dominated by a single outlier: **boolq +0.111** (still negative-centered for both runs, but mtp is "less wrong"). Excluding boolq, mtp's mean Δ across the remaining 21 tasks is **+0.003** — pure noise.

Most striking: **mtp hurts factoid-recall tasks**: bigbench_qa_wikidata −0.046, squad −0.025, jeopardy −0.005. Plausible mechanism: MTP's aux objective pulls representations toward "smooth multi-token continuations," which actively damages precise discrete factual memorization. Factoid recall wants sharp, peaky next-token distributions; MTP fights that.

### CORE-per-bpb-cost vs zloss (comparable trio, same compute budget)

| | CORE % gain | val/bpb % loss | CORE gain per bpb cost |
|---|---:|---:|---:|
| zloss | +5.15% | ≈ 0% (artifact-corrected) | unbounded — basically free |
| mtp | +3.25% (mostly boolq) | +8.78% | 0.37 — poor |

zloss dominates mtp on every axis at this scale.

### Why the literature's "MTP helps at scale" claim doesn't hold here

Gloeckle et al. (2024) report MTP becomes net-positive at multi-billion-parameter scale with parallel linear heads. d24 (~560M params) is in their "still negative" regime. d6 → d24 made the val/bpb regression *worse* relative-wise, not better — so scale alone, between these two points, didn't help. Either the threshold is further out (d30+, costing ~$500/trio to test), or our config diverges from the paper's in ways that matter (k, α, shared vs separate unembedding, no DeepSeek-block).

### Rescue paths (none are next-up)

If MTP gets revisited later, ordered by cost:

1. **α=0.1, k=1** — cheapest probe. Single-aux-head, lower aux weight. Tests whether the trunk can afford *some* multi-token supervision without the capacity tax we saw at α=0.3, k=3. One extra `(B, T, V)` softmax + one extra CE call vs baseline (near-baseline compute, ~1-2% tok/sec hit at scale), ~$80 to run.
2. **DeepSeek-MTP architecture** — separate transformer block per prediction depth (not parallel heads on shared trunk). Different cost profile, different objective shape. Substantial implementation work.
3. **Scale to d30** — actually test the literature's claim at a closer-to-paper scale. ~$500 for a baseline+mtp pair.

Per [`../../STATUS.md`](../../STATUS.md), the higher-priority next steps are deep supervision, token-level loss weighting, and differential attention. MTP rescue probes only happen if those raise specific questions answerable by an MTP variant.

### Components (unchanged)

- [`../../overlay/mtp.py`](../../overlay/mtp.py) — `MTPGPT(GPT)`. Training path: `baseline_CE + (α/k) · Σ_{d=1..k} CE(logits[:, :-d, :], targets[:, d:])`. Eval path (`loss_reduction != 'mean'`): pure baseline CE — `val/bpb` is directly comparable. (This is the contract zloss got wrong; mtp got it right because the lesson was written *after* zloss's first implementation. See [`../zloss/README.md`](../zloss/README.md) "Eval-path bug" for the story.)
- [`../../wrappers/train_mtp.py`](../../wrappers/train_mtp.py) — six lines: patch `nanochat.gpt.GPT` → `MTPGPT`, `runpy` `scripts.base_train`. Does **not** patch `GPTConfig`.
- [`../../wrappers/smoke_mtp.py`](../../wrappers/smoke_mtp.py) — verifies (a) forward math, (b) config-portability, (c) monkeypatch. Worth adding (d) eval-path equivalence retroactively, matching zloss's smoke (e), to prevent future regressions.

train/loss in wandb is **not** directly comparable to baseline — it includes `α · Σ aux_ce` (about 2.0 nats extra on d24 at α=0.3, k=3, matching baseline's 2.3 → mtp's 4.3). Use val/bpb for cross-run comparison.

### Earlier — d6 / 5000 iters on M4 (wandb `bcv9920m`, archived `results/d6_mtp/`)

| metric | d6_baseline | d6_mtp | Δ |
|---|---:|---:|---:|
| val/bpb (final, step 5000) | 1.16534 | 1.21398 | +0.04864 (+4.17%) |
| train/tok_per_sec | 27,628 | 18,657 | −32.5% |

The d6 result said "doesn't help at toy scale" (correctly). The hope was scale would flip it; d24 shows it doesn't, at least between these two points. The d6 val/bpb signature (stable +0.045-0.049 from step ~500) was directionally identical to d24's (+0.055-0.063 stable from step ~250) — same mechanism, just bigger absolute regression. **The "more training, less aux tax" expectation didn't materialize.**

### Implementation choices (unchanged — d24 deprioritization is about results, not design)

- **Naive only.** k separate `F.cross_entropy` calls on shifted target slices. Each saves its own softmax for backward, so peak saved-for-backward memory at the loss ≈ (k+1) × `(B, T, V)` slice.
- **Shared unembedding** with the main `lm_head`. No extra parameters.
- **k = 3 default** (predicts t+2, t+3, t+4). Env var `MTP_HEADS`.
- **Loss weighting**: total aux contribution = α (default 0.3); per-head = α/k. Env var `MTP_ALPHA`.
- **Eval path** (`loss_reduction != 'mean'`) returns main CE only.

## Goal

Improve base-model capability without changing model scale or training data. MTP adds auxiliary output heads that predict tokens `t+2, t+3, …` alongside the main `t+1` next-token loss. The denser learning signal improves the shared trunk's representations. The aux heads are **dropped at inference**, so the deployed model is unchanged in cost.

This is one of the objective-level levers identified for the "improve capability with scale + data off the table" constraint.

## Why MTP fits an overlay cleanly

MTP is a **pure model-side change**. It only alters the scalar loss returned by `GPT.forward()`. The training loop in `scripts/base_train.py` (`loss = model(x,y)` → `loss.backward()` → `optimizer.step()`) runs **completely unmodified**. Grad accumulation, DDP, the LR schedule, checkpointing — all untouched.

That means we can experiment without editing the nanochat repo at all, and keep `git pull`-ing upstream improvements.

## Design

Cheap variant — **Gloeckle et al. 2024** ("Better & Faster LLMs via Multi-token Prediction"): `k` independent linear heads sharing the trunk.

- Trunk (attention + MLP over all layers) is computed **once**.
- Each head is its own `d_model → vocab` projection + cross-entropy.
- During the backward pass the `k` head gradients are **summed at the final residual-stream activation**, then a **single** backward traverses the trunk (trunk backward is NOT duplicated per head).
- Total loss = weighted sum of the `t+1 … t+k` head losses.

Do **not** use the DeepSeek-V3 MTP variant here — it uses a full transformer block per prediction depth (~one extra layer each), far more expensive than linear heads.

### Recommended starting config (historical — superseded by d24 result)

This was the original plan; **what's actually implemented** is shown above under "Implementation choices." The differences worth noting:

- ~~Use **fused / cut cross-entropy** (Liger-style) for every head~~ — we went **naive** (k separate `F.cross_entropy` calls). Rationale per the z-loss work: a Python `torch.autograd.Function` reimplementing CE is *slower* than naive in pure PyTorch (no CUDA/Triton kernel to back it). A true Liger-style fused CE would still help at scale but is out of scope for this overlay.
- `k = 3` extra heads — kept as default.
- Loss weighting `α/k` per head (total aux contribution `α=0.3`) — kept; rescue path explores `α=0.1, k=1`.

## Overlay implementation plan

Keep nanochat pristine; all work lives in a **separate experiments repo** with nanochat pinned as a dependency (git dependency in `pyproject.toml`, or a git submodule installed editable).

```
ideas/mtp/
  README.md          <- this file
overlay/
  mtp.py             <- MTPGPT subclass + MTPConfig
wrappers/
  train_mtp.py       <- monkeypatch + runpy wrapper
results/             <- A/B run logs
```

### 1. Model subclass — `overlay/mtp.py`

Subclass `GPT`; every needed method is overridable:

```python
from nanochat.gpt import GPT, GPTConfig

class MTPGPT(GPT):
    def __init__(self, config):
        super().__init__(config)
        # k extra d_model -> vocab heads (config.mtp_heads)
    def forward(self, idx, targets=None, **kw):
        # run trunk once; compute t+1 loss + t+2..t+k losses;
        # return weighted sum (use fused cross-entropy per head)
    def setup_optimizer(self, **kw):
        ...   # include the new head params
    def init_weights(self):
        ...   # init the heads
    def estimate_flops(self):
        ...   # update so MFU / tok/sec logging stays honest
```

`MTPConfig` subclasses `GPTConfig` and adds `mtp_heads` (and loss weights) with **defaults**, so upstream's `GPTConfig(sequence_len=..., ...)` construction in `base_train.py` still works without passing the new field.

### 2. Wrapper — `wrappers/train_mtp.py`

`base_train.py` is a script with the model class hardcoded (`from nanochat.gpt import GPT, GPTConfig, Linear`, then `GPT(config)`). There is no plugin hook, so substitute the class *before* the script imports it:

```python
import runpy
import nanochat.gpt as g
from overlay.mtp import MTPGPT, MTPConfig

g.GPT = MTPGPT          # base_train's `from nanochat.gpt import GPT` resolves to this
g.GPTConfig = MTPConfig # subclass w/ defaulted extra fields

runpy.run_module("scripts.base_train", run_name="__main__")
```

This runs Karpathy's `base_train.py` **verbatim** with our model swapped in. Pass MTP hyperparams via **env vars** (`MTP_HEADS=3`) read inside `MTPConfig` — avoids argparse conflicts with the upstream parser.

## Expected cost

Training compute increase is dominated by the extra `d_model → vocab` matmuls. The unembedding's share of forward FLOPs is **scale-dependent**:

| Model size                | 3-head training-time cost |
|----------------------------|---------------------------|
| depth 12 (default-ish)     | ~+30–35%                  |
| depth 26 (GPT-2 speedrun)  | **~+15–20%** (~+20 min on the ~2h run) |
| larger                     | <+10%                     |

Notes:
- Trunk forward AND backward are computed once — they do **not** scale with `k`.
- Wall-clock lands slightly under the raw FLOP increase (fixed overheads — optimizer step, syncs, dataloader — don't scale), nudged back up by ~+15% extra params → larger optimizer state + gradient all-reduce volume.
- Fused CE removes the memory cost; the matmul FLOPs are the irreducible floor.
- **Inference cost: unchanged** (heads dropped).

## Testing / A-B methodology

- Pin a single nanochat commit; run **baseline and MTP on the same commit and same seed** — otherwise upstream improvements confound the comparison.
- Smoke-test the overlay wiring cheaply first with the repo's tiny config (from `base_train.py` docstring): `--depth=4 --max-seq-len=512 --device-batch-size=1 --eval-tokens=512 --core-metric-every=-1 --total-batch-size=512 --num-iterations=20`
- Primary metrics: `val_bpb` and CORE (same as the leaderboard). Compare the `t+1` head only — the aux heads are a training aid, not the eval target.
- Record the pinned commit + config + metrics for every run under `results/`.

## Open questions / things to tune

- Loss weighting across heads (equal vs. decaying).
- `k`: is 3 worth the ~+20% vs. `k=1` at ~+8%?
- Head architecture: bare linear vs. a small MLP per head.
- Interaction with the existing `lm_head` logit softcap (`gpt.py`) — apply the same softcap in the aux heads?
- Whether to keep one aux head at inference for speculative decoding (free side benefit, not the primary goal).

## References

- Gloeckle et al. 2024 — "Better & Faster Large Language Models via Multi-token Prediction" (the cheap k-linear-heads variant used here).
- DeepSeek-V3 technical report — MTP with per-depth transformer blocks (the expensive variant; NOT used here).
