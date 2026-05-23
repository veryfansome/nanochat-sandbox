# Multi-Token Prediction (MTP)

Status: **implemented** — overlay + wrapper + smoke + d6/M4 A/B measured. GPU validation pending.
Target repo: `nanochat` (Karpathy) — kept **pristine**, never edited.

## Current state

The mechanism is live and smoke-verified:

- [`../../overlay/mtp.py`](../../overlay/mtp.py) — `MTPGPT(GPT)`. Forward returns `baseline_CE + (α/k) · Σ_{d=1..k} CE(logits[:, :-d, :], targets[:, d:])` on the training path; pure baseline CE on the eval path (`loss_reduction='none'`) so `val/bpb` stays directly comparable.
- [`../../wrappers/train_mtp.py`](../../wrappers/train_mtp.py) — six lines: patch `nanochat.gpt.GPT` → `MTPGPT`, `runpy` `scripts.base_train`. Does **not** patch `GPTConfig`.
- [`../../wrappers/smoke_mtp.py`](../../wrappers/smoke_mtp.py) — verifies (a) forward math, (b) config-portability, (c) monkeypatch. <5 sec, no data.

### Measured at d6 / 5000 iters on M4 (wandb `bcv9920m`, archived `results/d6_mtp/`)

| metric | d6_baseline | d6_mtp | Δ |
|---|---:|---:|---:|
| val/bpb (final, step 5000) | 1.16534 | 1.21398 | **+0.04864 (+4.17%)** |
| train/tok_per_sec | 27,628 | 18,657 | **−32.5%** |
| total time | 50.6 min | 71.3 min | +41% |
| CORE (post-train eval) | not recoverable | −0.0179 | (within noise; below-random on centered tasks) |

The val/bpb gap is **stable, not transient** — diverges from 0 at step 0, locks in at +0.045–0.049 by step ~500, holds for the remaining ~4500 steps. This is the expected toy-scale signature of MTP: the model has 37M params and 5000 warmup-heavy iters; asking the residual stream to also encode lookahead for tokens 2–4 ahead diverts gradient budget the main objective can't spare. The literature (Gloeckle 2024, DeepSeek-V3) shows the win at scale, where the model can afford the dual objective and the denser supervision becomes a sample-efficiency lever rather than a capacity tax. tok/sec hit (−32%) tracks the cost of 4 separate `F.cross_entropy` calls per step — similar magnitude to the Python-fused-zloss number, same root cause. Expect this to shrink to a few percent at GPT-2 / GPU scale where the trunk dominates wall-clock.

train/loss in wandb is **not** directly comparable to baseline — it includes the aux contribution. Use val/bpb for cross-run comparison.

### Implementation choices (per the agreed plan)

- **Naive only.** k separate `F.cross_entropy` calls on shifted target slices. Each saves its own softmax for backward, so peak saved-for-backward memory at the loss = (k+1) × `(B, T, V)` slice. Worked example of the loss-shape gotcha from `../../overlay/README.md`. A Python fused autograd would halve this but cost throughput per the z-loss experience; revisit only if scale-up memory forces it.
- **Shared unembedding** with the main `lm_head`. No extra parameters; all heads use the same logits with different shifted targets. The trunk is regularized to encode multi-step-ahead information in the residual stream.
- **k = 3 default** (predicts t+2, t+3, t+4). Env var `MTP_HEADS`.
- **Loss weighting**: total aux contribution = α (default 0.3); per-head = α/k. Env var `MTP_ALPHA`.
- **Eval path** (`loss_reduction='none'`, used by `evaluate_bpb` to compute per-token bpb) returns main CE only — aux supervision is a training regularizer; mixing it into bpb would inflate the metric in a non-comparable way.

Run the smoke any time: `uv run python -m wrappers.smoke_mtp`.

### Next milestone — GPU validation at d24/Lambda

The d6/M4 result tells us MTP doesn't help at toy scale (expected), not whether it helps at GPT-2 scale (the actual question). On Lambda:

```bash
WANDB_RUN=d24_baseline bash runs/speedrun.sh
WANDB_RUN=d24_mtp MODEL_TAG=d24_mtp OVERLAY=mtp bash runs/speedrun.sh
uv run python -m tools.compare_runs d24_baseline d24_mtp
```

At d24 scale the saved-for-backward stack from k=3 aux heads is roughly `4 × (16 × 2048 × 32768 × 4 bytes) ≈ 16 GB` — comfortable on a single H100 (80 GB HBM) and trivial across 8. The throughput hit should drop to a few percent (trunk dominates).

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

### Recommended starting config

- `k = 3` extra heads (predict `t+2, t+3, t+4`).
- Use **fused / cut cross-entropy** (Liger-style) for every head — never materialize the full `(B, T, vocab)` logits tensor. This removes the memory blowup; the `d×V` matmul FLOPs remain (irreducible).
- Loss weighting: start with equal weights, or mild decay for deeper heads (e.g. `1.0, 0.5, 0.25`) — tune.

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
