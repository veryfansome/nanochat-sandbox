# z-loss

Status: **implemented** — overlay + wrapper + smoke are checked in; real-data validation pending.
Target repo: `nanochat` (Karpathy) — kept **pristine**, never edited.

## Current state

The mechanism is live and smoke-verified:

- [`../../overlay/zloss.py`](../../overlay/zloss.py) — `ZLossGPT(GPT)`. Forward returns `baseline_CE + z_coeff · mean(logsumexp(logits)²)` over non-ignored target positions. Two implementations:
  - **Naive** (default) — `F.cross_entropy(logits, ...)` + `torch.logsumexp(logits, ...)` as two separate ops. PyTorch's C++ CE is highly optimized; the extra `(B,T,V)` saved-for-backward tensor costs ~7% throughput at d6/M4 but no other measurable problem. This is the right default in pure PyTorch.
  - **Fused — Python autograd** (opt-in via `ZLOSS_FUSED=1`) — a custom `torch.autograd.Function` (`_FusedCEZLoss`) that computes CE + z-term from one `log_softmax` and saves only `log_probs` for backward. **Numerically correct** (bit-exact loss + ~2e-8 grad agreement, verified by smoke check `(d)`) and saves ~50% backward memory, but in pure PyTorch it replaces optimized C++ CE with Python ops — measured ~32% slower than baseline at d6/M4. Kept as a numerical reference / for memory-constrained scenarios; a real throughput win requires a CUDA/Triton fused kernel (Liger, cut-cross-entropy), which is out of scope here.
- [`../../wrappers/train_zloss.py`](../../wrappers/train_zloss.py) — patches `nanochat.gpt.GPT` then `runpy.run_module("scripts.base_train")`. Does **not** patch `GPTConfig`. Six lines.
- [`../../wrappers/smoke_zloss.py`](../../wrappers/smoke_zloss.py) — verifies (a) forward math, (b) config-portability (no overlay-only fields leak into the saved checkpoint), (c) monkeypatch, (d) fused-vs-naive numerical equivalence (loss bit-exact, gradients within ~2e-8). Passes in <5 sec with no data.

Run the smoke any time: `uv run python -m wrappers.smoke_zloss`.

Next milestone — real-data validation:

- Cheap: `OVERLAY=zloss bash runs/runcpu.sh` (~50 min on M4); compare `val_bpb` vs. a baseline run on the same nanochat commit + seed.
- Real: `OVERLAY=zloss bash runs/speedrun.sh` on a GPU node; primary metrics are CORE and `val_bpb`, plus training-loss spike frequency/max (stability is half the point of z-loss).

For A/B-ing the fused vs naive implementations (to confirm the memory/throughput observations below):

```bash
# default = fused
WANDB_RUN=d6_zloss_fused MODEL_TAG=d6_zloss_fused OVERLAY=zloss bash runs/runcpu.sh

# opt out: naive
ZLOSS_FUSED=0 WANDB_RUN=d6_zloss_naive MODEL_TAG=d6_zloss_naive OVERLAY=zloss bash runs/runcpu.sh

uv run python -m tools.compare_runs d6_baseline d6_zloss_naive d6_zloss_fused
```

## Goal

Add the PaLM-style auxiliary **z-loss** to stabilize training and gain a small quality improvement, without changing model scale or training data.

## What it is

z-loss penalizes the magnitude of the softmax **log-partition function** (the logsumexp of the logits):

```
L_z = z_coeff * mean( logsumexp(logits, dim=-1) ** 2 )
```

It pushes the partition function `Z` toward 1 (`log Z` toward 0), which keeps logits well-scaled, reduces loss spikes, and tends to give a small `val_bpb` improvement. PaLM used `z_coeff = 1e-4`.

## Why it fits the overlay

Pure loss change inside `GPT.forward()` — same patch + `runpy` overlay as MTP (see `../mtp/README.md`), **zero harness edits**.

## Design

- In `forward()`, after logits are computed — they are already fp32 in nanochat (`gpt.py:471`) — compute `lse = torch.logsumexp(logits, dim=-1)` over the vocab dim, restricted to non-ignored target positions, and add `z_coeff * (lse ** 2).mean()` to the cross-entropy loss.
- `z_coeff` default `1e-4`; read from env var `Z_LOSS_COEFF` at `__init__` and stored as a plain Python attribute on the model (**not** a config field — see implementation note below).

### Implementation note — attribute, not config field

The z-loss coefficient is stored on the model as an instance attribute, deliberately NOT added as a field on a `GPTConfig` subclass. Why: `scripts/base_train.py` saves the model config to `meta_*.json` via `asdict(model.config)`, and any downstream tool that loads the checkpoint with vanilla `GPTConfig(**kwargs)` (`scripts.base_eval`, `scripts.chat_sft`, `scripts.chat_cli`) then crashes with `TypeError: GPTConfig.__init__() got an unexpected keyword argument 'z_loss_coeff'`. Keeping the coefficient off the config means the saved checkpoint is byte-identical to a baseline run and fully portable to upstream tooling, with zero overlay machinery needed for eval / SFT / chat. Trade-off: the coefficient isn't recorded in the checkpoint — capture it via `MODEL_TAG` (e.g. `d24_zloss_1e4`) and wandb. The smoke test (`wrappers/smoke_zloss.py` check `(b)`) verifies this invariant on every run.

## Interaction with the existing logit softcap — important

nanochat already squashes logits with a tanh **softcap** (`softcap = 15`, `gpt.py:472`). Softcap **hard-bounds** each logit to `[-15, 15]`; z-loss instead **softly penalizes** the aggregate partition function. Same intent (logit control), different mechanism — they can coexist.

- Default: run z-loss **with softcap on**.
- Worthwhile ablation: z-loss with the softcap relaxed/removed — z-loss may make the hard cap unnecessary, and the cap distorts the output distribution.

## Overlay implementation

Subclass `GPT`, override `forward()`. If combined with MTP, fold it into the `MTPGPT` subclass — apply z-loss to the `t+1` head (optionally the aux heads too). Same `overlay/` + `wrappers/` setup as the MTP plan.

## Expected cost

**Compute**: minimal additional FLOPs — a `logsumexp` per row + a scalar multiply.

**Measured at d6/5000 iters on an M4 mini** (`train/tok_per_sec`):

| run | tok/sec | vs baseline |
|---|---:|---:|
| baseline (no overlay) | 27,628 | — |
| **naive** (default; `F.cross_entropy` + `torch.logsumexp`) | 25,579 | **−7.4%** |
| fused — Python autograd (`ZLOSS_FUSED=1`) | 18,738 | **−32.2%** |

Naive is the right default: PyTorch's C++ `F.cross_entropy` is highly optimized, and the extra `(B,T,V)` saved-for-backward tensor from `logsumexp` adds memory pressure but not catastrophic throughput cost. The Python fused path is **numerically correct** (bit-exact loss + ~2e-8 grad agreement vs naive; smoke `(d)` enforces this) and halves backward-time memory, but in pure PyTorch it replaces optimized C++ ops with Python and pays per-op dispatch + MPS sync overhead. **A true throughput win requires a CUDA/Triton fused kernel** (Liger, cut-cross-entropy) — those exist on GPU but not on CPU/MPS. At GPT-2 / GPU scale the naive path's overhead drops further (trunk dominates) and the memory pattern matters less (no swap pressure with 80+ GB HBM), so naive should remain a good default at scale until/unless we wire in a real fused kernel.

**Inference cost**: unchanged — no z-loss on the inference path.

## Testing / A-B

- Same pinned nanochat commit + same seed as baseline.
- Metrics: `val_bpb`, CORE — and watch **training-loss spike frequency / max**; stability is half the point.
- Smoke-test the wiring first with the repo's tiny config (`--depth=4 --num-iterations=20`, see `base_train.py` docstring).

## Open questions

- `z_coeff` value — `1e-4` default; sweep `1e-5 … 1e-3`.
- Keep the softcap, or relax it once z-loss is in.
- If combined with MTP: apply z-loss to the aux heads or only the `t+1` head.

## References

- Chowdhery et al. 2022 — PaLM; introduces the auxiliary z-loss (`1e-4 * log² Z`) for training stability.
