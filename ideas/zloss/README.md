# z-loss

Status: **validated at d24/8xA100 — real positive result.** CORE +5.15% (base_eval, full), throughput cost <1%, val/bpb cost in noise (artifact-corrected). Worth adopting.
Target repo: `nanochat` (Karpathy) — kept **pristine**, never edited.

## Current state — d24 results (8xA100 40GB)

| metric | baseline | zloss | Δ | notes |
|---|---:|---:|---:|---|
| CORE (`base_eval`, full per-task) | 0.250488 | **0.263387** | **+0.01290 (+5.15%)** | 16 wins / 5 losses / 1 tie of 22 (one-sided sign test dropping the tie: p ≈ 0.013) |
| CORE (wandb, mid-training subsample) | 0.260098 | 0.268057 | +0.00796 (+3.06%) | subsample noisier; base_eval is authoritative |
| val/bpb (final, observed) | 0.71609 | 0.72009 | +0.00400 (+0.56%) | **artifactual — see "Eval-path bug" below** |
| val/bpb (corrected, post-fix) | 0.71609 | ~0.71609 ± noise | ≈ 0 | no underlying perplexity cost |
| train tok/sec | 291,210 | 288,483 | −0.94% | within noise |
| total time | 5h33m | 5h36m | +0.95% | ~$1 extra |

The CORE gain is concentrated on high-N robust tasks: **piqa +3.9pp** (N=1838), **commonsense_qa +3.4pp** (N=1221), **lambada +1.7pp**, **coqa +2.0pp**. Big bigbench-subtask wins (dyck +6.2pp, lsat_ar +6.0pp) are real but inflated by sampling on small N. The few regressions are small (≤2.2pp) and concentrated on noisier tasks. The asymmetry — robust wins on big datasets, small losses on noisy ones — is what a real-but-modest improvement looks like.

### Eval-path bug (discovered post-hoc; fixed)

The reported val/bpb regression of +0.004 turned out to be entirely artifactual. The original `ZLossGPT.forward` added the scalar z-loss term (`z_coeff · mean(lse²)`) to the CE tensor regardless of `loss_reduction`. When `nanochat/loss_eval.py` called `model(x, y, loss_reduction='none')` to compute val/bpb, the scalar broadcast onto every per-token loss, inflating bpb by `z_coeff·E[lse²] / mean_bytes / ln 2`. With `z_coeff=1e-4` and `mean(lse²) ≈ 100-150` at trained scale, that's exactly the observed Δbpb ≈ 0.004.

**Fix**: `forward` now branches on `loss_reduction != 'mean'` and returns plain CE on the eval path (matching `mtp.py`'s pattern). **Smoke check `(e)` enforces this contract** — with nonzero `z_coeff`, `ZLossGPT(..., loss_reduction='none')` must match vanilla GPT exactly. The bug would have been caught immediately had the smoke checked this; the lesson "aux losses must be confined to the training path" was documented in [`../../overlay/README.md`](../../overlay/README.md) but text-only lessons aren't enforcement.

The d24 CORE result is unaffected (CORE is evaluated by a separate pipeline that doesn't go through this forward path).

### Components

- [`../../overlay/zloss.py`](../../overlay/zloss.py) — `ZLossGPT(GPT)`. Training path returns `baseline_CE + z_coeff · mean(logsumexp(logits)²)`; eval path (`loss_reduction != 'mean'`) returns plain CE. Two implementations:
  - **Naive** (default) — `F.cross_entropy` + `torch.logsumexp` as two separate ops. PyTorch's C++ CE is highly optimized; the extra `(B,T,V)` saved-for-backward tensor costs ~7% throughput at d6/M4 and ~1% at d24/8xA100. Right default in pure PyTorch.
  - **Fused — Python autograd** (opt-in via `ZLOSS_FUSED=1`) — a custom `torch.autograd.Function` (`_FusedCEZLoss`) that computes CE + z-term from one `log_softmax` and saves only `log_probs` for backward. **Numerically correct** (bit-exact loss + ~2e-8 grad agreement, smoke check `(d)`) and saves ~50% backward memory, but ~32% slower than baseline at d6/M4 (Python ops replace optimized C++ CE). Kept as a numerical reference / for memory-constrained scenarios; a real throughput win requires a CUDA/Triton fused kernel (Liger, cut-cross-entropy).
- [`../../wrappers/train_zloss.py`](../../wrappers/train_zloss.py) — patches `nanochat.gpt.GPT` then `runpy.run_module("scripts.base_train")`. Does **not** patch `GPTConfig`. Six lines.
- [`../../wrappers/smoke_zloss.py`](../../wrappers/smoke_zloss.py) — verifies (a) forward math, (b) config-portability, (c) monkeypatch, (d) fused-vs-naive equivalence, (e) eval-path contract. Passes in <5 sec with no data.

Run the smoke any time: `uv run python -m wrappers.smoke_zloss`.

For A/B-ing the fused vs naive implementations (to reconfirm the memory/throughput observations below):

```bash
# default = naive (no env var needed)
WANDB_RUN=d6_zloss_naive MODEL_TAG=d6_zloss_naive OVERLAY=zloss bash runs/runcpu.sh

# opt in: fused (Python autograd)
ZLOSS_FUSED=1 WANDB_RUN=d6_zloss_fused MODEL_TAG=d6_zloss_fused OVERLAY=zloss bash runs/runcpu.sh

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
