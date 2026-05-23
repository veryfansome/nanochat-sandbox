# z-loss

Status: **implemented** — overlay + wrapper + smoke are checked in; real-data validation pending.
Target repo: `nanochat` (Karpathy) — kept **pristine**, never edited.

## Current state

The mechanism is live and smoke-verified:

- [`../../overlay/zloss.py`](../../overlay/zloss.py) — `ZLossGPT(GPT)` + `ZLossConfig(GPTConfig)`. Forward returns `baseline_CE + z_coeff · mean(logsumexp(logits)²)` over non-ignored target positions.
- [`../../wrappers/train_zloss.py`](../../wrappers/train_zloss.py) — patches `nanochat.gpt.GPT` / `GPTConfig` then `runpy.run_module("scripts.base_train")`. Six lines.
- [`../../wrappers/smoke_zloss.py`](../../wrappers/smoke_zloss.py) — verifies (a) the forward math (diff matches the z-term to ~1e-7), (b) the monkeypatch substitution. Passes in <5 sec with no data.

Run the smoke any time: `uv run python -m wrappers.smoke_zloss`.

Next milestone — real-data validation:

- Cheap: `OVERLAY=zloss bash runs/runcpu.sh` (~30 min on M3 Max); compare `val_bpb` vs. a baseline run on the same nanochat commit + seed.
- Real: `OVERLAY=zloss bash runs/speedrun.sh` on a GPU node; primary metrics are CORE and `val_bpb`, plus training-loss spike frequency/max (stability is half the point of z-loss).

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
- `z_coeff` default `1e-4`; expose via env var `Z_LOSS_COEFF`.

## Interaction with the existing logit softcap — important

nanochat already squashes logits with a tanh **softcap** (`softcap = 15`, `gpt.py:472`). Softcap **hard-bounds** each logit to `[-15, 15]`; z-loss instead **softly penalizes** the aggregate partition function. Same intent (logit control), different mechanism — they can coexist.

- Default: run z-loss **with softcap on**.
- Worthwhile ablation: z-loss with the softcap relaxed/removed — z-loss may make the hard cap unnecessary, and the cap distorts the output distribution.

## Overlay implementation

Subclass `GPT`, override `forward()`. If combined with MTP, fold it into the `MTPGPT` subclass — apply z-loss to the `t+1` head (optionally the aux heads too). Same `overlay/` + `wrappers/` setup as the MTP plan.

## Expected cost

Negligible — one extra `logsumexp` over logits that are already materialized. ~0% training-time impact. (With fused CE, compute the `lse` inside the fused kernel; with naive CE the logits are already in hand.)

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
