# Layer-wise LR / staged maturation — proposed setup

Status: **proposal / not yet implemented**
Target repo: `nanochat` (Karpathy) — kept **pristine**, never edited.

## Goal

Emulate developmental staging — lower/earlier layers consolidate first, later layers stay plastic longer — by giving each layer (or layer band) its own learning-rate schedule. Tests whether the cortical critical-period analogy buys anything at LM scale.

## Brain-inspired motivation

Maps to the "staged development / critical periods" insight: lower sensory cortex matures and becomes less plastic *before* higher areas come online. This converts a simultaneous co-adaptation problem into a sequential one — lower layers settle on local features first, higher layers then learn to use them.

## Why it fits the overlay

`nanochat` already groups params by *kind* (`gpt.py:374`, `setup_optimizer` creates groups for `matrix`, `embedding`, `unembedding`, `scalar`) and the training loop sets per-group LR each step (`base_train.py:520-527`). The cheap form of this idea — static per-depth base LR — slots into that machinery as a `setup_optimizer` overlay with **no harness edits**.

## Two variants — pick one to start

### A. Static per-depth base LR *(overlay, recommended first)*

Override `setup_optimizer()` so each transformer block's matrix params live in their own group (or in `B` depth bands), each with a depth-dependent `initial_lr`. Earlier blocks get a smaller `initial_lr`; later blocks larger. The schedule `lrm` is still uniform across groups — only the *base* differs.

- Trivial to implement; pure model-side overlay (no harness change).
- Limitation: gives a per-depth *magnitude*, not per-depth *schedule shape* — all layers decay together.

### B. Per-depth schedule shape *(harness change)*

To let earlier layers decay *faster* (a sharper analog of "critical period closes"), the loop at `base_train.py:524` (`group["lr"] = group["initial_lr"] * lrm`) must compute `lrm` *per group*, not globally. That edits the loop → needs a **copied `base_train.py`** (same pattern as `../online-data-selection/`).

Defer this until Variant A shows the magnitude effect is real.

## Interaction with the existing optimizer mix

The repo uses Muon for 2D matrix params and AdamW for the rest (`optim.py`). Splitting matrix params by depth band creates more Muon groups — fine in principle, but with the distributed Muon (ZeRO-2-style sharding), check that group reorganization does not break the sharding assumption. Smoke-test on the tiny config first.

## Design choices to make

- Number of depth bands `B`: e.g., 2 (lower half / upper half), 4, or per-block.
- Depth → LR mapping: linear ramp, exponential, or step. Start with linear (lower band = `0.5 ×`, upper band = `1.0 ×` of the current `matrix_lr`).
- Whether to also stage **weight decay** (the repo already schedules `muon_weight_decay`, `base_train.py:522`).
- Whether to apply staging to embeddings/unembedding (probably not — they have their own groups with their own LRs already).

## Overlay implementation

Subclass `GPT` and override `setup_optimizer()` only. Pass band count / multipliers via env vars. Wrapper is the same `runpy` pattern as MTP. Inference and the training loop are untouched.

## Expected cost

**~0% training-time impact** — just a different optimizer config. Same FLOPs, same memory.

## Combines well with

- Everything — this is orthogonal to objective/architecture changes.
- Especially natural with the **non-backprop block-local** track (`../non-backprop/README.md`): block-local training plus a staged LR is the most direct existing-ML model of cortical critical periods.

## Testing / A-B

- Same pinned nanochat commit + same seed as baseline.
- Metrics: `val_bpb`, CORE.
- Honest expectation: layer-wise / discriminative LR has shown clear benefits in *fine-tuning* (ULMFiT) but mixed-to-marginal benefits in transformer *pretraining*. This is a cheap probe of "does the developmental analogy buy anything here?" — set expectations accordingly.

## Open questions

- The right depth → LR mapping (and whether it should be data-driven rather than hand-set).
- Whether scheduled *freezing* (Variant B) materially beats static per-depth LR (Variant A).
- Interaction with the distributed-Muon sharding (correctness check, not performance).

## References

- Howard & Ruder 2018 — "ULMFiT" (discriminative fine-tuning with layer-wise LRs).
- General brain motivation: critical periods and the staged maturation of cortical hierarchies.
