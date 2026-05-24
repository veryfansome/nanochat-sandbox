# Adaptive sequence length & batch size — proposed setup

Status: **proposal / not yet implemented**
Target repo: `nanochat` (Karpathy) — kept **pristine**, never edited.

## Goal

Two of nanochat's biggest "fixed" training hyperparameters — sequence length (`--sequence-len`, default 2048) and total batch size (`--total-batch-size`) — have well-established reasons to *vary during training*. The speedrun pins both at their final values for the whole run, which is provably suboptimal in two ways:

- **Sequence length**: attention is O(T²). Training the first 25% of steps at T=512 instead of T=2048 saves ~94% of attention FLOPs on those steps. The model already learns local structure first, so the long-context budget is wasted early.
- **Batch size**: per McCandlish 2018, the *critical batch size* (above which data-parallel scaling breaks down) **grows** as loss decreases. A fixed batch size is either too large early (wasting data per step) or too small late (wasting compute per step). Schedule it upward to track the critical batch.

Both are about **scheduling what is currently fixed** — same architecture, same dataset, same total tokens; just allocated differently across the run.

## Brain-inspired motivation

Maps to **developmental staging** — brains don't receive full sensory richness from day one. Visual acuity, working-memory span, and attentional capacity all develop over months. The wetware capacity is built in, but the *bandwidth of experience* ramps up. Sequence-length warmup is the direct analog (shorter "context window" early); batch scheduling is a looser analog to maturation of consolidation pacing.

---

## Part A — Sequence length warmup

### What it is

Start training at short sequence length (e.g., 512), grow to full (2048) over training. Simplest schedule: piecewise (512 → 1024 → 2048 at 25% / 50% boundaries). Cleaner alternative: continuous ramp tied to step count.

### Why it fits the project

Pure compute-efficiency win in the early run. The model itself is already shape-agnostic — `GPT.forward()` reads `T = idx.size()` dynamically (`gpt.py:417`) and rotary embeddings are pre-computed 10× over (`gpt.py:195`). So **the model needs zero changes**. The schedule lives in the dataloader / harness.

### Implementation

- **Dataloader**: re-instantiate `tokenizing_distributed_data_loader_*` (`nanochat/dataloader.py`) at each stage transition with the new `sequence_len`.
- **Window attention (`window_pattern`, `gpt.py:39`)**: nanochat's short-window layers compute their window as `sequence_len / 4` (`gpt.py:300`). At T=512 with the default `SSSL` pattern, short windows would be 128 — which is fine, but means each stage transition reshapes attention windows. Two options:
  - **Simple**: fix windows to the final-stage values; at short stages, all layers are effectively full-context (windows ≥ T). One-line override of `_compute_window_sizes`.
  - **Faithful**: recompute windows per stage. Requires reconstructing the model's `window_sizes` attribute at each transition. More code, marginal benefit.
  - Recommend simple.
- **LR**: keep schedule unchanged; sequence length doesn't change gradient scale meaningfully.

### Cost / win

- Net **FLOP saving**, biggest at early steps. Win is capped by attention's share of total FLOPs:
  - At d6 (small `n_embd`): attention is a meaningful slice; Part A win is large.
  - At d24+ (speedrun scale): MLP dominates; Part A win is real but smaller (single-digit % wall-clock).
- Risk: loss spike at stage transitions. Mitigations: smooth ramp instead of piecewise; brief LR warmup after each transition.

---

## Part B — Adaptive batch size

### What it is

Schedule `total_batch_size` upward during training. McCandlish 2018 gives the framework: gradient noise scale determines the critical batch size, and that scale shrinks as loss falls — so the critical batch grows. Track it.

Simple schedules:
- **Multiplicative steps** — 2× every 25% of training.
- **Continuous** — `B(t) = B_0 * (1 + α·t)` or grow proportional to inverse loss.
- **Empirical** — measure gradient noise scale online; grow batch when noise drops below a threshold.

### Why it needs a harness change

`--total-batch-size` is set once at script start in `scripts/base_train.py` and used to derive grad-accumulation steps. Changing mid-run requires re-deriving accumulation and re-tuning LR per the chosen scaling rule (sqrt or linear).

### Implementation

Copy `base_train.py` into `wrappers/` and add:
- A schedule function: `total_batch_size(step) → int`.
- Re-derivation of grad-accumulation steps when the batch changes.
- LR adjustment per scaling rule. Recommend **sqrt scaling** (`lr ∝ √B`) as the conservative default; linear is more aggressive but more spike-prone.

Diff against upstream periodically per the project's harness-change pattern (same as online-data-selection Part B).

### Cost / win

- **Not** a FLOP saving — total tokens consumed is unchanged.
- Win is in **loss-per-token** (sample efficiency) by avoiding above-critical-batch waste early in training. Per McCandlish, the gap between "fixed at speedrun's chosen batch" and "schedule tracking critical batch" can be meaningful (10–30% sample-efficiency improvement reported in some regimes), and is larger when the speedrun's chosen batch is well above the early-training critical batch.
- Risk: stage transitions introduce loss spikes. Continuous schedules are smoother than piecewise.

---

## Relationship to existing ideas

- **`online-data-selection` Part A** (static batch-size tuning, [`../online-data-selection/README.md`](../online-data-selection/README.md)) — finds the right *fixed* batch size. This proposal is the *scheduled* version. Order: do the static sweep first to find the knee, then schedule around it.
- **`online-data-selection` Part B** (online data selection) — also harness-touching, same copied-`base_train.py` pattern. Probably worth folding both into a single forked harness when both land.
- **`layerwise-lr`** ([`../layerwise-lr/README.md`](../layerwise-lr/README.md)) — same family ("schedule something the speedrun currently fixes"). Layer-wise LR varies *across parameters*; adaptive schedules vary *across time*. Orthogonal.
- **MTP / deep supervision** — both add per-token aux losses, which change the gradient noise scale and thus shift the critical batch size. If both are in flight, the optimal schedule for Part B should be re-measured rather than carried over from baseline.

## Why this is worth doing here

Both parts target the cheapest possible kind of win — **better allocation of the existing budget**, no new parameters, no new data, no new losses. The speedrun's fixed values are *correct on average* for the run as a whole, but average-optimal is per-step-suboptimal at both ends. Adaptive scheduling captures the gap.

Part A also has a unique property: it's one of the few ideas that can make the speedrun **faster on wall-clock** at the same final loss (rather than slower in exchange for better loss). That's directly aligned with the speedrun's native metric.

## Testing / A-B

- Same pinned nanochat commit + same seed as baseline.
- Metrics: `val/bpb` / CORE vs **tokens trained on** (sample efficiency, primary for Part B) **and** vs **wall-clock** (primary for Part A).
- Smoke-test wiring: verify model accepts T < `sequence_len` cleanly through one stage transition; verify grad-accumulation re-derives correctly when batch size changes mid-run.
- Per the project's `ideas/README.md` "Lessons learned" — d6/runcpu is mechanism verification only; real wins on Lambda d24.
- Recommend Part A first (smaller scope, model needs no change), then Part B.

## Open questions

- **Part A schedule shape** — piecewise (simple) vs. continuous ramp (smoother). Start point and final-stage onset.
- **Part B schedule basis** — step count vs. loss-conditioned vs. measured gradient noise scale. Multiplicative vs. continuous.
- **LR scaling rule for Part B** — sqrt (safer) vs. linear (more aggressive).
- **Window-attention handling for Part A** — simple (final-stage windows fixed) vs. faithful (recompute per stage).
- **Interaction with sliding window pattern** — if the project later changes `window_pattern`, the schedule may need to coordinate.
- Whether to schedule both jointly or independently (suspect roughly independent at first order; verify with an A/B/AB grid).

## References

- McCandlish et al. 2018 — "An Empirical Model of Large-Batch Training" (critical batch size, gradient noise scale).
- Press et al. 2021 — "Shortformer" (variable sequence length training shows substantial speedup).
- Bengio et al. 2009 — "Curriculum Learning" (the general framing of training-time scheduling of difficulty).
- Smith et al. 2018 — "Don't Decay the Learning Rate, Increase the Batch Size" (related batch-size-as-schedule line).
