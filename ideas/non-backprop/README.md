# Non-backprop LLM (DFA → block-local transformer) — proposed setup

Status: **proposal / research track / not yet implemented**
Target repo: `nanochat` (Karpathy) — kept **pristine**, never edited.

## Goal

Train a working autoregressive LLM whose **credit assignment is not backpropagation**, and evaluate it with a nanochat-style speedrun (wall-clock to GPT-2 CORE), reusing the same tokenizer, dataloader, and evals.

## Honest framing — read this first

The motivating argument ("the brain doesn't do backprop and is more sample efficient") does not survive contact with the evidence:

- The brain's sample efficiency comes mostly from evolutionary priors, multimodal grounding, and active/curricular learning — not from the learning rule.
- **Every known non-backprop rule today is *less* sample-efficient than backprop**, not more.

So the realistic expectation: this model **loses the time-to-GPT-2 race**. Do not pitch this experiment on sample efficiency. Its real payoff is the set of properties backprop structurally *cannot* offer:

- much lower activation memory,
- pipeline parallelism with no backward-pass dependency,
- local / online weight updates,
- reduced cross-layer interference → better resistance to catastrophic forgetting.

Measure and report *those*.

## Ruled out for a speedrun

A speedrun is a **wall-clock** race; per-step cost must be ≈ backprop or cheaper.

- **Predictive coding, Equilibrium Propagation** — the `T×` relaxation inner loop destroys wall-clock (10–100× per step). Dead on arrival.
- **Zeroth-order / evolution strategies** (MeZO-style) — viable for fine-tuning, hopeless for pretraining from scratch (gradient-estimate variance scales with parameter count).
- **Forward-Forward / pure Hebbian** — FF does not adapt cleanly to autoregressive sequence modeling and has not scaled past toy tasks.

## Design — DFA → block-local transformer

Keep nanochat's architecture **verbatim** (RoPE, RMSNorm, attention, ReLU² MLP, x0 residual, value embeddings). Change **only the credit-assignment rule**, in two stages.

### Stage 1 — Direct Feedback Alignment (low-risk v1)

Replace the backward pass entirely. Instead of propagating error through transposed weights via the chain rule, project the output-layer error **directly to every block** through *fixed random feedback matrices* — one cheap matmul per block, off the output. No sequential backward, no chain rule.

- Most *demonstrated* non-backprop method for transformers (Launay et al. 2020).
- Architecture unchanged; per-step compute ≈ backprop; the random projections are trivially parallel across blocks.
- Smallest viable change → the right **first speedrun entry**.
- Note: DFA still uses a *global* error signal (it just isn't propagated through the chain). It is "non-backprop" but not yet "fully local."

### Stage 2 — add block-local objectives (the interesting version)

`stop-gradient` between transformer blocks, and give each block its own lightweight local head with a **local next-token predictive loss** (decoupled-greedy / Greedy-InfoMax style). Credit assignment becomes **fully local** — no error crosses a block boundary. Keep the DFA projection as a weak auxiliary "what the output needs" hint so blocks do not drift into purely greedy solutions.

Where the non-backprop wins show up:

- **Activation memory collapses** — no global autograd graph; free each block's activations as soon as its local loss is computed → bigger batch / no recompute.
- **Blocks decouple** — train asynchronously, pipeline-parallel with **no backward bubbles**. Backprop cannot do this.
- **Localized weight updates** — plausibly reduces catastrophic interference under sequential domain tuning.

Inference is unchanged: standard forward pass; drop the local heads (keep the final one).

## How it fits the harness

- **Model**: a subclass of `GPT` — per-block local heads, `.detach()` between blocks, fixed random feedback buffers. Overlay-style (see `../mtp/README.md`).
- **Training loop**: the real change — no single `loss.backward()`; instead per-block local losses + DFA projection. Needs a **copied `base_train.py`** (same pattern as `../online-data-selection/`), diffed against upstream periodically.
- Implement *with* PyTorch autograd — autograd *within* a block, `stop-gradient` at boundaries. Still "not backprop": no error crosses the network.
- Reuses the tokenizer, dataloader, and CORE / `val_bpb` evals unchanged.

## Expected cost / result

- Per-step compute ≈ backprop (Stage 1); lower activation memory (Stage 2).
- **Capability ceiling is expected to be lower** than the backprop baseline — DFA's noisier credit assignment degrades with depth; block-local has no cross-depth coordination. It **may plateau short of CORE 0.2565 entirely**, in which case it does not finish the speedrun slowly — it cannot finish it at all. The ceiling's position relative to the target is unknown at this scale; this is the headline risk.
- **If** the ceiling does clear the target, expect it to still **lose the time-to-GPT-2 race**: worse sample efficiency means more tokens to reach any target, even at comparable per-step cost. Slower-but-finishes is the optimistic case.

## Testing / A-B

- Same pinned nanochat commit + seed as baseline.
- Run the standard **time-to-CORE 0.2565** for comparability.
- Also report what backprop structurally cannot do — the actual story:
  - peak activation memory,
  - pipeline-parallel scaling efficiency,
  - retention under sequential domain tuning (catastrophic-forgetting probe).
- Smoke-test wiring with the tiny config (`--depth=4 --num-iterations=20`).

## Interactions with the other ideas

Non-backprop changes the **learning rule**; the other ideas change the objective, the architecture, and the data — orthogonal axes, so they compose technically. The real constraints are experimental hygiene and a few mechanism-level interactions.

- **[z-loss](../zloss/README.md) — clean combine, mild synergy.** Just an output-side loss term; rides along under DFA. Under Stage 2 it becomes *more* useful — apply it to **every local head**, since block-local training is less stable and each head's logits can drift. Cheap per-head stabilizer.
- **[Batch-size tuning](../online-data-selection/README.md) — clean combine, real synergy.** Learning-rule-agnostic config. Stage 2's activation memory collapses → much more headroom to push batch size; and larger batches damp DFA's random-feedback gradient noise. Widen the batch-size sweep once non-backprop runs.
- **[Online data selection](../online-data-selection/README.md) — composable, complementary, but sequence it.** Both need a copied `base_train.py`, so they share one modified training loop. Score by the **final-block head's loss** (Stage 2 has no single global loss). Targets sample efficiency — exactly where non-backprop is weak — so it is a natural offset. Get non-backprop working alone first, then layer this on; do not build both at once.
- **[Differential attention](../diff-attention/README.md) — composable, but the riskiest pairing.** Its tuned machinery (learned λ re-parameterization, headwise norm, the two-map cancellation) was developed under backprop and may not learn cleanly under DFA's noisier signal or block-local's greedy loss. Keep it a **separate experiment** — never folded into the learning-rule change.

**Do not confound.** Non-backprop is expected to underperform the baseline; if architecture/objective/data changes are stacked on top, a worse result cannot be attributed. Sequence:

1. Validate **non-backprop alone** vs. the backprop baseline.
2. Add the safe pair — **z-loss + batch-size tuning** (they aid stability/noise).
3. Add **online data selection** as a deliberate follow-on.
4. Keep **differential attention** as its own separate A/B.

## Open questions

- DFA feedback-matrix scale / init; keep fixed or slowly adapt.
- Block-local objective: bare next-token vs. contrastive/InfoMax vs. predicting multiple steps.
- Granularity: per-block vs. per-pair-of-blocks (looser locality, higher ceiling).
- How much the DFA auxiliary signal recovers vs. pure block-local.
- Whether nanochat's x0 residual (every block sees the embedding) helps local blocks stay meaningful — likely yes; worth ablating.
- Does the local rule actually reduce forgetting? (the headline secondary claim)

## References

- Lillicrap et al. 2016 — Feedback Alignment ("Random synaptic feedback weights support error backpropagation for deep learning").
- Nøkland 2016 — "Direct Feedback Alignment Provides Learning in Deep Neural Networks."
- Launay et al. 2020 — "Direct Feedback Alignment Scales to Modern Deep Learning Tasks and Architectures" (DFA applied to transformers).
- Löwe et al. 2019 — Greedy InfoMax ("Putting An End to End-to-End: Gradient-Isolated Learning of Representations").
- Belilovsky et al. 2019/2020 — "Decoupled Greedy Learning of CNNs."
- Hinton 2022 — "The Forward-Forward Algorithm" (considered, not used).
