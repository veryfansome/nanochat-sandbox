# nanochat experiment ideas

Curated catalog of experiments. Each entry has its own folder with a design doc; this index summarizes what each one is and how it fits the overlay model. For current implementation status and what's in flight, see [`../STATUS.md`](../STATUS.md).

## Ideas

| Idea | Targets | Overlay fit |
|------|---------|-------------|
| [MTP](mtp/README.md) — multi-token prediction | capability + sample efficiency | pure model-side (forward/loss) |
| [z-loss](zloss/README.md) | training stability + small quality gain | pure model-side (forward/loss) |
| [Deep supervision](deep-supervision/README.md) — intermediate-layer aux losses | capability (depth-axis signal density) | pure model-side (forward/loss) |
| [Differential attention](diff-attention/README.md) | capability + scaling | architecture subclass |
| [Online data selection + batch-size tuning](online-data-selection/README.md) | sample efficiency | config-only (A) / harness change (B) |
| [Layer-wise LR / staged maturation](layerwise-lr/README.md) | depth-staged plasticity (brain-inspired) | overlay (`setup_optimizer`) / harness change for staged freezing |
| [Non-backprop LLM](non-backprop/README.md) — DFA → block-local | non-backprop credit assignment; memory / parallelism / forgetting | model subclass + harness change (separate research track) |

## Overlay fit — what the categories mean

- **pure model-side** — only changes the loss/`forward` of `GPT`. Rides the patch + `runpy` wrapper with **zero harness edits**. (MTP, z-loss, deep supervision)
- **architecture subclass** — subclass an attention/MLP module; patch the class before `runpy`. Still zero harness edits. (Differential attention)
- **config-only** — a CLI arg on `base_train.py`; no code at all. (Batch-size tuning)
- **harness change** — restructures the training loop; needs a **copied** `base_train.py` diffed against upstream periodically. (Online data selection, non-backprop, layer-wise-LR variant B)

## Lessons learned (ideation)

Things we've discovered through actual experiments that should shape how new experiments are designed.

- **`runs/runcpu.sh` is for mechanism verification, not scientific A/B.** d6 / 5000 iters / single seed can't reliably distinguish effects smaller than ~0.05 bpb from noise — within the band where most regularizer / objective-tweak techniques operate. CORE is disabled in runcpu (too expensive on CPU). Plan to validate real capability claims on Lambda speedruns; use runcpu only to confirm the overlay survives a real training loop. See [`zloss/README.md`](zloss/README.md) "Current state".

For *implementation* lessons (gotchas in writing the overlay subclass / forward), see [`../overlay/README.md`](../overlay/README.md). Skim those before designing a new overlay — some of them constrain what's feasible.

## Sequencing principle

Cheap things first (config + pure-loss overlays), then the main capability lever (MTP) + its depth-axis complement (deep supervision) folded into one subclass, then architecture A/B (differential attention), then harness-changing items. The non-backprop track is independent (different goal: memory / parallelism / forgetting, not the speedrun). See [`../STATUS.md`](../STATUS.md) for the explicit plan and current state.

## Considered, not adopted

- **MoE** — the largest architectural capability lever, but it grows total parameter count (gray area vs. the "no scale" constraint). Revisit only if param growth is acceptable.
- **Checkpoint weight averaging** — nearly-free post-hoc gain; not yet documented as its own idea.

## Ruled out (explicitly out of scope)

- **Knowledge distillation**, **RLVR**, **test-time compute** — removed from scope by decision.
- **Optimizer swaps** (Muon is already frontier here), attention/norm micro-tweaks, **MLA** (efficiency, not capability), **µP**, and regularization/dropout (not in an overfitting regime) — low ROI for this goal.
