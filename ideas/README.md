# nanochat experiment ideas

Curated index of experiments aimed at improving the **capability** and **sample efficiency** of the `nanochat` model — under the constraint that **model scale and the training dataset are fixed**. Gains must come from the training objective, architecture, or training dynamics.

**Overlay philosophy:** the `nanochat` repo (Karpathy) is kept **pristine** — never edited — so upstream improvements can be `git pull`-ed at any time. All experiment code lives alongside these design docs, with nanochat loaded via a sys.path bootstrap (no install). The full overlay mechanism (sys.path bootstrap + subclassing + monkeypatch + `runpy` wrapper) — plus first-time setup, wrapper template, distributed launch, and an end-to-end z-loss example — is documented in [`../SETUP.md`](../SETUP.md).

## Ideas

| Idea | Targets | Overlay fit | Status |
|------|---------|-------------|--------|
| [MTP](mtp/README.md) — multi-token prediction | capability + sample efficiency | pure model-side (forward/loss) | proposed |
| [z-loss](zloss/README.md) | training stability + small quality gain | pure model-side (forward/loss) | **implemented** (smoke ✓; real-data validation pending) |
| [Deep supervision](deep-supervision/README.md) — intermediate-layer aux losses | capability (depth-axis signal density) | pure model-side (forward/loss) | proposed |
| [Differential attention](diff-attention/README.md) | capability + scaling | architecture subclass | proposed |
| [Online data selection + batch-size tuning](online-data-selection/README.md) | sample efficiency | config-only (A) / harness change (B) | proposed |
| [Layer-wise LR / staged maturation](layerwise-lr/README.md) | depth-staged plasticity (brain-inspired) | overlay (`setup_optimizer`) / harness change for staged freezing | proposed |
| [Non-backprop LLM](non-backprop/README.md) — DFA → block-local | non-backprop credit assignment; memory / parallelism / forgetting | model subclass + harness change | proposed (research track) |

## Overlay fit — what the categories mean

- **pure model-side** — only changes the loss/`forward` of `GPT`. Rides the patch + `runpy` wrapper with **zero harness edits**. (MTP, z-loss, deep supervision)
- **architecture subclass** — subclass an attention/MLP module; patch the class before `runpy`. Still zero harness edits. (Differential attention)
- **config-only** — a CLI arg on `base_train.py`; no code at all. (Batch-size tuning)
- **harness change** — restructures the training loop; needs a **copied** `base_train.py` that is diffed against upstream periodically. (Online data selection)

## Suggested sequencing

0. **Currently in flight — validate z-loss on real data.** Overlay + wrapper + smoke are built and pass locally; the next milestone is `OVERLAY=zloss bash runs/runcpu.sh` (cheap, ~30 min M3 Max) or `OVERLAY=zloss bash runs/speedrun.sh` (GPU box, run from the sandbox/ root) to confirm `val_bpb` / CORE move as expected vs. baseline.
1. **Cheapest first** — batch-size A/B (config-only), z-loss (done; see above), and layer-wise LR (overlay, ~0 cost). Quick signal, no real engineering.
2. **MTP** — the main capability + sample-efficiency lever. Next big build per the plan.
3. **Deep supervision** — depth-axis complement to MTP; fold into the same subclass for one combined experiment.
4. **Differential attention** — independent architecture A/B.
5. **Online data selection** — bigger, harness-touching; pursue if the earlier results justify it.

Always A/B against baseline on the **same pinned nanochat commit and seed** — otherwise upstream changes confound the comparison.

The [non-backprop LLM](non-backprop/README.md) is a **separate research track**, not part of this capability-tuning sequence. It is not expected to win the speedrun — its payoff is memory, parallelism, and forgetting-resistance. Pursue it independently of the items above.

## Considered, not adopted

- **MoE** — the largest architectural capability lever, but it grows total parameter count (gray area vs. the "no scale" constraint). Revisit only if param growth is acceptable.
- **Checkpoint weight averaging** — nearly-free post-hoc gain; not yet documented as its own idea.

## Ruled out (explicitly out of scope)

- **Knowledge distillation**, **RLVR**, **test-time compute** — removed from scope by decision.
- **Optimizer swaps** (Muon is already frontier here), attention/norm micro-tweaks, **MLA** (efficiency, not capability), **µP**, and regularization/dropout (not in an overfitting regime) — low ROI for this goal.
