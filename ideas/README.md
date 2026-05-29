# nanochat experiment ideas

Curated catalog of experiments. Each entry has its own folder with a design doc; this index summarizes what each one is and how it fits the overlay model. For current implementation status and what's in flight, see [`../STATUS.md`](../STATUS.md).

## Ideas

| Idea | Targets | Overlay fit |
|------|---------|-------------|
| [MTP](mtp/README.md) — multi-token prediction | capability + sample efficiency | pure model-side (forward/loss) |
| [z-loss](zloss/README.md) | training stability + small quality gain | pure model-side (forward/loss) |
| [Deep supervision](deep-supervision/README.md) — intermediate-layer aux losses | capability (depth-axis signal density) | pure model-side (forward/loss) |
| [Token-level loss weighting](token-loss-weighting/README.md) — entropy / focal / RHO-Loss | sample efficiency (reshape per-token signal) | pure model-side (forward/loss) |
| [Differential attention](diff-attention/README.md) | capability + scaling | architecture subclass |
| [Online data selection + batch-size tuning](online-data-selection/README.md) | sample efficiency | config-only (A) / harness change (B) |
| [Adaptive sequence length / batch size](adaptive-schedule/README.md) — schedule what's currently fixed | compute efficiency (A) + sample efficiency (B) | harness change |
| [Layer-wise LR / staged maturation](layerwise-lr/README.md) | depth-staged plasticity (brain-inspired) | overlay (`setup_optimizer`) / harness change for staged freezing |
| [Tokenizer variants](tokenizer-variants/README.md) — merge-producer experiments (regex / seed tokens / forced phrases / corpus / vocab) | capability ceiling (representational atoms); sample efficiency (forced common-phrase tokens) | preprocessing artifact (parallel track, no GPU until A/B). **Two live variants: `force_merges` adopted (CORE +5.81%) + `auto_tune` (active block-list auto-tuning). See [STATUS.md](../STATUS.md).** |
| [BPE-dropout](bpe-dropout/README.md) — stochastic segmentation at pretrain time | sample efficiency (train undertrained subword atoms) + tokenization robustness | encode-time overlay (monkeypatch `tokenizer.encode`, train-split-gated; no copied harness) |
| [Non-backprop LLM](non-backprop/README.md) — DFA → block-local | non-backprop credit assignment; memory / parallelism / forgetting | model subclass + harness change (separate research track) |

## Overlay fit — what the categories mean

- **pure model-side** — only changes the loss/`forward` of `GPT`. Rides the patch + `runpy` wrapper with **zero harness edits**. (MTP, z-loss, deep supervision, token-level loss weighting)
- **architecture subclass** — subclass an attention/MLP module; patch the class before `runpy`. Still zero harness edits. (Differential attention)
- **config-only** — a CLI arg on `base_train.py`; no code at all. (Batch-size tuning)
- **harness change** — restructures the training loop; needs a **copied** `base_train.py` diffed against upstream periodically. (Online data selection, adaptive sequence length / batch size, non-backprop, layer-wise-LR variant B)
- **preprocessing artifact** — produces a non-model artifact (e.g. a trained tokenizer) that the existing pipeline consumes via its cache. No nanochat edits, no harness edits. Iteration happens offline; GPU is paid only for the final A/B. (Tokenizer variants)
- **encode-time overlay** — monkeypatches `tokenizer.encode` (and gates train vs val/eval) before `runpy`, exploiting that nanochat tokenizes raw-text parquet on-the-fly in the dataloader. No copied `base_train.py`; the one non-`GPT` patch is the train-split gate. (BPE-dropout)

## Lessons learned (ideation)

Things we've discovered through actual experiments that should shape how new experiments are designed.

- **`runs/runcpu.sh` is for mechanism verification, not scientific A/B.** d6 / 5000 iters / single seed can't reliably distinguish effects smaller than ~0.05 bpb from noise — within the band where most regularizer / objective-tweak techniques operate. CORE is disabled in runcpu (too expensive on CPU). Plan to validate real capability claims on Lambda speedruns; use runcpu only to confirm the overlay survives a real training loop. See [`zloss/README.md`](zloss/README.md) "Current state".

For *implementation* lessons (gotchas in writing the overlay subclass / forward), see [`../overlay/README.md`](../overlay/README.md). Skim those before designing a new overlay — some of them constrain what's feasible.

## Sequencing principle

Cheap things first (config + pure-loss overlays — z-loss, token-level loss weighting [entropy/focal variants]), then the main capability lever (MTP) + its depth-axis complement (deep supervision) folded into one subclass — token weighting's RHO-Loss variant folds in here too if signal warrants — then architecture A/B (differential attention), then harness-changing items (online data selection, adaptive sequence length / batch size — both share the copied-`base_train.py` pattern, so fold them into one fork when both land). The non-backprop track is independent (different goal: memory / parallelism / forgetting, not the speedrun). **Tokenizer variants** run in parallel with the model-overlay sequence — offline iteration doesn't compete with GPU time, only the final variant's speedrun A/B does, so it can progress whenever model-side work is GPU-bound. See [`../STATUS.md`](../STATUS.md) for the explicit plan and current state.

## Considered, not adopted

- **MoE** — the largest architectural capability lever, but it grows total parameter count (gray area vs. the "no scale" constraint). Revisit only if param growth is acceptable.
- **Checkpoint weight averaging** — nearly-free post-hoc gain; not yet documented as its own idea.

## Ruled out (explicitly out of scope)

- **Knowledge distillation**, **RLVR**, **test-time compute** — removed from scope by decision.
- **Optimizer swaps** (Muon is already frontier here), attention/norm micro-tweaks, **MLA** (efficiency, not capability), **µP**, and **weight/activation dropout** (not in an overfitting regime) — low ROI for this goal. Note: this rules out *overfitting-control* dropout; it does **not** rule out [BPE-dropout](bpe-dropout/README.md), which is segmentation-space augmentation targeting undertrained subword atoms / robustness, not a train/val gap.
