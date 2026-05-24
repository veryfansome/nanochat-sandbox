# Deep supervision (intermediate-layer auxiliary losses) — proposed setup

Status: **proposal / not yet implemented**
Target repo: `nanochat` (Karpathy) — kept **pristine**, never edited.

## Goal

Densify the learning signal along the **depth axis** by attaching auxiliary next-token `lm_head`s at intermediate residual-stream depths. Each aux head adds a local predictive loss to the total objective, so every layer in the trunk has a locally meaningful task. Aux heads are **dropped at inference** — deployed cost is unchanged.

This is the depth-axis complement to **MTP** (which densifies the time axis): MTP gives each token `k` prediction losses; deep supervision gives each *layer* its own prediction loss.

## Brain-inspired motivation

Maps to the "local objectives aligned with global" insight from the cortex / predictive-coding side: lower cortical regions build globally-useful features from purely local predictive objectives because the world is hierarchically structured. Deep supervision is the existing-ML analog — instead of waiting for the top loss to reach every layer through one long chain, give every layer its own local next-token target.

## Why it fits the overlay cleanly

**Pure model-side change.** Only alters the scalar loss returned by `GPT.forward()`. The training loop in `scripts/base_train.py` runs **completely unmodified** — same patch + `runpy` overlay as MTP (see `../mtp/README.md`), zero harness edits.

## Design

Standard deep supervision (Lee et al. 2015 "Deeply-Supervised Nets"; GoogLeNet's auxiliary classifiers).

- Attach `K` auxiliary `lm_head`s at intermediate depths. Simple default: every `L / (K+1)` blocks. For `L = 26`, `K = 3` → heads at layers ~6, 13, 20.
- Each aux head: `norm → d_model → vocab` projection → cross-entropy next-token loss on its own output. Optionally apply the same tanh softcap the main head uses (`gpt.py:472`) so aux distributions match shape.
- Total loss = main loss + `α · Σ` aux losses; start with `α = 0.3 / K` and tune.
- Use **fused / cut cross-entropy** per aux head — `K` extra `(B, T, vocab)` logits tensors would otherwise blow memory (same concern and same fix as MTP).
- Aux heads are dropped at inference — keep only the final `lm_head` in `generate()` / `engine.py`. Deployed cost unchanged.

### Weight sharing — choice point

- **Independent heads** — each aux head learns its own projection. Adds `K · d_model · V` params.
- **Shared with main `lm_head`** — all aux heads use the *same* unembedding matrix; only the per-depth `norm` differs. Far fewer extra params; experimentally usually fine and sometimes better. **Recommend as default.**

## Overlay implementation

Subclass `GPT`, override `forward()` + `init_weights()` + `setup_optimizer()` + `estimate_flops()`. Same `overlay/` + `wrappers/` setup as MTP. Pass `K`, depths, and `α` via env vars to avoid argparse conflicts.

If combined with MTP and/or z-loss (recommended — see below), fold all three into one `MTPGPT`-style subclass.

## Expected cost

Same shape as MTP — each aux head is one `d_model → vocab` matmul + CE. The unembedding's share of forward FLOPs is scale-dependent.

| Model size                | 3-head training-time cost |
|----------------------------|---------------------------|
| depth 12 (default-ish)     | ~+30–35%                  |
| depth 26 (GPT-2 speedrun)  | ~+15–20% (≈ +20 min on the ~2h run) |
| larger                     | <+10%                     |

Notes:
- Sharing the unembedding with the main `lm_head` removes the extra-params cost (DDP all-reduce, optimizer state), leaving just the matmul + CE — closer to the lower end of the band.
- Fused CE removes the activation-memory blowup.
- **Inference cost: unchanged** (aux heads dropped).

## Combines well with

- **MTP** (`../mtp/README.md`) — orthogonal axes (depth × time), same overlay surface. Fold both into one subclass; share the unembedding to limit param/compute growth. Single biggest expected gain on the capability-track list.
- **z-loss** (`../zloss/README.md`) — apply z-loss to **every** aux head (each produces its own vocab logits subject to drift). Near-zero added cost.
- Orthogonal to differential attention, batch-size tuning, and layer-wise LR.

## Downstream — adaptive-depth / early-exit (essentially free once aux heads exist)

Aux heads are placed for a training-side reason (depth-axis signal density), but the same heads enable adaptive computation downstream — without a separate overlay. Two flavors, increasing in implementation cost:

- **Inference-time early exit (free).** Keep the aux heads at inference time and exit each sequence at the first head whose top-token confidence (or entropy) crosses a threshold. Saves trunk forward FLOPs at decode. CALM (Schuster et al. 2022) reports ~2–3× inference speedup at <1% quality loss on standard LMs. Cost: zero training-side, a few lines of inference-side logic to consult an aux head per block and short-circuit. **Worth doing immediately once deep supervision lands and has measurable signal.**

- **Training-time adaptive depth (~tens of lines).** During training, stop backproping through later blocks for sequences whose intermediate aux head is already confident. Forward still runs all blocks (later aux heads need signal); backward stops early per sequence. Expected wall-clock saving over a full speedrun: **~5–12%** (small early in training when no exits are confident; grows late). Modest as a pure speedup, but it's incremental on machinery already built rather than a new overlay.

The relevant brain-inspired connection is the **conditional-computation / adaptive-compute** line (MoD, ACT). That whole family is hardware-awkward when implemented at per-token granularity (variable tensor shapes per block, gather/scatter overhead, FA3 friction). Sequence-level early-exit on top of deep supervision sidesteps all of that — no shape change, no router, no per-block gather — at the cost of a coarser granularity (whole sequences, not individual tokens). For this project's scale, that trade is the right one.

Net framing: **build deep supervision for the depth-axis signal-density argument**; once it works, the inference-time early-exit comes for free and the training-time variant is a small follow-on. Don't build either before deep supervision has a real-data win on its own — the early-exit thresholds are meaningless without trustworthy intermediate heads.

## Testing / A-B

- Same pinned nanochat commit + same seed as baseline.
- Primary metrics: `val_bpb`, CORE on the **main head only** (aux heads are a training aid, not the eval target).
- Watch the aux-head losses too — they should track the main head's loss with a gap that shrinks with depth (deeper aux heads ≈ main head). If a deep aux head's loss > main head's, placement or weighting is off.
- Smoke-test wiring with the tiny config (`--depth=4 --num-iterations=20`); place 1 aux head at the middle.

## Open questions

- `K` and exact depths — uniform spacing vs. emphasising the early half (where the signal is most needed).
- Aux loss weighting `α` — uniform vs. heavier on shallower heads.
- Weight sharing with main `lm_head` vs. independent heads.
- Apply norm + softcap per aux head, or just norm.
- Whether to feed aux heads through the same `x0` residual / `backout` machinery (`gpt.py`).
- Interaction with MTP: do aux heads also predict `t+2, t+3`, or only `t+1`? (Recommend: only `t+1` for aux heads; MTP at the main head only.)

## References

- Lee et al. 2015 — "Deeply-Supervised Nets."
- Szegedy et al. 2014 — "Going Deeper with Convolutions" (GoogLeNet auxiliary classifiers).
- Belilovsky et al. 2019/2020 — "Decoupled Greedy Learning of CNNs" (related local-prediction line; with stop-gradient between blocks it becomes the **non-backprop** Stage 2 — see `../non-backprop/README.md`).
