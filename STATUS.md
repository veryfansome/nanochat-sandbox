# Status

Working state of the project — implementation progress, current focus, sequencing, next concrete steps. **Not auto-loaded** as agent context; read explicitly when you need to know "where are we right now?". Update this file as work lands.

## Implementation

| Idea | Designed | Implemented | Smoke ✓ | Real-data ✓ |
|------|:--------:|:-----------:|:-------:|:-----------:|
| [z-loss](ideas/zloss/README.md) | ✓ | ✓ (naive default; fused autograd opt-in via `ZLOSS_FUSED=1`) | ✓ (a/b/c/d) | partial — d6/M4 only |
| [MTP](ideas/mtp/README.md) | ✓ | ✓ (naive only; shared unembedding; k=3 default) | ✓ (a/b/c) | d6/M4 measured — val/bpb +4.17%, tok/sec −32% vs baseline (expected toy-scale signature of a scale-dependent technique). GPU validation pending. |
| [Deep supervision](ideas/deep-supervision/README.md) | ✓ | — | — | — |
| [Token-level loss weighting](ideas/token-loss-weighting/README.md) | ✓ | — | — | — |
| [Differential attention](ideas/diff-attention/README.md) | ✓ | — | — | — |
| [Online data selection + batch-size tuning](ideas/online-data-selection/README.md) | ✓ | — | — | — |
| [Adaptive sequence length / batch size](ideas/adaptive-schedule/README.md) | ✓ | — | — | — |
| [Layer-wise LR / staged maturation](ideas/layerwise-lr/README.md) | ✓ | — | — | — |
| [Tokenizer variants](ideas/tokenizer-variants/README.md) | ✓ | — | — | — |
| [Non-backprop (DFA → block-local)](ideas/non-backprop/README.md) | ✓ (research track) | — | — | — |

## Currently in flight

Three overlays measured at d6/M4 scale; runs synced to wandb and archived under `results/`. Every result here is mechanism-verification, not capability claim — the d6/5000-iter regime is below the noise/signal threshold for any of these techniques per `ideas/README.md` "Lessons learned (ideation)".

- **`d6_baseline`** (wandb `qdoniwrj`) — `val/bpb` 1.16534. Pure baseline reference. (Original CORE CSV was overwritten before auto-archive landed; recovering it would require a retrain.)
- **`d6_zloss`** (wandb `szesl4yg`) — `val/bpb` 1.17007 (+0.41% vs baseline; within noise). Naive `F.cross_entropy` + `torch.logsumexp`. tok/sec ~7% under baseline. CORE 0.0361.
- **`d6_zloss_fused`** (wandb `de562k89`, killed at ~300 steps) — confirmed the Python fused-autograd path is **numerically correct** (bit-exact loss + ~2e-8 grad agreement vs naive) but **~32% slower than baseline** in pure PyTorch (Python ops replace C++ CE). Default flipped to naive; opt in via `ZLOSS_FUSED=1`. See `ideas/zloss/README.md` "Expected cost".
- **`d6_mtp`** (wandb `bcv9920m`) — `val/bpb` 1.21398 (+4.17% vs baseline; stable +0.045–0.049 gap from step ~500 onward). tok/sec ~32% under baseline (4 separate `F.cross_entropy` calls). CORE −0.0179. The val/bpb regression is the expected toy-scale MTP signature: aux supervision diverts gradient capacity from the main objective, and at 37M params / 5000 iters the model can't afford the dual objective. See `ideas/mtp/README.md` "Current state" for details.

Compare any combination:

```bash
uv run python -m tools.compare_runs d6_baseline d6_zloss [d6_zloss_fused] [d6_mtp]
```

## Suggested sequencing

1. **Cheapest first** — batch-size A/B (config-only), z-loss (done at d6 scale), layer-wise LR (overlay, ~0 cost), and **token-level loss weighting** (entropy / focal variants — pure-loss overlay, no extra model). Quick signal, no real engineering.
2. **MTP** — the main capability + sample-efficiency lever. Next big build per the plan.
3. **Deep supervision** — depth-axis complement to MTP; fold into the same subclass for one combined experiment. Token weighting's **RHO-Loss variant** can fold in here too (it shares the same `forward` surface and needs a reference model that can be added once for both).
4. **Differential attention** — independent architecture A/B.
5. **Adaptive sequence length (Part A of adaptive-schedule)** — harness change, but the model needs no edits and Part A is the rare idea that should make the speedrun **faster on wall-clock** at the same final loss. Cheap-ish entry into the harness-change tier.
6. **Online data selection + adaptive batch size (Part B of adaptive-schedule)** — bigger, harness-touching; pursue if earlier results justify it. Both share the copied-`base_train.py` pattern, so fold into a single forked harness. Online data selection is complementary to token weighting (sample-level vs. token-level signal shaping), so worth A/B-ing alone, weighting alone, and combined.

**[Tokenizer variants](ideas/tokenizer-variants/README.md)** is a **parallel track**, not part of this serialized sequence. Offline variant generation + offline filtering (compression / vocab / structural metrics) runs on a laptop and does not compete with GPU time; only the final variant's full speedrun A/B does. Pick it up whenever the model-overlay track is GPU-bound or blocked on Lambda time. The cheapest entry point is building `tools/eval_tokenizer.py` (Tier 1 metrics only) and running variant 4 (tokenizer-training corpus mix — no regex change, only a data filter).

The [non-backprop LLM](ideas/non-backprop/README.md) is a **separate research track**, not part of this capability-tuning sequence. Pursue independently.

## Next concrete steps

1. **Real-data validation of z-loss and MTP on GPU.** d6/M4 results aren't capability-meaningful for either; both are scale-dependent techniques whose wins (if any) appear at GPT-2+ scale per the literature. Lambda speedrun via [`runs/lambda.sh`](runs/lambda.sh) → on the instance:

   ```bash
   WANDB_RUN=d24_baseline bash runs/speedrun.sh
   WANDB_RUN=d24_zloss MODEL_TAG=d24_zloss OVERLAY=zloss bash runs/speedrun.sh
   WANDB_RUN=d24_mtp   MODEL_TAG=d24_mtp   OVERLAY=mtp   bash runs/speedrun.sh
   uv run python -m tools.compare_runs d24_baseline d24_zloss d24_mtp
   ```

2. **Build the next overlay** — per sequencing, **deep supervision** (folds into the same `MTPGPT` subclass) or **differential attention** (independent architecture A/B). Deep supervision is the more natural next step since the MTPGPT scaffolding is already in place. **Token-level loss weighting (entropy/focal variant)** is a viable parallel cheap track — pure-loss, no extra model, independent of the MTPGPT line — pick it up alongside if you want a second cheap signal in flight.

3. **Tokenizer-variants track (parallel, no GPU)** — build `tools/eval_tokenizer.py` (Tier 1 metrics: per-domain compression, vocab inspection, structural reversibility battery) and run variant 4 (tokenizer-training corpus mix) end-to-end as a workflow shakeout. Anything that survives Tier 1 + 2 + a d6 probe gets queued for the next Lambda session as an extra A/B alongside the model-side runs.

4. After that: differential attention, then online data selection if the earlier results justify the harness-touching investment.
