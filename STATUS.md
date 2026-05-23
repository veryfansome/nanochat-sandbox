# Status

Working state of the project — implementation progress, current focus, sequencing, next concrete steps. **Not auto-loaded** as agent context; read explicitly when you need to know "where are we right now?". Update this file as work lands.

## Implementation

| Idea | Designed | Implemented | Smoke ✓ | Real-data ✓ |
|------|:--------:|:-----------:|:-------:|:-----------:|
| [z-loss](ideas/zloss/README.md) | ✓ | ✓ (naive default; fused autograd opt-in via `ZLOSS_FUSED=1`) | ✓ (a/b/c/d) | partial — d6/M4 only |
| [MTP](ideas/mtp/README.md) | ✓ | — | — | — |
| [Deep supervision](ideas/deep-supervision/README.md) | ✓ | — | — | — |
| [Differential attention](ideas/diff-attention/README.md) | ✓ | — | — | — |
| [Online data selection + batch-size tuning](ideas/online-data-selection/README.md) | ✓ | — | — | — |
| [Layer-wise LR / staged maturation](ideas/layerwise-lr/README.md) | ✓ | — | — | — |
| [Non-backprop (DFA → block-local)](ideas/non-backprop/README.md) | ✓ (research track) | — | — | — |

## Currently in flight

z-loss has been validated at d6 scale on the M4 mini; runs archived under `results/` and synced to wandb:

- **`d6_baseline`** (wandb `qdoniwrj`) — `val/bpb` 1.16534, CORE 0.0361. The CORE number is toy-scale, not capability-meaningful.
- **`d6_zloss`** (wandb `szesl4yg`) — `val/bpb` 1.17007 (+0.41% vs baseline; within noise at this scale). Used the naive `F.cross_entropy` + `torch.logsumexp` path. tok/sec ~7% under baseline.
- **`d6_zloss_fused`** (wandb `de562k89`, killed at ~300 steps) — confirmed the Python fused-autograd path is **numerically correct** (matched naive bit-exactly: Δ val/bpb at step 300 = 6.6e-05) but **~32% slower than baseline** because pure-Python autograd replaces optimized C++ CE with Python ops. Default flipped to naive; the fused path remains opt-in via `ZLOSS_FUSED=1` for memory-bound scenarios. See `ideas/zloss/README.md` "Expected cost".

Compare any two/three:

```bash
uv run python -m tools.compare_runs d6_baseline d6_zloss [d6_zloss_fused]
```

## Suggested sequencing

1. **Cheapest first** — batch-size A/B (config-only), z-loss (done at d6 scale), and layer-wise LR (overlay, ~0 cost). Quick signal, no real engineering.
2. **MTP** — the main capability + sample-efficiency lever. Next big build per the plan.
3. **Deep supervision** — depth-axis complement to MTP; fold into the same subclass for one combined experiment.
4. **Differential attention** — independent architecture A/B.
5. **Online data selection** — bigger, harness-touching; pursue if earlier results justify it.

The [non-backprop LLM](ideas/non-backprop/README.md) is a **separate research track**, not part of this capability-tuning sequence. Pursue independently.

## Next concrete steps

1. **Real-data validation of z-loss on GPU.** d6/M4 results aren't capability-meaningful (the +0.41% gap is within seed noise at this scale, and z-loss's effects show up most at scale + under aggressive LR schedules). The actual question — does z-loss help at GPT-2 scale — needs a Lambda speedrun. Use [`runs/lambda.sh`](runs/lambda.sh) to provision, then on the instance:

   ```bash
   WANDB_RUN=d24_baseline bash runs/speedrun.sh
   WANDB_RUN=d24_zloss MODEL_TAG=d24_zloss OVERLAY=zloss bash runs/speedrun.sh
   uv run python -m tools.compare_runs d24_baseline d24_zloss
   ```

2. **Build the MTP overlay** — the next big lever per sequencing. Follow `SETUP.md` "Adding a new overlay" recipe (three files + smoke). Plan to compose with z-loss into one `MTPGPT` subclass so the depth-axis (deep supervision) work later folds in cleanly.

3. After MTP lands: deep supervision (folds into MTPGPT), then differential attention as a separate architecture A/B.
