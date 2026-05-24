# runs/

Operational guide for the shell pipelines: recipes and gotchas from running speedruns on rented GPUs. Script *mechanics* live in [`../SETUP.md`](../SETUP.md); implementation lessons in [`../overlay/README.md`](../overlay/README.md); ideation lessons in [`../ideas/README.md`](../ideas/README.md).

## First-time Lambda checklist

1. **`~/.lambda.env`** (chmod 600) with `LAMBDA_API_KEY`, `LAMBDA_SSH_KEY`, `WANDB_API_KEY`, optionally `NANOCHAT_COMMIT`. `lambda.sh` auto-sources it.
2. **SSH key uploaded to Lambda** under the name you set as `LAMBDA_SSH_KEY` (dashboard column, not a file path). List via `curl -u "$LAMBDA_API_KEY:" https://cloud.lambdalabs.com/api/v1/ssh-keys | jq '.data[].name'`.
3. **Local SSH agent has the matching private key**: `ssh-add ~/.ssh/<your_lambda_key>`.
4. **`bash runs/lambda.sh types --available`** before launching to see current capacity.

## Hardware recipes

Env vars per hardware tier for `runs/speedrun.sh`. Defaults match H100 SXM5 (upstream config).

| Hardware | `USE_FP8` | `WINDOW_PATTERN` | `DEVICE_BATCH_SIZE` |
|---|:-:|:-:|:-:|
| 8x H100 SXM5 (default) | 1 | SSSL | 16 |
| 8x A100 80GB SXM4 | 0 | L | 16 |
| 8x A100 40GB SXM4 | 0 | L | 8 |
| 8x V100 | 0 | L | 8 |

`USE_FP8=0` because A100/V100 lack fp8 hardware. `WINDOW_PATTERN=L` because PyTorch SDPA (the non-FA3 fallback) has no sliding-window kernels. `DEVICE_BATCH_SIZE=8` on 40GB HBM to avoid OOM at d24.

A/B integrity holds as long as baseline + overlay runs share the same env vars. Results won't match upstream's H100+fp8+SSSL leaderboard but are comparable to each other.

## Capacity strategy

H100 availability fluctuates. When scarce:

```bash
# wait-and-retry:
while ! bash runs/lambda.sh types --available 2>/dev/null | grep -q "gpu_8x_h100_sxm5"; do
    echo "$(date '+%H:%M:%S') waiting..." && sleep 600
done
```

**Fall-back ladder**: H100 SXM5 → A100 80GB → A100 40GB → V100. Match the row in "Hardware recipes." Total trio cost is roughly flat across tiers (slower hardware is cheaper per hour).

## Run patterns

SSH'd into the host. Substitute env vars per "Hardware recipes."

`USE_SFT=0` is the default: our overlays don't touch SFT, so chat_eval at this scale adds noise + cost without informative A/B signal. Set `USE_SFT=1` for the full pipeline. Base checkpoints persist regardless — SFT is re-runnable against any saved checkpoint.

**Single run** (cheap probe — recommended before committing to a trio):

```bash
tmux new -d -s sr 'cd ~/sandbox && \
    USE_FP8=0 WINDOW_PATTERN=L DEVICE_BATCH_SIZE=8 \
        WANDB_RUN=d24_baseline bash runs/speedrun.sh 2>&1 | tee runs/d24_baseline.log' && tmux attach -t sr
```

Check `tok/sec` in the live log or wandb after ~5 min for your trio projection.

**Sequential trio** (baseline → zloss → mtp; stops on first failure):

```bash
tmux new -d -s sr 'set -o pipefail; cd ~/sandbox && \
    USE_FP8=0 WINDOW_PATTERN=L DEVICE_BATCH_SIZE=8 WANDB_RUN=d24_baseline                  bash runs/speedrun.sh 2>&1 | tee runs/d24_baseline.log && \
    USE_FP8=0 WINDOW_PATTERN=L DEVICE_BATCH_SIZE=8 OVERLAY=zloss WANDB_RUN=d24_zloss       bash runs/speedrun.sh 2>&1 | tee runs/d24_zloss.log    && \
    USE_FP8=0 WINDOW_PATTERN=L DEVICE_BATCH_SIZE=8 OVERLAY=mtp   WANDB_RUN=d24_mtp         bash runs/speedrun.sh 2>&1 | tee runs/d24_mtp.log' && tmux attach -t sr
```

**Auto-queue after a previous run** (polls wandb; substitute `<entity>/<project>/<run>`):

```bash
tmux new -d -s sr2 'until uv run --directory ~/sandbox python -c "import sys, wandb; sys.exit(0 if wandb.Api().run(\"<entity>/<project>/d24_baseline\").state == \"finished\" else 1)" 2>/dev/null; do sleep 60; done && \
    cd ~/sandbox && set -o pipefail && \
    USE_FP8=0 WINDOW_PATTERN=L DEVICE_BATCH_SIZE=8 OVERLAY=zloss WANDB_RUN=d24_zloss bash runs/speedrun.sh 2>&1 | tee runs/d24_zloss.log && \
    USE_FP8=0 WINDOW_PATTERN=L DEVICE_BATCH_SIZE=8 OVERLAY=mtp   WANDB_RUN=d24_mtp   bash runs/speedrun.sh 2>&1 | tee runs/d24_mtp.log'
```

## Cost expectations

One d24 base-only speedrun (`USE_SFT=0`). Trio = baseline + zloss + mtp sequential. Probe with a single run first; prices fluctuate.

| Hardware | $/hr | per-run hours | trio cost |
|---|---:|---:|---:|
| 8x H100 SXM5 | ~$24 | ~3 (leaderboard) | ~$220 |
| 8x A100 80GB SXM4 | ~$15 | ~4-5 | ~$200-225 |
| 8x A100 40GB SXM4 | ~$15 | ~5 (measured at 288k tok/sec) | ~$225 |
| 8x V100 | ~$8 | ~15+ | ~$360+ |

## Operational lessons

- **wandb auth must be on the host before training.** A `wandb.init()` failure 5 min in wastes bootstrap + tokenizer cost. `lambda.sh bootstrap` hard-fails if `WANDB_API_KEY` isn't set locally; pass `SKIP_WANDB=1` if you'll `wandb login` manually on the host.
- **OOM means batch size, not a script bug.** `DEVICE_BATCH_SIZE=16` is for H100 80GB. Drop to 8 on 40GB HBM. Halving ~doubles grad-accum and wall-clock per step but is cheaper than re-bootstrapping on bigger hardware.
- **Termination is manual.** Lambda bills per-hour from launch to terminate, regardless of training or tmux state. After wandb shows `finished`: `bash runs/lambda.sh terminate "$ID"`. Phone alarm for ETA + buffer if unattended.

## Comparing runs

```bash
uv run python -m tools.compare_runs d24_baseline d24_zloss d24_mtp
```

Per-run artifacts auto-archive to `~/sandbox/results/<MODEL_TAG>/` on the host (eval CSV, base-model reports, meta.json, NANOCHAT_COMMIT, ENV). `rsync` back to local if you want a copy beyond wandb.
