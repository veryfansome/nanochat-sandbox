# runs/

Operational guide for the shell pipelines here — practical recipes and gotchas from actually running speedruns on rented GPUs. The script *mechanics* (each env var, each subcommand) live in [`../SETUP.md`](../SETUP.md); this file is the *operational* counterpart.

Companion files:
- [`../overlay/README.md`](../overlay/README.md) — implementation lessons for writing overlays
- [`../ideas/README.md`](../ideas/README.md) — ideation lessons for designing experiments

## First-time Lambda checklist

Confirm before kicking off any speedrun:

1. **`~/.lambda.env` exists locally** (chmod 600) with `LAMBDA_API_KEY`, `LAMBDA_SSH_KEY`, `WANDB_API_KEY`, optionally `NANOCHAT_COMMIT`. `lambda.sh` auto-sources it on every invocation.
2. **SSH key uploaded to Lambda** under `LAMBDA_SSH_KEY`'s exact name (the dashboard column, not a local file path). Verify with `curl -u "$LAMBDA_API_KEY:" https://cloud.lambdalabs.com/api/v1/ssh-keys | jq '.data[] | .name'`.
3. **Local SSH agent has the matching private key**: `ssh-add ~/.ssh/<your_lambda_key>`.
4. **`bash runs/lambda.sh types --available`** before launching — surfaces capacity right now so you don't get a launch-time "no capacity" surprise (see "Capacity strategy").

## Hardware recipes

Set these env vars per hardware tier when calling `runs/speedrun.sh`. Defaults match H100 SXM5 (upstream speedrun leaderboard config).

| Hardware | `USE_FP8` | `WINDOW_PATTERN` | `DEVICE_BATCH_SIZE` | Notes |
|---|:-:|:-:|:-:|---|
| 8x H100 SXM5 (default) | 1 | SSSL | 16 | Upstream leaderboard config; FA3 available |
| 8x H100 PCIe | 1 | SSSL | 16 | Slight interconnect penalty vs SXM5 |
| 8x A100 80GB SXM4 | 0 | L | 16 | No fp8; SDPA can't do SSSL efficiently |
| 8x A100 40GB SXM4 | 0 | L | 8 | All of the above + OOM-safe batch |
| 8x V100 | 0 | L | 8 | Volta; last resort |

Example for 8x A100 40GB:

```bash
USE_FP8=0 WINDOW_PATTERN=L DEVICE_BATCH_SIZE=8 \
    WANDB_RUN=d24_baseline bash runs/speedrun.sh
```

Internally-consistent A/B is preserved as long as baseline + overlay runs share the same env vars. The numbers won't be directly comparable to upstream's H100+fp8+SSSL leaderboard, but they're comparable to each other.

## Capacity strategy

H100 SXM5 availability fluctuates on Lambda. `bash runs/lambda.sh types --available` shows what's there right now.

**Wait-and-retry** (cheapest if you can wait):

```bash
while ! bash runs/lambda.sh types --available 2>/dev/null | grep -q "gpu_8x_h100_sxm5"; do
    echo "$(date '+%H:%M:%S') waiting for h100 capacity..."
    sleep 600
done && say "h100 available" 2>/dev/null   # macOS notification; substitute your alarm of choice
```

**Fall-back ladder** (when H100 is scarce): H100 SXM5 → H100 PCIe → A100 80GB → A100 40GB → V100. Pick the matching row from "Hardware recipes" and set its env vars. Total trio cost is roughly flat across tiers — slower hardware is cheaper per hour, so the two effects partially cancel.

## Run patterns

Each pattern assumes you've SSH'd into the host. Substitute env vars per the hardware recipes table.

**Single run** (cheap probe — recommended before committing to a trio):

```bash
tmux new -d -s sr 'cd ~/sandbox && \
    USE_FP8=0 WINDOW_PATTERN=L DEVICE_BATCH_SIZE=8 \
        WANDB_RUN=d24_baseline bash runs/speedrun.sh 2>&1 | tee runs/d24_baseline.log' && \
    tmux attach -t sr
```

After ~5 min check `tok/sec` in the live log or wandb — that's your throughput projection for the trio.

**Sequential trio** (baseline → zloss → mtp; stops on first failure):

```bash
tmux new -d -s sr 'set -o pipefail; cd ~/sandbox && \
    USE_FP8=0 WINDOW_PATTERN=L DEVICE_BATCH_SIZE=8 WANDB_RUN=d24_baseline                  bash runs/speedrun.sh 2>&1 | tee runs/d24_baseline.log && \
    USE_FP8=0 WINDOW_PATTERN=L DEVICE_BATCH_SIZE=8 OVERLAY=zloss WANDB_RUN=d24_zloss       bash runs/speedrun.sh 2>&1 | tee runs/d24_zloss.log    && \
    USE_FP8=0 WINDOW_PATTERN=L DEVICE_BATCH_SIZE=8 OVERLAY=mtp   WANDB_RUN=d24_mtp         bash runs/speedrun.sh 2>&1 | tee runs/d24_mtp.log' && \
    tmux attach -t sr
```

**Auto-queue after a previous run** (fire-and-forget — polls wandb for completion before kicking off the chain). Substitute `<entity>/<project>/<run_name>` for the run you're waiting on:

```bash
tmux new -d -s sr2 'until uv run --directory ~/sandbox python -c "import sys, wandb; sys.exit(0 if wandb.Api().run(\"<entity>/<project>/d24_baseline\").state == \"finished\" else 1)" 2>/dev/null; do sleep 60; done && \
    cd ~/sandbox && set -o pipefail && \
    USE_FP8=0 WINDOW_PATTERN=L DEVICE_BATCH_SIZE=8 OVERLAY=zloss WANDB_RUN=d24_zloss bash runs/speedrun.sh 2>&1 | tee runs/d24_zloss.log && \
    USE_FP8=0 WINDOW_PATTERN=L DEVICE_BATCH_SIZE=8 OVERLAY=mtp   WANDB_RUN=d24_mtp   bash runs/speedrun.sh 2>&1 | tee runs/d24_mtp.log'
```

## Cost expectations

Rough wall-clock × hourly rate for one d24 speedrun (base train + eval + SFT + chat_eval). Trio = baseline + zloss + mtp run sequentially. Always confirm with a single-run probe before committing.

| Hardware | $/hr (approx) | per-run hours | trio hours | trio cost |
|---|---:|---:|---:|---:|
| 8x H100 SXM5 | ~$24 | ~3 (leaderboard) | ~9 | ~$220 |
| 8x A100 80GB SXM4 | ~$15 | ~5 | ~15 | ~$225 |
| 8x A100 40GB SXM4 | ~$15 | ~5.5 (measured at 288k tok/sec) | ~17 | ~$255 |
| 8x V100 | ~$8 | ~15+ | ~45+ | ~$360+ |

Prices fluctuate; check Lambda's pricing page for current rates.

## Operational lessons

- **Verify wandb is set up before training kicks off.** A `wandb.init()` failure 5 min into training wastes the bootstrap cost and the data/tokenizer setup. `lambda.sh bootstrap` now hard-fails if `WANDB_API_KEY` isn't set locally (override with `SKIP_WANDB=1` if you intend to `wandb login` manually on the host). If you got past bootstrap and want to double-check: `ssh ubuntu@<ip> 'grep -A1 wandb ~/.netrc'`.
- **OOM means batch size, not the script being broken.** Default `DEVICE_BATCH_SIZE=16` is tuned for H100 80GB. On 40GB HBM, drop to 8. Halving the batch ~doubles grad-accum and ~doubles wall-clock per step, but it's usually cheaper than terminating and re-bootstrapping on bigger hardware.
- **Termination is manual.** Lambda bills per-hour from launch to terminate, regardless of whether training is running or the tmux session is alive. After the run completes (check wandb shows `finished`), terminate locally: `bash runs/lambda.sh terminate "$ID"`. Set a phone alarm for ETA + 1 hour buffer if you're unattended.

## Comparing runs

After completion, pull metrics from wandb side-by-side via `tools/compare_runs.py`:

```bash
uv run python -m tools.compare_runs d24_baseline d24_zloss d24_mtp
```

Auto-archived per-run artifacts (eval CSV, base-model reports, meta.json, nanochat commit, env) land under `sandbox/results/<MODEL_TAG>/` on the host — pull them back with `scp -r` or `rsync` if you want a permanent local copy beyond wandb.
