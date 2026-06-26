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

**Per-overlay overrides**: some overlays add saved-for-backward tensors that change the memory ceiling.

| Overlay | 40GB override | Rationale |
|---|---|---|
| `mtp` (k=3 default) | `DEVICE_BATCH_SIZE=4` | k=3 extra `(B,T,V)` fp32 softmaxes ≈ +6 GB at B=8 → OOM. B=4 cuts it to +3 GB with comfortable headroom. Wall-clock +12%, not +100% — grad-accum overhead is sub-linear. |

A/B integrity holds as long as baseline + overlay runs share the same env vars *within their override row*. If you change `DEVICE_BATCH_SIZE` only for one arm (as we did for mtp), note it in the comparison — `base_train` holds total batch (in tokens) constant via grad-accum, so the optimization signal is preserved, but it's worth flagging. Results won't match upstream's H100+fp8+SSSL leaderboard but are comparable to each other.

## Capacity strategy

H100 availability fluctuates. When scarce:

```bash
# wait-and-retry:
while ! bash runs/lambda.sh types --available 2>/dev/null | grep -q "gpu_8x_h100_sxm5"; do
    echo "$(date '+%H:%M:%S') waiting..." && sleep 600
done
```

**Fall-back ladder**: H100 SXM5 → A100 80GB → A100 40GB → V100. Match the row in "Hardware recipes." Trio cost is *not* flat across tiers and cheaper hardware isn't always cheaper per run — at current Lambda prices the 80GB A100 is ~15% faster but ~20% pricier per run than the 40GB. Check live $/hr (see "Cost expectations") before picking.

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

**Sequential trio** (baseline → zloss → mtp; stops on first failure). Note the mtp arm uses `DEVICE_BATCH_SIZE=4` per the per-overlay-overrides table:

```bash
tmux new -d -s sr 'set -o pipefail; cd ~/sandbox && \
    USE_FP8=0 WINDOW_PATTERN=L DEVICE_BATCH_SIZE=8 WANDB_RUN=d24_baseline                  bash runs/speedrun.sh 2>&1 | tee runs/d24_baseline.log && \
    USE_FP8=0 WINDOW_PATTERN=L DEVICE_BATCH_SIZE=8 OVERLAY=zloss WANDB_RUN=d24_zloss       bash runs/speedrun.sh 2>&1 | tee runs/d24_zloss.log    && \
    USE_FP8=0 WINDOW_PATTERN=L DEVICE_BATCH_SIZE=4 OVERLAY=mtp   WANDB_RUN=d24_mtp         bash runs/speedrun.sh 2>&1 | tee runs/d24_mtp.log' && tmux attach -t sr
```

**Auto-queue after a previous run** (polls wandb; substitute `<entity>/<project>/<run>`):

```bash
tmux new -d -s sr2 'until uv run --directory ~/sandbox python -c "import sys, wandb; sys.exit(0 if wandb.Api().run(\"<entity>/<project>/d24_baseline\").state == \"finished\" else 1)" 2>/dev/null; do sleep 60; done && \
    cd ~/sandbox && set -o pipefail && \
    USE_FP8=0 WINDOW_PATTERN=L DEVICE_BATCH_SIZE=8 OVERLAY=zloss WANDB_RUN=d24_zloss bash runs/speedrun.sh 2>&1 | tee runs/d24_zloss.log && \
    USE_FP8=0 WINDOW_PATTERN=L DEVICE_BATCH_SIZE=4 OVERLAY=mtp   WANDB_RUN=d24_mtp   bash runs/speedrun.sh 2>&1 | tee runs/d24_mtp.log'
```

**Tokenizer-variant arm** (`auto_tune`, `force_merges`): these swap the *tokenizer*, not the model — there is **no `train_<variant>` module**, so `OVERLAY` stays **empty** and the variant enters via a pre-built tokenizer at `$VBASE/tokenizer/`. Run it in its own base dir so it doesn't clobber the baseline tokenizer, and **let speedrun download data + eval_bundle into `$VBASE` itself** — do NOT pre-symlink them to the main cache on a fresh host (see the dangling-symlink lesson below):

```bash
# get the tokenizer into $VBASE/tokenizer/ first: either bake on the host
#   ( wrappers.tok_train_<variant>, which also sets up the data symlink safely ),
#   or rsync a locally-baked tokenizer dir up (md5-verify after).
VBASE=$HOME/.cache/nanochat-variants/<variant>
tmux new -d -s sr "set -o pipefail; cd ~/sandbox && export PATH=\$HOME/.local/bin:\$PATH && \
  NANOCHAT_BASE_DIR=$VBASE USE_FP8=0 WINDOW_PATTERN=L DEVICE_BATCH_SIZE=8 \
    MODEL_TAG=d24_<variant> WANDB_RUN=d24_<variant> bash runs/speedrun.sh 2>&1 | tee runs/d24_<variant>.log" && tmux attach -t sr
```

The tokenizer is the only thing that must be in `$VBASE` ahead of time; the run self-provisions everything else. (Pinning the block/merge list into the variant's `pairs.py` + committing is the cleaner alternative to rsyncing a baked tokenizer — then the host bake is reproducible from code. See `rustbpe_variants/<variant>/README.md`.)

## Orchestrated A/B — `runs/lambda_ab.sh`

`lambda_ab.sh` runs the **whole surface-factoring d24 A/B lifecycle** end-to-end and **idempotently**, wrapping `lambda.sh` + `speedrun.sh`: provision (8x A100 40GB) → bootstrap (`setup.sh` + Rust + the `rustbpe_force_merges` crate, which the surface arm imports via `tools/stack_triple`) → sync the cached `force_merges` + `surface_triple` tokenizers (they live outside `sandbox/`, so plain bootstrap misses them) → preflight (8 GPUs, venv, both tokenizers, crate, surface smoke) → train **both arms** in one tmux session → poll → download results **+ the model checkpoints** → verify everything is local → terminate.

Two arms, both at d24 on the same commit/config (only the tokenizer + overlay differ):
- **`d24_force_merges_ab`** — baseline: the adopted `force_merges` tokenizer, no overlay.
- **`d24_surface_factoring_ab`** — `OVERLAY=surface_factoring` + `EVAL_MODULE=wrappers.base_eval_surface` (surface-aware CORE/load). Both eval `--eval core,bpb` (no `sample`).

```bash
bash runs/lambda_ab.sh                  # full pipeline (resumable; reuses live instance, skips finished arms)
STAGE=download bash runs/lambda_ab.sh   # run a single stage: provision|bootstrap|sync|verify|train|poll|download|verifydl|terminate
AUTO_TERMINATE=1 YES=1 bash runs/lambda_ab.sh   # don't prompt at provision/terminate
```

**Idempotency** is the point: a live instance + completed stages (host markers `~/.ab_setup_done`, `~/sandbox/.ab_done`) + finished arms (`results/<tag>/eval.csv`) are detected and skipped. So **fix → push → re-run** and only the unfinished work re-executes. The instance is **never terminated until `verifydl` confirms both arms' `eval.csv` + checkpoint are local** — a crash leaves it up; resume, don't re-provision.

Gotchas: (1) the long `poll` stage blocks the local terminal for ~hours — run the orchestrator itself under a local `tmux`/`nohup`, or split it (`STAGE=train` now, `STAGE=poll` … `terminate` later) — the *training* runs in the host's tmux and survives regardless. (2) `LAMBDA_INSTANCE_TYPE` defaults to `gpu_8x_a100_sxm4`; if that name is wrong for current capacity the provision stage prints the available types and exits — set the right one. (3) If the surface arm OOMs at `DEVICE_BATCH_SIZE=8`, drop to 4 (`base_train` holds total batch constant, so it stays comparable — note it); first-step OOM is the cheap ~$3-8 kind.

## Cost expectations

One d24 base-only speedrun (`USE_SFT=0`). Trio = baseline + zloss + mtp sequential. Per-run hours include `base_train` + `base_eval` (CORE on full per-task budget takes ~45 min on 8xA100). Trio cost includes ~$18 recurring overhead (bootstrap + inspection). Probe with a single run first; prices fluctuate.

| Hardware | $/hr | per-run hours | trio cost (no SFT) |
|---|---:|---:|---:|
| 8x H100 SXM5 | ~$24 | ~3.75 | ~$285 |
| 8x A100 80GB SXM4 | $22.32 | ~5.75 | ~$385 (computed) |
| 8x A100 40GB SXM4 | $15.92 | ~6.75 (measured: ~$313 steady-state, $337 first trip) | ~$320 (measured ~$313–337) |
| 8x V100 | ~$8 | ~16+ | ~$390+ |

A100 $/hr are **live Lambda prices (2026-06-14): 40GB $15.92, 80GB $22.32**; H100/V100 rows are older approximations — verify before relying. Per *run* in dollars: 40GB ~$107, 80GB ~$128 — the 80GB is ~15% faster (it fits DEVICE_BATCH_SIZE=16 vs 8 + has ~30% more HBM bandwidth; same GA100 die so identical FLOPS) but ~20% **pricier** per run, because its ~40% hourly premium outweighs the time saved. Hardware never changes the CORE/bpb result, only wall-clock — so 40GB is the cheaper route to the same answer; pick 80GB only when the ~1 hr/run saving is worth ~$21/run.

**First trip on a new hardware tier or with a new overlay adds ~$15-25** for learning incidents (overlay-specific OOMs, hardware quirks, auth retries). Budget for it; subsequent trips amortize the lessons.

**SFT adds ~$10/run if `USE_SFT=1`.** The default is 0 because our overlays don't touch SFT — chat_eval results at this scale add noise + cost without informative A/B signal. Enable only when you actually want the chat-eval comparison (and run it on *all* arms of the trio, not just baseline).

## Operational lessons

- **wandb auth must be on the host before training.** A `wandb.init()` failure 5 min in wastes bootstrap + tokenizer cost. `lambda.sh bootstrap` hard-fails if `WANDB_API_KEY` isn't set locally; pass `SKIP_WANDB=1` if you'll `wandb login` manually on the host.

- **A fresh instance has NO training data until a run downloads it.** A just-bootstrapped host holds only `sandbox/` (rsync'd) + anything you pushed; the 170 shards and `eval_bundle` are fetched on first run. Result/checkpoint archives from past runs (`results/d24_*`, wandb) live elsewhere — your local machine or the cloud — and do **not** mean the dataset is present on *this* host. So don't assume a variant's separate base dir can borrow data from `~/.cache/nanochat`: on a fresh host that cache is empty. Concrete failure mode: a tokenizer-variant arm run in its own `NANOCHAT_BASE_DIR=$VBASE` with `$VBASE/base_data_climbmix` symlinked to `~/.cache/nanochat/base_data_climbmix` — but that target doesn't exist on a fresh host, so the symlink dangles and `nanochat.dataset`'s `os.makedirs(DATA_DIR, exist_ok=True)` throws `FileExistsError` (`makedirs` tolerates a symlink only if it resolves to a real dir; for a dangling link `os.path.isdir()` is False, so `exist_ok=True` re-raises). **Rule:** for a standalone variant arm, don't pre-symlink data/eval_bundle — let the run download them into `$VBASE`. Only symlink after a prior arm on the *same host* populated `~/.cache/nanochat`, and guard it the way `setup_variant_base` does (`os.path.exists(src) and not os.path.lexists(dst)`), never a bare `ln -sfn` to a maybe-missing target.

- **OOM diagnosis: batch size first, then look for bugs.** Most OOMs we've hit have been the memory ceiling — `DEVICE_BATCH_SIZE=16` is for H100 80GB, 8 fits 40GB HBM at baseline, `mtp` at k=3 needs 4 on 40GB. Halving roughly doubles grad-accum but only adds ~12% wall-clock (sub-linear, not 2×). But OOM can also come from overlay code (tensors retained across iterations, autograd Function reference leaks, KV cache growth) or upstream changes that altered the activation graph. If halving doesn't fit and you've added overlay code recently, skim recent diffs before re-launching on bigger hardware.

- **OOM cost depends on *when* it hits, and preflight only catches the cheap kind.**
  - **First-step OOM** (config doesn't fit): bootstrap + ~5 min wasted, restart with smaller batch. ~$3-8.
  - **Mid-training OOM, last checkpoint recent**: lose progress since checkpoint + restart. Bounded by checkpoint frequency.
  - **Mid-training OOM, no recoverable checkpoint** (or upstream's checkpoint cadence is sparse): potentially lose hours — $30-50+ on a $15/hr instance, depending on how far in.
  - **Run-tossing OOM** (e.g. OOM during the final eval, or systematic mid-training crash you can't restart past): full run wasted, ~$80+ per arm.

  **Static preflight** (estimate saved-for-backward as `extra_ops × B × T × V × bytes` against headroom; or a 100-step probe at half batch) reliably catches the first kind. It **cannot predict late OOMs** from cumulative state growth (KV cache, fragmentation, outlier-batch sequence-length spikes). Defenses for the painful kind: conservative initial sizing (round down, don't fit-to-the-byte), and ensuring upstream's checkpoint cadence is dense enough that an early-percent OOM doesn't toss hours of work.

- **Termination is manual.** Lambda bills per-hour from launch to terminate, regardless of training or tmux state. After wandb shows `finished`: `bash runs/lambda.sh terminate "$ID"`. Phone alarm for ETA + buffer if unattended.

## Comparing runs

```bash
uv run python -m tools.compare_runs d24_baseline d24_zloss d24_mtp
```

Per-run artifacts auto-archive to `~/sandbox/results/<MODEL_TAG>/` on the host (eval CSV, base-model reports, meta.json, NANOCHAT_COMMIT, ENV). Pull them — plus the final concatenated `report.md` — back to local `sandbox/results/` with:

```bash
bash runs/lambda.sh pull "$ID"
```

`pull` is additive (no `--delete`) and idempotent — safe to re-run between runs to incrementally accumulate per-`MODEL_TAG` archives as each completes. The `report.md` gets a UTC timestamp suffix so successive pulls don't clobber prior copies.
