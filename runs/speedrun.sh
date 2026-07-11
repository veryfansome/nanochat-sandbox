#!/bin/bash
# Full pretraining + SFT pipeline on a multi-GPU node, with an optional overlay
# applied to base_train. Mirrors ../../nanochat/runs/speedrun.sh exactly — the
# only difference is the train module is swapped to wrappers.train_$OVERLAY
# when OVERLAY is set.
#
# Usage:
#     OVERLAY=zloss WANDB_RUN=zloss bash runs/speedrun.sh
#
# Or under tmux so it survives an SSH disconnect:
#     OVERLAY=zloss tmux new -s sr "cd $PWD && bash runs/speedrun.sh 2>&1 | tee runs/sr.log"
#     # Ctrl-B D to detach, `tmux attach -t sr` to re-attach
#
# Env vars:
#   OVERLAY     idea name; empty (default) = upstream baseline (scripts.base_train)
#   MODEL_TAG   checkpoint dir name (default: d${DEPTH}_${OVERLAY:-baseline});
#               ensures baseline and overlay A/B runs don't overwrite each other
#   NPROC       --nproc_per_node for torchrun (default: 8)
#   DEPTH       --depth for base_train (default: 24, matches upstream speedrun)
#   WANDB_RUN   wandb run name (default: dummy = disabled). Stable so
#               tools.compare_runs (which keys on wandb name) keeps working.
#   WANDB_SUFFIX  set to 1 to append _YYYYMMDD_HHMM (UTC) to WANDB_RUN, so
#               reruns of the same name don't share a wandb display name.
#               Default: 0 (stable name).
#   NANOCHAT_BASE_DIR  artifact dir (default: ~/.cache/nanochat)
#   FORCE_RETRAIN_TOKENIZER  set to 1 to retrain tokenizer even if cached (default: 0)
#   USE_FP8     pass --fp8 to base_train (default: 1; set to 0 for A100 / V100
#               or any non-H100 hardware that lacks FP8 tensor cores — upstream
#               speedrun's leaderboard times assume H100 SXM5 with fp8)
#   WINDOW_PATTERN  --window-pattern for base_train (default: SSSL, upstream
#               speedrun). Set WINDOW_PATTERN=L on A100/V100/any non-FA3
#               hardware: PyTorch SDPA has no sliding-window kernels, so the
#               default SSSL pattern simulates with masking and tanks GPU
#               utilization. L = full context attention everywhere, well-
#               optimized in SDPA. Internally-consistent A/B is preserved as
#               long as baseline and overlay runs share the same value.
#   DEVICE_BATCH_SIZE  --device-batch-size for base_train / base_eval /
#               chat_sft / chat_eval (default: 16, upstream speedrun, tuned
#               for H100 80GB). Drop to 8 on A100 40GB to avoid OOM. Halving
#               this doubles grad-accum steps (total batch size held constant
#               by base_train), so wall-clock per training step roughly
#               doubles. Lower = OOM-safe, slower; higher = more memory.
#   EXTRA_TRAIN_ARGS  extra flags appended to the base_train torchrun line (default:
#               empty). E.g. EXTRA_TRAIN_ARGS="--num-iterations=1850" for a reduced
#               COOLED schedule (warmdown stays proportional via --warmdown-ratio,
#               default 0.65) — used by the λ-screen (runs/runpod_lambda_screen.sh).
#               --num-iterations takes precedence over --target-param-data-ratio for
#               the step count; batch/LR/weight-decay still derive from ratio=8, so it
#               is the d24 recipe run for fewer (cooled) steps, not a truncated schedule.
#   USE_SFT     run chat_sft + chat_eval after base_train (default: 0). This
#               *differs* from upstream's speedrun: our default workflow is
#               overlay A/B, where neither overlay touches SFT, so chat_eval
#               just adds noise + cost (~$10-20/run) without informative
#               signal. Set USE_SFT=1 if you want the chat pipeline (full
#               nanochat experience, or to validate a base checkpoint that
#               looked interesting). The base checkpoint persists either way,
#               so you can manually re-run SFT against it later.

set -euo pipefail

# Resolve sandbox dir and activate the venv set up by runs/setup.sh
SANDBOX_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SANDBOX_DIR"
export PATH="$HOME/.local/bin:$PATH"
# shellcheck disable=SC1091
source .venv/bin/activate

# Standard env from upstream
export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat}"
mkdir -p "$NANOCHAT_BASE_DIR"

# Keep wandb's local cache under NANOCHAT_BASE_DIR (see runcpu.sh for rationale).
export WANDB_DIR="$NANOCHAT_BASE_DIR"

OVERLAY="${OVERLAY:-}"
if [ -n "$OVERLAY" ]; then
    TRAIN_MODULE="wrappers.train_$OVERLAY"
    RUN_TAG="${OVERLAY}"
else
    # Baseline routes through wrappers.base_train (not scripts.base_train) so
    # wrappers/__init__.py installs the core_eval prefix-safety patch before
    # mid-train CORE runs. No-op for prefix-stable tokenizers. See wrappers/_patches.py.
    TRAIN_MODULE="wrappers.base_train"
    RUN_TAG="baseline"
fi

NPROC="${NPROC:-8}"
DEPTH="${DEPTH:-24}"
MODEL_TAG="${MODEL_TAG:-d${DEPTH}_${OVERLAY:-baseline}}"
WANDB_RUN="${WANDB_RUN:-dummy}"
# Opt-in UTC timestamp suffix; see header for the trade-off.
if [ "$WANDB_RUN" != "dummy" ] && [ "${WANDB_SUFFIX:-0}" = "1" ]; then
    WANDB_RUN="${WANDB_RUN}_$(date -u +%Y%m%d_%H%M)"
fi

# FP8 toggle: default on (H100+); set USE_FP8=0 to drop --fp8 for A100/V100/etc.
USE_FP8="${USE_FP8:-1}"
FP8_FLAG=""
[ "$USE_FP8" = "1" ] && FP8_FLAG="--fp8"

# Window pattern: default SSSL (upstream speedrun, H100+FA3); set
# WINDOW_PATTERN=L on A100/V100/any non-FA3 hardware (SDPA can't do
# sliding-window efficiently).
WINDOW_PATTERN="${WINDOW_PATTERN:-SSSL}"

# Per-device per-microbatch size. Default 16 (H100 80GB). Drop to 8 on A100
# 40GB to avoid OOM. base_train holds the total batch size constant by
# adjusting grad-accum steps, so halving this doubles wall-clock per step
# but preserves the effective batch.
DEVICE_BATCH_SIZE="${DEVICE_BATCH_SIZE:-16}"

# Run chat_sft + chat_eval after base_train? Default OFF — neither overlay
# touches SFT, so chat_eval at our scale is noisy and not informative for
# A/B comparison. The base checkpoint persists; SFT can be re-run manually
# against any saved checkpoint if needed. Set USE_SFT=1 for the full pipeline.
USE_SFT="${USE_SFT:-0}"

echo "==> overlay:      ${OVERLAY:-(baseline)}"
echo "==> train module: $TRAIN_MODULE"
echo "==> model tag:    $MODEL_TAG"
echo "==> wandb run:    $WANDB_RUN"
echo "==> nproc:        $NPROC"
echo "==> depth:        $DEPTH"
echo "==> fp8:          $USE_FP8"
echo "==> window:       $WINDOW_PATTERN"
echo "==> dev batch:    $DEVICE_BATCH_SIZE"
echo "==> sft:          $USE_SFT"
echo "==> wandb run:    $WANDB_RUN"
echo "==> base dir:     $NANOCHAT_BASE_DIR"
echo "==> nanochat:     $(git -C ../nanochat rev-parse --short HEAD)"
echo

# -----------------------------------------------------------------------------
# Init the per-run report. Guarded: upstream removed nanochat.report (commit
# f10bd75, "delete the whole report thing"), so this is a no-op on new nanochat
# while staying compatible with older pins that still have it.
if python -c "import nanochat.report" 2>/dev/null; then python -m nanochat.report reset; else echo "==> nanochat.report absent (removed upstream) — skipping report reset"; fi

# -----------------------------------------------------------------------------
# Tokenizer + data
# wrappers.download_dataset applies the FineWeb override (NANOCHAT_DATASET=fineweb) before
# nanochat.dataset's __main__ — so a FineWeb run downloads FineWeb, not ClimbMix. No-op for ClimbMix.
python -m wrappers.download_dataset -n 8
python -m wrappers.download_dataset -n 170 &
DATASET_PID=$!

TOKENIZER_FILE="$NANOCHAT_BASE_DIR/tokenizer/tokenizer.pkl"  # RustBPETokenizer save format (default)
if [ -f "$TOKENIZER_FILE" ] && [ "${FORCE_RETRAIN_TOKENIZER:-0}" != "1" ]; then
    echo "==> tokenizer cached at $TOKENIZER_FILE; skipping tok_train + tok_eval"
    echo "    (set FORCE_RETRAIN_TOKENIZER=1 to redo)"
else
    python -m scripts.tok_train
    python -m scripts.tok_eval
fi

# -----------------------------------------------------------------------------
# Pretraining
echo "Waiting for background dataset download to complete..."
wait $DATASET_PID

# Mirrors upstream: --depth=24 --target-param-data-ratio=8 (+ --fp8 if
# USE_FP8=1, default). The only swap is $TRAIN_MODULE in place of
# scripts.base_train. $FP8_FLAG is "" or "--fp8" — bash word-splitting drops
# the arg cleanly when empty. --window-pattern and --device-batch-size are
# env-controlled for non-H100 hardware compatibility.
# EVAL_ONLY=1 skips training and evaluates an EXISTING checkpoint (recovery when training
# finished but the post-train eval crashed — avoids a full retrain). The dataset/tokenizer
# steps above are idempotent no-ops when already present.
if [ "${EVAL_ONLY:-0}" = "1" ]; then
    echo "==> EVAL_ONLY=1: skipping training, evaluating the existing $MODEL_TAG checkpoint"
else
torchrun --standalone --nproc_per_node="$NPROC" -m "$TRAIN_MODULE" -- \
    --depth="$DEPTH" --target-param-data-ratio=8 --device-batch-size="$DEVICE_BATCH_SIZE" $FP8_FLAG \
    --window-pattern="$WINDOW_PATTERN" \
    --model-tag="$MODEL_TAG" --run="$WANDB_RUN" ${EXTRA_TRAIN_ARGS:-}
fi

# wrappers.base_eval (not scripts.base_eval directly) so the prefix-safety
# patch is applied + boundary-crossing summary printed. See wrappers/_patches.py.
# EVAL_MODULE overrides it (default wrappers.base_eval); surface-factoring sets
# EVAL_MODULE=wrappers.base_eval_surface for surface-aware CORE scoring + load.
# EVAL_ARGS appends extra base_eval flags (e.g. "--eval core,bpb").
torchrun --standalone --nproc_per_node="$NPROC" -m "${EVAL_MODULE:-wrappers.base_eval}" -- \
    --model-tag="$MODEL_TAG" --device-batch-size="$DEVICE_BATCH_SIZE" ${EVAL_ARGS:-}

# Archive base-stage artifacts to sandbox/results/$MODEL_TAG/ BEFORE SFT (or
# the next run) overwrites them. Eval CSV path doesn't include $MODEL_TAG
# (upstream writes base_model_<step>.csv), and `nanochat.report reset` clears
# report/ at the start of every run — without this, the only durable record
# of the base-stage numbers is wandb.
ARCHIVE_DIR="$SANDBOX_DIR/results/$MODEL_TAG"
mkdir -p "$ARCHIVE_DIR"
EVAL_CSV=$(ls -t "$NANOCHAT_BASE_DIR/base_eval/base_model_"*.csv 2>/dev/null | head -1)
[ -n "$EVAL_CSV" ] && cp "$EVAL_CSV" "$ARCHIVE_DIR/eval.csv"
cp "$NANOCHAT_BASE_DIR/report/base-model-"*.md "$ARCHIVE_DIR/" 2>/dev/null || true
cp "$NANOCHAT_BASE_DIR/base_checkpoints/$MODEL_TAG/meta_"*.json "$ARCHIVE_DIR/meta.json" 2>/dev/null || true
# LM tokenization boundary-crossing stats (written by wrappers/base_eval.py)
cp "$NANOCHAT_BASE_DIR/base_eval/lm_boundary_stats.json" "$ARCHIVE_DIR/" 2>/dev/null || true
git -C ../nanochat rev-parse HEAD > "$ARCHIVE_DIR/NANOCHAT_COMMIT" 2>/dev/null || true
{
    echo "MODEL_TAG=$MODEL_TAG"
    echo "OVERLAY=${OVERLAY:-}"
    echo "WANDB_RUN=$WANDB_RUN"
    echo "DEPTH=$DEPTH"
    echo "NPROC=$NPROC"
    echo "USE_FP8=$USE_FP8"
    echo "WINDOW_PATTERN=$WINDOW_PATTERN"
    echo "DEVICE_BATCH_SIZE=$DEVICE_BATCH_SIZE"
    echo "USE_SFT=$USE_SFT"
    [ -n "${Z_LOSS_COEFF:-}" ] && echo "Z_LOSS_COEFF=$Z_LOSS_COEFF"
    echo "TIMESTAMP=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} > "$ARCHIVE_DIR/ENV"
echo "==> archived base-stage artifacts to $ARCHIVE_DIR/"

# -----------------------------------------------------------------------------
# SFT + chat_eval (opt-in via USE_SFT=1 — see header for why default is off)
if [ "$USE_SFT" = "1" ]; then
    curl -L -o "$NANOCHAT_BASE_DIR/identity_conversations.jsonl" \
        https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl

    torchrun --standalone --nproc_per_node="$NPROC" -m scripts.chat_sft -- \
        --model-tag="$MODEL_TAG" --device-batch-size="$DEVICE_BATCH_SIZE" --run="$WANDB_RUN"
    torchrun --standalone --nproc_per_node="$NPROC" -m scripts.chat_eval -- \
        --model-tag="$MODEL_TAG" -i sft
else
    echo "==> skipping chat_sft + chat_eval (USE_SFT=0 — set USE_SFT=1 to run them)" >&2
fi

# -----------------------------------------------------------------------------
# Final report (markdown). Guarded — nanochat.report removed upstream (see above);
# eval.csv + meta.json are archived independently, so losing the markdown is fine.
if python -c "import nanochat.report" 2>/dev/null; then python -m nanochat.report generate; else echo "==> nanochat.report absent — skipping markdown report (eval.csv + meta already archived)"; fi

echo
echo "==> $RUN_TAG run complete (model tag: $MODEL_TAG). Report: $NANOCHAT_BASE_DIR/report/"
