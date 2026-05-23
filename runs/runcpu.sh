#!/bin/bash
# CPU / macOS smoke run with an optional overlay. Mirrors
# ../../nanochat/runs/runcpu.sh exactly — only the train module is swapped.
#
# This is not a real comparison run — it's the cheap end-to-end check that
# the overlay survives a few thousand actual training iterations on real data.
# For a meaningful A/B, use runs/speedrun.sh on a GPU node.
#
# Usage:
#     OVERLAY=zloss bash runs/runcpu.sh
#
# Env vars:
#   OVERLAY     idea name; empty (default) = upstream baseline (scripts.base_train)
#   MODEL_TAG   checkpoint dir name (default: d6_${OVERLAY:-baseline}); ensures
#               baseline and overlay A/B runs don't overwrite each other
#   WANDB_RUN   wandb run name (default: dummy = disabled)
#   NANOCHAT_BASE_DIR  artifact dir (default: ~/.cache/nanochat)
#   FORCE_RETRAIN_TOKENIZER  set to 1 to retrain tokenizer even if cached (default: 0)
#   KEEP_CHECKPOINT  set to 1 to preserve the trained checkpoint (default: 0 = cleanup
#                    after base_eval; wandb has the curves and the report has metrics,
#                    so the ~500 MB checkpoint is wasted disk for a CPU sanity run)

set -euo pipefail

SANDBOX_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SANDBOX_DIR"
export PATH="$HOME/.local/bin:$PATH"
# shellcheck disable=SC1091
source .venv/bin/activate

export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat}"
mkdir -p "$NANOCHAT_BASE_DIR"

# Keep wandb's local cache alongside other generated artifacts (data shards,
# tokenizer, checkpoints, reports) under NANOCHAT_BASE_DIR rather than dumping
# a `wandb/` dir into sandbox/. wandb creates `$WANDB_DIR/wandb/` on init.
export WANDB_DIR="$NANOCHAT_BASE_DIR"

OVERLAY="${OVERLAY:-}"
if [ -n "$OVERLAY" ]; then
    TRAIN_MODULE="wrappers.train_$OVERLAY"
    RUN_TAG="${OVERLAY}"
else
    TRAIN_MODULE="scripts.base_train"
    RUN_TAG="baseline"
fi
MODEL_TAG="${MODEL_TAG:-d6_${OVERLAY:-baseline}}"
WANDB_RUN="${WANDB_RUN:-dummy}"

echo "==> overlay:      ${OVERLAY:-(baseline)}"
echo "==> train module: $TRAIN_MODULE"
echo "==> model tag:    $MODEL_TAG"

# Tokenizer + data (small)
python -m nanochat.dataset -n 8

TOKENIZER_FILE="$NANOCHAT_BASE_DIR/tokenizer/tokenizer.pkl"  # RustBPETokenizer save format (default)
if [ -f "$TOKENIZER_FILE" ] && [ "${FORCE_RETRAIN_TOKENIZER:-0}" != "1" ]; then
    echo "==> tokenizer cached at $TOKENIZER_FILE; skipping tok_train + tok_eval"
    echo "    (set FORCE_RETRAIN_TOKENIZER=1 to redo)"
else
    python -m scripts.tok_train --max-chars=2000000000
    python -m scripts.tok_eval
fi

# Tuned by upstream for ~30 min on an M3 Max. Single process (no torchrun).
python -m "$TRAIN_MODULE" \
    --depth=6 \
    --head-dim=64 \
    --window-pattern=L \
    --max-seq-len=512 \
    --device-batch-size=32 \
    --total-batch-size=16384 \
    --eval-every=100 \
    --eval-tokens=524288 \
    --core-metric-every=-1 \
    --sample-every=100 \
    --num-iterations=5000 \
    --model-tag="$MODEL_TAG" \
    --run="$WANDB_RUN"

python -m scripts.base_eval --device-batch-size=1 --split-tokens=16384 --max-per-task=16 --model-tag="$MODEL_TAG"

# Archive run artifacts to sandbox/results/$MODEL_TAG/ BEFORE the next run can
# wipe them. The eval CSV path doesn't include $MODEL_TAG (upstream writes
# base_model_<step>.csv), and `nanochat.report reset` clears report/ at the
# start of every run — wandb is otherwise the only durable record. Cheap (a
# few KB of CSV + markdown + a small meta.json), no env-var gate needed.
ARCHIVE_DIR="$SANDBOX_DIR/results/$MODEL_TAG"
mkdir -p "$ARCHIVE_DIR"
EVAL_CSV=$(ls -t "$NANOCHAT_BASE_DIR/base_eval/base_model_"*.csv 2>/dev/null | head -1)
[ -n "$EVAL_CSV" ] && cp "$EVAL_CSV" "$ARCHIVE_DIR/eval.csv"
cp "$NANOCHAT_BASE_DIR/report/base-model-"*.md "$ARCHIVE_DIR/" 2>/dev/null || true
cp "$NANOCHAT_BASE_DIR/base_checkpoints/$MODEL_TAG/meta_"*.json "$ARCHIVE_DIR/meta.json" 2>/dev/null || true
git -C ../nanochat rev-parse HEAD > "$ARCHIVE_DIR/NANOCHAT_COMMIT" 2>/dev/null || true
{
    echo "MODEL_TAG=$MODEL_TAG"
    echo "OVERLAY=${OVERLAY:-}"
    echo "WANDB_RUN=$WANDB_RUN"
    [ -n "${Z_LOSS_COEFF:-}" ] && echo "Z_LOSS_COEFF=$Z_LOSS_COEFF"
    echo "TIMESTAMP=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} > "$ARCHIVE_DIR/ENV"
echo "==> archived run artifacts to $ARCHIVE_DIR/"

# Cleanup: runcpu is a sanity / mechanism run, not something you'd do downstream
# work on. Wandb has the training curves; the archive above keeps the report
# and eval CSV. Keep the ~500 MB checkpoint only if explicitly asked to.
CKPT_DIR="$NANOCHAT_BASE_DIR/base_checkpoints/$MODEL_TAG"
if [ "${KEEP_CHECKPOINT:-0}" != "1" ] && [ -d "$CKPT_DIR" ]; then
    CKPT_SIZE=$(du -sh "$CKPT_DIR" 2>/dev/null | cut -f1)
    echo "==> removing checkpoint dir: $CKPT_DIR ($CKPT_SIZE)"
    echo "    (set KEEP_CHECKPOINT=1 to preserve for resume / chat / further eval)"
    rm -rf "$CKPT_DIR"
fi

echo
echo "==> $RUN_TAG runcpu complete (model tag: $MODEL_TAG)."
