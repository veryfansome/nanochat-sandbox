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
#   WANDB_RUN   wandb run name (default: dummy = disabled)
#   NANOCHAT_BASE_DIR  artifact dir (default: ~/.cache/nanochat)
#   FORCE_RETRAIN_TOKENIZER  set to 1 to retrain tokenizer even if cached (default: 0)

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

# Overlay selection: empty => baseline (upstream scripts.base_train)
OVERLAY="${OVERLAY:-}"
if [ -n "$OVERLAY" ]; then
    TRAIN_MODULE="wrappers.train_$OVERLAY"
    RUN_TAG="${OVERLAY}"
else
    TRAIN_MODULE="scripts.base_train"
    RUN_TAG="baseline"
fi

NPROC="${NPROC:-8}"
DEPTH="${DEPTH:-24}"
MODEL_TAG="${MODEL_TAG:-d${DEPTH}_${OVERLAY:-baseline}}"
WANDB_RUN="${WANDB_RUN:-dummy}"

echo "==> overlay:      ${OVERLAY:-(baseline)}"
echo "==> train module: $TRAIN_MODULE"
echo "==> model tag:    $MODEL_TAG"
echo "==> nproc:        $NPROC"
echo "==> depth:        $DEPTH"
echo "==> wandb run:    $WANDB_RUN"
echo "==> base dir:     $NANOCHAT_BASE_DIR"
echo "==> nanochat:     $(git -C ../nanochat rev-parse --short HEAD)"
echo

# -----------------------------------------------------------------------------
# Init the per-run report
python -m nanochat.report reset

# -----------------------------------------------------------------------------
# Tokenizer + data
python -m nanochat.dataset -n 8
python -m nanochat.dataset -n 170 &
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

# Mirrors upstream: --depth=24 --target-param-data-ratio=8 --device-batch-size=16 --fp8
# The only swap is $TRAIN_MODULE in place of scripts.base_train.
torchrun --standalone --nproc_per_node="$NPROC" -m "$TRAIN_MODULE" -- \
    --depth="$DEPTH" --target-param-data-ratio=8 --device-batch-size=16 --fp8 \
    --model-tag="$MODEL_TAG" --run="$WANDB_RUN"

torchrun --standalone --nproc_per_node="$NPROC" -m scripts.base_eval -- \
    --model-tag="$MODEL_TAG" --device-batch-size=16

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
git -C ../nanochat rev-parse HEAD > "$ARCHIVE_DIR/NANOCHAT_COMMIT" 2>/dev/null || true
{
    echo "MODEL_TAG=$MODEL_TAG"
    echo "OVERLAY=${OVERLAY:-}"
    echo "WANDB_RUN=$WANDB_RUN"
    echo "DEPTH=$DEPTH"
    echo "NPROC=$NPROC"
    [ -n "${Z_LOSS_COEFF:-}" ] && echo "Z_LOSS_COEFF=$Z_LOSS_COEFF"
    echo "TIMESTAMP=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} > "$ARCHIVE_DIR/ENV"
echo "==> archived base-stage artifacts to $ARCHIVE_DIR/"

# -----------------------------------------------------------------------------
# SFT
curl -L -o "$NANOCHAT_BASE_DIR/identity_conversations.jsonl" \
    https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl

torchrun --standalone --nproc_per_node="$NPROC" -m scripts.chat_sft -- \
    --model-tag="$MODEL_TAG" --device-batch-size=16 --run="$WANDB_RUN"
torchrun --standalone --nproc_per_node="$NPROC" -m scripts.chat_eval -- \
    --model-tag="$MODEL_TAG" -i sft

# -----------------------------------------------------------------------------
# Final report (markdown)
python -m nanochat.report generate

echo
echo "==> $RUN_TAG run complete (model tag: $MODEL_TAG). Report: $NANOCHAT_BASE_DIR/report/"
