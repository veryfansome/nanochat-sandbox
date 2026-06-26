#!/bin/bash
# CPU / macOS MECHANISM check for the surface-factoring harness. Mirrors
# runs/runcpu.sh, but (a) trains the surface-only tokenizer instead of the
# stock one, (b) routes training through wrappers.train_surface_factoring, and
# (c) SKIPS base_eval (CORE) — a surface-aware base_eval is still TODO, and the
# saved checkpoint carries surface params that won't load into a vanilla GPT.
#
# This is NOT a scientific A/B (see runs/runcpu.sh header) — it only confirms the
# surface overlay + dual-stream dataloader + joint loss + surface bpb survive a
# few thousand real training iterations. For the A/B, use a GPU speedrun once the
# surface-aware eval/checkpoint-load path lands.
#
# Usage:   bash runs/runcpu_surface.sh
#   reproducibility A/B (new seed + new tag):
#     SURFACE_SEED=1 MODEL_TAG=d6_surface_factoring_s1 WANDB_RUN=d6_surface_factoring_s1 bash runs/runcpu_surface.sh
# Env:     SURFACE_LAMBDA (default 0.5), NUM_ITERS (default 5000),
#          SURFACE_SEED (override model-init seed; default nanochat's 42),
#          MODEL_TAG / WANDB_RUN (default d6_surface_factoring; set both for a
#          distinct run that won't clobber the prior checkpoint/results/wandb),
#          NANOCHAT_BASE_DIR (default ~/.cache/nanochat), MAX_CHARS (tok train)

set -euo pipefail

SANDBOX_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SANDBOX_DIR"
export PATH="$HOME/.local/bin:$PATH"
# shellcheck disable=SC1091
source .venv/bin/activate

export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat}"
mkdir -p "$NANOCHAT_BASE_DIR"
VARIANT_BASE="$HOME/.cache/nanochat-variants/surface_only"
export WANDB_DIR="$VARIANT_BASE"
MODEL_TAG="${MODEL_TAG:-d6_surface_factoring}"
# Real wandb run name by default (so the val/bpb curve is captured + comparable
# via tools/compare_runs.py). Set WANDB_RUN=dummy to disable; WANDB_SUFFIX=1
# appends a UTC timestamp so reruns don't collide.
WANDB_RUN="${WANDB_RUN:-d6_surface_factoring}"
if [ "$WANDB_RUN" != "dummy" ] && [ "${WANDB_SUFFIX:-0}" = "1" ]; then
    WANDB_RUN="${WANDB_RUN}_$(date -u +%Y%m%d_%H%M)"
fi
NUM_ITERS="${NUM_ITERS:-5000}"
export SURFACE_LAMBDA="${SURFACE_LAMBDA:-0.5}"
ARCHIVE_DIR="$SANDBOX_DIR/results/$MODEL_TAG"
mkdir -p "$ARCHIVE_DIR"

echo "==> 1/3 ensure pretraining data (downloads to $NANOCHAT_BASE_DIR)"
python -m nanochat.dataset -n 8

echo "==> 2/3 train the surface-only tokenizer -> $VARIANT_BASE/tokenizer/"
SURF_TOK="$VARIANT_BASE/tokenizer/tokenizer.pkl"
if [ -f "$SURF_TOK" ] && [ "${FORCE_RETRAIN_TOKENIZER:-0}" != "1" ]; then
    echo "    cached at $SURF_TOK (set FORCE_RETRAIN_TOKENIZER=1 to redo)"
else
    python -m wrappers.tok_train_surface --max-chars="${MAX_CHARS:-2000000000}"
fi

echo "==> 3/3 surface-factoring training (depth 6, $NUM_ITERS iters, λ=$SURFACE_LAMBDA, seed=${SURFACE_SEED:-42}, tag=$MODEL_TAG, wandb=$WANDB_RUN)"
# Run with NANOCHAT_BASE_DIR pointed at the variant base so get_tokenizer /
# get_token_bytes resolve to the surface tokenizer; data is symlinked there by
# setup_variant_base during tok_train_surface. tee stdout so the val/bpb curve is
# durable even if wandb is off / the run dies mid-way.
# python -u: unbuffered stdout, so the `| tee` below streams live instead of
# block-buffering the training prints into chunks.
NANOCHAT_BASE_DIR="$VARIANT_BASE" python -u -m wrappers.train_surface_factoring \
    --depth=6 \
    --head-dim=64 \
    --window-pattern=L \
    --max-seq-len=512 \
    --device-batch-size=32 \
    --total-batch-size=16384 \
    --eval-every=100 \
    --eval-tokens=524288 \
    --core-metric-every=-1 \
    --sample-every=-1 \
    --num-iterations="$NUM_ITERS" \
    --model-tag="$MODEL_TAG" \
    --run="$WANDB_RUN" 2>&1 | tee "$ARCHIVE_DIR/train.log"

# Archive durable artifacts alongside the tee'd log (base_eval/CORE is skipped, so
# this + wandb are the record). Best-effort copies; the checkpoint stays in place.
grep -E "Validation bpb|Minimum validation bpb|CORE metric" "$ARCHIVE_DIR/train.log" \
    > "$ARCHIVE_DIR/bpb_trajectory.txt" 2>/dev/null || true
cp "$VARIANT_BASE/report/"*.md "$ARCHIVE_DIR/" 2>/dev/null || true
cp "$VARIANT_BASE/base_checkpoints/$MODEL_TAG/meta_"*.json "$ARCHIVE_DIR/meta.json" 2>/dev/null || true
git -C ../nanochat rev-parse HEAD > "$ARCHIVE_DIR/NANOCHAT_COMMIT" 2>/dev/null || true
{
    echo "MODEL_TAG=$MODEL_TAG"; echo "WANDB_RUN=$WANDB_RUN"
    echo "SURFACE_LAMBDA=$SURFACE_LAMBDA"; echo "NUM_ITERS=$NUM_ITERS"
    echo "TIMESTAMP=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} > "$ARCHIVE_DIR/ENV"

echo
echo "==> surface runcpu complete (model tag: $MODEL_TAG)."
echo "    artifacts: $ARCHIVE_DIR/{train.log, bpb_trajectory.txt, *.md report, meta.json}"
echo "    checkpoint kept: $VARIANT_BASE/base_checkpoints/$MODEL_TAG/"
echo "    (base_eval/CORE skipped — needs the surface-aware eval + checkpoint-load path.)"
