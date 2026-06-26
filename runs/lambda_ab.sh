#!/bin/bash
# runs/lambda_ab.sh — one-shot, IDEMPOTENT orchestrator for the surface-factoring
# d24 A/B on Lambda (8x A100 40GB by default).
#
# Pipeline (each stage is resumable; re-run any time):
#   provision  → reuse the live instance from state, else launch one
#   bootstrap  → lambda.sh bootstrap (setup.sh: uv sync, clone nanochat, wandb)
#                — no Rust/crate: surface-only is crate-free; tokenizers are pre-baked
#   sync       → rsync the cached force_merges + surface_only TOKENIZERS up
#                (they live outside sandbox/, so lambda.sh bootstrap misses them)
#                + re-rsync sandbox/ so code fixes propagate on every retrigger
#   verify     → preflight on the host (8 GPUs, venv, tokenizers, surface smoke;
#                crate-free) BEFORE the expensive run
#   train      → three arms in one tmux session, sequential, each SKIPPED if its
#                results/<tag>/eval.csv already exists:
#                  A) surface-only = OVERLAY=surface_factoring + base_eval_surface
#                  B) force_merges sibling (reference, NOT the baseline)
#                  C) vanilla baseline = the control for the factoring-isolation test
#   poll       → wait for the host's ~/sandbox/.ab_done marker (tails progress)
#   download   → lambda.sh pull (results/ + report) + rsync the CHECKPOINTS
#                (models) + the per-arm logs down
#   verifydl   → assert both arms' eval.csv + checkpoint + log are local; print
#                the two CORE numbers side by side
#   terminate  → terminate the instance (prompts unless AUTO_TERMINATE=1)
#
# Idempotency: a live instance + completed stages + finished arms are detected
# and skipped. Fix code → git push (or just re-run, sync re-rsyncs) → re-run this
# script; only unfinished work re-executes. The instance is NEVER terminated
# until verifydl passes.
#
# Usage:
#   bash runs/lambda_ab.sh                  # full pipeline (resumable)
#   STAGE=download bash runs/lambda_ab.sh   # run a single stage
#   AUTO_TERMINATE=1 bash runs/lambda_ab.sh # don't prompt before terminate
#   YES=1 bash runs/lambda_ab.sh            # don't prompt before provision either
#
# Env: ~/.lambda.env must provide LAMBDA_API_KEY, LAMBDA_SSH_KEY, WANDB_API_KEY
#   LAMBDA_INSTANCE_TYPE  default gpu_8x_a100_sxm4 (8x A100 40GB SXM4)
#   DEVICE_BATCH_SIZE     default 8 (A100 40GB; see runs/README.md hardware table)
#
# NOTE: provisioning bills ~$16/hr (A100 40GB); a 3-arm d24 A/B is ~20 GPU-hrs
# ≈ ~$320 + overhead. This script will not terminate before artifacts are verified
# local, so a crash leaves the instance up (resume, don't re-provision).
#
# Lambda 8x A100/H100 capacity is frequently UNAVAILABLE (mid-2026) — prefer
# runs/runpod_ab.sh (the maintained, crate-free, 3-arm surface-only orchestrator);
# this script is kept aligned but secondary.

set -euo pipefail

SANDBOX_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SANDBOX_DIR"
LAMBDA="$SANDBOX_DIR/runs/lambda.sh"
STATE_FILE="$SANDBOX_DIR/runs/.lambda_ab.state"
SSH_OPTS="-o StrictHostKeyChecking=accept-new"
SSH_USER="${LAMBDA_SSH_USER:-ubuntu}"

export LAMBDA_INSTANCE_TYPE="${LAMBDA_INSTANCE_TYPE:-gpu_8x_a100_sxm4}"
DEVICE_BATCH_SIZE="${DEVICE_BATCH_SIZE:-8}"

# --- A/B arm config ---------------------------------------------------------
# Surface-only factoring A/B: the DECISIVE test is surface-only (arm A) vs the vanilla
# baseline (arm C); force_merges (arm B) is an optional SIBLING reference, NOT stacked.
SURF_TAG="d24_surface_only_ab"          # arm A: surface-only + overlay (the test)
BASE_TAG="d24_force_merges_ab"          # arm B: force_merges sibling (reference)
BASELINE_TAG="d24_baseline_ab"          # arm C: vanilla baseline = the control for arm A
FM_VARIANT="force_merges"               # local dir under ~/.cache/nanochat-variants/
SURF_VARIANT="surface_only"
HW_ENV="USE_FP8=0 WINDOW_PATTERN=L DEVICE_BATCH_SIZE=$DEVICE_BATCH_SIZE"

log()  { echo "==> $*" >&2; }
die()  { echo "ERROR: $*" >&2; exit 1; }
confirm() {  # confirm "msg" ; honors YES=1
    [ "${YES:-0}" = "1" ] && return 0
    read -r -p "$1 [y/N] " a < /dev/tty; [ "$a" = "y" ] || [ "$a" = "Y" ]
}

# --- state ------------------------------------------------------------------
INSTANCE_ID=""; INSTANCE_IP=""
_load_state() { [ -f "$STATE_FILE" ] && . "$STATE_FILE" || true; }
_save_state() { printf 'INSTANCE_ID=%q\nINSTANCE_IP=%q\n' "$INSTANCE_ID" "$INSTANCE_IP" > "$STATE_FILE"; }

_ssh()  { ssh $SSH_OPTS "$SSH_USER@$INSTANCE_IP" "$@"; }
_have() { [ -n "$INSTANCE_ID" ] && [ -n "$INSTANCE_IP" ]; }

_refresh_ip() {  # re-resolve IP from the API (survives reboots / stale state)
    local ip; ip=$(bash "$LAMBDA" status "$INSTANCE_ID" 2>/dev/null | jq -r '.ip // ""')
    [ -n "$ip" ] && INSTANCE_IP="$ip" && _save_state || true
}

# ----------------------------------------------------------------------------
stage_provision() {
    _load_state
    if _have; then
        local st; st=$(bash "$LAMBDA" status "$INSTANCE_ID" 2>/dev/null | jq -r '.status // "missing"')
        if [ "$st" = "active" ]; then log "reusing live instance $INSTANCE_ID ($INSTANCE_IP)"; _refresh_ip; return 0; fi
        log "state instance $INSTANCE_ID is '$st', not active — launching a new one"
    fi
    log "instance type: $LAMBDA_INSTANCE_TYPE — checking capacity"
    bash "$LAMBDA" types --available | jq -r '.type' | grep -qx "$LAMBDA_INSTANCE_TYPE" \
        || { bash "$LAMBDA" types --available >&2; die "no capacity for $LAMBDA_INSTANCE_TYPE (available types above; set LAMBDA_INSTANCE_TYPE)"; }
    confirm "Launch $LAMBDA_INSTANCE_TYPE (~\$16/hr, ~\$320+ for the 3-arm d24 A/B)?" || die "aborted"
    read -r INSTANCE_ID INSTANCE_IP < <(bash "$LAMBDA" launch | tail -1)
    [ -n "$INSTANCE_ID" ] || die "launch failed"
    _save_state
    log "launched $INSTANCE_ID ($INSTANCE_IP)"
}

stage_bootstrap() {
    _load_state; _have || die "no instance (run provision)"
    _refresh_ip
    if _ssh "test -f ~/.ab_setup_done" 2>/dev/null; then
        log "host already bootstrapped (~/.ab_setup_done present) — skipping setup.sh"
        return 0
    fi
    log "bootstrapping host (rsync sandbox + setup.sh + wandb login) — no Rust/crate (surface-only is crate-free)"
    bash "$LAMBDA" bootstrap "$INSTANCE_ID"
    _ssh "touch ~/.ab_setup_done"
    log "bootstrap complete"
}

stage_sync() {
    _load_state; _have || die "no instance (run provision)"
    _refresh_ip
    # re-rsync sandbox so code fixes propagate on every retrigger (cheap; mirrors
    # lambda.sh's excludes). Does NOT touch the host venv / nanochat clone.
    log "re-rsync sandbox/ (propagate code fixes)"
    rsync -a --delete --exclude=.venv --exclude=__pycache__ --exclude=results \
        --exclude='*.pyc' --exclude=target --exclude='runs/.lambda_ab.state' \
        -e "ssh $SSH_OPTS" "$SANDBOX_DIR/" "$SSH_USER@$INSTANCE_IP:~/sandbox/"
    # rsync the two cached TOKENIZERS into their variant base dirs on the host
    # (they live outside sandbox/, so bootstrap misses them; the speedrun arms
    # skip tok_train when $VBASE/tokenizer/tokenizer.pkl is present).
    for v in "$FM_VARIANT" "$SURF_VARIANT"; do
        local src="$HOME/.cache/nanochat-variants/$v/tokenizer"
        [ -f "$src/tokenizer.pkl" ] || die "local tokenizer missing: $src (bake it first)"
        log "rsync tokenizer $v → host"
        _ssh "mkdir -p ~/.cache/nanochat-variants/$v/tokenizer"
        rsync -a -e "ssh $SSH_OPTS" "$src/" "$SSH_USER@$INSTANCE_IP:~/.cache/nanochat-variants/$v/tokenizer/"
    done
}

stage_verify() {
    _load_state; _have || die "no instance (run provision)"
    _refresh_ip
    log "preflight on host (GPUs / venv / tokenizers / crate / surface smoke)"
    _ssh 'set -e; export PATH="$HOME/.local/bin:$PATH"; cd ~/sandbox
        n=$(nvidia-smi -L | wc -l); echo "GPUs: $n"; [ "$n" -ge 8 ] || { echo "FAIL: <8 GPUs"; exit 1; }
        test -d .venv || { echo "FAIL: no venv"; exit 1; }
        for v in force_merges surface_only; do
            test -f ~/.cache/nanochat-variants/$v/tokenizer/tokenizer.pkl || { echo "FAIL: $v tokenizer missing"; exit 1; }
            test -f ~/.cache/nanochat-variants/$v/tokenizer/token_bytes.pt || { echo "FAIL: $v token_bytes.pt missing"; exit 1; }
        done
        echo "--- surface smoke (crate-free) ---"; uv run python -m wrappers.smoke_surface_factoring 2>&1 | tail -1
        echo "PREFLIGHT OK"'
    log "preflight passed"
}

_write_remote_train() {
    # Write the remote train script on the host. Single-quoted heredoc → $HOME and
    # the env vars are evaluated ON THE HOST, not locally. Each arm is skipped if
    # its results/<tag>/eval.csv exists (idempotent); set -e stops before .ab_done
    # if an arm fails.
    _ssh "cat > ~/sandbox/runs/.ab_train.sh" <<REMOTE
#!/bin/bash
set -eo pipefail
cd ~/sandbox
export PATH="\$HOME/.local/bin:\$PATH"

# --- arm A: surface-only (the test arm) ---
if [ -f results/$SURF_TAG/eval.csv ]; then
  echo "[ab] arm A ($SURF_TAG) already done — skipping"
else
  echo "[ab] arm A: surface-only"
  NANOCHAT_BASE_DIR="\$HOME/.cache/nanochat-variants/$SURF_VARIANT" \\
    OVERLAY=surface_factoring EVAL_MODULE=wrappers.base_eval_surface EVAL_ARGS="--eval core,bpb" $HW_ENV \\
    MODEL_TAG=$SURF_TAG WANDB_RUN=$SURF_TAG \\
    bash runs/speedrun.sh 2>&1 | tee runs/$SURF_TAG.log
fi

# --- arm B: force_merges sibling (reference, NOT the baseline) ---
if [ -f results/$BASE_TAG/eval.csv ]; then
  echo "[ab] arm B ($BASE_TAG) already done — skipping"
else
  echo "[ab] arm B: force_merges sibling"
  NANOCHAT_BASE_DIR="\$HOME/.cache/nanochat-variants/$FM_VARIANT" EVAL_ARGS="--eval core,bpb" $HW_ENV \\
    MODEL_TAG=$BASE_TAG WANDB_RUN=$BASE_TAG \\
    bash runs/speedrun.sh 2>&1 | tee runs/$BASE_TAG.log
fi

# --- arm C: vanilla baseline (the control for arm A) ---
if [ -f results/$BASELINE_TAG/eval.csv ]; then
  echo "[ab] arm C ($BASELINE_TAG) already done — skipping"
else
  echo "[ab] arm C: vanilla baseline"
  EVAL_ARGS="--eval core,bpb" $HW_ENV \\
    MODEL_TAG=$BASELINE_TAG WANDB_RUN=$BASELINE_TAG \\
    bash runs/speedrun.sh 2>&1 | tee runs/$BASELINE_TAG.log
fi

touch ~/sandbox/.ab_done
echo "[ab] ALL 3 ARMS DONE"
REMOTE
}

stage_train() {
    _load_state; _have || die "no instance (run provision)"
    _refresh_ip
    if _ssh "test -f ~/sandbox/.ab_done" 2>/dev/null; then log "~/sandbox/.ab_done present — training already complete"; return 0; fi
    if _ssh "tmux has-session -t ab" 2>/dev/null; then log "tmux session 'ab' already running — leaving it (poll to watch)"; return 0; fi
    _write_remote_train
    log "launching both arms in tmux session 'ab' (detached)"
    _ssh "tmux new-session -d -s ab 'bash ~/sandbox/runs/.ab_train.sh 2>&1 | tee ~/sandbox/runs/ab_orchestrator.log'"
    log "training launched. watch: bash runs/lambda.sh ssh $INSTANCE_ID  then  tmux attach -t ab"
}

stage_poll() {
    _load_state; _have || die "no instance (run provision)"
    log "polling for ~/sandbox/.ab_done (checks every 5 min; Ctrl-C is safe — re-run to resume)"
    while true; do
        _refresh_ip
        if _ssh "test -f ~/sandbox/.ab_done" 2>/dev/null; then log "training complete (.ab_done present)"; return 0; fi
        if ! _ssh "tmux has-session -t ab" 2>/dev/null; then
            log "tmux 'ab' gone but no .ab_done — an arm likely FAILED. Last log lines:"
            _ssh "tail -n 25 ~/sandbox/runs/ab_orchestrator.log 2>/dev/null" >&2 || true
            die "training session ended without completion — ssh in, fix, and re-run (stage train resumes the unfinished arm)"
        fi
        local prog; prog=$(_ssh "grep -hoE 'step [0-9]+/[0-9]+' ~/sandbox/runs/${SURF_TAG}.log ~/sandbox/runs/${BASE_TAG}.log ~/sandbox/runs/${BASELINE_TAG}.log 2>/dev/null | tail -1" || true)
        log "still training${prog:+ — $prog} ($(date -u +%H:%M:%SZ))"
        sleep 300
    done
}

stage_download() {
    _load_state; _have || die "no instance (run provision)"
    _refresh_ip
    log "pulling results/ + report (lambda.sh pull)"
    bash "$LAMBDA" pull "$INSTANCE_ID"
    # checkpoints (the models) + per-arm logs — lambda.sh pull doesn't cover the
    # cache dirs. Pull each arm's checkpoint dir into results/<tag>/checkpoint/.
    for entry in \
        "~/.cache/nanochat-variants/$SURF_VARIANT|$SURF_TAG" \
        "~/.cache/nanochat-variants/$FM_VARIANT|$BASE_TAG" \
        "~/.cache/nanochat|$BASELINE_TAG"; do
        local bdir="${entry%%|*}" tag="${entry##*|}"
        local rmt="$bdir/base_checkpoints/$tag/"
        local dst="$SANDBOX_DIR/results/$tag/checkpoint/"
        if _ssh "test -d $rmt" 2>/dev/null; then
            mkdir -p "$dst"; log "rsync checkpoint $tag → results/$tag/checkpoint/"
            rsync -a -e "ssh $SSH_OPTS" "$SSH_USER@$INSTANCE_IP:$rmt" "$dst"
        else
            log "no checkpoint dir for $tag on host (arm finished? base_train only saves at the end)"
        fi
        rsync -a -e "ssh $SSH_OPTS" "$SSH_USER@$INSTANCE_IP:~/sandbox/runs/$tag.log" "$SANDBOX_DIR/results/$tag/" 2>/dev/null || true
    done
}

stage_verifydl() {
    local ok=1
    echo "" >&2; log "verifying local artifacts"
    for tag in "$SURF_TAG" "$BASE_TAG" "$BASELINE_TAG"; do
        local d="$SANDBOX_DIR/results/$tag"
        local csv="$d/eval.csv" ckpt; ckpt=$(ls "$d"/checkpoint/model_*.pt 2>/dev/null | head -1 || true)
        local core; core=$(grep -hoE "CORE metric: [0-9.]+|core_metric[\",: ]+[0-9.]+" "$d"/*.md "$csv" 2>/dev/null | grep -oE "[0-9]+\.[0-9]+" | head -1 || true)
        printf '  %-26s eval.csv:%s  checkpoint:%s  CORE:%s\n' "$tag" \
            "$([ -f "$csv" ] && echo yes || echo NO)" \
            "$([ -n "$ckpt" ] && echo yes || echo NO)" \
            "${core:-?}" >&2
        { [ -f "$csv" ] && [ -n "$ckpt" ]; } || ok=0
    done
    [ "$ok" = "1" ] || { log "artifacts INCOMPLETE — NOT terminating. Re-run download (or fix the failed arm)."; return 1; }
    log "all artifacts local. Decisive A/B: results/$SURF_TAG (surface-only) vs results/$BASELINE_TAG (vanilla); $BASE_TAG is the force_merges sibling."
    return 0
}

stage_terminate() {
    _load_state; _have || { log "no instance in state — nothing to terminate"; return 0; }
    stage_verifydl || die "refusing to terminate: artifacts not verified local (resume download/train first)"
    if [ "${AUTO_TERMINATE:-0}" != "1" ]; then
        confirm "Artifacts verified local. Terminate $INSTANCE_ID (stops billing)?" || { log "left instance $INSTANCE_ID running"; return 0; }
    fi
    bash "$LAMBDA" terminate "$INSTANCE_ID"
    : > "$STATE_FILE"
    log "terminated $INSTANCE_ID and cleared state. Done."
}

# --- dispatch ---------------------------------------------------------------
ALL_STAGES="provision bootstrap sync verify train poll download verifydl terminate"
if [ -n "${STAGE:-}" ]; then
    "stage_$STAGE"
else
    for s in $ALL_STAGES; do
        log "=== STAGE: $s ==="
        "stage_$s"
    done
fi
