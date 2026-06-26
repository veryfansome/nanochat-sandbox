#!/bin/bash
# runs/runpod_ab.sh — IDEMPOTENT orchestrator for the surface-factoring d24 A/B on
# RunPod (8x H100 SXM). Same design as runs/lambda_ab.sh, RunPod-native: a pod is
# a container reached as root over an exposed SSH port, and NO Rust/crate build is
# needed (the surface train/eval path is crate-free; we rsync pre-baked tokenizers).
#
# Stages (resumable; re-run any time):
#   provision  → reuse the live pod from state, else deploy 8x H100
#   bootstrap  → runpod.sh bootstrap (apt rsync/git + rsync sandbox + setup.sh + wandb)
#   sync       → rsync the cached force_merges + surface_only TOKENIZERS up
#                + re-rsync sandbox/ so code fixes propagate each retrigger
#   verify     → crate-free preflight (8 GPUs, venv, both tokenizers load + a
#                surface encode/decode round-trip, overlay imports)
#   train      → both arms in one tmux session, each SKIPPED if results/<tag>/eval.csv
#                exists: A) surface-only (OVERLAY + base_eval_surface)  B) force_merges sibling  C) vanilla baseline
#   poll       → wait for ~/sandbox/.ab_done (tails progress)
#   download   → runpod.sh pull (results/+report) + rsync CHECKPOINTS + per-arm logs
#   verifydl   → assert both arms' eval.csv + checkpoint local; print the CORE numbers
#   terminate  → terminate the pod (prompts unless AUTO_TERMINATE=1)
#
# Never terminates before verifydl passes. Fix code → re-run; only unfinished work
# re-executes (live pod + host markers + per-arm eval.csv are detected & skipped).
#
# Usage:
#   bash runs/runpod_ab.sh                  # full pipeline (run under local tmux — poll blocks ~hrs)
#   STAGE=download bash runs/runpod_ab.sh   # one stage: provision|bootstrap|sync|verify|train|poll|download|verifydl|terminate
#   AUTO_TERMINATE=1 YES=1 bash runs/runpod_ab.sh
#
# Env: ~/.lambda.env needs RUNPOD_API_KEY + WANDB_API_KEY (+ an ssh key — see runpod.sh).
#   HW_ENV  default "USE_FP8=0 WINDOW_PATTERN=L DEVICE_BATCH_SIZE=16" — the SAFE H100
#           config (no FA3/fp8 dependence). For ~25% speedup try
#           HW_ENV="USE_FP8=1 WINDOW_PATTERN=SSSL DEVICE_BATCH_SIZE=16" IF FA3 is
#           present AND fp8 plays nice with the surface heads — but both arms must
#           share it, and changing it after one arm finished means clearing BOTH
#           results/<tag>/ and redoing (else the A/B isn't apples-to-apples).

set -euo pipefail
SANDBOX_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SANDBOX_DIR"
RUNPOD="$SANDBOX_DIR/runs/runpod.sh"
STATE_FILE="$SANDBOX_DIR/runs/.runpod_ab.state"
RUNPOD_SSH_KEY="${RUNPOD_SSH_KEY:-}"
if [ -z "$RUNPOD_SSH_KEY" ]; then
    for c in "$HOME/.ssh/lambda_cloud_ed25519" "$HOME/.ssh/id_ed25519" "$HOME/.ssh/id_rsa"; do
        [ -f "$c" ] && RUNPOD_SSH_KEY="$c" && break
    done
fi
SSH_OPTS="-o StrictHostKeyChecking=accept-new -o ServerAliveInterval=30${RUNPOD_SSH_KEY:+ -i $RUNPOD_SSH_KEY -o IdentitiesOnly=yes}"

HW_ENV="${HW_ENV:-USE_FP8=0 WINDOW_PATTERN=L DEVICE_BATCH_SIZE=16}"
# Surface-only factoring A/B: the DECISIVE test is surface-only (arm 1) vs the vanilla
# baseline (arm 3); force_merges (arm 2) is an optional SIBLING reference, NOT stacked.
SURF_TAG="d24_surface_only_ab"             # arm 1 (run FIRST): surface-only + overlay
BASE_TAG="d24_force_merges_ab"             # arm 2: force_merges sibling (reference)
BASELINE_TAG="d24_baseline_ab"             # arm 3: vanilla baseline = the control for arm 1
FM_VARIANT="force_merges"
SURF_VARIANT="surface_only"

log()  { echo "==> $*" >&2; }
die()  { echo "ERROR: $*" >&2; exit 1; }
confirm() { [ "${YES:-0}" = "1" ] && return 0; read -r -p "$1 [y/N] " a < /dev/tty; [ "$a" = "y" ] || [ "$a" = "Y" ]; }

POD_ID=""; POD_IP=""; POD_PORT=""
_load() { [ -f "$STATE_FILE" ] && . "$STATE_FILE" || true; }
_save() { printf 'POD_ID=%q\nPOD_IP=%q\nPOD_PORT=%q\n' "$POD_ID" "$POD_IP" "$POD_PORT" > "$STATE_FILE"; }
_have() { [ -n "$POD_ID" ]; }
_refresh() { local ip port; read -r ip port < <(bash "$RUNPOD" host "$POD_ID" 2>/dev/null) || true; [ -n "${ip:-}" ] && { POD_IP="$ip"; POD_PORT="$port"; _save; } || true; }
_ssh()   { ssh $SSH_OPTS -p "$POD_PORT" "root@$POD_IP" "$@"; }
_rsync() { rsync -a -e "ssh $SSH_OPTS -p $POD_PORT" "$@"; }

stage_provision() {
    _load
    if _have; then
        local st; st=$(bash "$RUNPOD" status "$POD_ID" 2>/dev/null | jq -r '.desiredStatus // "missing"')
        if [ "$st" = "RUNNING" ]; then log "reusing live pod $POD_ID"; _refresh; return 0; fi
        log "state pod $POD_ID is '$st' — deploying a new one"
    fi
    log "RunPod availability:"; bash "$RUNPOD" types --available >&2 || true
    confirm "Deploy 8x H100 on RunPod (~\$21.52/hr, ~\$300 for the 3-arm d24 A/B)?" || die "aborted"
    read -r POD_ID POD_IP POD_PORT < <(bash "$RUNPOD" launch | tail -1)
    [ -n "$POD_ID" ] || die "launch failed"
    _save; log "pod $POD_ID up at $POD_IP:$POD_PORT"
}

stage_bootstrap() {
    _load; _have || die "no pod (run provision)"; _refresh
    if _ssh "test -f ~/.ab_setup_done" 2>/dev/null; then log "host already bootstrapped — skipping setup.sh"; return 0; fi
    bash "$RUNPOD" bootstrap "$POD_ID"
    _ssh "touch ~/.ab_setup_done"
    log "bootstrap complete (no Rust/crate — surface path is crate-free)"
}

stage_sync() {
    _load; _have || die "no pod (run provision)"; _refresh
    log "re-rsync sandbox/ (propagate code fixes; python only)"
    _rsync --delete --exclude=.venv --exclude=__pycache__ --exclude=results --exclude='*.pyc' \
        --exclude=target --exclude=.git --exclude=audit --exclude='*.so' \
        --exclude='runs/.runpod_ab.state' "$SANDBOX_DIR/" "root@$POD_IP:~/sandbox/"
    for v in "$FM_VARIANT" "$SURF_VARIANT"; do
        local src="$HOME/.cache/nanochat-variants/$v/tokenizer"
        [ -f "$src/tokenizer.pkl" ] || die "local tokenizer missing: $src"
        log "rsync tokenizer $v → pod"
        _ssh "mkdir -p ~/.cache/nanochat-variants/$v/tokenizer"
        _rsync "$src/" "root@$POD_IP:~/.cache/nanochat-variants/$v/tokenizer/"
    done
    # standard tokenizer for the vanilla baseline arm (default ~/.cache/nanochat)
    local std="$HOME/.cache/nanochat/tokenizer"
    if [ -f "$std/tokenizer.pkl" ]; then
        log "rsync standard tokenizer → pod (vanilla baseline arm)"
        _ssh "mkdir -p ~/.cache/nanochat/tokenizer"
        _rsync "$std/" "root@$POD_IP:~/.cache/nanochat/tokenizer/"
    else
        log "no local standard tokenizer — the vanilla baseline arm will train it on the pod"
    fi
}

stage_verify() {
    _load; _have || die "no pod (run provision)"; _refresh
    log "crate-free preflight (GPUs / venv / tokenizers load + round-trip / overlay imports)"
    _ssh 'set -e; export PATH="$HOME/.local/bin:$PATH"; cd ~/sandbox
        n=$(nvidia-smi -L | wc -l); echo "GPUs: $n"; [ "$n" -ge 8 ] || { echo FAIL_GPUS; exit 1; }
        test -d .venv || { echo FAIL_VENV; exit 1; }
        for v in force_merges surface_only; do
            test -f ~/.cache/nanochat-variants/$v/tokenizer/token_bytes.pt || { echo "FAIL_TOK_$v"; exit 1; }
        done
        uv run python - <<"PY"
import os
from tools.stack_fm_spaceless import SPACELESS_BODY   # surface-only pattern (crate-free)
from nanochat.tokenizer import RustBPETokenizer
from overlay.surface_tokenizer import SurfaceTokenizer
import overlay.surface_factoring, overlay.surface_core_eval, overlay.surface_checkpoint  # noqa
enc = RustBPETokenizer.from_directory(os.path.expanduser("~/.cache/nanochat-variants/surface_only/tokenizer")).enc
tok = SurfaceTokenizer(enc, SPACELESS_BODY, kmax=1)
s = "Of The Rings in the city"; assert tok.decode(tok.encode(s)) == s
print("PREFLIGHT OK: surface-only tokenizer round-trips, overlay imports, no crate needed.")
PY'
    log "preflight passed"
}

_write_remote_train() {
    _ssh "cat > ~/sandbox/runs/.ab_train.sh" <<REMOTE
#!/bin/bash
set -eo pipefail
cd ~/sandbox
export PATH="\$HOME/.local/bin:\$PATH"

# --- arm 1/3: surface factoring (FIRST, per request) ---
if [ -f results/$SURF_TAG/eval.csv ]; then echo "[ab] $SURF_TAG done — skip"; else
  echo "[ab] arm 1/3: surface-only (the test arm)"
  NANOCHAT_BASE_DIR="\$HOME/.cache/nanochat-variants/$SURF_VARIANT" \\
    OVERLAY=surface_factoring EVAL_MODULE=wrappers.base_eval_surface EVAL_ARGS="--eval core,bpb" $HW_ENV \\
    MODEL_TAG=$SURF_TAG WANDB_RUN=$SURF_TAG bash runs/speedrun.sh 2>&1 | tee runs/$SURF_TAG.log
fi

# --- arm 2/3: force_merges sibling (reference, NOT the baseline) ---
if [ -f results/$BASE_TAG/eval.csv ]; then echo "[ab] $BASE_TAG done — skip"; else
  echo "[ab] arm 2/3: force_merges sibling (reference, not the baseline)"
  NANOCHAT_BASE_DIR="\$HOME/.cache/nanochat-variants/$FM_VARIANT" EVAL_ARGS="--eval core,bpb" $HW_ENV \\
    MODEL_TAG=$BASE_TAG WANDB_RUN=$BASE_TAG bash runs/speedrun.sh 2>&1 | tee runs/$BASE_TAG.log
fi

# --- arm 3/3: vanilla nanochat baseline (default base dir + standard tokenizer, no overlay) ---
if [ -f results/$BASELINE_TAG/eval.csv ]; then echo "[ab] $BASELINE_TAG done — skip"; else
  echo "[ab] arm 3/3: vanilla baseline (the control for arm 1)"
  EVAL_ARGS="--eval core,bpb" $HW_ENV \\
    MODEL_TAG=$BASELINE_TAG WANDB_RUN=$BASELINE_TAG bash runs/speedrun.sh 2>&1 | tee runs/$BASELINE_TAG.log
fi

touch ~/sandbox/.ab_done
echo "[ab] ALL 3 ARMS DONE"
REMOTE
}

stage_train() {
    _load; _have || die "no pod (run provision)"; _refresh
    if _ssh "test -f ~/sandbox/.ab_done" 2>/dev/null; then log "training already complete (.ab_done)"; return 0; fi
    if _ssh "tmux has-session -t ab" 2>/dev/null; then log "tmux 'ab' already running — poll to watch"; return 0; fi
    _write_remote_train
    log "launching both arms in tmux 'ab' (HW_ENV: $HW_ENV)"
    _ssh "tmux new-session -d -s ab 'bash ~/sandbox/runs/.ab_train.sh 2>&1 | tee ~/sandbox/runs/ab_orchestrator.log'"
}

stage_poll() {
    _load; _have || die "no pod (run provision)"
    log "polling for ~/sandbox/.ab_done (every 5 min; Ctrl-C safe — re-run resumes)"
    local miss=0
    while :; do
        _refresh
        if _ssh "test -f ~/sandbox/.ab_done" 2>/dev/null; then log "training complete"; return 0; fi
        # tmux gone WITHOUT .ab_done => a failed arm — but require 3 consecutive
        # misses so a transient ssh blip over the ~15h run isn't a false alarm.
        if _ssh "tmux has-session -t ab" 2>/dev/null; then
            miss=0
        else
            miss=$((miss + 1))
            if [ "$miss" -ge 3 ]; then
                _ssh "tail -n 30 ~/sandbox/runs/ab_orchestrator.log 2>/dev/null" >&2 || true
                die "tmux 'ab' gone for 3 checks without .ab_done — an arm failed. Fix + re-run (train resumes the unfinished arm)."
            fi
            log "tmux check missed ($miss/3) — likely transient ssh; retrying in 30s"
            sleep 30; continue
        fi
        local p; p=$(_ssh "grep -hoE 'step [0-9]+/[0-9]+' ~/sandbox/runs/${SURF_TAG}.log ~/sandbox/runs/${BASE_TAG}.log ~/sandbox/runs/${BASELINE_TAG}.log 2>/dev/null | tail -1" || true)
        log "still training${p:+ — $p} ($(date -u +%H:%M:%SZ))"; sleep 300
    done
}

stage_download() {
    _load; _have || die "no pod (run provision)"; _refresh
    log "pull results/ + report"; bash "$RUNPOD" pull "$POD_ID"
    # each arm's checkpoint lives under ITS base dir (variant or default ~/.cache/nanochat)
    for entry in \
        "~/.cache/nanochat-variants/$SURF_VARIANT|$SURF_TAG" \
        "~/.cache/nanochat-variants/$FM_VARIANT|$BASE_TAG" \
        "~/.cache/nanochat|$BASELINE_TAG"; do
        local bdir="${entry%%|*}" tag="${entry##*|}"
        local rmt="$bdir/base_checkpoints/$tag/"
        if _ssh "test -d $rmt" 2>/dev/null; then
            mkdir -p "$SANDBOX_DIR/results/$tag/checkpoint/"; log "rsync checkpoint $tag"
            _rsync "root@$POD_IP:$rmt" "$SANDBOX_DIR/results/$tag/checkpoint/"
        else log "no checkpoint for $tag yet"; fi
        _rsync "root@$POD_IP:~/sandbox/runs/$tag.log" "$SANDBOX_DIR/results/$tag/" 2>/dev/null || true
    done
}

stage_verifydl() {
    local ok=1; echo "" >&2; log "verifying local artifacts"
    for tag in "$SURF_TAG" "$BASE_TAG" "$BASELINE_TAG"; do
        local d="$SANDBOX_DIR/results/$tag" csv="$SANDBOX_DIR/results/$tag/eval.csv" ckpt
        ckpt=$(ls "$d"/checkpoint/model_*.pt 2>/dev/null | head -1 || true)
        local core; core=$(grep -rhoE "CORE metric: [0-9.]+" "$d" 2>/dev/null | grep -oE "[0-9.]+" | head -1 || true)
        printf '  %-26s eval.csv:%s checkpoint:%s CORE:%s\n' "$tag" \
            "$([ -f "$csv" ] && echo yes || echo NO)" "$([ -n "$ckpt" ] && echo yes || echo NO)" "${core:-?}" >&2
        { [ -f "$csv" ] && [ -n "$ckpt" ]; } || ok=0
    done
    [ "$ok" = "1" ] || { log "artifacts INCOMPLETE — NOT terminating. Re-run download (or fix the failed arm)."; return 1; }
    log "all artifacts local. Decisive A/B: results/$SURF_TAG (surface-only) vs results/$BASELINE_TAG (vanilla); $BASE_TAG is the force_merges sibling."
}

stage_terminate() {
    _load; _have || { log "no pod in state"; return 0; }
    stage_verifydl || die "refusing to terminate: artifacts not verified local"
    [ "${AUTO_TERMINATE:-0}" = "1" ] || confirm "Artifacts verified. Terminate $POD_ID (stops billing)?" || { log "left pod $POD_ID up"; return 0; }
    bash "$RUNPOD" terminate "$POD_ID"; : > "$STATE_FILE"; log "terminated $POD_ID. Done."
}

ALL="provision bootstrap sync verify train poll download verifydl terminate"
if [ -n "${STAGE:-}" ]; then "stage_$STAGE"; else for s in $ALL; do log "=== STAGE: $s ==="; "stage_$s"; done; fi
