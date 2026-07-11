#!/bin/bash
# runs/runpod_stp_ab.sh — IDEMPOTENT orchestrator for a full-d24 STP vs baseline A/B on
# RunPod (8x H100), with PERSISTED artifacts on a network volume. Modeled on
# runs/runpod_ab.sh (the surface_factoring orchestrator) but for the Semantic Tube
# Prediction overlay. Two arms at the SAME base_train-deterministic init:
#   stp (OVERLAY=stp STP_COEFF=0.02 STP_DIAG=1)  vs  vanilla baseline
# Both run the full compute-optimal d24 (--target-param-data-ratio=8) + SFT (USE_SFT=1),
# eval core+bpb + chat_eval, and write checkpoints/logs/results to the network VOLUME so
# they survive termination. Both arms SHARE one on-pod-trained ClimbMix tokenizer and one
# ClimbMix corpus download (symlinked into each variant's base dir) — so the tokenizer and
# corpus are identical across arms and the only difference is the STP loss term.
#
# WANDB/MODEL tags embed the sandbox short commit SHA (per the "SHA in labels" convention):
#   d24_stp_lam0p02_<sha>   d24_baseline_<sha>
#
# Stages (resumable; re-run any time — only unfinished work re-executes):
#   provision  → reuse the live pod from state, else deploy 8x H100 + the volume
#   bootstrap  → runpod.sh bootstrap (apt + rsync sandbox + setup.sh @ NANOCHAT_COMMIT + wandb)
#   sync       → rsync sandbox/ + download ClimbMix once + train tokenizer once (both on the
#                volume, shared) + symlink each arm's base_data/tokenizer → the shared copies
#   verify     → preflight: 8 GPUs, venv, shared tokenizer, and `wrappers.smoke_stp` GREEN
#                against the pinned nanochat (gates training on STP compatibility)
#   train      → both arms in one tmux 'ab', each skipped if results/<tag>/eval.csv exists
#                (or EVAL-ONLY recovery if its checkpoint is already on the volume)
#   poll       → wait for ~/sandbox/.ab_done (tails progress)
#   download   → pull both arms' eval.csv + report + log to local results/<tag>/
#   verifydl   → assert both eval.csv LOCAL + both checkpoints + logs ON THE VOLUME
#   terminate  → terminate the pod (prompts unless AUTO_TERMINATE=1) — NEVER before verifydl
#
# Requires a network volume: RUNPOD_VOLUME_ID + RUNPOD_DATACENTER (env or the env file below;
# the pod is pinned to the volume's datacenter). Create one with `runpod.sh create-volume`.
# Use LAMBDA_ENV_FILE=<path> to point at an alternate env file (e.g. ~/.lambda.env.iceland
# with the Iceland volume) WITHOUT editing the default ~/.lambda.env (which holds another vol).
#
# Usage:
#   LAMBDA_ENV_FILE=~/.lambda.env.iceland bash runs/runpod_stp_ab.sh          # full pipeline
#   STAGE=verify LAMBDA_ENV_FILE=~/.lambda.env.iceland bash runs/runpod_stp_ab.sh   # one stage
#   YES=1 AUTO_TERMINATE=1 LAMBDA_ENV_FILE=... bash runs/runpod_stp_ab.sh     # unattended
#
# Env: env file needs RUNPOD_API_KEY + WANDB_API_KEY + RUNPOD_VOLUME_ID + RUNPOD_DATACENTER.
#   STP_COEFF  STP lambda (default 0.02, the paper's NL-RX-SYNTH headline). Tag: lam<coeff>.
#   HW_ENV     default "USE_FP8=0 WINDOW_PATTERN=L DEVICE_BATCH_SIZE=16" (the proven fp8-off
#              A/B config; both arms share it — comparable to each other, not the leaderboard).

set -euo pipefail
SANDBOX_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SANDBOX_DIR"
RUNPOD="$SANDBOX_DIR/runs/runpod.sh"
STATE_FILE="$SANDBOX_DIR/runs/.runpod_stp_ab.state"
# Respect LAMBDA_ENV_FILE (so an alternate volume/DC can be selected without editing the
# default env file) — and export it so the runpod.sh child reads the SAME file.
LAMBDA_ENV_FILE="${LAMBDA_ENV_FILE:-$HOME/.lambda.env}"
export LAMBDA_ENV_FILE
[ -f "$LAMBDA_ENV_FILE" ] && { set +u; . "$LAMBDA_ENV_FILE"; set -u; }   # keys + RUNPOD_VOLUME_ID + RUNPOD_DATACENTER
RUNPOD_SSH_KEY="${RUNPOD_SSH_KEY:-}"
if [ -z "$RUNPOD_SSH_KEY" ]; then
    for c in "$HOME/.ssh/lambda_cloud_ed25519" "$HOME/.ssh/id_ed25519" "$HOME/.ssh/id_rsa"; do
        [ -f "$c" ] && RUNPOD_SSH_KEY="$c" && break
    done
fi
SSH_OPTS="-o StrictHostKeyChecking=accept-new -o ServerAliveInterval=30${RUNPOD_SSH_KEY:+ -i $RUNPOD_SSH_KEY -o IdentitiesOnly=yes}"

# Karpathy public base (fork's core-eval-controls adds only eval code; gpt.py is identical, so
# STP's smoke verification carries over). CORE is scored the standard way (no counterfactual controls).
export NANOCHAT_COMMIT="${NANOCHAT_COMMIT:-92d63d4}"
HW_ENV="${HW_ENV:-USE_FP8=0 WINDOW_PATTERN=L DEVICE_BATCH_SIZE=16}"
STP_COEFF="${STP_COEFF:-0.02}"
[ -n "${RUNPOD_VOLUME_ID:-}" ] || { echo "ERROR: RUNPOD_VOLUME_ID unset — set it + RUNPOD_DATACENTER in $LAMBDA_ENV_FILE (create a volume with runpod.sh create-volume)" >&2; exit 1; }
MOUNT="${RUNPOD_VOLUME_MOUNT:-/workspace}"
SHA="$(git -C "$SANDBOX_DIR" rev-parse --short HEAD)"

# --- corpus (ClimbMix is nanochat's default) staged ONCE on the volume, shared by both arms ---
DATA_SUBDIR="base_data_climbmix"
SHARED_DATA="$MOUNT/shared/$DATA_SUBDIR"
SHARED_TOK="$MOUNT/shared/tokenizer"
NTRAIN_SHARDS=170   # matches speedrun.sh's `-n 170` so its own download is a no-op

# --- the two arms: tag | variant (base-dir name) | per-arm train env (USE_SFT on both) ---
STP_TAG="d24_stp_lam$(echo "$STP_COEFF" | tr '.' 'p')_${SHA}"
BASE_TAG="d24_baseline_${SHA}"
ARM_TAG=("$STP_TAG" "$BASE_TAG")
ARM_VARIANT=("stp" "baseline")
ARM_ENV=("OVERLAY=stp STP_COEFF=$STP_COEFF STP_DIAG=1 USE_SFT=1" \
         "USE_SFT=1")

log()  { echo "==> $*" >&2; }
die()  { echo "ERROR: $*" >&2; exit 1; }
confirm() { [ "${YES:-0}" = "1" ] && return 0; read -r -p "$1 [y/N] " a < /dev/tty; [ "$a" = "y" ] || [ "$a" = "Y" ]; }

POD_ID=""; POD_IP=""; POD_PORT=""
_load() { [ -f "$STATE_FILE" ] && . "$STATE_FILE" || true; }
_save() { printf 'POD_ID=%q\nPOD_IP=%q\nPOD_PORT=%q\n' "$POD_ID" "$POD_IP" "$POD_PORT" > "$STATE_FILE"; }
_have() { [ -n "$POD_ID" ]; }
_refresh() { local ip port; read -r ip port < <(bash "$RUNPOD" host "$POD_ID" 2>/dev/null) || true; [ -n "${ip:-}" ] && { POD_IP="$ip"; POD_PORT="$port"; _save; } || true; }
_ssh()   { ssh $SSH_OPTS -p "$POD_PORT" "root@$POD_IP" "$@"; }
# --no-owner --no-group: network volumes reject chown → plain `rsync -a` fails (exit 23).
_rsync() { rsync -a --no-owner --no-group -e "ssh $SSH_OPTS -p $POD_PORT" "$@"; }
_vbase() { echo "$MOUNT/nanochat-variants/$1"; }

stage_provision() {
    _load
    if _have; then
        local st; st=$(bash "$RUNPOD" status "$POD_ID" 2>/dev/null | jq -r '.desiredStatus // "missing"')
        if [ "$st" = "RUNNING" ]; then log "reusing live pod $POD_ID"; _refresh; return 0; fi
        log "state pod $POD_ID is '$st' — deploying a new one"
    fi
    confirm "Deploy 8x H100 + volume $RUNPOD_VOLUME_ID @ $RUNPOD_DATACENTER for the STP vs baseline full-d24+SFT A/B (coeff $STP_COEFF, sha $SHA; ~\$250, ~11-12 hr)?" || die "aborted"
    read -r POD_ID POD_IP POD_PORT < <(bash "$RUNPOD" launch | tail -1)
    [ -n "$POD_ID" ] || die "launch failed"
    _save; log "pod $POD_ID up at $POD_IP:$POD_PORT (volume @ $MOUNT)"
}

stage_bootstrap() {
    _load; _have || die "no pod (run provision)"; _refresh
    if _ssh "test -f ~/.ab_setup_done" 2>/dev/null; then log "host already bootstrapped — skipping setup.sh"; return 0; fi
    bash "$RUNPOD" bootstrap "$POD_ID"
    _ssh "touch ~/.ab_setup_done"
    log "bootstrap complete (nanochat @ $NANOCHAT_COMMIT)"
}

stage_sync() {
    _load; _have || die "no pod (run provision)"; _refresh
    log "re-rsync sandbox/ (propagate code; python only)"
    _rsync --delete --exclude=.venv --exclude=__pycache__ --exclude=results --exclude='*.pyc' \
        --exclude=target --exclude=.git --exclude=audit --exclude='*.so' \
        --exclude='runs/.runpod_stp_ab.state' --exclude='runs/*.log' --exclude='runs/.ab_train.sh' \
        --exclude='runs/ab_orchestrator.log' --exclude='.ab_done' "$SANDBOX_DIR/" "root@$POD_IP:~/sandbox/"

    # 1) ClimbMix corpus → shared (download once; download_single_file skips present shards)
    local have; have=$(_ssh "ls '$SHARED_DATA'/*.parquet 2>/dev/null | wc -l" | tr -d ' ')
    if [ "${have:-0}" -lt "$((NTRAIN_SHARDS + 1))" ]; then
        log "downloading ClimbMix → $SHARED_DATA ($NTRAIN_SHARDS train + val shards) [on pod]"
        _ssh "set -e; export PATH=\"\$HOME/.local/bin:\$PATH\"; cd ~/sandbox; NANOCHAT_BASE_DIR='$MOUNT/shared' uv run python -m wrappers.download_dataset -n $NTRAIN_SHARDS -w 16"
    else
        log "ClimbMix already staged ($have shards on volume)"
    fi

    # 2) canonical tokenizer → shared (train once; both arms use the IDENTICAL tokenizer)
    if _ssh "test -f '$SHARED_TOK/tokenizer.pkl'" 2>/dev/null; then
        log "shared tokenizer already on volume"
    else
        log "training canonical ClimbMix tokenizer → $SHARED_TOK [on pod, once]"
        _ssh "set -e; export PATH=\"\$HOME/.local/bin:\$PATH\"; cd ~/sandbox
            NANOCHAT_BASE_DIR='$MOUNT/shared' uv run python -m scripts.tok_train
            NANOCHAT_BASE_DIR='$MOUNT/shared' uv run python -m scripts.tok_eval"
    fi

    # 3) symlink each arm's corpus + tokenizer → the shared copies (so speedrun.sh's own
    #    download + tok_train see them present and skip → identical inputs for both arms)
    local i variant vbase
    for i in 0 1; do
        variant="${ARM_VARIANT[$i]}"; vbase="$(_vbase "$variant")"
        _ssh "mkdir -p '$vbase'
            [ -e '$vbase/$DATA_SUBDIR' ] || ln -s '$SHARED_DATA' '$vbase/$DATA_SUBDIR'
            [ -e '$vbase/tokenizer' ]    || ln -s '$SHARED_TOK'  '$vbase/tokenizer'"
    done
    log "shared corpus: $(_ssh "ls '$SHARED_DATA'/*.parquet 2>/dev/null | wc -l") shards; tokenizer + symlinks in place"
}

stage_verify() {
    _load; _have || die "no pod (run provision)"; _refresh
    log "preflight (GPUs / venv / shared tokenizer / STP smoke against nanochat $NANOCHAT_COMMIT)"
    _ssh "set -e; export PATH=\"\$HOME/.local/bin:\$PATH\"; cd ~/sandbox
        n=\$(nvidia-smi -L | wc -l); echo \"GPUs: \$n\"; [ \"\$n\" -ge 8 ] || { echo FAIL_GPUS; exit 1; }
        test -d .venv || { echo FAIL_VENV; exit 1; }
        test -f '$SHARED_TOK/tokenizer.pkl' || { echo FAIL_TOK; exit 1; }
        echo '--- wrappers.smoke_stp (gates training) ---'
        uv run python -m wrappers.smoke_stp"
    log "preflight passed (STP smoke GREEN — overlay compatible with the pinned nanochat)"
}

_write_remote_train() {
    local body="" i tag variant env vbase
    for i in 0 1; do
        tag="${ARM_TAG[$i]}"; variant="${ARM_VARIANT[$i]}"; env="${ARM_ENV[$i]}"; vbase="$(_vbase "$variant")"
        body+='
mkdir -p '"$vbase"'/logs '"$vbase"'/results
if [ -f results/'"$tag"'/eval.csv ]; then echo "[ab] '"$tag"' done — skip";
elif ls '"$vbase"'/base_checkpoints/'"$tag"'/model_*.pt >/dev/null 2>&1; then
  echo "[ab] '"$tag"' checkpoint on volume but no eval.csv — EVAL-ONLY recovery"
  NANOCHAT_BASE_DIR='"$vbase"' '"$env"' EVAL_ONLY=1 EVAL_ARGS="--eval core,bpb" '"$HW_ENV"' MODEL_TAG='"$tag"' WANDB_RUN='"$tag"' bash runs/speedrun.sh 2>&1 | tee -a runs/'"$tag"'.log
else
  echo "[ab] training '"$tag"' (variant='"$variant"', full d24 + SFT)"
  NANOCHAT_BASE_DIR='"$vbase"' '"$env"' EVAL_ARGS="--eval core,bpb" '"$HW_ENV"' MODEL_TAG='"$tag"' WANDB_RUN='"$tag"' bash runs/speedrun.sh 2>&1 | tee runs/'"$tag"'.log
fi
cp -f runs/'"$tag"'.log '"$vbase"'/logs/'"$tag"'.log 2>/dev/null || true
mkdir -p '"$vbase"'/results/'"$tag"' && cp -rf results/'"$tag"'/. '"$vbase"'/results/'"$tag"'/ 2>/dev/null || true
'
    done
    local script='#!/bin/bash
set -eo pipefail
cd ~/sandbox
export PATH=~/.local/bin:$PATH
'"$body"'
touch ~/sandbox/.ab_done
echo "[ab] BOTH ARMS DONE"
'
    printf '%s' "$script" | _ssh "cat > ~/sandbox/runs/.ab_train.sh"
}

stage_train() {
    _load; _have || die "no pod (run provision)"; _refresh
    if _ssh "test -f ~/sandbox/.ab_done" 2>/dev/null; then log "training already complete (.ab_done)"; return 0; fi
    if _ssh "tmux has-session -t ab" 2>/dev/null; then log "tmux 'ab' already running — poll to watch"; return 0; fi
    _write_remote_train
    log "launching both arms [$STP_TAG, $BASE_TAG] in tmux 'ab' (HW_ENV: $HW_ENV)"
    _ssh "tmux new-session -d -s ab 'bash ~/sandbox/runs/.ab_train.sh 2>&1 | tee ~/sandbox/runs/ab_orchestrator.log'"
}

stage_poll() {
    _load; _have || die "no pod (run provision)"
    log "polling for ~/sandbox/.ab_done (every 5 min; Ctrl-C safe — re-run resumes)"
    local miss=0
    while :; do
        _refresh
        if _ssh "test -f ~/sandbox/.ab_done" 2>/dev/null; then log "training complete"; return 0; fi
        if _ssh "tmux has-session -t ab" 2>/dev/null; then
            miss=0
        else
            miss=$((miss + 1))
            if [ "$miss" -ge 3 ]; then
                _ssh "tail -n 30 ~/sandbox/runs/ab_orchestrator.log 2>/dev/null" >&2 || true
                die "tmux 'ab' gone for 3 checks without .ab_done — an arm failed. Fix + re-run (resumes the unfinished arm)."
            fi
            log "tmux check missed ($miss/3) — likely transient ssh; retrying in 30s"; sleep 30; continue
        fi
        local p; p=$(_ssh "for t in $STP_TAG $BASE_TAG; do grep -hoE 'step [0-9]+/[0-9]+' ~/sandbox/runs/\$t.log 2>/dev/null | tail -1; done | tail -1" || true)
        log "still training${p:+ — $p} ($(date -u +%H:%M:%SZ))"; sleep 300
    done
}

stage_download() {
    _load; _have || die "no pod (run provision)"; _refresh
    log "pull results/ + report"; bash "$RUNPOD" pull "$POD_ID"
    local i tag
    for i in 0 1; do
        tag="${ARM_TAG[$i]}"
        mkdir -p "$SANDBOX_DIR/results/$tag"
        _rsync "root@$POD_IP:~/sandbox/results/$tag/" "$SANDBOX_DIR/results/$tag/" 2>/dev/null || true
        _rsync "root@$POD_IP:~/sandbox/runs/$tag.log" "$SANDBOX_DIR/results/$tag/" 2>/dev/null || true
    done
    log "checkpoints persist on the volume (not downloaded)"
}

stage_verifydl() {
    _load; _have || die "no pod (verifydl needs the live pod to check the volume)"; _refresh
    local ok=1; echo "" >&2; log "verifying: eval.csv LOCAL + checkpoint & log ON THE VOLUME"
    local i tag variant vbase csv coreloc ckpt vlog
    for i in 0 1; do
        tag="${ARM_TAG[$i]}"; variant="${ARM_VARIANT[$i]}"; vbase="$(_vbase "$variant")"
        csv="$SANDBOX_DIR/results/$tag/eval.csv"
        ckpt=$(_ssh "ls $vbase/base_checkpoints/$tag/model_*.pt 2>/dev/null | head -1" || true)
        vlog=$(_ssh "ls $vbase/logs/$tag.log 2>/dev/null" || true)
        coreloc=$(grep -rhoE "CORE metric: [0-9.]+" "$SANDBOX_DIR/results/$tag" 2>/dev/null | grep -oE "[0-9.]+" | head -1 || true)
        printf '  %-30s eval.csv(local):%s  checkpoint(vol):%s  log(vol):%s  CORE:%s\n' "$tag" \
            "$([ -f "$csv" ] && echo yes || echo NO)" "$([ -n "$ckpt" ] && echo yes || echo NO)" \
            "$([ -n "$vlog" ] && echo yes || echo NO)" "${coreloc:-?}" >&2
        { [ -f "$csv" ] && [ -n "$ckpt" ] && [ -n "$vlog" ]; } || ok=0
    done
    [ "$ok" = "1" ] || { log "artifacts INCOMPLETE — NOT terminating. Re-run download / fix the failed arm."; return 1; }
    log "both arms verified: eval.csv local + checkpoints + logs persisted on volume $RUNPOD_VOLUME_ID."
}

stage_terminate() {
    _load; _have || { log "no pod in state"; return 0; }
    stage_verifydl || die "refusing to terminate: artifacts not verified"
    [ "${AUTO_TERMINATE:-0}" = "1" ] || confirm "Artifacts verified (checkpoints persist on the volume). Terminate $POD_ID?" || { log "left pod $POD_ID up"; return 0; }
    bash "$RUNPOD" terminate "$POD_ID"; : > "$STATE_FILE"; log "terminated $POD_ID. Done."
}

ALL="provision bootstrap sync verify train poll download verifydl terminate"
if [ -n "${STAGE:-}" ]; then "stage_$STAGE"; else for s in $ALL; do log "=== STAGE: $s ==="; "stage_$s"; done; fi
