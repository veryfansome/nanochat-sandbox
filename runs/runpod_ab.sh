#!/bin/bash
# runs/runpod_ab.sh — IDEMPOTENT orchestrator for a full-d24 A/B on RunPod (8x H100),
# with PERSISTED artifacts on a network volume. Two arms at a shared NEW init seed:
#   surface_only (SURFACE_LAMBDA=0.1, K_max=1)  vs  vanilla baseline
# Both train the full compute-optimal d24 (target-param-data-ratio=8 — NO --num-iterations),
# eval core+bpb, and write checkpoints + logs + results to the network VOLUME so they
# survive pod termination and power future analysis (the surface-only checkpoint was lost
# to a failed download last time — the volume fixes that). The volume's shared climbmix
# corpus is symlinked into each variant's base dir, so the ~15GB corpus is NOT re-downloaded.
#
# Stages (resumable; re-run any time — only unfinished work re-executes):
#   provision  → reuse the live pod from state, else deploy 8x H100 + the volume
#   bootstrap  → runpod.sh bootstrap (apt + rsync sandbox + setup.sh + wandb)
#   sync       → rsync sandbox/ + stage BOTH tokenizers (surface_only + canonical baseline)
#                onto the volume + symlink each variant's base_data_climbmix → shared corpus
#   verify     → preflight (8 GPUs, venv, both tokenizers, surface round-trip)
#   train      → both arms in one tmux 'ab', each skipped if results/<tag>/eval.csv exists
#                (or EVAL-ONLY recovery if its checkpoint is already on the volume)
#   poll       → wait for ~/sandbox/.ab_done (tails progress)
#   download   → pull both arms' eval.csv + report + log to local results/<tag>/
#   verifydl   → assert both eval.csv LOCAL + both checkpoints + logs ON THE VOLUME
#   terminate  → terminate the pod (prompts unless AUTO_TERMINATE=1) — NEVER before verifydl
#
# Requires a network volume: RUNPOD_VOLUME_ID + RUNPOD_DATACENTER in ~/.lambda.env (the
# pod is pinned to the volume's datacenter). Create one with `runpod.sh create-volume`.
#
# Usage:
#   bash runs/runpod_ab.sh                  # full pipeline (run under local tmux — poll blocks ~hrs)
#   STAGE=download bash runs/runpod_ab.sh   # one stage
#   SEED=7 bash runs/runpod_ab.sh           # the new paired seed (default 7; ≠42)
#   AUTO_TERMINATE=1 YES=1 bash runs/runpod_ab.sh
#
# Env: ~/.lambda.env needs RUNPOD_API_KEY + WANDB_API_KEY + RUNPOD_VOLUME_ID + RUNPOD_DATACENTER.
#   SEED     new init seed, shared by both arms (default 7). Tags get _s${SEED}.
#   HW_ENV   default "USE_FP8=0 WINDOW_PATTERN=L DEVICE_BATCH_SIZE=16" (safe; fp8-off — the
#            fidelity-matching config for the prior d24 runs). Both arms share it.

set -euo pipefail
SANDBOX_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SANDBOX_DIR"
RUNPOD="$SANDBOX_DIR/runs/runpod.sh"
STATE_FILE="$SANDBOX_DIR/runs/.runpod_ab.state"
[ -f "$HOME/.lambda.env" ] && { set +u; . "$HOME/.lambda.env"; set -u; }   # RUNPOD_VOLUME_ID + RUNPOD_DATACENTER + keys
RUNPOD_SSH_KEY="${RUNPOD_SSH_KEY:-}"
if [ -z "$RUNPOD_SSH_KEY" ]; then
    for c in "$HOME/.ssh/lambda_cloud_ed25519" "$HOME/.ssh/id_ed25519" "$HOME/.ssh/id_rsa"; do
        [ -f "$c" ] && RUNPOD_SSH_KEY="$c" && break
    done
fi
SSH_OPTS="-o StrictHostKeyChecking=accept-new -o ServerAliveInterval=30${RUNPOD_SSH_KEY:+ -i $RUNPOD_SSH_KEY -o IdentitiesOnly=yes}"

HW_ENV="${HW_ENV:-USE_FP8=0 WINDOW_PATTERN=L DEVICE_BATCH_SIZE=16}"
SEED="${SEED:-7}"                       # new paired init seed (≠ the prior runs' 42)
SURF_LAMBDA="${SURF_LAMBDA:-0.1}"
export NANOCHAT_COMMIT="${NANOCHAT_COMMIT:-dc54a1a}"    # same pin as the prior d24 runs
[ -n "${RUNPOD_VOLUME_ID:-}" ] || { echo "ERROR: RUNPOD_VOLUME_ID unset — this run needs a network volume (set it + RUNPOD_DATACENTER in ~/.lambda.env; create with runpod.sh create-volume)" >&2; exit 1; }
MOUNT="${RUNPOD_VOLUME_MOUNT:-/workspace}"
SHARED_DATA="$MOUNT/shared/base_data_climbmix"

# --- the two arms: tag | variant (base-dir name) | per-arm train/eval env ---
SURF_TAG="d24_surf_lam$(echo "$SURF_LAMBDA" | tr '.' 'p')_s${SEED}"
BASE_TAG="d24_baseline_s${SEED}"
ARM_TAG=("$SURF_TAG" "$BASE_TAG")
ARM_VARIANT=("surface_only" "baseline")
ARM_ENV=("OVERLAY=surface_factoring SURFACE_LAMBDA=$SURF_LAMBDA SURFACE_SEED=$SEED EVAL_MODULE=wrappers.base_eval_surface" \
         "SEED=$SEED")
# local tokenizer dir staged for each variant (surface-only baked; baseline = canonical 32768)
ARM_LOCALTOK=("$HOME/.cache/nanochat-variants/surface_only/tokenizer" "$HOME/.cache/nanochat/tokenizer")

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
    confirm "Deploy 8x H100 + volume $RUNPOD_VOLUME_ID @ $RUNPOD_DATACENTER for the 2-arm full-d24 A/B (seed $SEED; ~\$200, ~9-10 hr)?" || die "aborted"
    read -r POD_ID POD_IP POD_PORT < <(bash "$RUNPOD" launch | tail -1)
    [ -n "$POD_ID" ] || die "launch failed"
    _save; log "pod $POD_ID up at $POD_IP:$POD_PORT (volume @ $MOUNT)"
}

stage_bootstrap() {
    _load; _have || die "no pod (run provision)"; _refresh
    if _ssh "test -f ~/.ab_setup_done" 2>/dev/null; then log "host already bootstrapped — skipping setup.sh"; return 0; fi
    bash "$RUNPOD" bootstrap "$POD_ID"
    _ssh "touch ~/.ab_setup_done"
    log "bootstrap complete (crate-free)"
}

stage_sync() {
    _load; _have || die "no pod (run provision)"; _refresh
    log "re-rsync sandbox/ (propagate code; python only)"
    _rsync --delete --exclude=.venv --exclude=__pycache__ --exclude=results --exclude='*.pyc' \
        --exclude=target --exclude=.git --exclude=audit --exclude='*.so' \
        --exclude='runs/.runpod_ab.state' --exclude='runs/*.log' --exclude='runs/.ab_train.sh' \
        --exclude='runs/ab_orchestrator.log' --exclude='.ab_done' "$SANDBOX_DIR/" "root@$POD_IP:~/sandbox/"
    local i tag variant vbase src
    for i in 0 1; do
        variant="${ARM_VARIANT[$i]}"; src="${ARM_LOCALTOK[$i]}"; vbase="$(_vbase "$variant")"
        [ -f "$src/tokenizer.pkl" ] || die "local tokenizer missing for $variant: $src"
        log "stage tokenizer $variant → $vbase/tokenizer [volume]"
        _ssh "mkdir -p '$vbase/tokenizer'"
        _rsync "$src/" "root@$POD_IP:$vbase/tokenizer/"
        # symlink the variant's corpus dir → the shared corpus (download once, reuse forever)
        _ssh "mkdir -p '$SHARED_DATA' '$vbase'; [ -L '$vbase/base_data_climbmix' ] || [ -d '$vbase/base_data_climbmix' ] || ln -s '$SHARED_DATA' '$vbase/base_data_climbmix'"
    done
    log "shared corpus: $(_ssh "ls '$SHARED_DATA'/*.parquet 2>/dev/null | wc -l") shards present (no re-download)"
}

stage_verify() {
    _load; _have || die "no pod (run provision)"; _refresh
    log "preflight (GPUs / venv / both tokenizers / surface round-trip)"
    local vsurf vbase_b; vsurf="$(_vbase surface_only)"; vbase_b="$(_vbase baseline)"
    _ssh "set -e; export PATH=\"\$HOME/.local/bin:\$PATH\"; cd ~/sandbox
        n=\$(nvidia-smi -L | wc -l); echo \"GPUs: \$n\"; [ \"\$n\" -ge 8 ] || { echo FAIL_GPUS; exit 1; }
        test -d .venv || { echo FAIL_VENV; exit 1; }
        test -f $vsurf/tokenizer/token_bytes.pt   || { echo FAIL_TOK_surface; exit 1; }
        test -f $vbase_b/tokenizer/token_bytes.pt || { echo FAIL_TOK_baseline; exit 1; }
        SURF_TOK=$vsurf/tokenizer uv run python - <<'PY'
import os
from tools.stack_fm_spaceless import SPACELESS_BODY
from nanochat.tokenizer import RustBPETokenizer
from overlay.surface_tokenizer import SurfaceTokenizer
import overlay.surface_factoring, overlay.surface_core_eval, overlay.surface_checkpoint  # noqa
enc = RustBPETokenizer.from_directory(os.environ['SURF_TOK']).enc
tok = SurfaceTokenizer(enc, SPACELESS_BODY, kmax=1)
s = 'Of The Rings in the city'
assert tok.decode(tok.encode(s)) == s
print('PREFLIGHT OK: surface tokenizer round-trips, overlay imports, both tokenizers staged.')
PY"
    log "preflight passed"
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
  echo "[ab] training '"$tag"' (variant='"$variant"', seed='"$SEED"', full d24)"
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
    log "launching both arms [$SURF_TAG, $BASE_TAG] in tmux 'ab' (seed $SEED; HW_ENV: $HW_ENV)"
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
        local p; p=$(_ssh "for t in $SURF_TAG $BASE_TAG; do grep -hoE 'step [0-9]+/[0-9]+' ~/sandbox/runs/\$t.log 2>/dev/null | tail -1; done | tail -1" || true)
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
    log "checkpoints persist on the volume (not downloaded — DOWNLOAD_CKPT pull is manual if ever needed)"
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
        printf '  %-26s eval.csv(local):%s  checkpoint(vol):%s  log(vol):%s  CORE:%s\n' "$tag" \
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
