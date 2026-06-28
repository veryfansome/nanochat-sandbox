#!/bin/bash
# runs/runpod_lambda_screen.sh — IDEMPOTENT λ-screen for surface factoring on RunPod
# (8x H100). A within-surface paired triage of the ONE untested surface dial,
# SURFACE_LAMBDA (the content-vs-surface loss weight, untuned at 0.5). Trains several
# λ arms at a REDUCED COOLED schedule (--num-iterations=$NUM_ITERS, ~1/3 of the ~5569
# full d24, warmdown stays proportional at 0.65) so a λ delta is readable for ~1/3 the
# per-arm cost. Same design/stages as runs/runpod_ab.sh.
#
# WHY this shape (see ideas/surface_factoring/README.md "Cheap λ/exposure triage"):
#   - PAIRED, within-surface: all arms share commit + tokenizer + (seed-independent) data
#     order + the fixed init seed 42 — they differ ONLY in SURFACE_LAMBDA. The cross-
#     tokenizer offloading confound cancels; the content-bpb Δ has a low (~0.001-0.002) floor.
#   - PRIMARY signal = the MC-CORE robust trio (arc_easy / piqa / hellaswag), read from each
#     arm's eval.csv. content-bpb is a SECONDARY diagnostic only (it is empirically CORE-
#     DECOUPLED, so a content-bpb move neither greenlights nor, alone, kills a λ).
#   - The overhead-vs-content split is logged every eval ([surface bpb] line) by the
#     instrumented surface_evaluate_bpb — the readout surfaces it per arm.
#   - This is a NEGATIVE-CONFIRMATION / closing experiment on a weak prior, not a rescue.
#
# Stages (resumable; re-run any time): provision bootstrap sync verify train poll
#   download verifydl terminate. Each ARM is skipped if its results/<tag>/eval.csv exists.
#   Never terminates before verifydl passes.
#
# Usage (default = 2-arm extreme bracket, ≈ $95 / ~3.5 hr; full CORE eval per arm):
#   bash runs/runpod_lambda_screen.sh                       # control λ=0.5 vs λ=0.1
#   LAMBDAS="0.5 0.25 0.1" bash runs/runpod_lambda_screen.sh   # add the middle (≈ $130) if 0.1 moved
#   SEED=2 LAMBDAS="0.5 0.1" bash runs/runpod_lambda_screen.sh # 2nd-seed REPLICATE of a promoted arm
#   STAGE=verifydl bash runs/runpod_lambda_screen.sh        # re-print the result table anytime
#   AUTO_TERMINATE=1 YES=1 bash runs/runpod_lambda_screen.sh
# NOTE (extend-grid footgun): adding a λ later → also `rm` the host ~/sandbox/.lamscreen_done
# (or run STAGE=train), else the done-marker short-circuits and the new arm never trains.
#
# Env: ~/.lambda.env needs RUNPOD_API_KEY + WANDB_API_KEY (+ an ssh key — see runpod.sh).
#   LAMBDAS      space-separated λ list; FIRST is the control (default "0.5 0.1")
#   SEED         empty = init seed 42 (paired); SEED=2 = a replicate arm (tag _s<SEED>)
#   NUM_ITERS    reduced cooled schedule length (default 1850 ≈ 1/3 of full d24 5569)
#   HW_ENV       default "USE_FP8=0 WINDOW_PATTERN=L DEVICE_BATCH_SIZE=16". fp8-OFF is the
#                FIDELITY-MATCHING regime (the d24 surface wash ran fp8-off) and avoids an
#                UNTESTED-on-hardware variable in a paid closing run. FP8 would ~halve cost
#                (surface heads have out_features=8 < 128 → excluded from FP8 conversion, so
#                only the standard trunk converts) — but enable it ONLY after a one-time ~$2
#                preflight on one arm (expect 'skipped <surface heads>' in the convert log),
#                then HW_ENV="USE_FP8=1 WINDOW_PATTERN=SSSL DEVICE_BATCH_SIZE=16" for all arms.
#   TAG_PREFIX   default "d24_surf"  (arm tag = <prefix>_lam<λ>_k<NUM_ITERS>[_s<SEED>])
#   RUNPOD_VOLUME_ID + RUNPOD_DATACENTER  (in ~/.lambda.env) → base dir on a persistent
#                network volume (mounts at /workspace): checkpoints + tokenizer survive
#                termination (no 4GB download), AND the ~15GB climbmix corpus is SHARED across
#                ALL variants/runs (symlinked to /workspace/shared/base_data_climbmix) so it
#                downloads once, ever. Follow-ups re-mount instantly. Create a volume with:
#                  bash runs/runpod.sh datacenters         # pick a storage DC with H100 stock
#                  bash runs/runpod.sh create-volume surface-vol 100 US-GA-2
#                then add the printed RUNPOD_VOLUME_ID + RUNPOD_DATACENTER to ~/.lambda.env.
#   DOWNLOAD_CKPT  with a volume, set =1 to ALSO pull a local checkpoint copy (default: 0,
#                since the volume is the durable store; the tiny eval.csv/logs always download)

set -euo pipefail
SANDBOX_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SANDBOX_DIR"
# source ~/.lambda.env for RUNPOD_VOLUME_ID / RUNPOD_DATACENTER (+ keys) so the volume
# branch here AND runpod.sh launch both see them.
[ -f "$HOME/.lambda.env" ] && { set +u; . "$HOME/.lambda.env"; set -u; }
RUNPOD="$SANDBOX_DIR/runs/runpod.sh"
STATE_FILE="$SANDBOX_DIR/runs/.runpod_lambda_screen.state"
RUNPOD_SSH_KEY="${RUNPOD_SSH_KEY:-}"
if [ -z "$RUNPOD_SSH_KEY" ]; then
    for c in "$HOME/.ssh/lambda_cloud_ed25519" "$HOME/.ssh/id_ed25519" "$HOME/.ssh/id_rsa"; do
        [ -f "$c" ] && RUNPOD_SSH_KEY="$c" && break
    done
fi
SSH_OPTS="-o StrictHostKeyChecking=accept-new -o ServerAliveInterval=30${RUNPOD_SSH_KEY:+ -i $RUNPOD_SSH_KEY -o IdentitiesOnly=yes}"

HW_ENV="${HW_ENV:-USE_FP8=0 WINDOW_PATTERN=L DEVICE_BATCH_SIZE=16}"
# DEFAULT = the 2-arm EXTREME bracket (control λ=0.5 vs λ=0.1): the grid is monotone in λ,
# so the biggest λ move is the single most powered test. If λ=0.1 shows a real (non-floor)
# move, add 0.25 (LAMBDAS="0.5 0.25 0.1") to map the curve. ~$95 for 2 arms (see header).
LAMBDAS="${LAMBDAS:-0.5 0.1}"
NUM_ITERS="${NUM_ITERS:-1850}"
TAG_PREFIX="${TAG_PREFIX:-d24_surf}"
# SEED: empty = the default init seed 42 (the screen is paired on it). Set SEED=2 to run a
# REPLICATE of a PROMOTE'd arm at a 2nd seed (the readout's required pre-full-run gate) —
# the tag gets a _s<SEED> suffix so it doesn't collide with the seed-42 screen.
SEED="${SEED:-}"
SURF_VARIANT="surface_only"
# Same commit as the d24 surface-only wash, so this stays comparable to it.
export NANOCHAT_COMMIT="${NANOCHAT_COMMIT:-dc54a1a}"

# Artifact persistence: with a RunPod NETWORK VOLUME (RUNPOD_VOLUME_ID set — see runpod.sh
# `create-volume`/`datacenters`), point the base dir at the volume mount so checkpoints +
# data + tokenizer SURVIVE pod termination — no fragile 4GB checkpoint download, and a
# follow-up pod (promote/re-eval) re-mounts them instantly. Without a volume, fall back to
# the ephemeral container disk (~/.cache) + the (slow, failure-prone) download.
if [ -n "${RUNPOD_VOLUME_ID:-}" ]; then
    VBASE="${RUNPOD_VOLUME_MOUNT:-/workspace}/nanochat-variants/$SURF_VARIANT"; ON_VOLUME=1
    # The climbmix corpus is variant-INDEPENDENT (only the tokenizer differs), so share ONE
    # copy across all variants/runs on the volume. Each variant's base_data_climbmix is
    # symlinked here → the 170 shards (~15GB) download once, ever; every future run skips it.
    SHARED_DATA="${RUNPOD_VOLUME_MOUNT:-/workspace}/shared/base_data_climbmix"
else
    VBASE="~/.cache/nanochat-variants/$SURF_VARIANT"; ON_VOLUME=0
    SHARED_DATA=""
fi

# arm tag for a given λ (0.5 -> lam0p5; appended _k<iters>, + _s<SEED> for a replicate)
_tag() { echo "${TAG_PREFIX}_lam$(echo "$1" | tr '.' 'p')_k${NUM_ITERS}${SEED:+_s$SEED}"; }
CONTROL_LAM="$(echo "$LAMBDAS" | awk '{print $1}')"

log()  { echo "==> $*" >&2; }
die()  { echo "ERROR: $*" >&2; exit 1; }
confirm() { [ "${YES:-0}" = "1" ] && return 0; read -r -p "$1 [y/N] " a < /dev/tty; [ "$a" = "y" ] || [ "$a" = "Y" ]; }

POD_ID=""; POD_IP=""; POD_PORT=""
_load() { [ -f "$STATE_FILE" ] && . "$STATE_FILE" || true; }
_save() { printf 'POD_ID=%q\nPOD_IP=%q\nPOD_PORT=%q\n' "$POD_ID" "$POD_IP" "$POD_PORT" > "$STATE_FILE"; }
_have() { [ -n "$POD_ID" ]; }
_refresh() { local ip port; read -r ip port < <(bash "$RUNPOD" host "$POD_ID" 2>/dev/null) || true; [ -n "${ip:-}" ] && { POD_IP="$ip"; POD_PORT="$port"; _save; } || true; }
_ssh()   { ssh $SSH_OPTS -p "$POD_PORT" "root@$POD_IP" "$@"; }
# --no-owner --no-group: RunPod NETWORK VOLUMES reject chown ("Operation not permitted"),
# which makes plain `rsync -a` fail (exit 23) when writing to /workspace. We're root +
# single-user, so owner/group preservation is irrelevant — skip the chown.
_rsync() { rsync -a --no-owner --no-group -e "ssh $SSH_OPTS -p $POD_PORT" "$@"; }

stage_provision() {
    _load
    if _have; then
        local st; st=$(bash "$RUNPOD" status "$POD_ID" 2>/dev/null | jq -r '.desiredStatus // "missing"')
        if [ "$st" = "RUNNING" ]; then log "reusing live pod $POD_ID"; _refresh; return 0; fi
        log "state pod $POD_ID is '$st' — deploying a new one"
    fi
    local narms; narms=$(echo "$LAMBDAS" | wc -w | tr -d ' ')
    if [ "$ON_VOLUME" = 1 ]; then
        log "network volume: $RUNPOD_VOLUME_ID @ ${RUNPOD_DATACENTER:-?} → checkpoints persist (no 4GB download)"
        log "${RUNPOD_DATACENTER:-?} stock:"; bash "$RUNPOD" datacenters 2>/dev/null | grep -E "^\s*${RUNPOD_DATACENTER:-XXX}\b" >&2 || true
    else
        log "NO network volume — checkpoints on ephemeral disk (download required). Set RUNPOD_VOLUME_ID to persist."
        log "RunPod availability:"; bash "$RUNPOD" types --available >&2 || true
    fi
    confirm "Deploy 8x H100 (~\$21.52/hr)$([ "$ON_VOLUME" = 1 ] && echo " in ${RUNPOD_DATACENTER:-?} w/ volume") for a ${narms}-arm λ-screen at $NUM_ITERS cooled iters (≈ \$$((narms * 35 + 25)) total)?" || die "aborted"
    read -r POD_ID POD_IP POD_PORT < <(bash "$RUNPOD" launch | tail -1)
    [ -n "$POD_ID" ] || die "launch failed"
    _save; log "pod $POD_ID up at $POD_IP:$POD_PORT"
}

stage_bootstrap() {
    _load; _have || die "no pod (run provision)"; _refresh
    if _ssh "test -f ~/.ls_setup_done" 2>/dev/null; then log "host already bootstrapped — skipping setup.sh"; return 0; fi
    bash "$RUNPOD" bootstrap "$POD_ID"
    _ssh "touch ~/.ls_setup_done"
    log "bootstrap complete (crate-free)"
}

stage_sync() {
    _load; _have || die "no pod (run provision)"; _refresh
    log "re-rsync sandbox/ (propagate code fixes; python only)"
    # --delete: also EXCLUDE the pod-only runtime artifacts so a resync (which runs before
    # download on a full-pipeline re-run) doesn't wipe the per-arm logs (they carry the
    # SECONDARY [surface bpb] split + debug for completed-but-undownloaded arms) or unlink an
    # in-flight arm's tee target mid-write.
    _rsync --delete --exclude=.venv --exclude=__pycache__ --exclude=results --exclude='*.pyc' \
        --exclude=target --exclude=.git --exclude=audit --exclude='*.so' \
        --exclude='runs/.runpod_lambda_screen.state' --exclude='runs/*.log' \
        --exclude='runs/.lamscreen_train.sh' --exclude='runs/lamscreen_orchestrator.log' \
        --exclude='.lamscreen_done' "$SANDBOX_DIR/" "root@$POD_IP:~/sandbox/"
    local src="$HOME/.cache/nanochat-variants/$SURF_VARIANT/tokenizer"
    [ -f "$src/tokenizer.pkl" ] || die "local surface-only tokenizer missing: $src (bake: uv run python -m wrappers.tok_train_surface)"
    log "rsync tokenizer $SURF_VARIANT → pod ($VBASE/tokenizer)$([ "$ON_VOLUME" = 1 ] && echo ' [persistent volume]')"
    _ssh "mkdir -p $VBASE/tokenizer"
    _rsync "$src/" "root@$POD_IP:$VBASE/tokenizer/"
    if [ "$ON_VOLUME" = 1 ]; then
        # symlink this variant's base_data_climbmix → the SHARED corpus dir (only if not already
        # a dir/symlink, so we never clobber a real corpus). nanochat.dataset writes/reads through
        # the symlink → the 170 shards download once into $SHARED_DATA and every run reuses them.
        log "sharing climbmix corpus on the volume: $VBASE/base_data_climbmix → $SHARED_DATA (download once, reuse forever)"
        _ssh "mkdir -p '$SHARED_DATA' '$VBASE'; [ -L '$VBASE/base_data_climbmix' ] || [ -d '$VBASE/base_data_climbmix' ] || ln -s '$SHARED_DATA' '$VBASE/base_data_climbmix'"
    fi
}

stage_verify() {
    _load; _have || die "no pod (run provision)"; _refresh
    log "crate-free preflight (GPUs / venv / tokenizer round-trip / overlay imports / λ plumbing)"
    local tok="$VBASE/tokenizer"   # on /workspace when ON_VOLUME, else ~/.cache (~ expands remotely)
    _ssh "set -e; export PATH=\"\$HOME/.local/bin:\$PATH\"; cd ~/sandbox
        n=\$(nvidia-smi -L | wc -l); echo \"GPUs: \$n\"; [ \"\$n\" -ge 8 ] || { echo FAIL_GPUS; exit 1; }
        test -d .venv || { echo FAIL_VENV; exit 1; }
        test -f $tok/token_bytes.pt || { echo FAIL_TOK; exit 1; }
        SURFACE_LAMBDA=0.123 SURF_TOK=$tok uv run python - <<'PY'
import os
from tools.stack_fm_spaceless import SPACELESS_BODY
from nanochat.tokenizer import RustBPETokenizer
from overlay.surface_tokenizer import SurfaceTokenizer
from overlay.surface_factoring import _lambda_from_env
import overlay.surface_factoring, overlay.surface_core_eval, overlay.surface_checkpoint  # noqa
enc = RustBPETokenizer.from_directory(os.path.expanduser(os.environ['SURF_TOK'])).enc
tok = SurfaceTokenizer(enc, SPACELESS_BODY, kmax=1)
s = 'Of The Rings in the city'
assert tok.decode(tok.encode(s)) == s
assert abs(_lambda_from_env() - 0.123) < 1e-9, 'SURFACE_LAMBDA not honored'
print('PREFLIGHT OK: tokenizer round-trips, overlay imports, SURFACE_LAMBDA plumbing live.')
PY"
    log "preflight passed"
}

_write_remote_train() {
    local body="" lam tag seedenv=""
    [ -n "$SEED" ] && seedenv="SURFACE_SEED=$SEED"
    for lam in $LAMBDAS; do
        tag="$(_tag "$lam")"
        body+='
if [ -f results/'"$tag"'/eval.csv ]; then echo "[ls] '"$tag"' done — skip";
elif ls '"$VBASE"'/base_checkpoints/'"$tag"'/model_*.pt >/dev/null 2>&1; then
  echo "[ls] '"$tag"' checkpoint exists but no eval.csv — EVAL-ONLY recovery (no retrain)"
  NANOCHAT_BASE_DIR='"$VBASE"' OVERLAY=surface_factoring SURFACE_LAMBDA='"$lam"' '"$seedenv"' EVAL_ONLY=1 \
    EVAL_MODULE=wrappers.base_eval_surface EVAL_ARGS="--eval core,bpb" '"$HW_ENV"' \
    MODEL_TAG='"$tag"' WANDB_RUN='"$tag"' bash runs/speedrun.sh 2>&1 | tee -a runs/'"$tag"'.log
else
  echo "[ls] training '"$tag"' (SURFACE_LAMBDA='"$lam"', cooled '"$NUM_ITERS"' iters'"${SEED:+, seed $SEED}"')"
  NANOCHAT_BASE_DIR='"$VBASE"' OVERLAY=surface_factoring SURFACE_LAMBDA='"$lam"' '"$seedenv"' \
    EXTRA_TRAIN_ARGS="--num-iterations='"$NUM_ITERS"'" \
    EVAL_MODULE=wrappers.base_eval_surface EVAL_ARGS="--eval core,bpb" '"$HW_ENV"' \
    MODEL_TAG='"$tag"' WANDB_RUN='"$tag"' bash runs/speedrun.sh 2>&1 | tee runs/'"$tag"'.log
fi
'
    done
    local script='#!/bin/bash
set -eo pipefail
cd ~/sandbox
export PATH=~/.local/bin:$PATH
'"$body"'
touch ~/sandbox/.lamscreen_done
echo "[ls] ALL λ-SCREEN ARMS DONE"
'
    printf '%s' "$script" | _ssh "cat > ~/sandbox/runs/.lamscreen_train.sh"
}

stage_train() {
    _load; _have || die "no pod (run provision)"; _refresh
    if _ssh "test -f ~/sandbox/.lamscreen_done" 2>/dev/null; then log "screen already complete (.lamscreen_done)"; return 0; fi
    if _ssh "tmux has-session -t lamscreen" 2>/dev/null; then log "tmux 'lamscreen' already running — poll to watch"; return 0; fi
    _write_remote_train
    log "launching λ-screen arms [$LAMBDAS] @ $NUM_ITERS iters in tmux 'lamscreen' (HW_ENV: $HW_ENV)"
    _ssh "tmux new-session -d -s lamscreen 'bash ~/sandbox/runs/.lamscreen_train.sh 2>&1 | tee ~/sandbox/runs/lamscreen_orchestrator.log'"
}

stage_poll() {
    _load; _have || die "no pod (run provision)"
    log "polling for ~/sandbox/.lamscreen_done (every 5 min; Ctrl-C safe — re-run resumes)"
    local miss=0
    while :; do
        _refresh
        if _ssh "test -f ~/sandbox/.lamscreen_done" 2>/dev/null; then log "screen complete"; return 0; fi
        if _ssh "tmux has-session -t lamscreen" 2>/dev/null; then
            miss=0
        else
            miss=$((miss + 1))
            if [ "$miss" -ge 3 ]; then
                _ssh "tail -n 30 ~/sandbox/runs/lamscreen_orchestrator.log 2>/dev/null" >&2 || true
                die "tmux 'lamscreen' gone for 3 checks without done-marker — an arm failed. Fix + re-run (resumes the unfinished arm)."
            fi
            log "tmux check missed ($miss/3) — likely transient ssh; retrying in 30s"; sleep 30; continue
        fi
        local p; p=$(_ssh "grep -hoE 'step [0-9]+/[0-9]+' ~/sandbox/runs/${TAG_PREFIX}_*.log 2>/dev/null | tail -1" || true)
        log "still training${p:+ — $p} ($(date -u +%H:%M:%SZ))"; sleep 300
    done
}

stage_download() {
    _load; _have || die "no pod (run provision)"; _refresh
    log "pull results/ + report (tiny: eval.csv + logs)"; bash "$RUNPOD" pull "$POD_ID"
    local lam tag rmt
    for lam in $LAMBDAS; do
        tag="$(_tag "$lam")"
        rmt="$VBASE/base_checkpoints/$tag"
        if [ "$ON_VOLUME" = 1 ] && [ "${DOWNLOAD_CKPT:-0}" != 1 ]; then
            # Checkpoint persists on the network volume — do NOT pull the 4GB (the failure
            # mode that lost the surface-only ckpt). Set DOWNLOAD_CKPT=1 to also keep a local copy.
            if _ssh "ls $rmt/model_*.pt" >/dev/null 2>&1; then
                log "checkpoint $tag persists on the volume (skipping download; DOWNLOAD_CKPT=1 to pull local)"
            else log "no checkpoint for $tag on the volume yet"; fi
        elif _ssh "test -d $rmt" 2>/dev/null; then
            mkdir -p "$SANDBOX_DIR/results/$tag/checkpoint/"; log "rsync checkpoint $tag (model + meta; skip optim shards)"
            _rsync --exclude='optim_*.pt' "root@$POD_IP:$rmt/" "$SANDBOX_DIR/results/$tag/checkpoint/"
        else log "no checkpoint for $tag yet"; fi
        _rsync "root@$POD_IP:~/sandbox/runs/$tag.log" "$SANDBOX_DIR/results/$tag/" 2>/dev/null || true
    done
}

stage_verifydl() {
    local ok=1 lam tag; _load
    echo "" >&2; log "verifying artifacts + reading the λ-screen$([ "$ON_VOLUME" = 1 ] && echo ' (checkpoints persist on the volume)')"
    [ "$ON_VOLUME" = 1 ] && _refresh
    for lam in $LAMBDAS; do
        tag="$(_tag "$lam")"
        local csv="$SANDBOX_DIR/results/$tag/eval.csv" ck="NO"
        if [ "$ON_VOLUME" = 1 ] && _have && _ssh "ls $VBASE/base_checkpoints/$tag/model_*.pt" >/dev/null 2>&1; then
            ck="on-volume"
        elif ls "$SANDBOX_DIR/results/$tag"/checkpoint/model_*.pt >/dev/null 2>&1; then
            ck="local"
        fi
        printf '  %-30s eval.csv:%s checkpoint:%s\n' "$tag" "$([ -f "$csv" ] && echo yes || echo NO)" "$ck" >&2
        { [ -f "$csv" ] && [ "$ck" != "NO" ]; } || ok=0
    done
    [ "$ok" = "1" ] || { log "artifacts INCOMPLETE — NOT terminating. Re-run download (or fix the failed arm)."; return 1; }
    echo "" >&2
    LAMBDAS="$LAMBDAS" TAG_PREFIX="$TAG_PREFIX" NUM_ITERS="$NUM_ITERS" CONTROL_LAM="$CONTROL_LAM" \
        SEED="$SEED" SANDBOX_DIR="$SANDBOX_DIR" python3 "$SANDBOX_DIR/runs/_lambda_screen_readout.py" >&2 || true
}

stage_terminate() {
    _load; _have || { log "no pod in state"; return 0; }
    stage_verifydl || die "refusing to terminate: artifacts not verified local"
    [ "${AUTO_TERMINATE:-0}" = "1" ] || confirm "Artifacts verified. Terminate $POD_ID (stops billing)?" || { log "left pod $POD_ID up"; return 0; }
    bash "$RUNPOD" terminate "$POD_ID"; : > "$STATE_FILE"; log "terminated $POD_ID. Done."
}

ALL="provision bootstrap sync verify train poll download verifydl terminate"
if [ -n "${STAGE:-}" ]; then "stage_$STAGE"; else for s in $ALL; do log "=== STAGE: $s ==="; "stage_$s"; done; fi
