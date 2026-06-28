#!/bin/bash
# runs/runpod.sh — RunPod provider wrapper (GraphQL API), same SHAPE as
# runs/lambda.sh but for RunPod's container model: a pod is a Docker container,
# you get `root` over SSH on an EXPOSED PUBLIC PORT (not user@ip:22), and the
# image is a CUDA/PyTorch base. No Rust/crate build is needed on the host — the
# surface train/eval path is crate-free (it loads a pre-baked tokenizer; the
# rustbpe_force_merges import is lazy/training-only). See runs/runpod_ab.sh.
#
# Subcommands (each prints machine-parseable results on the last stdout line):
#   types [--available]            list H100/A100 8x availability + price
#   datacenters                    storage-capable DCs + 8xGPU stock (where a volume can live)
#   volumes                        list existing network volumes (id/name/size/dc)
#   create-volume <name> <gb> <dc> create a network volume; prints its id
#   launch                         deploy an 8xH100 pod; prints "<podId> <ip> <port>"
#   host       <podId>             re-resolve "<ip> <port>" (survives restarts)
#   ssh        <podId>             interactive ssh into the pod
#   bootstrap  <podId>             apt rsync/git + rsync sandbox + setup.sh + wandb
#   pull       <podId>             rsync ~/sandbox/results/ + report → local
#   terminate  <podId>             terminate (stops billing)
#   status     <podId> | list
#
# Auth: RUNPOD_API_KEY (in ~/.lambda.env, auto-sourced). GraphQL for no-volume deploy +
# host/status/terminate; REST (rest.runpod.io/v1) for volumes + volume-attached deploy.
# Env:
#   RUNPOD_GPU_TYPE     default "NVIDIA H100 80GB HBM3"  (8x H100 SXM)
#   RUNPOD_GPU_COUNT    default 8
#   RUNPOD_CLOUD        SECURE | COMMUNITY | ALL  (default SECURE)
#   RUNPOD_IMAGE        default runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04
#   RUNPOD_DISK_GB      container disk GB (default 300; ~170 data shards + 2 d24 ckpts)
#   RUNPOD_POD_NAME     default surface-ab
#   RUNPOD_PUBKEY_FILE  ssh public key to inject (default: ~/.ssh/id_ed25519.pub then id_rsa.pub)
#   WANDB_API_KEY       required for bootstrap (piped to host, never on argv)
#   NANOCHAT_COMMIT     optional pin forwarded to the host setup.sh
#   --- NETWORK VOLUME (persist checkpoints across termination; no fragile 4GB download) ---
#   RUNPOD_VOLUME_ID    if set, deploy via REST with this network volume attached. The
#                       volume is DATACENTER-PINNED, so deploys are constrained to its DC
#                       (only storage-capable DCs have it; run `datacenters` to see stock).
#   RUNPOD_DATACENTER   the volume's datacenter id (REQUIRED with RUNPOD_VOLUME_ID), e.g. US-GA-2
#   RUNPOD_VOLUME_MOUNT mount path on the pod (default /workspace — RunPod's only supported path)

set -euo pipefail
SANDBOX_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LAMBDA_ENV_FILE="${LAMBDA_ENV_FILE:-$HOME/.lambda.env}"
[ -f "$LAMBDA_ENV_FILE" ] && { set +u; . "$LAMBDA_ENV_FILE"; set -u; }
: "${RUNPOD_API_KEY:?RUNPOD_API_KEY required (put it in ~/.lambda.env)}"

GQL="https://api.runpod.io/graphql?api_key=$RUNPOD_API_KEY"
GPU_TYPE="${RUNPOD_GPU_TYPE:-NVIDIA H100 80GB HBM3}"
GPU_COUNT="${RUNPOD_GPU_COUNT:-8}"
CLOUD="${RUNPOD_CLOUD:-SECURE}"
IMAGE="${RUNPOD_IMAGE:-runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04}"
DISK_GB="${RUNPOD_DISK_GB:-300}"
POD_NAME="${RUNPOD_POD_NAME:-surface-ab}"
# SSH identity: discover a private key (prefer the lambda_cloud key this project
# uses), pin it with IdentitiesOnly so a multi-key agent doesn't trip MaxAuthTries.
RUNPOD_SSH_KEY="${RUNPOD_SSH_KEY:-}"
if [ -z "$RUNPOD_SSH_KEY" ]; then
    for c in "$HOME/.ssh/lambda_cloud_ed25519" "$HOME/.ssh/id_ed25519" "$HOME/.ssh/id_rsa"; do
        [ -f "$c" ] && RUNPOD_SSH_KEY="$c" && break
    done
fi
SSH_OPTS="-o StrictHostKeyChecking=accept-new -o ServerAliveInterval=30${RUNPOD_SSH_KEY:+ -i $RUNPOD_SSH_KEY -o IdentitiesOnly=yes}"

log() { echo "==> $*" >&2; }
die() { echo "ERROR: $*" >&2; exit 1; }

# gql '<graphql string>'  → raw JSON (errors surfaced)
gql() {
    local q; q=$(jq -Rn --arg q "$1" '{query:$q}')
    curl -sS -X POST "$GQL" -H "Content-Type: application/json" -d "$q"
}

# rest METHOD PATH [JSON_BODY] → raw JSON. The newer REST API (Bearer auth) — used for
# network volumes (GET/POST /networkvolumes) and volume-attached pod deploy (POST /pods),
# which the GraphQL podFindAndDeployOnDemand can't do (no networkVolumeId field there).
REST_BASE="https://rest.runpod.io/v1"
rest() {
    local method="$1" path="$2" body="${3:-}"
    if [ -n "$body" ]; then
        curl -sS -X "$method" "$REST_BASE$path" -H "Authorization: Bearer $RUNPOD_API_KEY" \
            -H "Content-Type: application/json" -d "$body"
    else
        curl -sS -X "$method" "$REST_BASE$path" -H "Authorization: Bearer $RUNPOD_API_KEY"
    fi
}

_pubkey() {
    local f="${RUNPOD_PUBKEY_FILE:-${RUNPOD_SSH_KEY:+$RUNPOD_SSH_KEY.pub}}"
    [ -n "$f" ] && [ -f "$f" ] || die "no ssh public key (set RUNPOD_PUBKEY_FILE, or RUNPOD_SSH_KEY with a .pub beside it)"
    cat "$f"
}

# resolve a running pod's public ssh "<ip> <port>" (privatePort 22 mapping)
_host() {
    local id="$1" js
    js=$(gql "query{ pod(input:{podId:\"$id\"}){ id desiredStatus runtime{ ports{ ip isIpPublic privatePort publicPort type } } } }")
    echo "$js" | jq -r '.data.pod.runtime.ports[]? | select(.privatePort==22 and .isIpPublic==true) | "\(.ip) \(.publicPort)"' | head -1
}

_ssh_to() { local ip="$1" port="$2"; shift 2; ssh $SSH_OPTS -p "$port" "root@$ip" "$@"; }

cmd_types() {
    local sel=""
    [ "${1:-}" = "--available" ] && sel='| select(.lowestPrice.stockStatus != null)'
    gql "query{ gpuTypes{ id displayName secureCloud communityCloud lowestPrice(input:{gpuCount:$GPU_COUNT}){ uninterruptablePrice stockStatus } } }" \
        | jq -r ".data.gpuTypes[]? | select(.displayName|test(\"H100|A100\")) $sel | \"\(.displayName) | ${GPU_COUNT}x stock:\(.lowestPrice.stockStatus // \"none\") \$\(.lowestPrice.uninterruptablePrice // \"-\")/hr | id=\(.id)\""
}

# poll a just-created pod id → RUNNING + a public SSH port, wait for sshd, then print
# "<id> <ip> <port>" (the parseable last line). Shared by the GraphQL + REST deploy paths.
_await_ssh() {
    local id="$1"
    log "pod id: $id — polling for RUNNING + a public SSH port (2-5 min)"
    local ip="" port="" st="" tries=0
    while :; do
        st=$(gql "query{ pod(input:{podId:\"$id\"}){ desiredStatus } }" 2>/dev/null | jq -r '.data.pod.desiredStatus // "?"' 2>/dev/null || echo "?")
        # reset + `|| true`: before the port is ready _host emits nothing and `read` hits
        # EOF (non-zero) → under set -e that would abort launch and orphan the pod.
        ip=""; port=""; read -r ip port < <(_host "$id") || true
        echo "    status=$st ssh=${ip:+$ip:$port}" >&2
        [ -n "${ip:-}" ] && [ -n "${port:-}" ] && break
        tries=$((tries+1)); [ "$tries" -gt 60 ] && die "pod never exposed a public SSH port (check the RunPod console for $id)"
        sleep 10
    done
    log "waiting for sshd at $ip:$port"
    for _ in $(seq 1 30); do _ssh_to "$ip" "$port" true 2>/dev/null && break; sleep 10; done
    echo "$id $ip $port"     # last line: parseable
}

# GraphQL on-demand deploy (NO network volume). Prints the pod id.
_deploy_gql() {
    local pubkey; pubkey=$(_pubkey | tr -d '\n')   # ssh keys are quote/backslash-free → embed directly
    log "deploying ${GPU_COUNT}x '$GPU_TYPE' ($CLOUD), disk=${DISK_GB}GB, image=$IMAGE (no volume)"
    local m id
    m=$(gql "mutation{ podFindAndDeployOnDemand(input:{
            cloudType:$CLOUD gpuCount:$GPU_COUNT gpuTypeId:\"$GPU_TYPE\"
            name:\"$POD_NAME\" imageName:\"$IMAGE\"
            containerDiskInGb:$DISK_GB volumeInGb:0
            ports:\"22/tcp\" supportPublicIp:true startSsh:true
            env:[{key:\"PUBLIC_KEY\" value:\"$pubkey\"}]
        }){ id } }")
    id=$(echo "$m" | jq -r '.data.podFindAndDeployOnDemand.id // empty')
    [ -n "$id" ] || { echo "$m" | jq -r '.errors[0].message // .' >&2; die "deploy failed"; }
    echo "$id"
}

# REST deploy WITH a network volume (mounts at $RUNPOD_VOLUME_MOUNT, pins the pod to the
# volume's datacenter). Prints the pod id. Required because the GraphQL deploy has no
# networkVolumeId field.
_deploy_rest() {
    local pubkey; pubkey=$(_pubkey | tr -d '\n')
    : "${RUNPOD_DATACENTER:?RUNPOD_DATACENTER required with RUNPOD_VOLUME_ID (the volume datacenter id, e.g. US-GA-2)}"
    local mount="${RUNPOD_VOLUME_MOUNT:-/workspace}"
    log "deploying ${GPU_COUNT}x '$GPU_TYPE' ($CLOUD) in $RUNPOD_DATACENTER, volume $RUNPOD_VOLUME_ID @ $mount, disk=${DISK_GB}GB"
    local body r id
    body=$(jq -n --arg name "$POD_NAME" --arg img "$IMAGE" --arg cloud "$CLOUD" \
        --argjson gc "$GPU_COUNT" --arg gt "$GPU_TYPE" --arg dc "$RUNPOD_DATACENTER" \
        --arg vol "$RUNPOD_VOLUME_ID" --arg mount "$mount" --argjson disk "$DISK_GB" --arg pk "$pubkey" \
        '{name:$name, imageName:$img, cloudType:$cloud, computeType:"GPU",
          gpuCount:$gc, gpuTypeIds:[$gt], dataCenterIds:[$dc],
          networkVolumeId:$vol, volumeMountPath:$mount, volumeInGb:0,
          containerDiskInGb:$disk, ports:["22/tcp"], env:{PUBLIC_KEY:$pk}}')
    r=$(rest POST /pods "$body")
    id=$(echo "$r" | jq -r '.id // empty')
    [ -n "$id" ] || { echo "$r" | jq -r '.error // .message // .' >&2; die "deploy (REST + volume) failed"; }
    echo "$id"
}

cmd_launch() {
    local id
    if [ -n "${RUNPOD_VOLUME_ID:-}" ]; then id=$(_deploy_rest); else id=$(_deploy_gql); fi
    _await_ssh "$id"
}

cmd_datacenters() {
    log "storage-capable datacenters + ${GPU_COUNT}x '$GPU_TYPE' stock (a network volume can only live where storage=true):"
    local dc st
    for dc in $(gql 'query{ dataCenters { id storageSupport } }' | jq -r '.data.dataCenters[]? | select(.storageSupport==true) | .id'); do
        st=$(gql "query{ gpuTypes(input:{id:\"$GPU_TYPE\"}){ lowestPrice(input:{gpuCount:$GPU_COUNT, dataCenterId:\"$dc\"}){ stockStatus uninterruptablePrice } } }" \
            | jq -r '.data.gpuTypes[0].lowestPrice | "stock=\(.stockStatus // "none") $\(.uninterruptablePrice // "-")/hr"' 2>/dev/null)
        printf '  %-10s %s\n' "$dc" "$st" >&2
    done
}

cmd_volumes() {
    rest GET /networkvolumes | jq -r 'if type=="array" then (.[] | "id=\(.id)  name=\(.name)  size=\(.size)GB  dc=\(.dataCenterId)") elif .error then "ERROR: \(.error)" else . end'
}

cmd_create_volume() {
    local name="${1:?create-volume <name> <sizeGB> <dataCenterId>}" size="${2:?<sizeGB>}" dc="${3:?<dataCenterId>}"
    local body r id
    body=$(jq -n --arg n "$name" --argjson s "$size" --arg dc "$dc" '{name:$n, size:$s, dataCenterId:$dc}')
    r=$(rest POST /networkvolumes "$body")
    id=$(echo "$r" | jq -r '.id // empty')
    [ -n "$id" ] || { echo "$r" | jq -r '.error // .message // .' >&2; die "volume create failed"; }
    log "created network volume id=$id ($name, ${size}GB, $dc)."
    log "→ add to ~/.lambda.env:  export RUNPOD_VOLUME_ID=$id ; export RUNPOD_DATACENTER=$dc"
    echo "$id"
}

cmd_host()   { local id="${1:?host <podId>}"; _host "$id"; }
cmd_status() { local id="${1:?status <podId>}"; gql "query{ pod(input:{podId:\"$id\"}){ id name desiredStatus machineId } }" | jq '.data.pod'; }
cmd_list()   { gql 'query{ myself{ pods{ id name desiredStatus machine{ gpuDisplayName } } } }' | jq '.data.myself.pods'; }

cmd_ssh() {
    local id="${1:?ssh <podId>}" ip port
    read -r ip port < <(_host "$id"); [ -n "${ip:-}" ] || die "no public SSH port for $id (running?)"
    log "ssh root@$ip -p $port"
    exec ssh $SSH_OPTS -p "$port" "root@$ip"
}

cmd_bootstrap() {
    local id="${1:?bootstrap <podId>}"
    if [ -z "${WANDB_API_KEY:-}" ] && [ "${SKIP_WANDB:-0}" != "1" ]; then
        die "WANDB_API_KEY not set — training dies in wandb.init() ~5 min in. Set it (or SKIP_WANDB=1)."
    fi
    local ip port; read -r ip port < <(_host "$id"); [ -n "${ip:-}" ] || die "no SSH port for $id"

    # the base image may lack rsync/git/tmux; install before the rsync (ssh works without them)
    log "ensuring rsync/git/curl/tmux on the pod"
    _ssh_to "$ip" "$port" '(command -v rsync >/dev/null && command -v git >/dev/null && command -v tmux >/dev/null) || (apt-get update -qq && apt-get install -y -qq rsync git curl tmux)'

    log "rsync $SANDBOX_DIR/ → root@$ip:~/sandbox/  (python only — pod is crate-free)"
    # exclude everything the crate-free pod doesn't need: build artifacts (.venv,
    # target, *.so), git history, and the big auto_tune audit checkpoints. Only
    # the Python sources (incl. rustbpe_variants/force_merges/pairs.py) go up.
    rsync -a --delete --exclude=.venv --exclude=__pycache__ --exclude=results \
        --exclude='*.pyc' --exclude=target --exclude=.git --exclude=audit --exclude='*.so' \
        -e "ssh $SSH_OPTS -p $port" "$SANDBOX_DIR/" "root@$ip:~/sandbox/"

    local pin=""; [ -n "${NANOCHAT_COMMIT:-}" ] && pin="NANOCHAT_COMMIT='$NANOCHAT_COMMIT' "
    log "setup.sh on the pod (uv, clone nanochat, uv sync --extra gpu, smoke)"
    _ssh_to "$ip" "$port" "${pin}bash ~/sandbox/runs/setup.sh"

    if [ -n "${WANDB_API_KEY:-}" ]; then
        log "seeding wandb credentials (piped over stdin)"
        _ssh_to "$ip" "$port" 'export PATH="$HOME/.local/bin:$PATH"; cd ~/sandbox && uv run wandb login --relogin "$(cat)"' <<< "$WANDB_API_KEY" \
            && log "wandb login: ok" || die "wandb login failed on host (see output above)"
    fi
    log "bootstrap complete for $id"
}

cmd_pull() {
    local id="${1:?pull <podId>}" ip port; read -r ip port < <(_host "$id"); [ -n "${ip:-}" ] || die "no SSH port for $id"
    local dest="$SANDBOX_DIR/results"; mkdir -p "$dest"
    log "rsync root@$ip:~/sandbox/results/ → $dest/"
    rsync -a -e "ssh $SSH_OPTS -p $port" "root@$ip:~/sandbox/results/" "$dest/"
    if _ssh_to "$ip" "$port" "test -f ~/sandbox/report.md" 2>/dev/null; then
        rsync -a -e "ssh $SSH_OPTS -p $port" "root@$ip:~/sandbox/report.md" "$dest/report_$(date -u +%Y%m%dT%H%M%SZ).md" >&2
    fi
}

cmd_terminate() {
    local id="${1:?terminate <podId>}"
    log "terminating $id"
    gql "mutation{ podTerminate(input:{podId:\"$id\"}) }" | jq -r 'if .errors then .errors[0].message else "terminated" end' >&2
}

case "${1:-}" in
    ""|-h|--help) sed -n '2,30p' "$0" | sed 's/^# \{0,1\}//' >&2; exit 0 ;;
esac
CMD="$1"; shift
case "$CMD" in
    types) cmd_types "$@";; datacenters) cmd_datacenters "$@";;
    volumes) cmd_volumes "$@";; create-volume) cmd_create_volume "$@";;
    launch) cmd_launch "$@";; host) cmd_host "$@";;
    ssh) cmd_ssh "$@";; bootstrap) cmd_bootstrap "$@";; pull) cmd_pull "$@";;
    terminate) cmd_terminate "$@";; status) cmd_status "$@";; list) cmd_list "$@";;
    *) die "unknown command: $CMD";;
esac
