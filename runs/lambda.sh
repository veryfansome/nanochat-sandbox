#!/bin/bash
# Minimal Lambda Cloud wrapper for provisioning a GPU instance, rsync-ing this
# sandbox onto it, and running setup.sh. Just curl + jq + ssh + rsync, no
# additional dependencies. Built against the documented REST API at
# https://cloud.lambdalabs.com/api/v1 (the API endpoint hasn't moved despite
# the dashboard rebrand to cloud.lambda.ai).
#
# Auth: set LAMBDA_API_KEY in your environment. Get one at
#   https://cloud.lambda.ai/api-keys
#
# Usage:
#   bash runs/lambda.sh types [--available]
#   bash runs/lambda.sh launch [--type T] [--region R] --ssh-key NAME
#   bash runs/lambda.sh list
#   bash runs/lambda.sh status     <instance_id>
#   bash runs/lambda.sh ssh        <instance_id>
#   bash runs/lambda.sh bootstrap  <instance_id>      # rsync sandbox/ + run setup.sh
#   bash runs/lambda.sh terminate  <instance_id>
#
# Env defaults (override per-command with flags above):
#   LAMBDA_API_KEY        required
#   LAMBDA_INSTANCE_TYPE  gpu_8x_h100_sxm5
#   LAMBDA_REGION         (empty = auto-pick any region with capacity)
#   LAMBDA_SSH_KEY        (required for launch; name of an SSH key already
#                          uploaded to your Lambda account — not a local file)
#   LAMBDA_SSH_USER       ubuntu
#
# End-to-end happy path:
#   export LAMBDA_API_KEY=...
#   export LAMBDA_SSH_KEY=mykey
#   read ID IP < <(bash runs/lambda.sh launch | tail -1)
#   bash runs/lambda.sh bootstrap "$ID"
#   bash runs/lambda.sh ssh "$ID"
#   # then on the instance:
#   #   OVERLAY=zloss WANDB_RUN=zloss tmux new -s sr "cd ~/sandbox && bash runs/speedrun.sh 2>&1 | tee runs/sr.log"
#   bash runs/lambda.sh terminate "$ID"   # don't forget — billing is per-hour

set -euo pipefail

API="https://cloud.lambdalabs.com/api/v1"
SANDBOX_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

usage() {
    sed -n '/^# Usage:/,/^# End-to-end/p' "$0" | sed 's/^# \{0,1\}//' >&2
}

# Print usage if no subcommand, -h, or --help (before requiring LAMBDA_API_KEY)
case "${1:-}" in
    ""|-h|--help) usage; exit 0 ;;
esac

: "${LAMBDA_API_KEY:?LAMBDA_API_KEY is required (https://cloud.lambda.ai/api-keys)}"

# --- API plumbing -----------------------------------------------------------

# api METHOD PATH [-d BODY]
api() {
    local method="$1" path="$2"; shift 2
    curl --fail-with-body -sS -u "$LAMBDA_API_KEY:" \
        -X "$method" -H "Content-Type: application/json" \
        "$@" "$API$path"
}

# fetch one instance's full record by id (filtered from /instances list)
_get_instance() {
    local id="$1"
    api GET "/instances" | jq --arg id "$id" '.data[] | select(.id == $id)'
}

# --- subcommands ------------------------------------------------------------

cmd_types() {
    local jq_filter='.data | to_entries[]'
    if [ "${1:-}" = "--available" ]; then
        jq_filter="$jq_filter | select(.value.regions_with_capacity_available | length > 0)"
    fi
    api GET "/instance-types" | jq "$jq_filter | {type: .key, regions: [.value.regions_with_capacity_available[].name]}"
}

cmd_launch() {
    local type="${LAMBDA_INSTANCE_TYPE:-gpu_8x_h100_sxm5}"
    local region="${LAMBDA_REGION:-}"
    local ssh_key="${LAMBDA_SSH_KEY:-}"

    while [ $# -gt 0 ]; do
        case "$1" in
            --type)    type="$2";    shift 2 ;;
            --region)  region="$2";  shift 2 ;;
            --ssh-key) ssh_key="$2"; shift 2 ;;
            *) echo "launch: unknown arg: $1" >&2; exit 1 ;;
        esac
    done
    [ -n "$ssh_key" ] || { echo "launch: --ssh-key (or LAMBDA_SSH_KEY env) is required" >&2; exit 1; }

    # auto-pick a region with capacity if none specified
    if [ -z "$region" ]; then
        region=$(api GET "/instance-types" | jq -r --arg t "$type" '
            (.data[$t].regions_with_capacity_available // []) as $r |
            if ($r | length) > 0 then $r[0].name else empty end')
        [ -n "$region" ] || { echo "launch: no capacity for $type in any region right now" >&2; exit 1; }
        echo "==> auto-picked region: $region (has capacity for $type)" >&2
    fi

    local payload
    payload=$(jq -n --arg r "$region" --arg t "$type" --arg k "$ssh_key" \
        '{region_name: $r, instance_type_name: $t, ssh_key_names: [$k]}')

    echo "==> launching $type in $region (ssh-key=$ssh_key)" >&2
    local resp id
    resp=$(api POST "/instance-operations/launch" -d "$payload")
    id=$(echo "$resp" | jq -r '.data.instance_ids[0]')
    [ "$id" != "null" ] || { echo "$resp" >&2; exit 1; }
    echo "==> instance id: $id" >&2

    echo "==> polling for active state (typically 2-5 min)" >&2
    local status="" ip="" detail
    while true; do
        detail=$(_get_instance "$id")
        status=$(echo "$detail" | jq -r '.status // "missing"')
        ip=$(echo "$detail"     | jq -r '.ip // ""')
        echo "    status=$status ip=${ip:-<pending>}" >&2
        case "$status" in
            active) break ;;
            terminated|terminating|failed)
                echo "instance entered $status unexpectedly" >&2; exit 1 ;;
        esac
        sleep 10
    done

    # last line on stdout is shell-parseable: "<id> <ip>"
    echo "$id $ip"
}

cmd_list() {
    api GET "/instances" | jq '.data[] | {id, name, type: .instance_type.name, region: .region.name, status, ip}'
}

cmd_status() {
    local id="${1:?usage: status <id>}"
    _get_instance "$id" | jq '{id, status, ip, type: .instance_type.name, region: .region.name}'
}

_resolve_ip() {
    local id="$1" ip
    ip=$(_get_instance "$id" | jq -r '.ip // ""')
    [ -n "$ip" ] || { echo "no IP for $id (still booting? check: status $id)" >&2; exit 1; }
    echo "$ip"
}

cmd_ssh() {
    local id="${1:?usage: ssh <id>}"
    local ip user
    ip=$(_resolve_ip "$id")
    user="${LAMBDA_SSH_USER:-ubuntu}"
    echo "==> ssh $user@$ip" >&2
    exec ssh -o StrictHostKeyChecking=accept-new "$user@$ip"
}

cmd_bootstrap() {
    local id="${1:?usage: bootstrap <id>}"
    local ip user
    ip=$(_resolve_ip "$id")
    user="${LAMBDA_SSH_USER:-ubuntu}"

    echo "==> rsync $SANDBOX_DIR/ → $user@$ip:~/sandbox/" >&2
    rsync -av --delete \
        --exclude=.venv --exclude=__pycache__ --exclude=results --exclude='*.pyc' \
        -e "ssh -o StrictHostKeyChecking=accept-new" \
        "$SANDBOX_DIR/" "$user@$ip:~/sandbox/"

    echo "==> running setup.sh on the instance (installs uv, clones nanochat, uv sync --extra gpu, drops .pth, smokes)" >&2
    ssh -o StrictHostKeyChecking=accept-new "$user@$ip" "bash ~/sandbox/runs/setup.sh"

    cat >&2 <<EOF

==> bootstrap complete.
    next:
      bash runs/lambda.sh ssh $id
    then on the instance:
      OVERLAY=zloss WANDB_RUN=zloss tmux new -s sr "cd ~/sandbox && bash runs/speedrun.sh 2>&1 | tee runs/sr.log"
    when done:
      bash runs/lambda.sh terminate $id
EOF
}

cmd_terminate() {
    local id="${1:?usage: terminate <id>}"
    local payload
    payload=$(jq -n --arg id "$id" '{instance_ids: [$id]}')
    echo "==> terminating $id" >&2
    api POST "/instance-operations/terminate" -d "$payload" \
        | jq '.data.terminated_instances[]? | {id, name, status}'
}

# --- dispatch ---------------------------------------------------------------

CMD="$1"
shift
case "$CMD" in
    types)     cmd_types     "$@" ;;
    launch)    cmd_launch    "$@" ;;
    list)      cmd_list      "$@" ;;
    status)    cmd_status    "$@" ;;
    ssh)       cmd_ssh       "$@" ;;
    bootstrap) cmd_bootstrap "$@" ;;
    terminate) cmd_terminate "$@" ;;
    *) echo "unknown command: $CMD" >&2; exit 1 ;;
esac
