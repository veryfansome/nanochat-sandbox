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
#   bash runs/lambda.sh pull       <instance_id>      # rsync host's ~/sandbox/results/ (auto-archives) + report.md → local
#   bash runs/lambda.sh terminate  <instance_id>
#
# Env defaults (override per-command with flags above):
#   LAMBDA_API_KEY        required
#   LAMBDA_INSTANCE_TYPE  gpu_8x_h100_sxm5
#   LAMBDA_REGION         (empty = auto-pick any region with capacity)
#   LAMBDA_SSH_KEY        (required for launch; name of an SSH key already
#                          uploaded to your Lambda account — not a local file)
#   LAMBDA_SSH_USER       ubuntu
#   WANDB_API_KEY         required for `bootstrap` (unless SKIP_WANDB=1); the
#                         key is piped to the host over stdin and `wandb login`
#                         is run there so training auto-authenticates. Hard-
#                         fails fast if missing — training otherwise dies in
#                         `wandb.init()` after the bootstrap+tokenizer cost
#   SKIP_WANDB            set to 1 to bypass the WANDB_API_KEY check during
#                         `bootstrap` (only if you'll `wandb login` manually
#                         on the host before training)
#   NANOCHAT_COMMIT       optional; if set during `bootstrap`, forwarded to
#                         the host's setup.sh so nanochat is checked out at
#                         that commit (for reproducible / pinned A/B runs)
#   LAMBDA_ENV_FILE       path to a credentials file sourced on startup
#                         (default: ~/.lambda.env; set to /dev/null to skip).
#                         Convention: keep `export FOO=...` lines for the
#                         vars above so you don't have to source manually or
#                         wire them into your shell rc.
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

# Source ~/.lambda.env (or $LAMBDA_ENV_FILE) if present — convenience so
# credentials don't have to be sourced manually or wired into the shell rc.
# Standard `source` semantics: variables set by the file override anything
# already set in the calling shell. For a one-off override, edit the file
# or set LAMBDA_ENV_FILE=/dev/null to skip.
LAMBDA_ENV_FILE="${LAMBDA_ENV_FILE:-$HOME/.lambda.env}"
if [ -f "$LAMBDA_ENV_FILE" ]; then
    # shellcheck disable=SC1090,SC1091
    source "$LAMBDA_ENV_FILE"
fi

: "${LAMBDA_API_KEY:?LAMBDA_API_KEY is required (https://cloud.lambda.ai/api-keys, or set in ~/.lambda.env)}"

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

    # Fail fast if wandb credentials won't be available — BEFORE any API call
    # or ssh setup. Otherwise training dies ~5 min in after the costly
    # bootstrap + tokenizer setup, having silently slid past a stderr
    # "skipping" message that was easy to miss.
    if [ -z "${WANDB_API_KEY:-}" ] && [ "${SKIP_WANDB:-0}" != "1" ]; then
        cat >&2 <<'EOF'

ERROR: WANDB_API_KEY is not set locally — refusing to bootstrap.

Training will fail when wandb.init() runs ~5 min in, *after* the bootstrap +
tokenizer cost. Fix one of:

  1. Set WANDB_API_KEY in your env or ~/.lambda.env, then re-run bootstrap.

  2. Pass SKIP_WANDB=1 if you intend to run `uv run wandb login` manually on
     the host before training:

         SKIP_WANDB=1 bash runs/lambda.sh bootstrap "$ID"

EOF
        exit 1
    fi

    local ip user
    ip=$(_resolve_ip "$id")
    user="${LAMBDA_SSH_USER:-ubuntu}"

    echo "==> rsync $SANDBOX_DIR/ → $user@$ip:~/sandbox/" >&2
    # Excludes: .venv (host-built, Lambda will uv sync fresh); __pycache__/*.pyc
    # (rebuilt on first import); results (per-run archives — pulled the other
    # direction); target (~900MB of Mac-built Rust artifacts under rustbpe/ +
    # rustbpe_variants/*/, useless on Linux — Lambda rebuilds via
    # runs/build_rustbpe.sh if a variant is actually needed there).
    rsync -av --delete \
        --exclude=.venv --exclude=__pycache__ --exclude=results --exclude='*.pyc' \
        --exclude=target \
        -e "ssh -o StrictHostKeyChecking=accept-new" \
        "$SANDBOX_DIR/" "$user@$ip:~/sandbox/"

    # Forward NANOCHAT_COMMIT to the remote setup.sh if set locally — pins
    # the nanochat checkout to a specific commit for reproducible A/B runs.
    local remote_env=""
    if [ -n "${NANOCHAT_COMMIT:-}" ]; then
        remote_env="NANOCHAT_COMMIT='$NANOCHAT_COMMIT' "
        echo "==> pinning nanochat to commit: $NANOCHAT_COMMIT" >&2
    fi

    echo "==> running setup.sh on the instance (installs uv, clones nanochat, uv sync --extra gpu, drops .pth, smokes)" >&2
    ssh -o StrictHostKeyChecking=accept-new "$user@$ip" "${remote_env}bash ~/sandbox/runs/setup.sh"

    # Seed wandb credentials on the host. Key piped over SSH stdin (never on
    # command line) so it doesn't appear in process listings on either side.
    # `wandb login` is the authoritative check — trust its exit code rather
    # than second-guessing where it writes credentials (file location varies
    # across wandb versions: ~/.netrc, ~/.config/wandb/, etc.). The
    # WANDB_API_KEY-required check at the top of this function means we get
    # here only if it's set or SKIP_WANDB=1.
    if [ -n "${WANDB_API_KEY:-}" ]; then
        echo "==> seeding wandb credentials on the host (from local WANDB_API_KEY)" >&2
        # Non-interactive ssh doesn't source ~/.bashrc / ~/.profile, so the
        # uv installer's PATH addition (~/.local/bin) isn't picked up. Add it
        # explicitly. (setup.sh / speedrun.sh / runcpu.sh do the same at top.)
        if ssh -o StrictHostKeyChecking=accept-new "$user@$ip" \
            'export PATH="$HOME/.local/bin:$PATH" && cd ~/sandbox && uv run wandb login --relogin "$(cat)"' \
            <<< "$WANDB_API_KEY"; then
            echo "    wandb login: ok" >&2
        else
            cat >&2 <<EOF

ERROR: wandb login on the host exited non-zero. See the wandb output above.
       Common causes:
         - WANDB_API_KEY value is malformed (extra whitespace? wrong format?)
         - wandb CLI couldn't reach api.wandb.ai (network / proxy issue)
         - the venv on the host doesn't have wandb installed (re-run bootstrap
           to ensure setup.sh finished)
       Training will crash on wandb.init() ~5 min in; fix before launching.

       Workaround: ssh in and run \`uv run wandb login\` interactively, then
       re-launch training (no need to re-bootstrap).
EOF
            exit 1
        fi
    else
        echo "==> SKIP_WANDB=1; not seeding wandb credentials on host" >&2
        echo "    (run 'uv run wandb login' on the host before training, or it will fail in wandb.init())" >&2
    fi

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

cmd_pull() {
    local id="${1:?usage: pull <id>}"
    local ip user
    ip=$(_resolve_ip "$id")
    user="${LAMBDA_SSH_USER:-ubuntu}"

    local dest="$SANDBOX_DIR/results"
    mkdir -p "$dest"

    # Mirror the host's auto-archive dirs. rsync is additive (no --delete) so
    # multiple pulls accumulate cleanly; running it twice on the same data is a
    # no-op. Per-run dirs are keyed by MODEL_TAG and won't collide between runs.
    echo "==> rsync $user@$ip:~/sandbox/results/ → $dest/" >&2
    rsync -av \
        -e "ssh -o StrictHostKeyChecking=accept-new" \
        "$user@$ip:~/sandbox/results/" "$dest/"

    # The concatenated report.md (from `python -m nanochat.report generate` at
    # the end of speedrun.sh) is written to cwd, which is ~/sandbox on the host.
    # Save it alongside the per-run archive dirs with a UTC timestamp so
    # successive pulls (after multiple runs) don't clobber each other.
    if ssh -o StrictHostKeyChecking=accept-new "$user@$ip" "test -f ~/sandbox/report.md" 2>/dev/null; then
        local ts="$(date -u +%Y%m%dT%H%M%SZ)"
        local report_dest="$dest/report_${ts}.md"
        rsync -av \
            -e "ssh -o StrictHostKeyChecking=accept-new" \
            "$user@$ip:~/sandbox/report.md" "$report_dest" >&2
        echo "==> pulled report.md → $report_dest" >&2
    else
        echo "==> no ~/sandbox/report.md on host yet (speedrun's report-generate step hasn't run)" >&2
    fi

    echo >&2
    echo "==> local $dest/:" >&2
    ls -la "$dest" >&2
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
    pull)      cmd_pull      "$@" ;;
    terminate) cmd_terminate "$@" ;;
    *) echo "unknown command: $CMD" >&2; exit 1 ;;
esac
