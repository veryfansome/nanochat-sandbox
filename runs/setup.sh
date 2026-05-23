#!/bin/bash
# One-shot bootstrap on a fresh machine (GPU instance or local dev box).
#
# Idempotent — safe to re-run. From any directory:
#     bash /path/to/sandbox/runs/setup.sh
#
# Env vars:
#   NANOCHAT_GIT    upstream URL (default: https://github.com/karpathy/nanochat.git)
#   NANOCHAT_COMMIT optional pin (default: tip of default branch)
#   EXTRA           uv extra to install (default: "gpu"; set to "cpu" for CPU/MPS)
#
# What it does:
#   1. installs uv (if missing)
#   2. clones nanochat as a sibling of sandbox/ (if missing)
#   3. uv sync --extra $EXTRA in sandbox/
#   4. drops a .pth file in the venv so `import nanochat` / `import scripts`
#      work from any `uv run python` invocation (not just `run.*` wrappers)
#   5. runs the z-loss smoke test to verify wiring

set -euo pipefail

SANDBOX_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PARENT_DIR="$(cd "$SANDBOX_DIR/.." && pwd)"
NANOCHAT_DIR="$PARENT_DIR/nanochat"
NANOCHAT_GIT="${NANOCHAT_GIT:-https://github.com/karpathy/nanochat.git}"
NANOCHAT_COMMIT="${NANOCHAT_COMMIT:-}"
EXTRA="${EXTRA:-gpu}"

echo "==> sandbox dir:   $SANDBOX_DIR"
echo "==> nanochat dir:  $NANOCHAT_DIR"
echo "==> uv extra:      $EXTRA"

echo
echo "[1/5] install uv"
if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"
uv --version

echo
echo "[2/5] clone nanochat as sibling (if missing)"
if [ ! -d "$NANOCHAT_DIR" ]; then
    git clone "$NANOCHAT_GIT" "$NANOCHAT_DIR"
else
    echo "  $NANOCHAT_DIR already exists, skipping clone"
fi
if [ -n "$NANOCHAT_COMMIT" ]; then
    git -C "$NANOCHAT_DIR" fetch --all --tags
    git -C "$NANOCHAT_DIR" checkout "$NANOCHAT_COMMIT"
fi
echo "  nanochat at $(git -C "$NANOCHAT_DIR" rev-parse --short HEAD)"

echo
echo "[3/5] uv sync --extra $EXTRA"
cd "$SANDBOX_DIR"
uv sync --extra "$EXTRA"

echo
echo "[4/5] drop _overlay_bootstrap.pth in site-packages"
SITE_PKGS="$(uv run python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
PTH="$SITE_PKGS/_overlay_bootstrap.pth"
echo "import overlay" > "$PTH"
echo "  wrote $PTH"
echo "  (makes \`import nanochat\` and \`import scripts\` work from any uv-run python)"

echo
echo "[5/5] smoke test"
uv run python -m wrappers.smoke_zloss

echo
echo "==> setup complete."
echo "    sandbox:  $SANDBOX_DIR"
echo "    nanochat: $NANOCHAT_DIR  ($(git -C "$NANOCHAT_DIR" rev-parse --short HEAD))"
echo
echo "Next: launch a full run in tmux (8xGPU speedrun):"
echo "    OVERLAY=zloss tmux new -s sr \"cd $SANDBOX_DIR && bash runs/speedrun.sh 2>&1 | tee runs/sr.log\""
echo "Or the CPU smoke recipe:"
echo "    OVERLAY=zloss bash $SANDBOX_DIR/runs/runcpu.sh"
