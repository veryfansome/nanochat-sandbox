#!/bin/bash
# Build the vendored rustbpe (or a variant) into the sandbox venv.
#
# Usage:
#     bash runs/build_rustbpe.sh                    # build pristine sandbox/rustbpe/
#     VARIANT=auto_tune RELEASE=1 bash runs/build_rustbpe.sh
#     VARIANT=force_merges RELEASE=1 bash runs/build_rustbpe.sh
#
# Env vars:
#   VARIANT  variant name under rustbpe_variants/<name>/ (default: empty = pristine)
#   RELEASE  set to 1 for a release build (slower compile, faster runtime)
#            (default: 0 = `maturin develop` which is dev/incremental mode)
#
# Behavior:
#   - With no VARIANT, builds sandbox/rustbpe/ and installs it as the `rustbpe`
#     module in the sandbox venv. This OVERRIDES the PyPI rustbpe pulled in by
#     `uv sync` — to revert, re-run `uv sync` and the PyPI version comes back.
#   - With VARIANT, builds sandbox/rustbpe_variants/<name>/ and installs it
#     under its distinct module name (e.g. `rustbpe_seed_tokens`). Does NOT
#     touch the default `rustbpe` module; coexists with pristine.
#
# Idempotent — safe to re-run. Re-runs are fast in dev mode (cargo's incremental
# build cache); release mode rebuilds more aggressively but still benefits from
# the cache after the first run.
#
# Prereqs:
#   - Rust toolchain (rustup, cargo) — install from https://rustup.rs/
#   - maturin (installed automatically as a build dep, but `uv tool install
#     maturin` is faster for repeated builds)

set -euo pipefail

SANDBOX_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VARIANT="${VARIANT:-}"
RELEASE="${RELEASE:-0}"

if [ -z "$VARIANT" ]; then
    CRATE_DIR="$SANDBOX_DIR/rustbpe"
    MODULE_NAME="rustbpe (pristine)"
else
    CRATE_DIR="$SANDBOX_DIR/rustbpe_variants/$VARIANT"
    MODULE_NAME="rustbpe_$VARIANT"
fi

if [ ! -d "$CRATE_DIR" ]; then
    echo "error: crate directory not found: $CRATE_DIR" >&2
    if [ -n "$VARIANT" ]; then
        echo "Available variants:" >&2
        ls "$SANDBOX_DIR/rustbpe_variants/" 2>/dev/null | grep -v README || echo "  (none yet)" >&2
        echo "See rustbpe_variants/README.md for how to add one." >&2
    fi
    exit 1
fi

if ! command -v cargo >/dev/null 2>&1; then
    echo "error: cargo not found. Install Rust toolchain from https://rustup.rs/" >&2
    exit 1
fi

echo "==> building $MODULE_NAME from $CRATE_DIR"

# Ensure maturin is available in the sandbox venv. Run all uv commands from
# the sandbox dir so uv resolves the right pyproject.toml (otherwise, since
# our crate dirs have their own pyproject.toml for maturin, `uv run` from a
# crate dir would spin up a NEW isolated venv and miss our deps).
SANDBOX_VENV="$SANDBOX_DIR/.venv"
SANDBOX_PY="$SANDBOX_VENV/bin/python"
SANDBOX_MATURIN="$SANDBOX_VENV/bin/maturin"

cd "$SANDBOX_DIR"
if [ ! -x "$SANDBOX_MATURIN" ]; then
    echo "==> installing maturin into sandbox venv"
    uv pip install "maturin>=1.7,<2.0"
fi

# Invoke maturin directly via the sandbox venv binary (not via `uv run`) so
# it installs into the sandbox venv, not a per-crate venv. Maturin develop
# detects the target venv via VIRTUAL_ENV; set it explicitly so the script
# works regardless of the caller's shell state.
cd "$CRATE_DIR"
export VIRTUAL_ENV="$SANDBOX_VENV"
MATURIN_ARGS=(develop)
if [ "$RELEASE" = "1" ]; then
    MATURIN_ARGS+=(--release)
    echo "==> $SANDBOX_MATURIN ${MATURIN_ARGS[*]}"
else
    echo "==> $SANDBOX_MATURIN ${MATURIN_ARGS[*]} (dev mode)"
fi
"$SANDBOX_MATURIN" "${MATURIN_ARGS[@]}"

echo
echo "==> done. Sanity check:"
"$SANDBOX_PY" -c "
import importlib
mod_name = 'rustbpe' if not '${VARIANT}' else 'rustbpe_${VARIANT}'
mod = importlib.import_module(mod_name)
print(f'  imported {mod_name} from {mod.__file__}')
tok = mod.Tokenizer()
print(f'  Tokenizer instance: {tok}')
"
echo
if [ -z "$VARIANT" ]; then
    echo "Note: this OVERRIDES the PyPI rustbpe in the venv."
    echo "      Re-run 'uv sync' to revert to the PyPI version."
fi
