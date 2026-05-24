"""
Shared scaffolding for tokenizer-variant wrappers.

Each wrapper:
  1. Imports this module first.
  2. Calls setup_variant_base(name) which:
     - Picks a per-variant base dir under ~/.cache/nanochat-variants/<name>/
       (overridable via VARIANT_BASE env var).
     - Symlinks the pretraining corpus from $NANOCHAT_BASE_DIR so the variant
       reuses the existing data without re-downloading.
     - Sets NANOCHAT_BASE_DIR to the variant base for the lifetime of the
       wrapper, so `tok_train.py`'s `get_base_dir()` returns the variant dir
       and the resulting tokenizer is saved at $VARIANT_BASE/tokenizer/.
  3. (Optionally) calls alias_rustbpe(variant_module_name) to swap in a
     parallel-crate rustbpe variant before `nanochat.tokenizer` is imported.
  4. Applies its own monkeypatches (SPLIT_PATTERN, train_from_iterator kwargs, etc).
  5. runpys `scripts.tok_train`.

The result is a tokenizer at ~/.cache/nanochat-variants/<name>/tokenizer/ that
the rest of the pipeline can consume by pointing get_tokenizer / eval_tokenizer
at that directory. Original ~/.cache/nanochat/tokenizer/ is never touched.
"""

import importlib
import os
import sys


def setup_variant_base(variant_name: str) -> str:
    """
    Pick a per-variant base dir and configure NANOCHAT_BASE_DIR + corpus symlink.

    Returns the variant base dir path so the wrapper can print downstream
    consumption hints. Idempotent — safe to re-run.
    """
    orig_base = os.environ.get(
        "NANOCHAT_BASE_DIR",
        os.path.expanduser("~/.cache/nanochat"),
    )
    variant_base = os.environ.get(
        "VARIANT_BASE",
        os.path.expanduser(f"~/.cache/nanochat-variants/{variant_name}"),
    )
    os.makedirs(variant_base, exist_ok=True)

    # Symlink data dirs that tok_train needs to READ. Any new data dirs added
    # by upstream should be appended here (currently just the pretraining corpus).
    for needed in ["base_data_climbmix"]:
        src = os.path.join(orig_base, needed)
        dst = os.path.join(variant_base, needed)
        if os.path.exists(src) and not os.path.lexists(dst):
            os.symlink(src, dst)

    # Override the env var so get_base_dir() returns the variant dir from now on.
    # This means tokenizer.save() and report logs land in the variant dir.
    os.environ["NANOCHAT_BASE_DIR"] = variant_base
    return variant_base


def alias_rustbpe(variant_module_name: str) -> None:
    """
    Swap `import rustbpe` to import a parallel-crate variant instead.

    MUST be called BEFORE `nanochat.tokenizer` is imported anywhere — that
    module does `import rustbpe` at module scope, so once it has loaded, the
    alias is too late.
    """
    if "nanochat.tokenizer" in sys.modules:
        raise RuntimeError(
            f"alias_rustbpe('{variant_module_name}') called after nanochat.tokenizer "
            "was already imported. Move the alias call to the top of the wrapper, "
            "before any nanochat import."
        )
    try:
        mod = importlib.import_module(variant_module_name)
    except ImportError as e:
        raise ImportError(
            f"Cannot import rustbpe variant '{variant_module_name}'. "
            f"Build it first: VARIANT={variant_module_name.removeprefix('rustbpe_')} "
            f"bash runs/build_rustbpe.sh"
        ) from e
    sys.modules["rustbpe"] = mod


def print_downstream_hint(variant_base: str) -> None:
    """Standard tail-of-wrapper message: where the tokenizer landed + how to eval it."""
    tokenizer_dir = os.path.join(variant_base, "tokenizer")
    print(f"\n==> tokenizer saved to: {tokenizer_dir}")
    print(f"==> compare via:")
    print(f"    uv run python -m tools.eval_tokenizer \\")
    print(f"        --tokenizers ours,{tokenizer_dir}")
