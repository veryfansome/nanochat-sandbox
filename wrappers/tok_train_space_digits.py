"""
Tokenizer variant: optional space before digits in pre-tokenization regex.

Changes the digit clause in SPLIT_PATTERN from `\\p{N}{1,2}` to ` ?\\p{N}{1,2}`
so leading digits merge with their preceding space (like words do via the
`[^\\r\\n\\p{L}\\p{N}]?+\\p{L}+` clause). Lifted from
veryfansome/nanochat@force_merges_wip, where it was measured at +1.09–1.23%
compression for ~220 token cost.

Pure-Python wrapper: no Rust build needed. Uses PyPI rustbpe (or whichever
`rustbpe` is currently in the venv).

Usage:
    uv run python -m wrappers.tok_train_space_digits

Output:
    ~/.cache/nanochat-variants/space_digits/tokenizer/
"""
import runpy

from wrappers._tokenizer_variant_common import (
    setup_variant_base,
    print_downstream_hint,
)


def patch_split_pattern():
    """Apply the digit-clause space-prefix change. Returns (orig, patched) for
    inspection / testing. Idempotent — safe to call twice."""
    import nanochat.tokenizer as tk
    orig = tk.SPLIT_PATTERN
    if r"| ?\p{N}{1,2}|" in orig:
        # Already patched (e.g. by another wrapper in the same process).
        return orig, orig
    patched = orig.replace(r"|\p{N}{1,2}|", r"| ?\p{N}{1,2}|")
    assert patched != orig, (
        f"Pattern substitution failed — `\\p{{N}}{{1,2}}` not found in current "
        f"SPLIT_PATTERN:\n{orig}\nDid upstream change the digit clause?"
    )
    tk.SPLIT_PATTERN = patched
    return orig, patched


def main():
    variant_base = setup_variant_base("space_digits")
    orig, patched = patch_split_pattern()
    print(f"==> variant_base:        {variant_base}")
    print(f"==> SPLIT_PATTERN diff:  |\\p{{N}}{{1,2}}|  →  | ?\\p{{N}}{{1,2}}|")
    runpy.run_module("scripts.tok_train", run_name="__main__")
    print_downstream_hint(variant_base)


if __name__ == "__main__":
    main()
