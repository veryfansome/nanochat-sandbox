"""
Tokenizer variant: forced cross-boundary common-phrase merges.

Lifted from veryfansome/nanochat@force_merges_wip. The Rust crate
`rustbpe_force_merges` (built via `VARIANT=force_merges bash runs/build_rustbpe.sh`)
adds `forced_pairs=` and `blocked_pairs=` kwargs to `train_from_iterator` and
runs `apply_forced_merges_at_end` after normal BPE training. The Python wrapper
patches SPLIT_PATTERN with the regex carve-out so cross-boundary phrases form
single pre-tok chunks at inference (otherwise tiktoken would never apply the
forced merges).

See ../rustbpe_variants/force_merges/pairs.py for the FORCED_PAIRS / BLOCKED_PAIRS
data + the derived SPLIT_PATTERN, and ../ideas/tokenizer-variants/README.md
"force_merges_wip" for the design rationale.

Usage:
    # Build the rustbpe variant once (and after any Rust edit):
    VARIANT=force_merges bash runs/build_rustbpe.sh

    # Then train:
    uv run python -m wrappers.tok_train_force_merges

Output:
    ~/.cache/nanochat-variants/force_merges/tokenizer/
"""
import runpy

from wrappers._tokenizer_variant_common import (
    setup_variant_base,
    alias_rustbpe,
    print_downstream_hint,
)


def apply_patches():
    """
    Install the rustbpe alias + SPLIT_PATTERN patch + train_from_iterator shim.

    Returns (SPLIT_PATTERN, FORCED_PAIRS, BLOCKED_PAIRS) for inspection / testing.
    Must be called AFTER setup_variant_base() (so NANOCHAT_BASE_DIR is set) and
    BEFORE any nanochat.tokenizer import (so alias_rustbpe wins).
    """
    alias_rustbpe("rustbpe_force_merges")

    import nanochat.tokenizer as tk
    from rustbpe_variants.force_merges.pairs import (
        SPLIT_PATTERN,
        FORCED_PAIRS,
        BLOCKED_PAIRS,
    )

    tk.SPLIT_PATTERN = SPLIT_PATTERN

    def _patched_train_classmethod(text_iterator, vocab_size):
        """Drop-in replacement that calls rustbpe_force_merges with the extra kwargs.

        Upstream train_from_iterator doesn't accept forced_pairs / blocked_pairs,
        so we reimplement the small post-train wiring rather than try to thread
        the kwargs through the existing classmethod signature."""
        import rustbpe  # aliased to rustbpe_force_merges
        import tiktoken
        from nanochat.tokenizer import SPECIAL_TOKENS

        tokenizer = rustbpe.Tokenizer()
        vocab_size_no_special = vocab_size - len(SPECIAL_TOKENS)
        assert vocab_size_no_special >= 256
        tokenizer.train_from_iterator(
            text_iterator, vocab_size_no_special,
            pattern=tk.SPLIT_PATTERN,
            forced_pairs=FORCED_PAIRS,
            blocked_pairs=BLOCKED_PAIRS,
        )
        pattern = tokenizer.get_pattern()
        mergeable_ranks_list = tokenizer.get_mergeable_ranks()
        mergeable_ranks = {bytes(k): v for k, v in mergeable_ranks_list}
        tokens_offset = len(mergeable_ranks)
        special_tokens = {name: tokens_offset + i for i, name in enumerate(SPECIAL_TOKENS)}
        enc = tiktoken.Encoding(
            name="rustbpe_force_merges",
            pat_str=pattern,
            mergeable_ranks=mergeable_ranks,
            special_tokens=special_tokens,
        )
        return tk.RustBPETokenizer(enc, "<|bos|>")

    tk.RustBPETokenizer.train_from_iterator = classmethod(
        lambda cls, text_iterator, vocab_size: _patched_train_classmethod(text_iterator, vocab_size)
    )
    return SPLIT_PATTERN, FORCED_PAIRS, BLOCKED_PAIRS


def main():
    variant_base = setup_variant_base("force_merges")
    split_pattern, forced, blocked = apply_patches()
    print(f"==> variant_base:    {variant_base}")
    print(f"==> rustbpe module:  {__import__('rustbpe').__file__}")
    print(f"==> SPLIT_PATTERN:   patched (FORCED_PAIRS_EXPR + ` ?\\p{{N}}{{1,2}}`)")
    print(f"==> forced pairs:    {len(forced)}")
    print(f"==> blocked pairs:   {len(blocked)}")
    runpy.run_module("scripts.tok_train", run_name="__main__")
    print_downstream_hint(variant_base)


if __name__ == "__main__":
    main()
