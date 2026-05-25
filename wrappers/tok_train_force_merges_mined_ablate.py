"""
Fine-grained DERIVED-pair ablation training.

Trains with an arbitrary subset of the 20 mined DERIVED pairs as specified
by INCLUDE_INDICES (comma-separated 0-based indices). Variant base dir
includes the index list in its name so multiple subsets coexist.

Usage:
    INCLUDE_INDICES=0,1,2,...,14,16 \\
        uv run python -m wrappers.tok_train_force_merges_mined_ablate
"""
import os
import runpy

from wrappers._tokenizer_variant_common import (
    setup_variant_base,
    alias_rustbpe,
    print_downstream_hint,
)


def apply_patches():
    alias_rustbpe("rustbpe_force_merges")

    import nanochat.tokenizer as tk
    from rustbpe_variants.force_merges.pairs_mined_ablate import (
        SPLIT_PATTERN,
        FORCED_PAIRS,
        BLOCKED_PAIRS,
    )
    tk.SPLIT_PATTERN = SPLIT_PATTERN

    def _patched_train_classmethod(text_iterator, vocab_size):
        import rustbpe
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
        ids = [v for _, v in mergeable_ranks_list]
        unique_ids = set(ids)
        n_unique = len(unique_ids)
        max_id = max(ids) if ids else -1
        if max_id >= n_unique:
            missing = sorted(set(range(max_id + 1)) - unique_ids)
            raise ValueError(
                f"sparse ids: {n_unique} unique, max {max_id}. "
                f"Missing: {missing[:10]}."
            )
        mergeable_ranks = {bytes(k): v for k, v in mergeable_ranks_list}
        tokens_offset = len(mergeable_ranks)
        special_tokens = {
            name: tokens_offset + i for i, name in enumerate(SPECIAL_TOKENS)
        }
        enc = tiktoken.Encoding(
            name="rustbpe_force_merges_mined_ablate",
            pat_str=pattern,
            mergeable_ranks=mergeable_ranks,
            special_tokens=special_tokens,
        )
        return tk.RustBPETokenizer(enc, "<|bos|>")

    tk.RustBPETokenizer.train_from_iterator = classmethod(
        lambda cls, text_iterator, vocab_size: _patched_train_classmethod(
            text_iterator, vocab_size
        )
    )
    return SPLIT_PATTERN, FORCED_PAIRS, BLOCKED_PAIRS


def main():
    indices = os.environ.get("INCLUDE_INDICES", "all")
    # Encode index list into a filesystem-safe variant name.
    tag = indices.replace(",", "_") if indices != "all" else "all"
    if indices == "":
        tag = "none"
    if len(tag) > 80:
        # Long lists: use a hash for the dir name; full list in stdout.
        import hashlib
        tag = "h" + hashlib.sha1(indices.encode()).hexdigest()[:10]
    variant_base = setup_variant_base(f"force_merges_mined_ablate_{tag}")
    split_pattern, forced, blocked = apply_patches()
    print(f"==> variant_base:    {variant_base}")
    print(f"==> INCLUDE_INDICES: {indices}")
    print(f"==> forced pairs:    {len(forced)}")
    print(f"==> blocked pairs:   {len(blocked)}")
    runpy.run_module("scripts.tok_train", run_name="__main__")
    print_downstream_hint(variant_base)


if __name__ == "__main__":
    main()
