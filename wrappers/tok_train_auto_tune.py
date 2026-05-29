"""
Tokenizer variant: auto_tune — block mangled-morpheme merges.

Pristine `SPLIT_PATTERN` (no carve-out), blocking only — no forced merges and no
vocab bump. Reads `BLOCKED_PAIRS` from `rustbpe_variants/auto_tune/pairs.py` and
forbids those merges during BPE training, so words route through the cleaner
[stem][suffix] split. Uses auto_tune's own crate `rustbpe_auto_tune` (a fork of
`rustbpe_force_merges` + `load_corpus`/`train_from_cached` for the block-list
audit) for its `blocked_pairs=` kwarg; `train_from_iterator` is bit-identical.

Usage:
    uv run python -m wrappers.tok_train_auto_tune
      →  ~/.cache/nanochat-variants/auto_tune/tokenizer/
"""
import runpy
import sys

from wrappers._tokenizer_variant_common import (
    setup_variant_base,
    alias_rustbpe,
    print_downstream_hint,
)


def apply_patches():
    """Install the rustbpe alias + train_from_iterator shim that passes
    BLOCKED_PAIRS. Returns BLOCKED_PAIRS for inspection / testing.

    Must be called AFTER setup_variant_base() and BEFORE any
    nanochat.tokenizer import (so alias_rustbpe wins)."""
    alias_rustbpe("rustbpe_auto_tune")

    import nanochat.tokenizer as tk
    from rustbpe_variants.auto_tune.pairs import BLOCKED_PAIRS

    # Audit override: tools/audit_blocked_pairs.py sets AUTO_TUNE_BLOCKED_PAIRS_JSON
    # to a JSON list of [L, R] pairs to train this run with, bypassing pairs.py's
    # BLOCKED_PAIRS. Lets the build-up sweep train kept+candidate sets without
    # editing the module each iteration.
    import json
    import os
    _override = os.environ.get("AUTO_TUNE_BLOCKED_PAIRS_JSON")
    if _override is not None:
        BLOCKED_PAIRS = [tuple(p) for p in json.loads(_override)]

    # SPLIT_PATTERN stays pristine (no carve-out, no delimiter-glue tweak).

    def _patched_train_classmethod(text_iterator, vocab_size):
        """Drop-in replacement that calls rustbpe_auto_tune with
        blocked_pairs (and empty forced_pairs — this variant only blocks)."""
        import rustbpe  # aliased to rustbpe_auto_tune
        import tiktoken
        from nanochat.tokenizer import SPECIAL_TOKENS

        tokenizer = rustbpe.Tokenizer()
        vocab_size_no_special = vocab_size - len(SPECIAL_TOKENS)
        assert vocab_size_no_special >= 256
        tokenizer.train_from_iterator(
            text_iterator, vocab_size_no_special,
            pattern=tk.SPLIT_PATTERN,
            forced_pairs=[],
            blocked_pairs=BLOCKED_PAIRS,
        )
        pattern = tokenizer.get_pattern()
        mergeable_ranks_list = tokenizer.get_mergeable_ranks()
        # Blocking only removes merge candidates — ids stay dense from 0 (no
        # post-train injection like force_merges does), so the sparse-id guard
        # there isn't needed. Keep a light shortfall warning for small corpora.
        ids = [v for _, v in mergeable_ranks_list]
        n_unique = len(set(ids))
        if n_unique < vocab_size_no_special:
            import warnings
            warnings.warn(
                f"auto_tune produced {n_unique}/{vocab_size_no_special} "
                f"unique merges ({vocab_size_no_special - n_unique} short). "
                f"Likely cause: training corpus too small to fill the vocab. "
                f"Proceeding with reduced vocab.",
                stacklevel=2,
            )
        mergeable_ranks = {bytes(k): v for k, v in mergeable_ranks_list}
        tokens_offset = len(mergeable_ranks)
        special_tokens = {name: tokens_offset + i for i, name in enumerate(SPECIAL_TOKENS)}
        enc = tiktoken.Encoding(
            name="rustbpe_auto_tune",
            pat_str=pattern,
            mergeable_ranks=mergeable_ranks,
            special_tokens=special_tokens,
        )
        return tk.RustBPETokenizer(enc, "<|bos|>")

    tk.RustBPETokenizer.train_from_iterator = classmethod(
        lambda cls, text_iterator, vocab_size: _patched_train_classmethod(text_iterator, vocab_size)
    )
    return BLOCKED_PAIRS


def main():
    variant_base = setup_variant_base("auto_tune")
    blocked = apply_patches()
    from rustbpe_variants.auto_tune.pairs import RECOMMENDED_VOCAB_SIZE

    user_args = sys.argv[1:]
    has_vocab_arg = any(
        a == "--vocab-size" or a.startswith("--vocab-size=")
        for a in user_args
    )
    if has_vocab_arg:
        effective_vocab = "caller-supplied (see --vocab-size in args below)"
    else:
        user_args = ["--vocab-size", str(RECOMMENDED_VOCAB_SIZE)] + user_args
        effective_vocab = f"{RECOMMENDED_VOCAB_SIZE} (nanochat default, no bump)"
    print(f"==> variant_base:    {variant_base}")
    print(f"==> rustbpe module:  {__import__('rustbpe').__file__}")
    print(f"==> SPLIT_PATTERN:   pristine (no carve-out)")
    print(f"==> blocked pairs:   {len(blocked)}")
    print(f"==> vocab_size:      {effective_vocab}")
    print(f"==> tok_train args:  {user_args}")
    sys.argv = ["tok_train.py"] + user_args
    runpy.run_module("scripts.tok_train", run_name="__main__")
    print_downstream_hint(variant_base)


if __name__ == "__main__":
    main()
