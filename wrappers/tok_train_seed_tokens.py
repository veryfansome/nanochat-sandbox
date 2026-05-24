"""
Tokenizer variant: morpheme-seeded BPE.

Lifted from veryfansome/nanochat@seed_tokens. The Rust crate
`rustbpe_seed_tokens` (built via `VARIANT=seed_tokens bash runs/build_rustbpe.sh`)
adds a `seed_tokens=` kwarg to `train_from_iterator` and a constructive
merge-chain builder that ensures each seed token is achievable in the BPE merge
DAG (priority over frequency-driven merges).

The Python wrapper loads the morpheme YAML, generates the position-aware
variant set (leading-space / leading-space-capitalized / no-space), and shims
train_from_iterator to thread `seed_tokens=` through to the Rust trainer.

See ../rustbpe_variants/seed_tokens/seed_tokens.yaml for the ~200-morpheme
curated list (Greek/Latin roots, English prefixes, suffixes — etymology
commented per entry).

Usage:
    # Build the rustbpe variant once (and after any Rust edit):
    VARIANT=seed_tokens bash runs/build_rustbpe.sh

    # Then train:
    uv run python -m wrappers.tok_train_seed_tokens

Output:
    ~/.cache/nanochat-variants/seed_tokens/tokenizer/
"""
import os
import runpy

import yaml

from wrappers._tokenizer_variant_common import (
    setup_variant_base,
    alias_rustbpe,
    print_downstream_hint,
)


def _load_seeds():
    """Build the seed-token list from the YAML, mirroring sandbox/seed_tokens.py
    on the seed_tokens branch. Position-aware variants are generated per category:
      - versatile_morphemes: leading-space, leading-space-capitalized, no-space
      - prefixes:            leading-space, leading-space-capitalized
      - inner_morphemes:     no-space
    """
    yaml_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..",
        "rustbpe_variants",
        "seed_tokens",
        "seed_tokens.yaml",
    )
    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)
    seeds = set()
    for seed in cfg.get("versatile_morphemes", []):
        seeds.add(f" {seed}")
        seeds.add(f" {seed.capitalize()}")
        seeds.add(seed)
    for seed in cfg.get("prefixes", []):
        seeds.add(f" {seed}")
        seeds.add(f" {seed.capitalize()}")
    for seed in cfg.get("inner_morphemes", []):
        seeds.add(seed)
    return sorted(seeds)


def apply_patches():
    """
    Install the rustbpe alias + train_from_iterator shim that threads the seed
    list through to the Rust trainer.

    Returns the loaded seed list for inspection / testing. Must be called AFTER
    setup_variant_base() and BEFORE any nanochat.tokenizer import.
    """
    alias_rustbpe("rustbpe_seed_tokens")

    import nanochat.tokenizer as tk
    seeds = _load_seeds()

    def _patched_train_classmethod(text_iterator, vocab_size):
        import rustbpe  # aliased to rustbpe_seed_tokens
        import tiktoken
        from nanochat.tokenizer import SPECIAL_TOKENS

        tokenizer = rustbpe.Tokenizer()
        vocab_size_no_special = vocab_size - len(SPECIAL_TOKENS)
        assert vocab_size_no_special >= 256
        tokenizer.train_from_iterator(
            text_iterator, vocab_size_no_special,
            pattern=tk.SPLIT_PATTERN,
            seed_tokens=seeds,
        )
        pattern = tokenizer.get_pattern()
        mergeable_ranks_list = tokenizer.get_mergeable_ranks()
        # Contiguity check from the seed_tokens branch — surfaces bugs in the
        # constructive merge-chain builder where some token ids aren't allocated.
        ids = [v for _, v in mergeable_ranks_list]
        missing = sorted(set(range(vocab_size_no_special)) - set(ids))
        if missing:
            raise ValueError(
                f"rustbpe_seed_tokens export missing token ids: "
                f"count={len(missing)}, sample={missing[:20]}"
            )
        mergeable_ranks = {bytes(k): v for k, v in mergeable_ranks_list}
        tokens_offset = len(mergeable_ranks)
        special_tokens = {name: tokens_offset + i for i, name in enumerate(SPECIAL_TOKENS)}
        enc = tiktoken.Encoding(
            name="rustbpe_seed_tokens",
            pat_str=pattern,
            mergeable_ranks=mergeable_ranks,
            special_tokens=special_tokens,
        )
        return tk.RustBPETokenizer(enc, "<|bos|>")

    tk.RustBPETokenizer.train_from_iterator = classmethod(
        lambda cls, text_iterator, vocab_size: _patched_train_classmethod(text_iterator, vocab_size)
    )
    return seeds


def main():
    variant_base = setup_variant_base("seed_tokens")
    seeds = apply_patches()
    print(f"==> variant_base:    {variant_base}")
    print(f"==> rustbpe module:  {__import__('rustbpe').__file__}")
    print(f"==> seed tokens:     {len(seeds)}")
    print(f"==> sample seeds:    {seeds[:8]} ...")
    runpy.run_module("scripts.tok_train", run_name="__main__")
    print_downstream_hint(variant_base)


if __name__ == "__main__":
    main()
