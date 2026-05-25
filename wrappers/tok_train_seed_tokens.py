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
        # Two distinct missing-id regimes, only one of which is benign:
        #   - **Tail shortfall** (unique ids dense in `[0, n_unique)` but
        #     `n_unique < vocab_size_no_special`): BPE legitimately stops early
        #     when the corpus runs out of useful merges (small `--max_chars`,
        #     smoke/dev runs). Baseline rustbpe does the same. `tokens_offset =
        #     len(mergeable_ranks)` places special tokens above max_id, no
        #     collision. Warn and proceed.
        #   - **Interior hole** (some id in `[0, max_id]` is unallocated): the
        #     constructive merge-chain builder allocated ids non-contiguously.
        #     This is a Rust-trainer bug, not a small-corpus condition. Raise
        #     so the bug surfaces instead of risking a corrupted encoding;
        #     whether tiktoken catches the eventual overlap depends on its
        #     version.
        #
        # **Discriminator note**: we test against `len(set(ids))`, not raw
        # `len(ids)`. The Rust trainer can legitimately emit duplicate ids via
        # alias merges (`ensure_merge_pair` reuses an `existing_id` when seed
        # construction lands on an already-allocated pair). Raw list length is
        # inflated by aliases and is not a reliable proxy for ID density — a
        # hole plus enough aliases can make a list-length check pass while the
        # underlying ID space is sparse. The unique-id count directly asserts
        # the invariant downstream consumers need (every id in `[0, max_id]`
        # is allocated to some token).
        ids = [v for _, v in mergeable_ranks_list]
        unique_ids = set(ids)
        n_unique = len(unique_ids)
        max_id = max(ids) if ids else -1
        if max_id >= n_unique:
            missing = sorted(set(range(max_id + 1)) - unique_ids)
            raise ValueError(
                f"rustbpe_seed_tokens produced sparse mergeable ids: "
                f"{n_unique} unique ids allocated but max id is {max_id} "
                f"(would be {n_unique - 1} if dense). Missing ids in "
                f"[0, {max_id}]: {missing[:10]}"
                f"{'...' if len(missing) > 10 else ''}. Indicates a bug in "
                f"the constructive merge-chain builder, not a small-corpus "
                f"stop — special tokens at `tokens_offset = "
                f"len(mergeable_ranks)` would risk colliding with existing "
                f"mergeable ids above the hole."
            )
        if n_unique < vocab_size_no_special:
            import warnings
            warnings.warn(
                f"rustbpe_seed_tokens produced {n_unique}/{vocab_size_no_special} "
                f"unique merges ({vocab_size_no_special - n_unique} short). "
                f"Likely cause: training corpus too small to fill the vocab. "
                f"Ids are dense from 0 — special tokens place safely above "
                f"max_id. Proceeding with reduced vocab.",
                stacklevel=2,
            )
        mergeable_ranks = {bytes(k): v for k, v in mergeable_ranks_list}
        tokens_offset = len(mergeable_ranks)  # safe: dict size >= n_unique > max_id per check above
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
