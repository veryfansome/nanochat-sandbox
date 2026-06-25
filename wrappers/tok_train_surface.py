"""
Tokenizer trainer for surface factoring — the "triple" tokenizer:
force_merges forced phrases + the SPACELESS split pattern + a CASE-FOLDED corpus.

It is the force_merges trainer (`wrappers/tok_train_force_merges.py`) with three
surface-specific changes:
  1. SPLIT_PATTERN = the spaceless pattern (leading space is its own pretoken →
     factored to a bit at inference, not glued into the word token).
  2. forced_pairs = the leading-space-stripped FOLDED phrase pairs.
  3. the training corpus is case-FOLDED (titlecase first letters lowercased) so
     the merges are learned over the same surface the SurfaceTokenizer produces.

Saves a normal RustBPETokenizer (+ token_bytes.pt — the BASE-token byte lengths
the surface bpb denominator needs) to $VARIANT_BASE/tokenizer/. Point a surface
training run at it via NANOCHAT_BASE_DIR (runs/runcpu_surface.sh does this).

Usage:
    uv run python -m wrappers.tok_train_surface --max-chars=2000000000
    →  ~/.cache/nanochat-variants/surface_triple/tokenizer/
"""
import runpy
import sys

from wrappers._tokenizer_variant_common import (
    setup_variant_base, alias_rustbpe, print_downstream_hint,
)


def apply_patches():
    """Install rustbpe alias + spaceless SPLIT_PATTERN + folded-corpus / folded-
    forced-pairs train shim. Returns the build config (for vocab/inspection)."""
    alias_rustbpe("rustbpe_force_merges")

    import nanochat.tokenizer as tk
    from nanochat.tokenizer import SPECIAL_TOKENS
    from tools.stack_fm_spaceless import build_config
    from tools.case_factor_compression import fold_case
    from rustbpe_variants.force_merges.pairs import FORCED_PAIRS

    # folded + de-duplicated forced pairs (folding collapses e.g. 'Of The'/'OF THE')
    seen, folded_pairs = set(), []
    for a, b in FORCED_PAIRS:
        fa, fb = fold_case(a), fold_case(b)
        if (fa, fb) not in seen:
            seen.add((fa, fb))
            folded_pairs.append((fa, fb))
    cfg = build_config(folded_pairs)
    tk.SPLIT_PATTERN = cfg["spaceless_pattern"]
    spaceless_forced = cfg["spaceless_forced"]

    def _patched_train(text_iterator, vocab_size):
        import rustbpe                 # aliased to rustbpe_force_merges
        import tiktoken
        folded_iter = (fold_case(t) for t in text_iterator)   # train on folded surface
        tokenizer = rustbpe.Tokenizer()
        vns = vocab_size - len(SPECIAL_TOKENS)
        assert vns >= 256
        tokenizer.train_from_iterator(
            folded_iter, vns, pattern=tk.SPLIT_PATTERN,
            forced_pairs=spaceless_forced,
            block_trailing_space=True, block_leading_space=True,
        )
        pattern = tokenizer.get_pattern()
        mr_list = tokenizer.get_mergeable_ranks()
        # density check (mirrors tok_train_force_merges): forced-merge injection
        # must keep ids dense so special tokens at len(mergeable_ranks) don't collide.
        ids = [v for _, v in mr_list]
        unique_ids = set(ids)
        max_id = max(ids) if ids else -1
        if max_id >= len(unique_ids):
            missing = sorted(set(range(max_id + 1)) - unique_ids)
            raise ValueError(f"sparse mergeable ids: {len(unique_ids)} unique, max {max_id}; "
                             f"missing {missing[:10]}{'...' if len(missing) > 10 else ''}")
        if len(unique_ids) < vns:
            import warnings
            warnings.warn(f"{len(unique_ids)}/{vns} unique merges (corpus small); proceeding.",
                          stacklevel=2)
        mergeable_ranks = {bytes(k): v for k, v in mr_list}
        offset = len(mergeable_ranks)
        special = {name: offset + i for i, name in enumerate(SPECIAL_TOKENS)}
        enc = tiktoken.Encoding(name="surface_triple", pat_str=pattern,
                                mergeable_ranks=mergeable_ranks, special_tokens=special)
        return tk.RustBPETokenizer(enc, "<|bos|>")

    tk.RustBPETokenizer.train_from_iterator = classmethod(
        lambda cls, text_iterator, vocab_size: _patched_train(text_iterator, vocab_size))
    return cfg


def main():
    variant_base = setup_variant_base("surface_triple")
    cfg = apply_patches()
    # default the vocab to the additive triple vocab unless the caller overrode it
    if not any(a.startswith("--vocab-size") for a in sys.argv[1:]):
        sys.argv.append(f"--vocab-size={cfg['vocab']}")
    print(f"[surface] training triple tokenizer: vocab={cfg['vocab']}, "
          f"{len(cfg['spaceless_forced'])} forced, spaceless pattern, folded corpus",
          file=sys.stderr)
    runpy.run_module("scripts.tok_train", run_name="__main__")
    print_downstream_hint(variant_base)


if __name__ == "__main__":
    main()
