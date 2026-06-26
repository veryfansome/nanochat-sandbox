"""
Tokenizer trainer for surface factoring — the canonical SURFACE-ONLY tokenizer:
space + case factoring on the BASELINE vocab, NO force_merges phrases.

The earlier "triple" (force_merges phrases + space + case) was a premature stacking of
force_merges and surface factoring — it confounded the d24 read and forced the K_max=3
cap machinery — and was rolled back. surface-only cleanly isolates the factoring effect;
baseline and force_merges remain SEPARATE sibling tokenizers (force_merges via
wrappers/tok_train_force_merges.py; the deprecated triple build path is tools/stack_triple.py).

Structurally simple and CRATE-FREE: with no multi-word phrase tokens every token has at
most one word-start, so K_max = 1 (the cap is a single bit, like space). The spaceless
SPLIT_PATTERN already isolates spaces as their own pretoken, so the force_merges crate's
block-space option is redundant — this uses plain base rustbpe (no rustbpe_force_merges).

Trains a plain BPE over the case-folded + spaceless-patterned corpus (forced_pairs=[]).
Saves a RustBPETokenizer (+ token_bytes.pt) to
~/.cache/nanochat-variants/surface_only/tokenizer/ (vocab 32768, = baseline).

Usage: uv run python -m wrappers.tok_train_surface --max-chars=2000000000
"""
import runpy
import sys

from wrappers._tokenizer_variant_common import (
    setup_variant_base, print_downstream_hint,
)


def apply_patches():
    # surface-only is decoupled from force_merges: plain base rustbpe (no crate). The
    # spaceless SPLIT_PATTERN already isolates spaces as their own pretoken, so the
    # crate's block_leading/trailing_space (needed only for cross-space forced merges)
    # is redundant here — a clean surface tokenizer with no force_merges dependency.
    import nanochat.tokenizer as tk
    from nanochat.tokenizer import SPECIAL_TOKENS
    from tools.stack_fm_spaceless import SPACELESS_BODY
    from tools.case_factor_compression import fold_case

    # Base spaceless pattern (space-factoring). NOT build_config([]) — with no pairs it
    # prepends build_expr([]) = "()(?=([^a-z]|$))", an empty-match group that makes the
    # tiktoken regex panic ("slice of length 0") on every input.
    tk.SPLIT_PATTERN = SPACELESS_BODY
    cfg = {"vocab": 32768, "spaceless_pattern": SPACELESS_BODY}

    def _patched_train(text_iterator, vocab_size):
        import rustbpe                            # base rustbpe (PyPI) — no force_merges crate
        import tiktoken
        folded_iter = (fold_case(t) for t in text_iterator)   # train on the folded surface
        tokenizer = rustbpe.Tokenizer()
        vns = vocab_size - len(SPECIAL_TOKENS)
        assert vns >= 256
        tokenizer.train_from_iterator(folded_iter, vns, pattern=tk.SPLIT_PATTERN)   # no phrases, no block-space
        mr_list = tokenizer.get_mergeable_ranks()
        mergeable_ranks = {bytes(k): v for k, v in mr_list}
        offset = len(mergeable_ranks)
        special = {name: offset + i for i, name in enumerate(SPECIAL_TOKENS)}
        enc = tiktoken.Encoding(name="surface_only", pat_str=tokenizer.get_pattern(),
                                mergeable_ranks=mergeable_ranks, special_tokens=special)
        return tk.RustBPETokenizer(enc, "<|bos|>")

    tk.RustBPETokenizer.train_from_iterator = classmethod(
        lambda cls, text_iterator, vocab_size: _patched_train(text_iterator, vocab_size))
    return cfg


def main():
    variant_base = setup_variant_base("surface_only")
    cfg = apply_patches()
    if not any(a.startswith("--vocab-size") for a in sys.argv[1:]):
        sys.argv.append(f"--vocab-size={cfg['vocab']}")
    print(f"[surface] training surface-only tokenizer: vocab={cfg['vocab']}, NO forced pairs, "
          f"spaceless pattern, folded corpus, K_max=1", file=sys.stderr)
    runpy.run_module("scripts.tok_train", run_name="__main__")
    print_downstream_hint(variant_base)


if __name__ == "__main__":
    main()
