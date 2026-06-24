"""
Re-measure compression after applying the KNOWN digit-space regex fix to the
baseline, to isolate the genuine idea-specific gain of the space-as-a-bit scheme.

The fix is the ' ?\\p{N}{1,2}' space-prefix-digit tweak: nanochat's number
alternative \\p{N}{1,2} has no leading-space capture, so " 42" tokenizes as
[" ", "42"] (two tokens, one wasted on the space). Gluing an optional leading
space recovers it. This is NOT novel here: it is the removed `space_digits`
probe, already folded into force_merges' SPLIT_PATTERN, with a prior measured
effect of +1.09-1.23% compression (rustbpe_variants/force_merges/pairs.py:217).

The adversarial verification of the space-factoring result attributed ~30%
(+0.96%) of the spaceless +3.18% bytes/token gain to exactly this digit-space
effect. This script confirms that by training a digit-fixed baseline and
reporting:
  - baseline -> digit-fixed baseline   (should reproduce the ~+1.1% space_digits gain)
  - digit-fixed baseline -> spaceless  (the genuine space-as-a-bit gain on top)

Reuses the validated baseline_repro / spaceless_repro tokenizers from the
headline run (saved under ~/.cache/nanochat-variants), training only the new
digit-fixed variant on the same data + code path.

Usage:
    uv run python -m tools.digitfix_remeasure
"""

import os

from nanochat.tokenizer import SPLIT_PATTERN, RustBPETokenizer
from tools.space_factor_compression import (
    train_encoding, encode_token_count, corpus_byte_and_bit_stats,
)
from tools._corpus_iter import iter_capped_batches

# The exact known fix: glue an optional leading space to the number alternative.
# Matches rustbpe_variants/force_merges/pairs.py (the standalone `space_digits`).
DIGITFIX_PATTERN = SPLIT_PATTERN.replace(r"\p{N}{1,2}", r" ?\p{N}{1,2}", 1)
assert DIGITFIX_PATTERN != SPLIT_PATTERN, "replacement did not change the pattern"
assert DIGITFIX_PATTERN.count(r" ?\p{N}{1,2}") == 1, "expected exactly one digit-space tweak"

TRAIN_CHARS = 2_000_000_000
VAL_CHARS = 20_000_000
DOC_CAP = 10_000
VOCAB = 32768
VARIANTS = os.path.expanduser("~/.cache/nanochat-variants")


def main():
    # Train the digit-fixed baseline on the same data + code path as the headline.
    enc_digitfix = train_encoding(DIGITFIX_PATTERN, TRAIN_CHARS, DOC_CAP, VOCAB, "digitfix")

    # Reuse the already-validated tokenizers (baseline_repro is bit-identical to
    # the cached canonical; spaceless_repro produced the +3.18% headline).
    enc_base = RustBPETokenizer.from_directory(
        os.path.join(VARIANTS, "baseline_repro", "tokenizer")).enc
    enc_space = RustBPETokenizer.from_directory(
        os.path.join(VARIANTS, "spaceless_repro", "tokenizer")).enc

    corpus = list(iter_capped_batches("val", VAL_CHARS))
    total_bytes, total_chars, bit_count = corpus_byte_and_bit_stats(corpus)

    base_t = encode_token_count(enc_base, corpus)
    df_t = encode_token_count(enc_digitfix, corpus)
    space_t = encode_token_count(enc_space, corpus) - bit_count

    def row(name, t):
        print(f"{name:<24}{t:>14,}{total_bytes / t:>12.4f}")

    print(f"\n=== Digit-space fix re-measurement ===")
    print(f"held-out: ClimbMix val, {total_chars:,} chars / {total_bytes:,} bytes\n")
    print(f"{'scheme':<24}{'tokens':>14}{'bytes/tok':>12}")
    print("-" * 50)
    row("baseline", base_t)
    row("baseline + digit-fix", df_t)
    row("spaceless (bit)", space_t)
    print("-" * 50)
    print(f"digit-fix vs baseline   : {100*(base_t-df_t)/base_t:+.3f}% tokens  "
          f"(prior space_digits: +1.09-1.23%)")
    print(f"spaceless vs baseline   : {100*(base_t-space_t)/base_t:+.3f}% tokens  (headline)")
    print(f"spaceless vs digit-fix  : {100*(df_t-space_t)/df_t:+.3f}% tokens  "
          f"<- genuine space-as-a-bit gain beyond the known fix")


if __name__ == "__main__":
    main()
