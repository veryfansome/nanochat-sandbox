"""
Compression A/B for the "space-as-a-bit" tokenizer reparametrization.

Idea under test (see the design discussion): factor the single leading SPACE
out of token identities and into a side-channel binary bit ("is the next token
preceded by a space?", predicted by a cheap auxiliary head). BPE merges are
then learned on space-stripped pretokens, freeing the ~12% of vocab currently
spent on " X"/"X" duplicate pairs (plus cleaner pooled statistics) for longer
merges. The space bit is lossless at the tokenizer level; its only cost is
head-2's prediction burden, deferred to a model A/B.

This script answers the ONE question that gates prototyping: does retraining
BPE at the SAME vocab budget on space-stripped pretokens compress held-out text
better (fewer tokens / higher bytes-per-token)?

Method — a controlled A/B, both sides trained by the same code path on the same
data, differing ONLY in the pretokenization pattern:

  baseline  : nanochat's SPLIT_PATTERN (leading space glued onto words/punct).
  spaceless : SPLIT_PATTERN with the leading-space capture removed from the
              word and punct alternatives (space falls through to \\s+, so it
              is isolated as its own pretoken). Everything else identical.

Held-out token count for the spaceless scheme = raw encode_ordinary token count
MINUS the number of "space immediately followed by non-whitespace" occurrences
in the corpus. Each such occurrence is the single space adjacent to a content
token (the regex isolates it as a lone " " pretoken → one space token), which
the scheme absorbs into that token's bit, removing exactly one token. Multi-space
runs keep all-but-the-adjacent space as tokens, so this does NOT over-credit
indentation. The fast count is validated against a slow pretoken-level
simulation on a sample (--validate).

Bytes are the original UTF-8 bytes (spaces included) for BOTH schemes, so
bytes-per-token is directly comparable.

Usage:
    uv run python -m tools.space_factor_compression \\
        --train-chars 500_000_000 --val-chars 20_000_000 --vocab-size 32768
    # add --save-dir ~/.cache/nanochat-variants to persist both tokenizers
    # add --validate to cross-check the fast bit-count on a sample
"""

import argparse
import os
import sys
import time
from collections import Counter
from math import log2

import regex

import tiktoken
import rustbpe
from nanochat.tokenizer import SPLIT_PATTERN, SPECIAL_TOKENS, RustBPETokenizer
from nanochat.dataset import parquets_iter_batched
from tools._corpus_iter import iter_capped_batches

# The spaceless pattern: same as SPLIT_PATTERN but the leading-space capture is
# removed from (a) the word alternative — add a literal space to its negated
# leading-char class — and (b) the punct alternative — drop the leading " ?".
# Space then falls through to the \s+ alternatives and is isolated.
SPACELESS_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N} ]?+\p{L}+|\p{N}{1,2}|[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

# regex: a single space (0x20) immediately followed by a non-whitespace char.
# This is exactly the "adjacent-to-content" space the scheme turns into a bit.
_BIT_RE = regex.compile(r" (?=\S)")


def train_encoding(pattern, max_chars, doc_cap, vocab_size, label):
    """Replicate RustBPETokenizer.train_from_iterator but with a custom
    pretokenization pattern. Returns a tiktoken.Encoding."""
    def text_iter():
        n = 0
        for batch in parquets_iter_batched(split="train"):
            for doc in batch:
                d = doc[:doc_cap] if len(doc) > doc_cap else doc
                n += len(d)
                yield d
                if n > max_chars:
                    return

    print(f"[{label}] training rustbpe (vocab={vocab_size}, ≤{max_chars:,} chars)...",
          file=sys.stderr)
    t0 = time.time()
    tok = rustbpe.Tokenizer()
    vocab_no_special = vocab_size - len(SPECIAL_TOKENS)
    assert vocab_no_special >= 256
    tok.train_from_iterator(text_iter(), vocab_no_special, pattern=pattern)
    dt = time.time() - t0
    mr_list = tok.get_mergeable_ranks()
    mergeable_ranks = {bytes(k): v for k, v in mr_list}
    offset = len(mergeable_ranks)
    special = {name: offset + i for i, name in enumerate(SPECIAL_TOKENS)}
    enc = tiktoken.Encoding(
        name=f"rustbpe-{label}",
        pat_str=tok.get_pattern(),
        mergeable_ranks=mergeable_ranks,
        special_tokens=special,
    )
    print(f"[{label}] done in {dt:.1f}s, {len(mergeable_ranks)} merges + "
          f"{len(special)} special", file=sys.stderr)
    return enc


def encode_token_count(enc, corpus):
    """Total ordinary tokens over a list of doc-batches (parallel batch encode)."""
    total = 0
    for docs in corpus:
        for ids in enc.encode_ordinary_batch(docs, num_threads=8):
            total += len(ids)
    return total


def corpus_byte_and_bit_stats(corpus):
    """One pass over raw text: total UTF-8 bytes, total chars, and bit_count
    (# of spaces immediately followed by non-whitespace = scheme's set bits)."""
    total_bytes = 0
    total_chars = 0
    bit_count = 0
    for docs in corpus:
        for d in docs:
            total_bytes += len(d.encode("utf-8"))
            total_chars += len(d)
            bit_count += len(_BIT_RE.findall(d))
    return total_bytes, total_chars, bit_count


def slow_spaceless_count(enc_spaceless, docs):
    """Pretoken-level reference simulation of the scheme's token count on a
    list of docs. A lone ' ' pretoken immediately followed by a non-whitespace
    pretoken is absorbed as a bit (0 tokens); everything else is BPE-encoded.
    Used to validate the fast (raw_count - bit_count) method."""
    pat = regex.compile(SPACELESS_PATTERN)
    n_tokens = 0
    n_bits = 0
    for d in docs:
        pretoks = [m.group(0) for m in pat.finditer(d)]
        for i, pt in enumerate(pretoks):
            if pt == " " and i + 1 < len(pretoks) and not pretoks[i + 1][:1].isspace():
                n_bits += 1
                continue
            n_tokens += len(enc_spaceless.encode_ordinary(pt))
    return n_tokens, n_bits


def binary_entropy(p):
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return -p * log2(p) - (1 - p) * log2(1 - p)


def conditional_bit_stats(enc_spaceless, docs):
    """Estimate head-2's realistic cost: the entropy of the space bit given the
    PREVIOUS token id (the single most predictive feature head-2 sees; the real
    head conditions on full context so does at least this well → upper bound).

    Walks pretokens, absorbs the adjacent single space as a bit on the FIRST
    token of the following content pretoken, and accumulates (prev_token_id, bit)
    over those boundary tokens. Interior tokens of multi-token pretokens carry
    bit=0 deterministically and add ~0 entropy, so we fold them in only via the
    total-token denominator.

    Returns (total_head2_bits, n_boundary, n_tokens_total):
      total_head2_bits = sum over prev_id groups of n*H(p1)  [bits]
    """
    pat = regex.compile(SPACELESS_PATTERN)
    pair = Counter()        # (prev_id, bit) -> count, boundary tokens only
    n_tokens_total = 0
    for d in docs:
        pretoks = [m.group(0) for m in pat.finditer(d)]
        last_id = -1        # sentinel for doc start
        pending_bit = 0
        for i, pt in enumerate(pretoks):
            if pt == " " and i + 1 < len(pretoks) and not pretoks[i + 1][:1].isspace():
                pending_bit = 1
                continue
            ids = enc_spaceless.encode_ordinary(pt)
            if not ids:
                continue
            pair[(last_id, pending_bit)] += 1     # bit applies to first token
            n_tokens_total += len(ids)
            last_id = ids[-1]
            pending_bit = 0

    # Conditional entropy: group by prev_id, sum n*H(p1).
    by_prev = {}
    for (prev_id, bit), c in pair.items():
        d = by_prev.setdefault(prev_id, [0, 0])
        d[bit] += c
    total_head2_bits = 0.0
    n_boundary = 0
    for prev_id, (n0, n1) in by_prev.items():
        n = n0 + n1
        n_boundary += n
        total_head2_bits += n * binary_entropy(n1 / n)
    return total_head2_bits, n_boundary, n_tokens_total


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train-chars", type=int, default=500_000_000)
    ap.add_argument("--val-chars", type=int, default=20_000_000)
    ap.add_argument("--doc-cap", type=int, default=10_000)
    ap.add_argument("--vocab-size", type=int, default=32768)
    ap.add_argument("--save-dir", type=str, default=None,
                    help="If set, persist both tokenizers under <dir>/{baseline,spaceless}_repro/")
    ap.add_argument("--validate", action="store_true",
                    help="Cross-check fast bit-count vs slow pretoken simulation on a sample")
    ap.add_argument("--cond-sample-chars", type=int, default=3_000_000,
                    help="Char sample for conditional bit-entropy (head-2 cost) estimate")
    ap.add_argument("--compare-cached", action="store_true",
                    help="Also report the cached ~/.cache/nanochat/tokenizer baseline for sanity")
    args = ap.parse_args()

    # Train both tokenizers on the same data, differing only in pattern.
    enc_base = train_encoding(SPLIT_PATTERN, args.train_chars, args.doc_cap,
                              args.vocab_size, "baseline")
    enc_space = train_encoding(SPACELESS_PATTERN, args.train_chars, args.doc_cap,
                               args.vocab_size, "spaceless")

    if args.save_dir:
        for label, enc in [("baseline_repro", enc_base), ("spaceless_repro", enc_space)]:
            d = os.path.join(os.path.expanduser(args.save_dir), label, "tokenizer")
            os.makedirs(d, exist_ok=True)
            import pickle
            with open(os.path.join(d, "tokenizer.pkl"), "wb") as f:
                pickle.dump(enc, f)
            print(f"saved {label} -> {d}", file=sys.stderr)

    # Held-out corpus (materialize once so both encoders see identical docs).
    print(f"loading held-out val ({args.val_chars:,} char cap)...", file=sys.stderr)
    corpus = list(iter_capped_batches("val", args.val_chars))

    total_bytes, total_chars, bit_count = corpus_byte_and_bit_stats(corpus)
    base_tokens = encode_token_count(enc_base, corpus)
    space_raw_tokens = encode_token_count(enc_space, corpus)
    space_tokens = space_raw_tokens - bit_count

    if args.validate:
        # Slow reference on the first batch (or two) — should match fast method.
        sample = corpus[: min(3, len(corpus))]
        slow_tok, slow_bits = slow_spaceless_count(enc_space, [d for b in sample for d in b])
        fast_raw = encode_token_count(enc_space, sample)
        s_bytes, s_chars, fast_bits = corpus_byte_and_bit_stats(sample)
        fast_tok = fast_raw - fast_bits
        print("\n=== VALIDATION (sample) ===", file=sys.stderr)
        print(f"  slow pretoken sim : tokens={slow_tok:,}  bits={slow_bits:,}", file=sys.stderr)
        print(f"  fast (raw-bits)   : tokens={fast_tok:,}  bits={fast_bits:,}", file=sys.stderr)
        match = (slow_tok == fast_tok and slow_bits == fast_bits)
        print(f"  MATCH: {match}", file=sys.stderr)

    # ---- Report ----
    base_bpt = total_bytes / base_tokens
    space_bpt = total_bytes / space_tokens
    tok_delta = space_tokens - base_tokens
    bpt_delta_pct = 100 * (space_bpt - base_bpt) / base_bpt

    # Space-bit cost (UPPER bound: context-free head-2). p = fraction of
    # spaceless tokens that are space-preceded.
    p = bit_count / space_tokens
    bits_per_byte_added = space_tokens * binary_entropy(p) / total_bytes

    print(f"\n=== Space-factoring compression A/B ===")
    print(f"held-out: ClimbMix val, {total_chars:,} chars / {total_bytes:,} bytes")
    print(f"train:    {args.train_chars:,} chars, vocab={args.vocab_size}\n")
    print(f"{'scheme':<14}{'tokens':>16}{'bytes/token':>14}")
    print(f"{'-'*44}")
    print(f"{'baseline':<14}{base_tokens:>16,}{base_bpt:>14.4f}")
    print(f"{'spaceless':<14}{space_tokens:>16,}{space_bpt:>14.4f}")
    print(f"{'-'*44}")
    print(f"token delta : {tok_delta:>+,} ({100*tok_delta/base_tokens:+.3f}%)")
    print(f"bytes/token : {space_bpt - base_bpt:+.4f} ({bpt_delta_pct:+.3f}%)  "
          f"[+ = spaceless compresses better]")
    print(f"\nspace bits  : {bit_count:,} set / {space_tokens:,} tokens "
          f"(p={p:.3f}, marginal H={binary_entropy(p):.3f} bit/token)")
    print(f"bit-stream cost (context-free head-2, UPPER bound): "
          f"{bits_per_byte_added:.4f} bits/byte added")

    # Realistic head-2 cost: condition on previous token (sample).
    cond_corpus = list(iter_capped_batches("val", args.cond_sample_chars))
    cond_docs = [d for b in cond_corpus for d in b]
    cb_bits, cb_boundary, cb_tokens = conditional_bit_stats(enc_space, cond_docs)
    cs_bytes, cs_chars, _ = corpus_byte_and_bit_stats(cond_corpus)
    cond_bpb = cb_bits / cs_bytes
    cond_per_tok = cb_bits / cb_tokens
    print(f"bit-stream cost (cond. on prev token, ~{args.cond_sample_chars/1e6:.0f}M sample): "
          f"{cond_bpb:.4f} bits/byte  ({cond_per_tok:.4f} bit/token)")
    print(f"  (real head-2 sees full context → true cost ≤ this)")

    if args.compare_cached:
        try:
            cached = RustBPETokenizer.from_directory(
                os.path.expanduser("~/.cache/nanochat/tokenizer"))
            ct = encode_token_count(cached.enc, corpus)
            print(f"\n[sanity] cached baseline tokens: {ct:,} "
                  f"(bytes/token {total_bytes/ct:.4f}); repro baseline {base_tokens:,}")
        except Exception as e:
            print(f"[sanity] cached baseline load failed: {e}")


if __name__ == "__main__":
    main()
