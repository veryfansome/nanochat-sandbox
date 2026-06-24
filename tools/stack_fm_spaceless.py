"""
Stack the space-as-a-bit scheme on top of force_merges (word-phrases only).

Two design decisions per the request:
  1. PHRASES ONLY: drop force_merges' punctuation/numeric-glued forced pairs
     (',' + ' and', '.' + ' The', ' However' + ',', '00' + '0', ...), keeping
     only pure word-phrases (' of' + ' the', ' one of' + ' the', ...). 69 of
     the 122 forced pairs survive.
  2. SPACE-FACTORING ON TOP: leading spaces become a per-token bit, INTERNAL
     phrase spaces stay in the token. So ' of the' -> token "of the" (internal
     space kept) + a leading-space bit. Transform: strip one leading space from
     the LEFT operand of each forced pair (' of',' the')->('of',' the'); the
     carve-out matches the phrase without its leading space ("of the").

Builds, on the same data + crate (rustbpe_force_merges, block_leading/trailing
_space=True) as adopted force_merges:
  - fm_phrases            : phrases-only force_merges, NORMAL spaces (the base
                            we stack on; force_merges body incl. ' ?\\p{N}{1,2}')
  - fm_phrases_spaceless  : phrases-only + space-factoring (the stack)
both at vocab 32768 + 69 (additive, force_merges convention).

Reports compression vs the saved baseline/spaceless (32768), and crucially
fm_phrases_spaceless vs fm_phrases (does space-factoring add ON TOP of phrases?).
Bit-count is pretoken-aware (lone space before content only; phrase-internal
spaces are NOT credited). Also runs a lossless round-trip render check.

Usage:
    uv run python -m tools.stack_fm_spaceless [--train-chars N] [--val-chars N]
"""

import argparse
import os
import re
import sys

import regex
import tiktoken
import rustbpe_force_merges

from nanochat.tokenizer import SPECIAL_TOKENS, RustBPETokenizer
from nanochat.dataset import parquets_iter_batched
from rustbpe_variants.force_merges.pairs import FORCED_PAIRS
from tools._corpus_iter import iter_capped_batches

# ---- phrase filtering + transforms ----------------------------------------

def is_word_phrase(a, b):
    return regex.fullmatch(r"[ \p{L}]+", a + b) is not None

def strip_lead(s):
    return s[1:] if s.startswith(" ") else s

WORD_PHRASES = [(a, b) for (a, b) in FORCED_PAIRS if is_word_phrase(a, b)]  # order preserved

def build_expr(phrase_strs):
    ph = sorted(set(phrase_strs), key=len, reverse=True)  # longest match wins
    return "(" + "|".join(re.escape(p) for p in ph) + ")(?=([^a-z]|$))"

# force_merges body (leading-space-glued words/punct, ' ?\p{N}{1,2}' digit-fix)
FM_BODY = (r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+| ?\p{N}{1,2}"""
           r"""| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+""")
# spaceless body (no leading-space capture; '\p{N}{1,2}' since digit-space -> bit)
SPACELESS_BODY = (r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N} ]?+\p{L}+|\p{N}{1,2}"""
                  r"""|[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+""")


def build_config(pairs):
    """Build forced-pair lists + patterns + vocab for a given set of forced
    pairs. Works for any subset: strip_lead(a)+b == strip_lead(a+b) holds for
    all pairs (word OR punct OR numeric), so word phrases get their leading
    space factored to a bit while punct stays content."""
    spaceless_forced = [(strip_lead(a), b) for (a, b) in pairs]
    normal_pattern = build_expr([a + b for (a, b) in pairs]) + "|" + FM_BODY
    spaceless_pattern = build_expr([strip_lead(a + b) for (a, b) in pairs]) + "|" + SPACELESS_BODY
    return {
        "normal_forced": list(pairs),
        "spaceless_forced": spaceless_forced,
        "normal_pattern": normal_pattern,
        "spaceless_pattern": spaceless_pattern,
        "vocab": 32768 + len(pairs),  # additive
        "spaceless_re": regex.compile(spaceless_pattern),
    }

# ---- training -------------------------------------------------------------

def train_fm(pattern, forced, vocab, max_chars, label):
    def text_iter():
        n = 0
        for batch in parquets_iter_batched(split="train"):
            for doc in batch:
                d = doc[:10_000]
                n += len(d)
                yield d
                if n > max_chars:
                    return
    print(f"[{label}] training (vocab={vocab}, {len(forced)} forced)...", file=sys.stderr)
    tok = rustbpe_force_merges.Tokenizer()
    tok.train_from_iterator(
        text_iter(), vocab - len(SPECIAL_TOKENS), pattern=pattern,
        forced_pairs=forced, block_trailing_space=True, block_leading_space=True,
    )
    mr = {bytes(k): v for k, v in tok.get_mergeable_ranks()}
    offset = len(mr)
    special = {nm: offset + i for i, nm in enumerate(SPECIAL_TOKENS)}
    enc = tiktoken.Encoding(name=label, pat_str=tok.get_pattern(),
                            mergeable_ranks=mr, special_tokens=special)
    print(f"[{label}] done: {len(mr)} merges", file=sys.stderr)
    return enc

# ---- measurement ----------------------------------------------------------

def raw_token_count(enc, corpus):
    return sum(len(ids) for docs in corpus
               for ids in enc.encode_ordinary_batch(docs, num_threads=8))

def stacked_bit_count_and_bytes(corpus, spaceless_re):
    """Pretoken-aware: lone ' ' pretoken (under the spaceless-FM pattern) that
    immediately precedes a non-whitespace pretoken = one absorbed bit. Phrase
    chunks are single pretokens, so their internal spaces are never counted."""
    total_bytes = total_chars = bit_count = 0
    for docs in corpus:
        for d in docs:
            total_bytes += len(d.encode("utf-8"))
            total_chars += len(d)
            pretoks = [m.group(0) for m in spaceless_re.finditer(d)]
            for i, pt in enumerate(pretoks):
                if pt == " " and i + 1 < len(pretoks) and not pretoks[i + 1][:1].isspace():
                    bit_count += 1
    return bit_count, total_bytes, total_chars

# ---- lossless render verification -----------------------------------------

def render_roundtrip_ok(enc, docs, spaceless_re, limit=None):
    """Encode each doc into (token, bit) streams per the scheme, decode, compare.
    Returns (n_ok, n_total, first_failure_or_None)."""
    n_ok = 0
    n = 0
    first_fail = None
    for d in docs:
        if limit and n >= limit:
            break
        n += 1
        pretoks = [m.group(0) for m in spaceless_re.finditer(d)]
        toks, bits = [], []
        pending = 0
        for i, pt in enumerate(pretoks):
            if pt == " " and i + 1 < len(pretoks) and not pretoks[i + 1][:1].isspace():
                pending = 1
                continue
            ids = enc.encode_ordinary(pt)
            for j, tid in enumerate(ids):
                toks.append(tid)
                bits.append(pending if j == 0 else 0)
            pending = 0
        # decode
        out = bytearray()
        for tid, bit in zip(toks, bits):
            if bit:
                out += b" "
            out += enc.decode_single_token_bytes(tid)
        recon = out.decode("utf-8", errors="replace")
        if recon == d:
            n_ok += 1
        elif first_fail is None:
            first_fail = (d[:120], recon[:120])
    return n_ok, n, first_fail

# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-chars", type=int, default=2_000_000_000)
    ap.add_argument("--val-chars", type=int, default=20_000_000)
    ap.add_argument("--all-pairs", action="store_true",
                    help="Use all 122 forced pairs (incl. punct/numeric) instead of 69 word-phrases only")
    args = ap.parse_args()

    pairs = list(FORCED_PAIRS) if args.all_pairs else WORD_PHRASES
    tag = "fm_full" if args.all_pairs else "fm_phrases"
    cfg = build_config(pairs)
    dropped = len(FORCED_PAIRS) - len(pairs)
    print(f"forced pairs: {len(pairs)} / {len(FORCED_PAIRS)} ({'ALL incl punct/numeric' if args.all_pairs else 'word-phrases only'}); "
          f"vocab={cfg['vocab']}", file=sys.stderr)

    enc_fm = train_fm(cfg["normal_pattern"], cfg["normal_forced"], cfg["vocab"], args.train_chars, tag)
    enc_fms = train_fm(cfg["spaceless_pattern"], cfg["spaceless_forced"], cfg["vocab"], args.train_chars, tag + "_spaceless")

    # Reference saved tokenizers (vocab 32768).
    V = os.path.expanduser("~/.cache/nanochat-variants")
    enc_base = RustBPETokenizer.from_directory(os.path.join(V, "baseline_repro", "tokenizer")).enc
    enc_sp = RustBPETokenizer.from_directory(os.path.join(V, "spaceless_repro", "tokenizer")).enc

    corpus = list(iter_capped_batches("val", args.val_chars))
    bit_count, total_bytes, total_chars = stacked_bit_count_and_bytes(corpus, cfg["spaceless_re"])
    from tools.space_factor_compression import corpus_byte_and_bit_stats
    _, _, sp_bits = corpus_byte_and_bit_stats(corpus)

    base_t = raw_token_count(enc_base, corpus)
    sp_t = raw_token_count(enc_sp, corpus) - sp_bits
    fm_t = raw_token_count(enc_fm, corpus)
    fms_t = raw_token_count(enc_fms, corpus) - bit_count

    # Render verification on a sample.
    sample_docs = [d for b in corpus[:3] for d in b]
    n_ok, n_tot, fail = render_roundtrip_ok(enc_fms, sample_docs, cfg["spaceless_re"], limit=2000)

    def line(name, t, vocab):
        print(f"{name:<26}{vocab:>7}{t:>14,}{total_bytes/t:>12.4f}")

    kind = "ALL 122 pairs" if args.all_pairs else "phrases only"
    print(f"\n=== Space-factoring stacked on force_merges ({kind}) ===")
    print(f"held-out: ClimbMix val, {total_chars:,} chars / {total_bytes:,} bytes")
    print(f"forced pairs: {len(pairs)} (dropped {dropped} punct/numeric)\n")
    print(f"{'scheme':<26}{'vocab':>7}{'tokens':>14}{'bytes/tok':>12}")
    print("-" * 59)
    line("baseline", base_t, 32768)
    line("spaceless", sp_t, 32768)
    line(f"{tag} (normal)", fm_t, cfg["vocab"])
    line(f"{tag} + spaceless", fms_t, cfg["vocab"])
    print("-" * 59)
    print(f"{tag:<12} vs baseline : {100*(base_t-fm_t)/base_t:+.3f}% tokens")
    print(f"spaceless    vs baseline : {100*(base_t-sp_t)/base_t:+.3f}% tokens")
    print(f"stacked      vs baseline : {100*(base_t-fms_t)/base_t:+.3f}% tokens")
    print(f"stacked      vs {tag}: {100*(fm_t-fms_t)/fm_t:+.3f}% tokens  "
          f"<- does space-factoring stack on phrases?")
    print(f"\nrender round-trip: {n_ok}/{n_tot} docs lossless"
          + ("" if fail is None else f"  FIRST FAIL:\n  orig:  {fail[0]!r}\n  recon: {fail[1]!r}"))


if __name__ == "__main__":
    main()
