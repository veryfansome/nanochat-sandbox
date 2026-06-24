"""
Compression A/B for factoring CAPITALIZATION out of token identity, and the
JOINT space+case stack — the generalization of space-as-a-bit to a second
"surface feature".

Case mechanism: lowercase the first letter of TITLECASE words (`\\p{Lu}\\p{Ll}*`,
first upper + rest lower) and record a per-token cap bit. The titlecase rule is
lossless and clean: it folds sentence-start + simple proper nouns (The, London),
and leaves acronyms (NASA), camelCase (getElementById), iPhone, McDonald
untouched (none are titlecase). The cap bit is a FREE annotation — it doesn't
add/remove tokens, so (unlike the space bit) there is NO token subtraction for
case; the compression gain comes purely from the denser case-folded vocab.

Variants (all vocab 32768, same data + code path as the headline runs):
  - caseless        : SPLIT_PATTERN on case-folded text (case only)
  - space+caseless  : SPACELESS_PATTERN on case-folded text (joint)
Reference: saved baseline / spaceless (vocab 32768).

For space+caseless, held-out tokens = encode_count(folded) - space_bit_count.
Lossless round-trip verifies (token, space_bit, cap_bit) -> original text.

Usage: uv run python -m tools.case_factor_compression [--train-chars N] [--val-chars N]
"""
import argparse
import os
import sys

import regex
import tiktoken
import rustbpe

from nanochat.tokenizer import SPLIT_PATTERN, SPECIAL_TOKENS, RustBPETokenizer
from nanochat.dataset import parquets_iter_batched
from tools._corpus_iter import iter_capped_batches
from tools.space_factor_compression import (
    SPACELESS_PATTERN, corpus_byte_and_bit_stats, encode_token_count,
)

# ---- case folding ---------------------------------------------------------
_FOLD_RE = regex.compile(r"(?<!\p{L})\p{Lu}\p{L}*")   # uppercase-initial word-starts
_TITLE = regex.compile(r"\p{Lu}\p{Ll}*")              # whole run is titlecase

def _fold_match(m):
    w = m.group(0)
    if _TITLE.fullmatch(w):
        first = w[0]
        lo = first.lower()
        # only fold if reversible AND byte-length-preserving (so offsets align)
        if lo.upper() == first and len(lo.encode("utf-8")) == len(first.encode("utf-8")):
            return lo + w[1:]
    return w

def fold_case(text):
    return _FOLD_RE.sub(_fold_match, text)

_SPACELESS_RE = regex.compile(SPACELESS_PATTERN)

# ---- training (case-folded) ----------------------------------------------

def train_folded(pattern, max_chars, doc_cap, vocab_size, label):
    def text_iter():
        n = 0
        for batch in parquets_iter_batched(split="train"):
            for doc in batch:
                d = doc[:doc_cap] if len(doc) > doc_cap else doc
                n += len(d)
                yield fold_case(d)
                if n > max_chars:
                    return
    print(f"[{label}] training on case-folded text (vocab={vocab_size})...", file=sys.stderr)
    tok = rustbpe.Tokenizer()
    tok.train_from_iterator(text_iter(), vocab_size - len(SPECIAL_TOKENS), pattern=pattern)
    mr = {bytes(k): v for k, v in tok.get_mergeable_ranks()}
    offset = len(mr)
    special = {nm: offset + i for i, nm in enumerate(SPECIAL_TOKENS)}
    enc = tiktoken.Encoding(name=label, pat_str=tok.get_pattern(),
                            mergeable_ranks=mr, special_tokens=special)
    print(f"[{label}] done: {len(mr)} merges", file=sys.stderr)
    return enc

def folded_token_count(enc, corpus):
    total = 0
    for docs in corpus:
        folded = [fold_case(d) for d in docs]
        for ids in enc.encode_ordinary_batch(folded, num_threads=8):
            total += len(ids)
    return total

# ---- lossless round-trip for the joint scheme ----------------------------

def encode_with_bits(text, enc):
    pretoks = [m.group(0) for m in _SPACELESS_RE.finditer(text)]
    toks, sbits, cbits = [], [], []
    pending = 0
    for i, pt in enumerate(pretoks):
        if pt == " " and i + 1 < len(pretoks) and not pretoks[i + 1][:1].isspace():
            pending = 1
            continue
        folded = fold_case(pt)
        cap_off = None
        if folded != pt:
            pb, fb = pt.encode("utf-8"), folded.encode("utf-8")
            for o in range(min(len(pb), len(fb))):
                if pb[o] != fb[o]:
                    cap_off = o
                    break
        ids = enc.encode_ordinary(folded)
        cap_k = None
        if cap_off is not None:
            acc = 0
            for k, tid in enumerate(ids):
                tl = len(enc.decode_single_token_bytes(tid))
                if acc <= cap_off < acc + tl:
                    cap_k = k
                    break
                acc += tl
        for k, tid in enumerate(ids):
            toks.append(tid)
            sbits.append(pending if k == 0 else 0)
            cbits.append(1 if k == cap_k else 0)
        pending = 0
    return toks, sbits, cbits

def decode_with_bits(toks, sbits, cbits, enc):
    # Build the full byte stream first, recording byte offsets where a cap
    # applies, then uppercase on the decoded string. Robust to tokens that split
    # mid-UTF-8-character (BPE can), unlike per-token decoding.
    import bisect
    out = bytearray()
    cap_offsets = []
    for tid, sb, cb in zip(toks, sbits, cbits):
        if sb:
            out += b" "
        if cb:
            cap_offsets.append(len(out))   # byte offset of this token's start
        out += enc.decode_single_token_bytes(tid)
    text = out.decode("utf-8", "replace")
    if not cap_offsets:
        return text
    char_starts, pos = [], 0
    for ch in text:
        char_starts.append(pos)
        pos += len(ch.encode("utf-8"))
    chars = list(text)
    for off in cap_offsets:
        ci = max(0, bisect.bisect_right(char_starts, off) - 1)  # char containing byte off
        while ci < len(chars) and not chars[ci].isalpha():
            ci += 1                                              # skip leading punct to the letter
        if ci < len(chars):
            chars[ci] = chars[ci].upper()
    return "".join(chars)

def render_ok(enc, docs, limit=2000):
    n_ok = n = 0
    fail = None
    for d in docs:
        if n >= limit:
            break
        n += 1
        t, s, c = encode_with_bits(d, enc)
        if decode_with_bits(t, s, c, enc) == d:
            n_ok += 1
        elif fail is None:
            fail = (d[:120], decode_with_bits(t, s, c, enc)[:120])
    return n_ok, n, fail

# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-chars", type=int, default=2_000_000_000)
    ap.add_argument("--val-chars", type=int, default=20_000_000)
    args = ap.parse_args()
    VOCAB = 32768

    enc_caseless = train_folded(SPLIT_PATTERN, args.train_chars, 10_000, VOCAB, "caseless")
    enc_joint = train_folded(SPACELESS_PATTERN, args.train_chars, 10_000, VOCAB, "space+caseless")

    V = os.path.expanduser("~/.cache/nanochat-variants")
    enc_base = RustBPETokenizer.from_directory(os.path.join(V, "baseline_repro", "tokenizer")).enc
    enc_sp = RustBPETokenizer.from_directory(os.path.join(V, "spaceless_repro", "tokenizer")).enc

    corpus = list(iter_capped_batches("val", args.val_chars))
    total_bytes, total_chars, sp_bits = corpus_byte_and_bit_stats(corpus)

    base_t = encode_token_count(enc_base, corpus)
    sp_t = encode_token_count(enc_sp, corpus) - sp_bits
    case_t = folded_token_count(enc_caseless, corpus)            # no subtraction (cap is free)
    joint_t = folded_token_count(enc_joint, corpus) - sp_bits     # subtract space bits only

    sample = [d for b in corpus[:3] for d in b]
    n_ok, n_tot, fail = render_ok(enc_joint, sample)

    def line(name, t):
        print(f"{name:<24}{t:>14,}{total_bytes/t:>12.4f}")

    print(f"\n=== Capitalization + joint space+case compression ===")
    print(f"held-out: ClimbMix val, {total_chars:,} chars / {total_bytes:,} bytes (vocab 32768)\n")
    print(f"{'scheme':<24}{'tokens':>14}{'bytes/tok':>12}")
    print("-" * 50)
    line("baseline", base_t)
    line("caseless", case_t)
    line("spaceless", sp_t)
    line("space+caseless", joint_t)
    print("-" * 50)
    print(f"caseless        vs baseline : {100*(base_t-case_t)/base_t:+.3f}% tokens")
    print(f"spaceless       vs baseline : {100*(base_t-sp_t)/base_t:+.3f}% tokens")
    print(f"space+caseless  vs baseline : {100*(base_t-joint_t)/base_t:+.3f}% tokens")
    print(f"space+caseless  vs spaceless: {100*(sp_t-joint_t)/sp_t:+.3f}% tokens  <- case stacks on space?")
    print(f"space+caseless  vs caseless : {100*(case_t-joint_t)/case_t:+.3f}% tokens  <- space stacks on case?")
    print(f"\nrender round-trip (joint space+cap bits): {n_ok}/{n_tot} docs lossless"
          + ("" if fail is None else f"\n  orig:  {fail[0]!r}\n  recon: {fail[1]!r}"))


if __name__ == "__main__":
    main()
