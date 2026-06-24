"""
Triple stack: force_merges (122 phrases) + space + case, the full surface-factored
tokenizer.

Built on globally case-folded text, so forced phrases are the FOLDED forms
(' of the', and '. the' from the sentence-start '. The'). Caps are a per-token
offset LIST, not a single bit: a phrase token spanning N words can carry up to N
caps (rare — <=~0.2% of phrase occurrences, titlecase-multi-word like 'Is The',
'At The', 'More Than'; all-caps 'OF THE' does NOT fold and stays uppercase
content with zero cap bits), and the list
representation renders all of them losslessly. Space bits and phrase-internal
spaces work as in the FM stack.

Trains (rustbpe_force_merges, block_leading/trailing_space=True), all vs the
saved baseline/spaceless (vocab 32768):
  - fm_space : force_merges(122) + space   (unfolded; = the prior +9.15% stack)
  - triple   : force_merges(122) + space + case (case-folded)
Reports compression + the key delta triple-vs-fm_space (does case stack on top),
and a lossless round-trip of the triple (token + space-bit + cap-offset streams).

Usage: uv run python -m tools.stack_triple [--train-chars N] [--val-chars N]
"""
import argparse
import bisect
import os
import sys

import tiktoken
import rustbpe_force_merges

from nanochat.tokenizer import SPECIAL_TOKENS, RustBPETokenizer
from nanochat.dataset import parquets_iter_batched
from rustbpe_variants.force_merges.pairs import FORCED_PAIRS
from tools._corpus_iter import iter_capped_batches
from tools.stack_fm_spaceless import build_config, raw_token_count, stacked_bit_count_and_bytes
from tools.space_factor_compression import corpus_byte_and_bit_stats
from tools.case_factor_compression import fold_case

# Folded phrase set (dedup, preserve order so derived chains stay valid).
_seen, FOLDED_PAIRS = set(), []
for a, b in ((fold_case(a), fold_case(b)) for (a, b) in FORCED_PAIRS):
    if (a, b) not in _seen:
        _seen.add((a, b))
        FOLDED_PAIRS.append((a, b))

CFG_FMSPACE = build_config(list(FORCED_PAIRS))   # unfolded, full 122
CFG_TRIPLE = build_config(FOLDED_PAIRS)          # folded phrases


def _train(pattern, forced, vocab, max_chars, fold, label):
    def text_iter():
        n = 0
        for batch in parquets_iter_batched(split="train"):
            for doc in batch:
                d = doc[:10_000]
                n += len(d)
                yield fold_case(d) if fold else d
                if n > max_chars:
                    return
    print(f"[{label}] training (vocab={vocab}, {len(forced)} forced, fold={fold})...", file=sys.stderr)
    tok = rustbpe_force_merges.Tokenizer()
    tok.train_from_iterator(text_iter(), vocab - len(SPECIAL_TOKENS), pattern=pattern,
                            forced_pairs=forced, block_trailing_space=True, block_leading_space=True)
    mr = {bytes(k): v for k, v in tok.get_mergeable_ranks()}
    offset = len(mr)
    special = {nm: offset + i for i, nm in enumerate(SPECIAL_TOKENS)}
    enc = tiktoken.Encoding(name=label, pat_str=tok.get_pattern(), mergeable_ranks=mr, special_tokens=special)
    print(f"[{label}] done: {len(mr)} merges", file=sys.stderr)
    return enc


def folded_raw_count(enc, corpus):
    t = 0
    for docs in corpus:
        for ids in enc.encode_ordinary_batch([fold_case(d) for d in docs], num_threads=8):
            t += len(ids)
    return t


# ---- cap-aware lossless round-trip for the triple -------------------------

def encode_triple(d, enc, sre):
    folded = fold_case(d)
    fb, ob = folded.encode("utf-8"), d.encode("utf-8")
    cap = set(i for i in range(min(len(fb), len(ob))) if fb[i] != ob[i])  # folded-letter byte offsets
    pretoks = [m.group(0) for m in sre.finditer(folded)]
    toks, sbits, caps = [], [], []
    pending = 0
    goff = 0
    for i, pt in enumerate(pretoks):
        ptb = pt.encode("utf-8")
        if pt == " " and i + 1 < len(pretoks) and not pretoks[i + 1][:1].isspace():
            pending = 1
            goff += len(ptb)
            continue
        ids = enc.encode_ordinary(pt)
        acc = 0
        for k, tid in enumerate(ids):
            tl = len(enc.decode_single_token_bytes(tid))
            rel = [o for o in range(tl) if (goff + acc + o) in cap]
            toks.append(tid)
            sbits.append(pending if k == 0 else 0)
            caps.append(rel)
            acc += tl
        pending = 0
        goff += len(ptb)
    return toks, sbits, caps


def decode_triple(toks, sbits, caps, enc):
    out = bytearray()
    cap_bp = []
    for tid, sb, rel in zip(toks, sbits, caps):
        if sb:
            out += b" "
        start = len(out)
        out += enc.decode_single_token_bytes(tid)
        cap_bp.extend(start + o for o in rel)
    text = out.decode("utf-8", "replace")
    if not cap_bp:
        return text
    char_starts, pos = [], 0
    for ch in text:
        char_starts.append(pos)
        pos += len(ch.encode("utf-8"))
    chars = list(text)
    for bp in cap_bp:
        ci = max(0, bisect.bisect_right(char_starts, bp) - 1)
        if ci < len(chars):
            chars[ci] = chars[ci].upper()
    return "".join(chars)


def render_ok(enc, docs, sre, limit=2000):
    n_ok = n = 0
    fail = None
    for d in docs:
        if n >= limit:
            break
        n += 1
        t, s, c = encode_triple(d, enc, sre)
        r = decode_triple(t, s, c, enc)
        if r == d:
            n_ok += 1
        elif fail is None:
            fail = (d[:120], r[:120])
    return n_ok, n, fail


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-chars", type=int, default=2_000_000_000)
    ap.add_argument("--val-chars", type=int, default=20_000_000)
    args = ap.parse_args()

    enc_fmspace = _train(CFG_FMSPACE["spaceless_pattern"], CFG_FMSPACE["spaceless_forced"],
                         CFG_FMSPACE["vocab"], args.train_chars, False, "fm_space")
    enc_triple = _train(CFG_TRIPLE["spaceless_pattern"], CFG_TRIPLE["spaceless_forced"],
                        CFG_TRIPLE["vocab"], args.train_chars, True, "triple")

    V = os.path.expanduser("~/.cache/nanochat-variants")
    enc_base = RustBPETokenizer.from_directory(os.path.join(V, "baseline_repro", "tokenizer")).enc
    enc_sp = RustBPETokenizer.from_directory(os.path.join(V, "spaceless_repro", "tokenizer")).enc

    corpus = list(iter_capped_batches("val", args.val_chars))
    fmspace_bits, total_bytes, total_chars = stacked_bit_count_and_bytes(corpus, CFG_FMSPACE["spaceless_re"])
    _, _, sp_bits = corpus_byte_and_bit_stats(corpus)
    folded_corpus = [[fold_case(d) for d in docs] for docs in corpus]
    triple_bits, _, _ = stacked_bit_count_and_bytes(folded_corpus, CFG_TRIPLE["spaceless_re"])

    base_t = raw_token_count(enc_base, corpus)
    sp_t = raw_token_count(enc_sp, corpus) - sp_bits
    fmspace_t = raw_token_count(enc_fmspace, corpus) - fmspace_bits
    triple_t = folded_raw_count(enc_triple, corpus) - triple_bits

    sample = [d for b in corpus[:3] for d in b]
    n_ok, n_tot, fail = render_ok(enc_triple, sample, CFG_TRIPLE["spaceless_re"])

    def line(name, t, vocab):
        print(f"{name:<26}{vocab:>7}{t:>14,}{total_bytes/t:>12.4f}")

    print(f"\n=== Triple stack: force_merges + space + case ===")
    print(f"held-out: ClimbMix val, {total_chars:,} chars / {total_bytes:,} bytes\n")
    print(f"{'scheme':<26}{'vocab':>7}{'tokens':>14}{'bytes/tok':>12}")
    print("-" * 59)
    line("baseline", base_t, 32768)
    line("spaceless", sp_t, 32768)
    line("fm_full + space", fmspace_t, CFG_FMSPACE["vocab"])
    line("fm_full + space + case", triple_t, CFG_TRIPLE["vocab"])
    print("-" * 59)
    print(f"fm+space        vs baseline : {100*(base_t-fmspace_t)/base_t:+.3f}% tokens")
    print(f"triple          vs baseline : {100*(base_t-triple_t)/base_t:+.3f}% tokens")
    print(f"triple          vs fm+space : {100*(fmspace_t-triple_t)/fmspace_t:+.3f}% tokens  <- case stacks on fm+space?")
    print(f"\nrender round-trip (token+space+cap-offset streams): {n_ok}/{n_tot} docs lossless"
          + ("" if fail is None else f"\n  orig:  {fail[0]!r}\n  recon: {fail[1]!r}"))


if __name__ == "__main__":
    main()
