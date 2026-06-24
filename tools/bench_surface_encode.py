"""
Benchmark the surface-factored tokenizer's encode (the dataloader's on-the-fly
cost) vs plain tiktoken batch encode, to decide whether the Python encode is a
training-throughput bottleneck or needs Rust-ifying.

Three encoders, all producing what training needs (base token ids + per-token
surface labels for surface variants):
  - PLAIN      : enc.encode_ordinary_batch (what nanochat's dataloader does today)
  - SURFACE-OPT: fold_case (regex) + batch encode + an O(tokens) Python pass that
                 drops lone-space tokens (=> space bit) and maps cap byte-offsets
                 to tokens via a precomputed id->bytelen table. The realistic
                 production-Python implementation.
  - SURFACE-NAIVE: tools.stack_triple.encode_triple (per-pretoken encode_ordinary
                 + per-token decode_single_token_bytes) — the correctness-reference,
                 NOT meant for the hot path. Timed on a smaller slice.

Reports chars/sec and tok/sec for each, and compares to nanochat's measured d24
throughput (~291,210 tok/sec total across 8 ranks => ~36,400 tok/sec/rank that
each rank's dataloader must sustain, prefetched behind the ~1.8s step).

Uses spaceless_repro for encode timing (encode speed is ~linear in text, vocab-
independent, so it represents the joint tokenizer's cost); fold_case adds the
case-side work.

Usage: uv run python -m tools.bench_surface_encode [--chars N] [--naive-chars N]
"""
import argparse
import os
import time

import regex

from nanochat.tokenizer import RustBPETokenizer
from tools._corpus_iter import iter_capped_batches
from tools.case_factor_compression import fold_case

D24_TOKS_PER_SEC_TOTAL = 291_210
D24_RANKS = 8

_SPACE = b" "


def opt_surface_encode(docs, enc, id2len, space_id):
    """Production-Python surface encode: returns (base_ids, space_bits, cap_labels)
    per doc. Fast path = batch encode + O(tokens) bit/cap assignment."""
    folded = [fold_case(d) for d in docs]
    ids_batch = enc.encode_ordinary_batch(folded, num_threads=8)
    out = []
    for d, fol, ids in zip(docs, folded, ids_batch):
        ob, fb = d.encode("utf-8"), fol.encode("utf-8")
        # cap byte offsets (folded letters), sorted ascending
        caps = [i for i in range(min(len(ob), len(fb))) if ob[i] != fb[i]]
        base_ids, sbits, clabels = [], [], []
        ci, ncaps = 0, len(caps)
        off = 0
        pending = 0
        for tid in ids:
            tl = id2len[tid]
            if tid == space_id:
                pending = 1
                off += tl
                continue
            base_ids.append(tid)
            sbits.append(pending)
            pending = 0
            rel = []
            while ci < ncaps and caps[ci] < off + tl:
                rel.append(caps[ci] - off)
                ci += 1
            clabels.append(rel)
            off += tl
        out.append((base_ids, sbits, clabels))
    return out


def timeit(fn, *a, repeats=3):
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn(*a)
        best = min(best, time.perf_counter() - t0)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chars", type=int, default=10_000_000)
    ap.add_argument("--naive-chars", type=int, default=1_000_000)
    args = ap.parse_args()

    V = os.path.expanduser("~/.cache/nanochat-variants")
    enc = RustBPETokenizer.from_directory(os.path.join(V, "spaceless_repro", "tokenizer")).enc
    id2len = [0] * enc.n_vocab
    for tid in range(enc.n_vocab):
        try:
            id2len[tid] = len(enc.decode_single_token_bytes(tid))
        except Exception:
            id2len[tid] = 0
    space_id = enc.encode_single_token(_SPACE)

    corpus = list(iter_capped_batches("val", args.chars))
    docs = [d for batch in corpus for d in batch]
    total_chars = sum(len(d) for d in docs)
    total_bytes = sum(len(d.encode("utf-8")) for d in docs)

    # token count (for tok/sec), via plain encode
    n_tokens = sum(len(x) for x in enc.encode_ordinary_batch(docs, num_threads=8))

    def run_plain():
        enc.encode_ordinary_batch(docs, num_threads=8)

    def run_opt():
        opt_surface_encode(docs, enc, id2len, space_id)

    t_plain = timeit(run_plain)
    t_opt = timeit(run_opt)

    # naive on a smaller slice
    from tools.stack_triple import encode_triple, CFG_TRIPLE
    small = docs[:max(1, len(docs) * args.naive_chars // max(1, total_chars))]
    small_chars = sum(len(d) for d in small)
    t0 = time.perf_counter()
    for d in small:
        encode_triple(d, enc, CFG_TRIPLE["spaceless_re"])
    t_naive = time.perf_counter() - t0

    def report(name, secs, chars, toks):
        print(f"{name:<16}{secs*1000:>9.1f} ms{chars/secs/1e6:>12.2f} Mchar/s{toks/secs:>14,.0f} tok/s")

    print(f"\n=== Surface encode throughput ({total_chars:,} chars / {total_bytes:,} bytes / {n_tokens:,} tokens) ===\n")
    print(f"{'encoder':<16}{'time':>12}{'chars/sec':>16}{'tok/sec':>16}")
    print("-" * 60)
    report("plain (tiktoken)", t_plain, total_chars, n_tokens)
    report("surface-opt", t_opt, total_chars, n_tokens)
    report("surface-naive", t_naive, small_chars, n_tokens * small_chars / total_chars)
    print("-" * 60)
    plain_tps = n_tokens / t_plain
    opt_tps = n_tokens / t_opt
    naive_tps = (n_tokens * small_chars / total_chars) / t_naive
    print(f"drag (opt vs plain):   {t_opt/t_plain:.1f}x slower")
    print(f"drag (naive vs plain): {t_plain and (t_naive/small_chars)*(total_chars/t_plain):.0f}x slower (naive is the reference impl, NOT the hot path)")
    print(f"\ntraining requirement: {D24_TOKS_PER_SEC_TOTAL:,} tok/s total / {D24_TOKS_PER_SEC_TOTAL//D24_RANKS:,} per rank (prefetched behind the step)")
    req = D24_TOKS_PER_SEC_TOTAL / D24_RANKS
    print(f"surface-opt headroom vs per-rank requirement: {opt_tps/req:.1f}x  ({'OK — hides behind prefetch' if opt_tps > req*2 else 'TIGHT/BOTTLENECK — consider Rust'})")


if __name__ == "__main__":
    main()
