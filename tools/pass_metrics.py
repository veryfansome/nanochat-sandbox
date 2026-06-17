"""
Corpus-delta metrics for ANY tokenizer variant vs baseline, on a fixed
held-out corpus. Generic — `--variant` takes any tokenizer dir. (Originally
written for blocked_morphemes' blocking cascade, hence the convergence framing
below: re-run after each curation pass to track whether the cascade converges
and whether firing mass concentrates toward common tokens; the same four
metrics read meaningfully for force/seed/manual variants too.)

Reports, baseline vs variant, on a fixed held-out corpus (ClimbMix val):

  1. Compression       — total tokens + chars/token + % vs baseline. The
                         cost side of the trade.
  2. Dead-token ratio  — vocab tokens that never fire on the corpus. A
                         convergence signal: mid-cascade it RISES (clean
                         stems planted but not yet activated), and should
                         fall back at convergence if the hypothesis holds.
  3. Mangled count     — how many mangled-morpheme tokens remain in the
                         vocab (via tools.find_mangled_morphemes detection).
                         The direct convergence metric: baseline 483 → ... → 0.
  4. Rank-bucket hist  — firing share by token-id bucket + firing-weighted
                         mean id. Tests "is usage concentrating in the common
                         (low-id) tokens?" — lower weighted-mean id = yes.

One corpus encode pass per tokenizer yields all four (firing counts +
word-start counts come from the same per-chunk encoding).

Usage:
    uv run python -m tools.pass_metrics \\
        --variant ~/.cache/nanochat-variants/auto_tune/tokenizer
    # add --no-mangled to skip the (slower) mangled-token detection.
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

RANK_BUCKETS = [
    (0, 256), (256, 512), (512, 1024), (1024, 2048),
    (2048, 4096), (4096, 8192), (8192, 16384), (16384, 32768),
]


def encode_corpus(tok, corpus) -> tuple[Counter, Counter, int]:
    """One pass: per-chunk encode the cached corpus. Returns
    (firing_counts, word_start_counts, total_chars). Per-chunk encoding
    gives the same token total as doc-level encode_ordinary (BPE is
    per-chunk) while also exposing chunk-first tokens for word-start probs."""
    import regex

    pat = regex.compile(
        tok.enc._pat_str if hasattr(tok.enc, "_pat_str") else tok.enc.pat_str
    )
    fire: Counter = Counter()
    word_start: Counter = Counter()
    total_chars = 0
    for docs in corpus:
        chunks: list[str] = []
        for doc in docs:
            chunks.extend(m.group(0) for m in pat.finditer(doc))
            total_chars += len(doc)
        for ids in tok.enc.encode_ordinary_batch(chunks, num_threads=8):
            if not ids:
                continue
            word_start[ids[0]] += 1
            fire.update(ids)
    return fire, word_start, total_chars


def mergeable_vocab_size(tok) -> int:
    """Vocab size excluding special tokens (the dead-token denominator)."""
    return tok.get_vocab_size() - len(tok.get_special_tokens())


def fire_counts(tok, corpus) -> Counter:
    """Per-token firing counts over a corpus, via a single doc-level
    encode_ordinary pass (tiktoken's Rust regex; no Python chunking). Fast."""
    fire: Counter = Counter()
    for docs in corpus:
        for ids in tok.enc.encode_ordinary_batch(docs, num_threads=8):
            fire.update(ids)
    return fire


def compute_fast_metrics(tok, corpus) -> dict:
    """Fast corpus metrics: total_tokens (compression) + dead, from a single
    doc-level encode_ordinary pass. Skips the Python regex chunking + word-start
    tracking that dominate encode_corpus (~10x faster), at the cost of NOT
    computing mangled (which needs word-start probs). total_tokens and dead are
    identical to the full path (encode_ordinary is per-chunk internally). For
    cheap held-out compression/dead vetting at scale."""
    fire = fire_counts(tok, corpus)
    n_merge = mergeable_vocab_size(tok)
    return {
        "total_tokens": sum(fire.values()),
        "dead": sum(1 for tid in range(n_merge) if fire.get(tid, 0) == 0),
    }


def compute_metrics(tok, corpus, count_mangled: bool, bound_threshold: float):
    fire, word_start, total_chars = encode_corpus(tok, corpus)
    total_tokens = sum(fire.values())
    n_merge = mergeable_vocab_size(tok)
    dead = sum(1 for tid in range(n_merge) if fire.get(tid, 0) == 0)
    wmean_id = (
        sum(tid * n for tid, n in fire.items()) / total_tokens
        if total_tokens else 0.0
    )
    bucket_fire = []
    for lo, hi in RANK_BUCKETS:
        bucket_fire.append(sum(n for tid, n in fire.items() if lo <= tid < hi))

    mangled_count = None
    if count_mangled:
        import contextlib
        import io

        from tools.find_mangled_morphemes import (
            SUFFIXES, build_vocab_maps, find_mangled_tokens,
        )
        b2i, i2b = build_vocab_maps(tok)
        ws_prob = {
            tid: word_start.get(tid, 0) / fire[tid]
            for tid in fire if fire[tid]
        }
        suffixes_sorted = sorted(set(SUFFIXES), key=len, reverse=True)
        # find_mangled_tokens prints a tally to stderr; silence it here.
        with contextlib.redirect_stderr(io.StringIO()):
            mangled = find_mangled_tokens(
                tok, ws_prob, b2i, i2b,
                bound_threshold=bound_threshold,
                suffixes_sorted=suffixes_sorted,
                firings=fire, min_firings=1,
            )
        mangled_count = len(mangled)

    return {
        "total_chars": total_chars,
        "total_tokens": total_tokens,
        "chars_per_tok": total_chars / total_tokens if total_tokens else 0.0,
        "n_merge": n_merge,
        "dead": dead,
        "wmean_id": wmean_id,
        "bucket_fire": bucket_fire,
        "mangled_count": mangled_count,
    }


def whole_word_counts(tok, words):
    """(base, inflected) space-prefixed dictionary-word tokens in the vocab.
    Mirrors blocked_pairs_report.whole_word_count but reads token strings
    straight from the tokenizer (no dump file needed). Used by the held-out
    audit drivers as the coverage metric."""
    from tools.blocked_pairs_report import _is_word
    base = inflected = 0
    n = tok.get_vocab_size() - len(tok.get_special_tokens())
    for tid in range(n):
        try:
            s = tok.decode([tid])
        except Exception:
            continue
        if not s.startswith(" "):
            continue
        w = s[1:].lower()
        if w in words:
            base += 1
        elif _is_word(w, words):
            inflected += 1
    return base, inflected


def main(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--variant", type=Path, required=True,
                   help="Variant tokenizer directory to measure")
    p.add_argument("--baseline", type=Path,
                   default=Path("~/.cache/nanochat/tokenizer").expanduser(),
                   help="Baseline tokenizer directory for comparison")
    p.add_argument("--max-chars", type=int, default=20_000_000,
                   help="Corpus sample cap (default 20M)")
    p.add_argument("--bound-threshold", type=float, default=0.05,
                   help="Mangled-detection word-start threshold (default 0.05)")
    p.add_argument("--no-mangled", action="store_true",
                   help="Skip mangled-token detection (faster)")
    args = p.parse_args(argv)

    from nanochat.tokenizer import RustBPETokenizer
    from tools._corpus_iter import iter_capped_batches

    print(f"Loading corpus (ClimbMix val, {args.max_chars:,} char cap)...",
          file=sys.stderr)
    corpus = list(iter_capped_batches("val", args.max_chars))

    print("Measuring baseline...", file=sys.stderr)
    base_tok = RustBPETokenizer.from_directory(str(args.baseline))
    base = compute_metrics(base_tok, corpus, not args.no_mangled, args.bound_threshold)

    print("Measuring variant...", file=sys.stderr)
    var_tok = RustBPETokenizer.from_directory(str(args.variant))
    var = compute_metrics(var_tok, corpus, not args.no_mangled, args.bound_threshold)

    bt, vt = base["total_tokens"], var["total_tokens"]
    print(f"\n=== Tokenizer pass metrics ===")
    print(f"corpus:   ClimbMix val, {base['total_chars']:,} chars")
    print(f"baseline: {args.baseline}")
    print(f"variant:  {args.variant}\n")
    print(f"{'metric':<26}{'baseline':>16}{'variant':>16}{'delta':>18}")
    print(f"{'-'*76}")

    # 1. Compression
    dtok = vt - bt
    print(f"{'compression (tokens)':<26}{bt:>16,}{vt:>16,}"
          f"{dtok:>+11,} ({100*dtok/bt:+.3f}%)")
    print(f"{'chars/token':<26}{base['chars_per_tok']:>16.4f}"
          f"{var['chars_per_tok']:>16.4f}{var['chars_per_tok']-base['chars_per_tok']:>+18.4f}")

    # 2. Dead tokens
    bd_pct = 100 * base["dead"] / base["n_merge"]
    vd_pct = 100 * var["dead"] / var["n_merge"]
    print(f"{'dead tokens (count)':<26}{base['dead']:>16,}{var['dead']:>16,}"
          f"{var['dead']-base['dead']:>+18,}")
    print(f"{'dead tokens (ratio)':<26}{bd_pct:>15.2f}%{vd_pct:>15.2f}%"
          f"{vd_pct-bd_pct:>+17.2f}pp")

    # 3. Mangled
    if base["mangled_count"] is not None:
        bm, vm = base["mangled_count"], var["mangled_count"]
        print(f"{'mangled tokens remaining':<26}{bm:>16,}{vm:>16,}{vm-bm:>+18,}")

    # 4. Weighted-mean id
    print(f"{'firing-weighted mean id':<26}{base['wmean_id']:>16,.0f}"
          f"{var['wmean_id']:>16,.0f}"
          f"{var['wmean_id']-base['wmean_id']:>+12,.0f} "
          f"({100*(var['wmean_id']-base['wmean_id'])/base['wmean_id']:+.2f}%)")

    # Rank-bucket histogram (firing share %)
    print(f"\n=== firing share by rank bucket (%) ===")
    print(f"{'bucket':<16}{'baseline':>12}{'variant':>12}{'delta(pp)':>12}")
    for i, (lo, hi) in enumerate(RANK_BUCKETS):
        bs = 100 * base["bucket_fire"][i] / bt
        vs = 100 * var["bucket_fire"][i] / vt
        print(f"[{lo:>5},{hi:>5})   {bs:>11.2f}%{vs:>11.2f}%{vs-bs:>+11.2f}")


if __name__ == "__main__":
    main()
