"""
Substring-productivity miner (DEPRECATED for seed selection).

**Status**: This was the first mining approach for seed_tokens. Tested
and net-negative on ClimbMix val at every threshold (see STATUS.md
§seed_tokens). The candidates compete with baseline's already-efficient
short morphemes rather than complementing them. The canonical mining
tool is now `tools/mine_seed_compositions.py` (within-chunk adjacent-pair
mining from baseline output).

The position-breakdown stats this miner produces (start/end/interior/full
distinct-types counts) are still valuable as a stand-alone morphological
diagnostic, so the tool is preserved. Don't use its output as direct
input to the seed_tokens wrapper without re-validating against the
empirical evidence in STATUS.md.

---

Mines morpheme-like substrings from the pretraining corpus. For each
baseline-pre-tokenized chunk, walks all substrings of length
[min_len, max_len] and tracks per-substring:

  - total          : occurrence count (chunk-frequency-weighted)
  - types          : distinct chunk types containing this substring
                     (productivity — the morpheme-likeness signal)
  - types_start    : distinct chunks where the substring is a prefix
  - types_end      : distinct chunks where the substring is a suffix
  - types_interior : distinct chunks where the substring is strictly interior
  - types_full     : distinct chunks where the substring IS the chunk
                     (standalone word)

Why productivity, not frequency: a high-frequency substring like ` the`
appears in only a handful of chunk types (its standalone forms — ` the`,
` The`, `.The`, etc.), so it's a common word, not a morpheme. A real
morpheme like `tion` appears in thousands of distinct word types.
`types` captures this directly; `total` is kept for tiebreaks and
diagnostic display only.

The position breakdown replaces the prior hand-curated three-bucket
categorization (versatile_morphemes / prefixes / inner_morphemes) — a
candidate's productivity distribution across chunk positions tells you
how it's used. `tools/find_seed_routes.py` doesn't currently consume
position info (route selection is purely id-based), but the breakdown
is preserved in the output for downstream filtering experiments.

Defaults are tuned for the climbmix canonical mining run:
  --row-groups 20  --min-len 4 --max-len 10
  --min-types 100  --top-k 1500

At min_len <= 3, results are dominated by short bigrams/trigrams that
BPE already mints naturally (the route-finder drops most of them as
already-in-vocab). min_len=4 focuses on morpheme-range candidates.

Output: a Python module with
  CANDIDATES = [(substring_str, stats_dict), ...]
ranked by productivity, mirroring
`rustbpe_variants/force_merges/mined/forced_pairs_climbmix_20rg_twopass.py`.

Usage:
    uv run python -m tools.mine_seed_morphemes \\
        --out-py rustbpe_variants/seed_tokens/mined/morphemes_climbmix_20rg.py

No GPU. ~1 min for 20 row groups on a laptop.
"""

import argparse
import sys
import time
from collections import Counter
from pathlib import Path

import regex

# Baseline pre-tokenization regex — lifted verbatim from
# nanochat/tokenizer.py:30 (pristine, no carve-outs). Kept inline so the
# miner doesn't depend on an installed nanochat.
BASELINE_SPLIT_PATTERN = (
    r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,2}"""
    r"""| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
)


# ---------------------------------------------------------------------------
# Corpus

def iter_corpus(row_groups: int, max_chars: int):
    """Yield documents from the local nanochat parquet shards.

    parquets_iter_batched yields one batch per parquet row group, not per
    shard. ~3M chars per row group; 20 row groups ≈ 60M chars (matches the
    force_merges canonical curation sample).
    """
    from nanochat.dataset import parquets_iter_batched

    total_chars = 0
    docs_yielded = 0
    row_groups_seen = 0
    for batch in parquets_iter_batched(split="train"):
        if row_groups > 0 and row_groups_seen >= row_groups:
            break
        for doc in batch:
            yield doc
            docs_yielded += 1
            total_chars += len(doc)
            if max_chars and total_chars >= max_chars:
                print(
                    f"  reached --max-chars cap ({total_chars:,} chars, "
                    f"{docs_yielded:,} docs)",
                    file=sys.stderr,
                )
                return
        row_groups_seen += 1
    print(
        f"  scanned {docs_yielded:,} docs / {total_chars:,} chars "
        f"across {row_groups_seen} row group(s)",
        file=sys.stderr,
    )


def count_chunks(docs, split_pattern: str) -> Counter:
    """First pass: count chunk types across the corpus."""
    rgx = regex.compile(split_pattern)
    counts: Counter = Counter()
    n_docs = 0
    t0 = time.time()
    for doc in docs:
        for m in rgx.finditer(doc):
            counts[m.group(0)] += 1
        n_docs += 1
        if n_docs % 5000 == 0:
            elapsed = time.time() - t0
            print(
                f"  {n_docs:,} docs / {len(counts):,} unique chunks "
                f"({elapsed:.1f}s)",
                file=sys.stderr,
            )
    return counts


# ---------------------------------------------------------------------------
# Substring extraction

def extract_substring_stats(
    chunk_counts: Counter, min_len: int, max_len: int,
    min_chunk_freq: int,
) -> dict[str, dict]:
    """Walk unique chunks, extract substrings of length [min_len, max_len],
    aggregate per-substring stats. Returns dict[substring -> stats_dict].

    `min_chunk_freq` drops noise chunks (typos, hapax) — orthogonal to the
    per-substring productivity floor, which is applied later in
    filter_and_rank.

    Per-position counts (start/end/full) are per-chunk-occurrence by
    construction — each can fire at most once per chunk per slen. Only
    `interior` can repeat within a single chunk (e.g. "ti" appears
    interior twice in "tititi"), so it's the only position that needs a
    set-based dedup.
    """
    total: Counter = Counter()
    types_count: Counter = Counter()
    types_start: Counter = Counter()
    types_end: Counter = Counter()
    types_interior: Counter = Counter()
    types_full: Counter = Counter()

    n_chunks = 0
    t0 = time.time()
    for chunk, chunk_freq in chunk_counts.items():
        if chunk_freq < min_chunk_freq:
            continue

        L = len(chunk)
        seen_in_this_chunk: set[str] = set()
        seen_interior: set[str] = set()

        for slen in range(min_len, min(max_len, L) + 1):
            # Each (chunk, slen) yields L-slen+1 substrings; at most one
            # at i=0 (start), one at i=L-slen (end), and (slen==L) means
            # it IS the chunk (full).
            for i in range(L - slen + 1):
                s = chunk[i:i + slen]
                total[s] += chunk_freq
                if slen == L:
                    types_full[s] += 1
                elif i == 0:
                    types_start[s] += 1
                elif i + slen == L:
                    types_end[s] += 1
                else:
                    if s not in seen_interior:
                        types_interior[s] += 1
                        seen_interior.add(s)
                seen_in_this_chunk.add(s)

        for s in seen_in_this_chunk:
            types_count[s] += 1

        n_chunks += 1
        if n_chunks % 50000 == 0:
            elapsed = time.time() - t0
            print(
                f"  {n_chunks:,} unique chunks processed "
                f"({len(types_count):,} unique substrings, {elapsed:.1f}s)",
                file=sys.stderr,
            )

    return {
        s: {
            "total": total[s],
            "types": types_count[s],
            "types_start": types_start[s],
            "types_end": types_end[s],
            "types_interior": types_interior[s],
            "types_full": types_full[s],
        }
        for s in types_count
    }


# ---------------------------------------------------------------------------
# Ranking + filtering

def filter_and_rank(
    stats: dict[str, dict], min_types: int, top_k: int,
) -> list[tuple[str, dict]]:
    """Filter by min-types, rank by productivity (types count, freq tiebreak).

    Drops single-byte substrings (already in the byte-level vocab).
    """
    filtered = [
        (s, st) for s, st in stats.items()
        if len(s) >= 2 and st["types"] >= min_types
    ]
    # Rank by productivity primarily, then by frequency.
    filtered.sort(key=lambda x: (-x[1]["types"], -x[1]["total"]))
    return filtered[:top_k]


# ---------------------------------------------------------------------------
# Output

def emit_text(ranked: list[tuple[str, dict]], file=sys.stdout) -> None:
    """Human-readable ranked dump."""
    print(
        f"# rank  types     total       start  end   inter  full   substring",
        file=file,
    )
    print(f"# {'-' * 75}", file=file)
    for i, (s, st) in enumerate(ranked, 1):
        print(
            f"  {i:>4}  {st['types']:>6,}  {st['total']:>10,}  "
            f"{st['types_start']:>5,}  {st['types_end']:>5,}  "
            f"{st['types_interior']:>5,}  {st['types_full']:>5,}  {s!r}",
            file=file,
        )


def emit_candidates_py(
    ranked: list[tuple[str, dict]], file=sys.stdout,
    mining_args: dict = None,
) -> None:
    """Emit a Python module with CANDIDATES = [(substring, stats), ...].

    Format mirrors rustbpe_variants/force_merges/mined/forced_pairs_*.py:
    a flat literal list, not a runtime loader. Re-generate by re-running
    this script.
    """
    print("# Auto-generated by tools/mine_seed_morphemes.py", file=file)
    if mining_args:
        print(f"# Mining args: {mining_args}", file=file)
    print(
        "# Each entry: (substring, stats_dict). Stats keys: 'total' (corpus",
        file=file,
    )
    print(
        "# occurrences), 'types' (distinct chunk types containing it),",
        file=file,
    )
    print(
        "# 'types_start' / 'types_end' / 'types_interior' / 'types_full'",
        file=file,
    )
    print(
        "# (chunk-position breakdown by distinct types).",
        file=file,
    )
    print("CANDIDATES = [", file=file)
    for s, st in ranked:
        # Order keys for stable output.
        stats_repr = (
            "{"
            f"'total': {st['total']}, 'types': {st['types']}, "
            f"'types_start': {st['types_start']}, "
            f"'types_end': {st['types_end']}, "
            f"'types_interior': {st['types_interior']}, "
            f"'types_full': {st['types_full']}"
            "}"
        )
        print(f"    ({s!r}, {stats_repr}),", file=file)
    print("]", file=file)


# ---------------------------------------------------------------------------
# CLI

def main(argv=None):
    p = argparse.ArgumentParser(
        description="Mine morpheme-like substrings for seed_tokens.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--row-groups", type=int, default=20,
        help="Number of parquet row groups to scan (default: 20; ~3M chars "
        "each; matches force_merges canonical curation). 0 = all.",
    )
    p.add_argument(
        "--max-chars", type=int, default=0,
        help="Cap total characters scanned (0 = no cap)",
    )
    p.add_argument(
        "--min-len", type=int, default=4,
        help="Min substring length (default: 4; len<=3 yields mostly "
        "already-in-vocab bigrams/trigrams the route-finder drops)",
    )
    p.add_argument(
        "--max-len", type=int, default=10,
        help="Max substring length (default: 10)",
    )
    p.add_argument(
        "--min-types", type=int, default=100,
        help="Min distinct chunk-types per substring (productivity floor; "
        "default: 100)",
    )
    p.add_argument(
        "--min-chunk-freq", type=int, default=2,
        help="Skip chunks below this corpus-occurrence threshold (drops "
        "hapax / typo noise; default: 2)",
    )
    p.add_argument(
        "--top-k", type=int, default=1500,
        help="Emit top-K candidates after ranking (default: 1500; route-"
        "finder drops most as already-in-vocab)",
    )
    p.add_argument(
        "--out-py", type=Path, default=None,
        help="Write a CANDIDATES list .py file to this path",
    )
    args = p.parse_args(argv)

    print(f"Pass 1: scanning {args.row_groups} row group(s) for chunks...",
          file=sys.stderr)
    chunk_counts = count_chunks(
        iter_corpus(args.row_groups, args.max_chars),
        BASELINE_SPLIT_PATTERN,
    )
    print(
        f"  {sum(chunk_counts.values()):,} total chunks, "
        f"{len(chunk_counts):,} unique.",
        file=sys.stderr,
    )

    print(
        f"Pass 2: extracting substrings (len {args.min_len}..{args.max_len}) "
        f"from {len(chunk_counts):,} unique chunks...",
        file=sys.stderr,
    )
    stats = extract_substring_stats(
        chunk_counts, args.min_len, args.max_len, args.min_chunk_freq,
    )
    print(
        f"  {len(stats):,} unique substrings stat'd.",
        file=sys.stderr,
    )

    ranked = filter_and_rank(stats, args.min_types, args.top_k)
    print(
        f"After min-types={args.min_types} + top-K={args.top_k}: "
        f"{len(ranked)} candidates.",
        file=sys.stderr,
    )

    mining_args_dict = {
        "row_groups": args.row_groups,
        "min_len": args.min_len,
        "max_len": args.max_len,
        "min_types": args.min_types,
        "min_chunk_freq": args.min_chunk_freq,
        "top_k": args.top_k,
    }

    if args.out_py:
        args.out_py.parent.mkdir(parents=True, exist_ok=True)
        with args.out_py.open("w") as f:
            emit_candidates_py(ranked, file=f, mining_args=mining_args_dict)
        print(f"Wrote {args.out_py}", file=sys.stderr)
    else:
        emit_text(ranked)


if __name__ == "__main__":
    main()
