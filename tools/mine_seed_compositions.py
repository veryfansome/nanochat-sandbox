"""
Mine compositions baseline BPE knows are common but couldn't fit in vocab.

Canonical mining tool for the seed_tokens tokenizer. Encodes a corpus
sample with the baseline tokenizer, counts adjacent-token-pair
occurrences **within each pre-tok chunk** (cross-chunk pairs are
excluded — tiktoken can't merge across pre-tok boundaries, that's the
gap `force_merges` addresses with a regex carve-out), filters to pairs
whose concatenation isn't already in baseline vocab, ranks by count.
Each surviving pair is directly a seed: the pair IS the route, since
`id_L` and `id_R` are already baseline-vocab tokens by construction.

BPE-token-level analog of `mine_forced_pairs.py`. force_merges mines
pre-tok-chunk-level bigrams (compositions BPE *structurally cannot see*
because pre-tok splits them); this miner targets within-chunk bigrams
(compositions BPE saw at training time but the merge didn't fit the
32K vocab budget).

## Why this works (and morpheme-mining didn't)

An earlier iteration (`mine_seed_morphemes.py` + `find_seed_routes.py`)
extracted substrings of pre-tok chunks by distinct-types productivity
and synthesized routes through baseline vocab. That approach was
net-negative on ClimbMix val at every threshold tested (see STATUS.md
§seed_tokens) because the mined morphemes (`cation`, `tation`,
`zation`, ...) had their bytes already covered by baseline at
finer-grained splits (`c`, `t`, `z` + `ation`); inserting the longer
seeds intercepted the shorter splits without enough downstream gain.

This miner targets a different gap: pairs that survived baseline
encoding as adjacent un-merged tokens. By construction:
  - `(L, R)` appears in baseline output → the seed fires at exactly
    those adjacencies.
  - `L + R` is not in baseline vocab → there's room for the merge.
  - Each firing replaces 2 tokens with 1 → strictly positive at the
    firing site.

The K-sweep ablation (STATUS.md table) identified K=1750 as the
plateau-end of per-seed marginal value — beyond K=1750 the marginal
value declines and the K=2000→2250 bucket goes actively
anti-productive.

## Output

`SEED_ROUTES = [(L_bytes, R_bytes, S_bytes), ...]` directly compatible
with `wrappers/tok_train_seed_tokens.py`.

Usage (regenerates the canonical):
    uv run python -m tools.mine_seed_compositions \\
        --baseline ~/.cache/nanochat/tokenizer \\
        --max-chars 20_000_000 --top-k 1750 --min-count 2 \\
        --out-py rustbpe_variants/seed_tokens/mined/compositions_climbmix_val_k1750.py
"""

import argparse
import sys
import time
from collections import Counter
from pathlib import Path


def encode_and_count_adjacent_pairs(
    tok, max_chars: int,
) -> tuple[Counter, int, int]:
    """Pre-tokenize the corpus, encode each chunk, count adjacent
    (id_L, id_R) pairs **within chunks only**. Pairs across pre-tok-chunk
    boundaries are excluded because tiktoken's encoder can't merge across
    them (same structural constraint that drove force_merges's regex
    carve-out approach). Counting cross-chunk pairs would surface
    `(',', ' ')`, `(' ', '1')`, etc. — high-frequency but un-mergeable.

    Returns (pair_counts, total_pair_instances, total_chars).
    """
    import regex

    from tools._corpus_iter import iter_capped_batches

    pat = regex.compile(tok.enc._pat_str if hasattr(tok.enc, "_pat_str")
                        else tok.enc.pat_str)

    pair_counts: Counter = Counter()
    total_pairs = 0
    total_chars = 0
    n_docs = 0
    t0 = time.time()
    for docs in iter_capped_batches("val", max_chars):
        all_chunks: list[str] = []
        for doc in docs:
            all_chunks.extend(m.group(0) for m in pat.finditer(doc))
        # encode_ordinary_batch parallelizes chunk-level encoding across
        # threads — one big call per batch keeps that parallelism intact.
        chunk_ids_list = tok.enc.encode_ordinary_batch(all_chunks, num_threads=8)
        for chunk_ids in chunk_ids_list:
            for i in range(len(chunk_ids) - 1):
                pair_counts[(chunk_ids[i], chunk_ids[i + 1])] += 1
            if len(chunk_ids) > 1:
                total_pairs += len(chunk_ids) - 1
        total_chars += sum(len(d) for d in docs)
        n_docs += len(docs)
        if n_docs % 5000 == 0:
            elapsed = time.time() - t0
            print(
                f"  {n_docs:,} docs / {total_chars:,} chars / "
                f"{len(pair_counts):,} unique within-chunk pairs ({elapsed:.1f}s)",
                file=sys.stderr,
            )
    return pair_counts, total_pairs, total_chars


def build_vocab_maps(tok) -> tuple[dict[bytes, int], dict[int, bytes]]:
    """Recover bytes↔id maps for the baseline vocab (specials excluded)."""
    enc = tok.enc
    special_set = set(tok.get_special_tokens())
    b2i: dict[bytes, int] = {}
    i2b: dict[int, bytes] = {}
    for tid in range(tok.get_vocab_size()):
        try:
            b = enc.decode_single_token_bytes(tid)
        except Exception:
            continue
        try:
            if b.decode("utf-8") in special_set:
                continue
        except UnicodeDecodeError:
            pass
        b2i[b] = tid
        i2b[tid] = b
    return b2i, i2b


def emit_routes_py(
    routes: list[dict], file=sys.stdout, baseline: Path = None,
    mining_args: dict = None,
) -> None:
    """Emit SEED_ROUTES list compatible with the seed_tokens wrapper."""
    print("# Auto-generated by tools/mine_seed_compositions.py", file=file)
    print(
        "# Source: adjacent-token-pair bigrams in baseline-tokenizer output",
        file=file,
    )
    if baseline:
        print(f"# Baseline tokenizer: {baseline}", file=file)
    if mining_args:
        print(f"# Mining args: {mining_args}", file=file)
    print(
        "# Each entry: (L_bytes, R_bytes, S_bytes). Both operands are "
        "baseline-vocab tokens",
        file=file,
    )
    print(
        "# by construction; concat S is NOT in baseline. Wrapper inserts "
        "S at rank max(id_L, id_R)+1.",
        file=file,
    )
    print("SEED_ROUTES = [", file=file)
    for r in routes:
        L, R, S = r["L"], r["R"], r["S"]
        comment = (
            f"  # count={r['count']:,}, max_id={r['max_id']}, "
            f"L_id={r['L_id']}, R_id={r['R_id']}"
        )
        print(f"    ({L!r}, {R!r}, {S!r}),{comment}", file=file)
    print("]", file=file)


def main(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--baseline", type=Path,
        default=Path("~/.cache/nanochat/tokenizer").expanduser(),
        help="Baseline tokenizer directory",
    )
    p.add_argument(
        "--max-chars", type=int, default=20_000_000,
        help="Cap corpus sample size in characters (default: 20M, matches "
        "eval_tokenizer Tier-2)",
    )
    p.add_argument(
        "--min-count", type=int, default=2,
        help="Drop pairs below this count (default: 2 — non-binding at the "
        "K values in the canonical sweep; raise for tighter filtering)",
    )
    p.add_argument(
        "--top-k", type=int, default=1750,
        help="Emit top-K pairs after filtering (default: 1750 — current "
        "canonical from the K-sweep plateau-end; see STATUS.md §seed_tokens)",
    )
    p.add_argument(
        "--out-py", type=Path, default=None,
        help="Write SEED_ROUTES .py to this path (else print to stdout)",
    )
    args = p.parse_args(argv)

    from nanochat.tokenizer import RustBPETokenizer

    print(f"Loading baseline tokenizer from {args.baseline}...", file=sys.stderr)
    tok = RustBPETokenizer.from_directory(str(args.baseline))
    b2i, i2b = build_vocab_maps(tok)
    print(f"  {len(b2i):,} baseline mergeable tokens.", file=sys.stderr)

    print(f"Encoding ClimbMix val ({args.max_chars:,} char cap)...", file=sys.stderr)
    pair_counts, total_pairs, total_chars = encode_and_count_adjacent_pairs(
        tok, args.max_chars,
    )
    print(
        f"  {total_pairs:,} pair instances, "
        f"{len(pair_counts):,} unique pairs over {total_chars:,} chars.",
        file=sys.stderr,
    )

    # Collect every pair that passes filters; defer the top-K cut until
    # after a deterministic selection sort. The earlier formulation broke
    # out as soon as `len(candidates) >= top_k`, which sliced through
    # whatever same-count bucket the cutoff landed in — `Counter.most_common`
    # ties are first-seen order, so re-mining could pick a different
    # subset within that bucket and the canonical artifact would drift.
    all_passing: list[dict] = []
    n_in_vocab = 0
    for (id_L, id_R), count in pair_counts.most_common():
        if count < args.min_count:
            break  # sorted descending; everything after also < min_count
        L = i2b.get(id_L)
        R = i2b.get(id_R)
        if L is None or R is None:
            continue  # special-token or vocab-gap; skip defensively
        S = L + R
        if S in b2i:
            # Concat already in baseline vocab — the (L, R) adjacency in
            # output means some lower-rank merge won the encode-time race,
            # not that the merge rule is missing. Skip; we can't add an
            # already-existing token.
            n_in_vocab += 1
            continue
        all_passing.append({
            "L": L, "R": R, "S": S, "count": count,
            "L_id": id_L, "R_id": id_R,
            "max_id": max(id_L, id_R),
        })

    # Selection criterion: pick top-K by utility, with explicit tiebreaks.
    # Primary: count desc (more firings = more compression benefit).
    # Secondary: max_id asc (lower-rank operands → seed fires earlier in
    # the BPE sequence → more downstream influence on subsequent merges).
    # Final: S asc — pure determinism, no semantic claim. This makes the
    # SET of top-K reproducible, not just its emission order.
    all_passing.sort(key=lambda r: (-r["count"], r["max_id"], r["S"]))
    candidates = all_passing[: args.top_k]

    print(
        f"\nMined candidates: {len(candidates)} kept "
        f"(top-K {args.top_k} out of {len(all_passing):,} pairs passing filters).",
        file=sys.stderr,
    )
    print(
        f"  filtered out: {n_in_vocab} already in baseline vocab",
        file=sys.stderr,
    )

    # Emission order: max_id asc (= natural-firing order in BPE merge
    # sequence; the wrapper's stable sort turns file order into final
    # rank). Inner tiebreak: count desc (higher-firing seeds get earlier
    # rank within insertion-point ties → more vocab-slot amortization).
    # Final tiebreak: S asc, matching the selection sort for consistency.
    candidates.sort(key=lambda r: (r["max_id"], -r["count"], r["S"]))

    mining_args = {
        "baseline": str(args.baseline),
        "max_chars": args.max_chars,
        "min_count": args.min_count,
        "top_k": args.top_k,
    }

    if args.out_py:
        args.out_py.parent.mkdir(parents=True, exist_ok=True)
        with args.out_py.open("w") as f:
            emit_routes_py(
                candidates, file=f, baseline=args.baseline,
                mining_args=mining_args,
            )
        print(f"Wrote {args.out_py}", file=sys.stderr)
    else:
        emit_routes_py(candidates, mining_args=mining_args)


if __name__ == "__main__":
    main()
