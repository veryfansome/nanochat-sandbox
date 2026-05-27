"""
Route-finder for substring-productivity candidates (DEPRECATED for the
canonical pipeline; substring-blocker filter still useful standalone).

**Status**: this tool was built to convert `mine_seed_morphemes.py`
output into seed routes. The morpheme-mining approach turned out to be
net-negative on ClimbMix val (see STATUS.md §seed_tokens) and the
canonical mining is now `tools/mine_seed_compositions.py`, which emits
routes directly (pair IS the route) and bypasses this tool entirely.

What's still useful here: the `--max-blocker-firings N` substring filter
(drops candidates whose bytes contain a high-firing baseline token as
substring). Generally applicable to any candidate list as a guardrail
against seeds that block existing baseline morphemes.

---

Loads:
  - A mined-candidates list (`tools.mine_seed_morphemes` output).
  - The baseline tokenizer's mergeable_ranks (bytes → id) from
    ~/.cache/nanochat/tokenizer/tokenizer.pkl by default.

For each candidate seed S (bytes):
  1. Skip if S is already in baseline vocab (forms naturally, no seeding).
  2. Enumerate splits S = L + R for each split point in 1..len(S)-1 where
     both L and R are in baseline vocab.
  3. Drop the candidate if no valid splits exist — un-routable through
     natural sub-tokens. The prior YAML-driven implementation would have
     brute-force constructed these via byte-level merge chains paying
     `len(S) - 1` slots each; we drop them instead.
  4. Pick a route by `--policy`. Default `min-max-id` — smallest
     max(id_L, id_R) (= earliest natural-firing rank, where the seed
     gets inserted in the final mergeable_ranks). Tiebreak: smallest
     min(id_L, id_R) (= most-productive sub-token in the other slot;
     better amortizes the routing-token's vocab footprint).

Output: a Python module with `SEED_ROUTES = [(L_bytes, R_bytes, S_bytes),
...]` (3-tuples; per-route stats annotated in a trailing comment for
inspection). The wrapper interleaves each seed into mergeable_ranks at
rank `max(id_L, id_R) + 1` — its mid-training-style natural-firing
rank. See `wrappers/tok_train_seed_tokens.py` for why mid-rank
insertion (not append-at-end) is required for the seed merge to fire
against competing natural merges.

Notes:
  - Routes are pinned against a specific baseline tokenizer. If the
    baseline retrains (vocab/corpus change), some operands may not exist
    at apply time — the wrapper skips and counts those routes.
  - The candidate's chunk-position breakdown (from mining) is not
    currently used by the route-finder. It's preserved in the candidates
    file for downstream filtering / ranking experiments.

Usage:
    uv run python -m tools.find_seed_routes \\
        --candidates rustbpe_variants/seed_tokens/mined/morphemes_climbmix_20rg.py \\
        --out-py rustbpe_variants/seed_tokens/mined/routes_climbmix_20rg.py
"""

import argparse
import base64
import hashlib
import importlib.util
import json
import sys
from pathlib import Path


def _tokenizer_fingerprint(tokenizer_dir: Path) -> str:
    """Content-hash of tokenizer.pkl. Detects in-place retraining of the
    baseline at the same directory path (the default `~/.cache/nanochat/
    tokenizer` makes this likely) — path-string equality alone would
    silently reuse stale firing counts after a rebuild.
    """
    return hashlib.sha1(
        (tokenizer_dir / "tokenizer.pkl").read_bytes()
    ).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Baseline vocab loading

def load_baseline_vocab(tokenizer_dir: Path) -> dict[bytes, int]:
    """Recover the mergeable_ranks dict (bytes → id) from a saved tokenizer.

    Walks all token IDs in the tiktoken Encoding, skipping special tokens
    (their bytes are like `<|bos|>` and aren't useful as merge operands).
    """
    # Local import — keeps the route-finder importable in tests without
    # nanochat on the path.
    from nanochat.tokenizer import RustBPETokenizer

    tok = RustBPETokenizer.from_directory(str(tokenizer_dir))
    enc = tok.enc
    special_set = set(tok.get_special_tokens())

    ranks: dict[bytes, int] = {}
    for tid in range(tok.get_vocab_size()):
        try:
            b = enc.decode_single_token_bytes(tid)
        except Exception:
            continue  # vocab gap (e.g. tiktoken reserved id)
        # Skip special tokens — their textual form (`<|bos|>` etc.) is in
        # special_set; raw bytes here would be that string encoded.
        try:
            if b.decode("utf-8") in special_set:
                continue
        except UnicodeDecodeError:
            pass  # not a special token, fall through
        ranks[b] = tid
    return ranks


# ---------------------------------------------------------------------------
# Route enumeration + selection

def find_routes(
    seed_bytes: bytes, vocab: dict[bytes, int],
) -> list[tuple[int, int, bytes, bytes]]:
    """Return list of (max_id, min_id, L_bytes, R_bytes) for every valid split.

    A split is valid iff both L and R are in vocab. Single-byte splits
    (L or R is one byte) are always valid because bytes 0..255 are always
    in the BPE base vocab.
    """
    out = []
    for i in range(1, len(seed_bytes)):
        L = seed_bytes[:i]
        R = seed_bytes[i:]
        if L in vocab and R in vocab:
            id_L = vocab[L]
            id_R = vocab[R]
            out.append((max(id_L, id_R), min(id_L, id_R), L, R))
    return out


def select_route(
    routes: list[tuple[int, int, bytes, bytes]], policy: str,
) -> tuple[int, int, bytes, bytes]:
    """Pick one route by policy."""
    if not routes:
        raise ValueError("no routes to select from")
    if policy == "min-max-id":
        # Primary: lowest max(id_L, id_R) — seed merge fires earliest in
        # BPE sequence (both sub-tokens exist by step max_id+1).
        # Tiebreak: lowest min(id_L, id_R) — the other sub-token is the
        # most central available; pair amortizes the routing-token slot.
        return min(routes, key=lambda r: (r[0], r[1]))
    raise ValueError(f"unknown policy: {policy}")


# ---------------------------------------------------------------------------
# Substring-blocker filter
#
# Empirical finding from tools/diag_seed_usage.py: inserting a seed S whose
# bytes contain a high-utility baseline token T as a substring frequently
# blocks T's natural firing at encode time. T's composition path passes
# through a state where the seed becomes adjacent, and the seed's lower
# rank wins greedy. Net token count goes up.
#
# Filter: for each candidate S, walk all proper substrings of length ≥ 2;
# if any substring is a baseline-vocab token T with `baseline_firings[T]`
# above a threshold, drop the candidate. The intuition: don't seed
# compositions whose decomposition steps over an already-productive
# baseline morpheme.

def build_baseline_firings_cache(
    baseline_tokenizer_dir: Path, max_chars: int,
) -> dict[bytes, int]:
    """Encode ClimbMix val with baseline, return {token_bytes: firing_count}.

    Uses `iter_capped_batches` so --max-chars is accurate to within one
    doc (the naive per-batch cap would round to the next row-group
    boundary; threshold semantics depend on absolute counts, so an
    inaccurate cap would silently shift filter outcomes).

    Caching is the caller's responsibility (see `load_or_build_firings`).
    """
    from collections import Counter
    from nanochat.tokenizer import RustBPETokenizer

    from tools._corpus_iter import iter_capped_batches

    tok = RustBPETokenizer.from_directory(str(baseline_tokenizer_dir))
    print(f"  encoding ClimbMix val (cap {max_chars:,} chars)...", file=sys.stderr)
    counts: Counter = Counter()
    for docs in iter_capped_batches("val", max_chars):
        rows = tok.encode(docs)
        for row in rows:
            counts.update(row)
    # Re-key by bytes.
    enc = tok.enc
    bytes_firings: dict[bytes, int] = {}
    for tid, c in counts.items():
        try:
            b = enc.decode_single_token_bytes(tid)
            bytes_firings[b] = c
        except Exception:
            continue
    return bytes_firings


def load_or_build_firings(
    cache_path: Path | None, baseline_tokenizer_dir: Path, max_chars: int,
) -> dict[bytes, int]:
    """Load baseline firing counts from JSON cache, or build + save.

    Validates `baseline_tokenizer` (path), `baseline_fingerprint` (sha1 of
    tokenizer.pkl), and `max_chars` against the current request — absolute
    firing counts depend on all three. Path equality alone would miss
    in-place retraining at the same directory.
    """
    fingerprint = _tokenizer_fingerprint(baseline_tokenizer_dir)
    if cache_path and cache_path.exists():
        with cache_path.open() as f:
            raw = json.load(f)
        cached_tok = raw.get("baseline_tokenizer")
        cached_fp = raw.get("baseline_fingerprint")
        cached_chars = raw.get("max_chars")
        if (cached_tok == str(baseline_tokenizer_dir)
                and cached_fp == fingerprint
                and cached_chars == max_chars):
            print(f"  loading firings cache from {cache_path}", file=sys.stderr)
            return {base64.b64decode(k): v for k, v in raw["firings"].items()}
        # Surface which field(s) drifted to make the rebuild reason obvious.
        diffs: list[str] = []
        if cached_tok != str(baseline_tokenizer_dir):
            diffs.append(f"tokenizer path: {cached_tok!r} → "
                         f"{str(baseline_tokenizer_dir)!r}")
        if cached_fp != fingerprint:
            diffs.append(f"tokenizer content fingerprint: "
                         f"{cached_fp!r} → {fingerprint!r}")
        if cached_chars != max_chars:
            diffs.append(f"max_chars: {cached_chars} → {max_chars}")
        print(f"  firings cache invalidated ({'; '.join(diffs)}); rebuilding",
              file=sys.stderr)
    else:
        print("  no firings cache; encoding baseline...", file=sys.stderr)
    firings = build_baseline_firings_cache(baseline_tokenizer_dir, max_chars)
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with cache_path.open("w") as f:
            json.dump(
                {
                    "baseline_tokenizer": str(baseline_tokenizer_dir),
                    "baseline_fingerprint": fingerprint,
                    "max_chars": max_chars,
                    "firings": {
                        base64.b64encode(k).decode(): v
                        for k, v in firings.items()
                    },
                },
                f,
            )
        print(f"  saved firings cache to {cache_path}", file=sys.stderr)
    return firings


def is_blocked_by_substring(
    seed_bytes: bytes, baseline_vocab: dict[bytes, int],
    firings: dict[bytes, int], threshold: int,
) -> bytes | None:
    """Return the first blocking substring found, or None if clean.

    A "blocker" is a proper substring T of `seed_bytes` (length ≥ 2) that's
    in baseline vocab with `firings[T] > threshold`. Single-byte substrings
    aren't morphemes so they're skipped — the concern is bridging stem-
    final bytes to actual baseline morphemes.
    """
    L = len(seed_bytes)
    for i in range(L):
        for j in range(i + 2, L + 1):
            if i == 0 and j == L:
                continue  # S itself, not a proper substring
            sub = seed_bytes[i:j]
            if sub in baseline_vocab and firings.get(sub, 0) > threshold:
                return sub
    return None


# ---------------------------------------------------------------------------
# Candidates loader

def load_candidates(path: Path) -> list[tuple[str, dict]]:
    """Load CANDIDATES from a mined-output Python module."""
    spec = importlib.util.spec_from_file_location("_candidates_mod", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    cands = getattr(mod, "CANDIDATES", None)
    if cands is None:
        raise ValueError(f"{path} does not define CANDIDATES")
    return list(cands)


# ---------------------------------------------------------------------------
# Output

def emit_routes_py(
    routes: list[dict], file=sys.stdout, mining_path: Path = None,
    baseline_tokenizer: Path = None, policy: str = "min-max-id",
) -> None:
    """Emit a Python module with SEED_ROUTES = [(L_bytes, R_bytes, S_bytes), ...].

    Per-route stats (productivity, operand ids, etc.) annotate each entry
    as a trailing comment for human inspection; the wrapper consumes only
    the 3-tuple.
    """
    print("# Auto-generated by tools/find_seed_routes.py", file=file)
    if mining_path:
        print(f"# Candidates from: {mining_path}", file=file)
    if baseline_tokenizer:
        print(f"# Baseline tokenizer: {baseline_tokenizer}", file=file)
    print(f"# Route-selection policy: {policy}", file=file)
    print(
        "# Each entry: (L_bytes, R_bytes, S_bytes). The wrapper inserts "
        "S into mergeable_ranks",
        file=file,
    )
    print(
        "# at rank max(id_L, id_R)+1 (= natural-firing rank); tiktoken "
        "applies the merge",
        file=file,
    )
    print(
        "# at encode time via standard rank-based greedy matching.",
        file=file,
    )
    print("SEED_ROUTES = [", file=file)
    for r in routes:
        L, R, S = r["L"], r["R"], r["S"]
        comment = (
            f"  # types={r['types']}, max_id={r['max_id']}, "
            f"L_id={r['L_id']}, R_id={r['R_id']}"
        )
        print(f"    ({L!r}, {R!r}, {S!r}),{comment}", file=file)
    print("]", file=file)


def emit_text(routes: list[dict], dropped: list[dict], file=sys.stdout) -> None:
    """Human-readable summary."""
    print(
        f"# {'rank':>4}  {'types':>6}  {'maxid':>6}  "
        f"{'L_id':>5}  {'R_id':>5}  {'L':>14}  {'R':>14}  {'S':>16}",
        file=file,
    )
    print(f"# {'-' * 95}", file=file)
    for i, r in enumerate(routes, 1):
        print(
            f"  {i:>4}  {r['types']:>6,}  {r['max_id']:>6}  "
            f"{r['L_id']:>5}  {r['R_id']:>5}  "
            f"{r['L']!r:>14}  {r['R']!r:>14}  {r['S']!r:>16}",
            file=file,
        )
    print(file=file)
    print(f"Routed: {len(routes)}", file=file)
    print(f"Dropped: {len(dropped)} candidates", file=file)
    in_vocab = sum(1 for d in dropped if d["reason"] == "in_vocab")
    no_route = sum(1 for d in dropped if d["reason"] == "no_route")
    blocked = sum(1 for d in dropped if d["reason"] == "blocked_by_substring")
    print(f"  already in baseline vocab: {in_vocab}", file=file)
    print(f"  no natural route through baseline vocab: {no_route}", file=file)
    if blocked:
        print(f"  blocked by high-firing baseline substring: {blocked}",
              file=file)


# ---------------------------------------------------------------------------
# CLI

def main(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--candidates", type=Path, required=True,
        help="Path to mined CANDIDATES .py file from tools.mine_seed_morphemes",
    )
    p.add_argument(
        "--baseline-tokenizer", type=Path,
        default=Path("~/.cache/nanochat/tokenizer").expanduser(),
        help="Path to baseline tokenizer directory (contains tokenizer.pkl)",
    )
    p.add_argument(
        "--policy", choices=["min-max-id"], default="min-max-id",
        help="Route-selection policy (default: min-max-id)",
    )
    p.add_argument(
        "--out-py", type=Path, default=None,
        help="Write SEED_ROUTES .py to this path (else print to stdout)",
    )
    p.add_argument(
        "--max-blocker-firings", type=int, default=None,
        help="Substring-blocker filter (default off): drop candidate S if "
        "any proper substring T of S is in baseline vocab with "
        "baseline_firings[T] > this threshold. Prevents seeds that block "
        "high-utility baseline morphemes via encode-time greedy. "
        "Threshold is firings-per-ClimbMix-val-sample; ~100 is a reasonable "
        "starting point (drops seeds containing any of the top ~3000 "
        "baseline tokens by usage).",
    )
    p.add_argument(
        "--firings-cache", type=Path,
        default=Path("results/seed_tokens/baseline_firings_climbmix_val.json"),
        help="Cache path for baseline token firing counts on ClimbMix val "
        "(built once on first run with --max-blocker-firings set).",
    )
    p.add_argument(
        "--firings-cache-chars", type=int, default=20_000_000,
        help="--firings-cache build size (default: 20M chars, matches "
        "eval_tokenizer Tier-2)",
    )
    args = p.parse_args(argv)

    print(f"Loading candidates from {args.candidates}...", file=sys.stderr)
    candidates = load_candidates(args.candidates)
    print(f"  {len(candidates)} candidates loaded.", file=sys.stderr)

    print(f"Loading baseline vocab from {args.baseline_tokenizer}...",
          file=sys.stderr)
    vocab = load_baseline_vocab(args.baseline_tokenizer)
    print(f"  {len(vocab):,} baseline mergeable tokens.", file=sys.stderr)

    firings: dict[bytes, int] | None = None
    if args.max_blocker_firings is not None:
        print(
            f"Loading baseline firing counts (threshold "
            f"--max-blocker-firings={args.max_blocker_firings})...",
            file=sys.stderr,
        )
        firings = load_or_build_firings(
            args.firings_cache, args.baseline_tokenizer,
            args.firings_cache_chars,
        )
        n_above = sum(1 for c in firings.values() if c > args.max_blocker_firings)
        print(
            f"  {len(firings):,} baseline tokens have firing data; "
            f"{n_above:,} above threshold (= candidate blockers).",
            file=sys.stderr,
        )

    routed: list[dict] = []
    dropped: list[dict] = []
    for cand_str, stats in candidates:
        S_bytes = cand_str.encode("utf-8")
        if S_bytes in vocab:
            dropped.append({"S": S_bytes, "reason": "in_vocab",
                            "stats": stats})
            continue
        if firings is not None:
            blocker = is_blocked_by_substring(
                S_bytes, vocab, firings, args.max_blocker_firings,
            )
            if blocker is not None:
                dropped.append({
                    "S": S_bytes, "reason": "blocked_by_substring",
                    "blocker": blocker, "stats": stats,
                })
                continue
        splits = find_routes(S_bytes, vocab)
        if not splits:
            dropped.append({"S": S_bytes, "reason": "no_route",
                            "stats": stats})
            continue
        max_id, _min_id, L, R = select_route(splits, args.policy)
        routed.append({
            "S": S_bytes, "L": L, "R": R,
            "max_id": max_id, "L_id": vocab[L], "R_id": vocab[R],
            "types": stats.get("types", 0),
            "total": stats.get("total", 0),
        })

    # Order by `max_id` ascending (= natural-firing order) so the wrapper's
    # stable sort assigns earlier ranks to earlier rows. **Tiebreak matters
    # behaviorally**: same-`max_id` routes share an insertion point, so
    # whichever comes first in the emitted file gets the lower final rank
    # in mergeable_ranks → wins greedy encoding against the other.
    # Tiebreak on `-types` then `-total` (higher productivity / frequency
    # gets the earlier rank); `S` last as a deterministic final tiebreak.
    routed.sort(
        key=lambda r: (r["max_id"], -r["types"], -r["total"], r["S"])
    )

    if args.out_py:
        args.out_py.parent.mkdir(parents=True, exist_ok=True)
        with args.out_py.open("w") as f:
            emit_routes_py(
                routed, file=f, mining_path=args.candidates,
                baseline_tokenizer=args.baseline_tokenizer,
                policy=args.policy,
            )
        print(f"Wrote {args.out_py}", file=sys.stderr)
        emit_text(routed, dropped, file=sys.stderr)
    else:
        emit_text(routed, dropped)


if __name__ == "__main__":
    main()
