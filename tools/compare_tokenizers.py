"""Three-way tokenizer comparison: baseline vs force_merges vs force_merges_combined.

Documents + reproduces how the 126-block "combined" variant was evaluated against its
parents — on the axes that actually matter for it (whole-word coverage + morpheme firing),
NOT just compression. Replaces the ad-hoc one-off scripts used during the audit handoff;
the findings it produced are written up in `rustbpe_variants/force_merges_combined/README.md`.

Three analyses, all over the same held-out shards (each tokenizer is encoded once and the
firing map reused across analyses):

1. COMPRESSION — total tokens per tokenizer + per-shard win rates. The *discounted* axis:
   the 126 blocks move it ~nothing (combined vs force_merges ≈ -0.025%).
2. FIRING (whole-word coverage) — distinct whole-word tokens that FIRE (>=1) under each, and
   for the ref->cmp pair the born/died whole-words classified real vs artifact by standalone
   usage (`tools.word_realness`). This is the axis the audit optimized.
3. MORPHEME ATOMS — firing summed over `tools.find_mangled_morphemes`'s SUFFIXES/PREFIXES
   inventories, per tokenizer + ref->cmp deltas. Shows whole-word consolidation vs morpheme
   decomposition.

force_merges and combined differ ONLY in the 126 blocks, so the ref->cmp deltas (default
force_merges->combined) ISOLATE exactly the blocks' effect. baseline uses a different
SPLIT_PATTERN, so baseline->force_merges deltas are CONFOUNDED (carve-out + forced phrases) —
read them as context, not isolation.

Usage:
  uv run python -m tools.compare_tokenizers                 # all 3 analyses, all held-out shards
  uv run python -m tools.compare_tokenizers --shards 5      # quick: first 5 shards
  uv run python -m tools.compare_tokenizers --analyses firing,morphemes
  uv run python -m tools.compare_tokenizers --ref baseline --cmp combined
  uv run python -m tools.compare_tokenizers --tokenizer mine=~/.cache/.../tokenizer
"""
import argparse
import glob
import sys
from collections import Counter
from pathlib import Path

HOME = str(Path.home())
DEFAULT_TOKENIZERS = {
    "baseline": f"{HOME}/.cache/nanochat/tokenizer",
    "force_merges": f"{HOME}/.cache/nanochat-variants/force_merges/tokenizer",
    "combined": f"{HOME}/.cache/nanochat-variants/force_merges_combined/tokenizer",
}
SHARD_GLOB = f"{HOME}/.cache/nanochat/base_data_climbmix_holdout*/*.parquet"
CAP = 10_000_000  # chars per shard (matches the audit's held-out eval)


def load_texts(limit=None):
    """Concatenate each held-out shard (capped) into one string; return the list of shards."""
    from tools.heldout_buildup import iter_capped_from_path
    shards = sorted(glob.glob(SHARD_GLOB))
    if limit:
        shards = shards[:limit]
    if not shards:
        raise SystemExit(f"no shards matched {SHARD_GLOB}")
    print(f"[compare] loading {len(shards)} shard(s) (<= {CAP:,} chars each)...", file=sys.stderr)
    return ["".join(d for b in iter_capped_from_path(s, CAP) for d in b) for s in shards]


def encode(name, tok_dir, texts):
    """Encode every shard once; return per-shard token counts + a firing map keyed by the
    DECODED token string (so the same key compares across tokenizers with different ids)."""
    from nanochat.tokenizer import RustBPETokenizer
    from tools.heldout_buildup import vocab_str2id
    tok = RustBPETokenizer.from_directory(str(Path(tok_dir).expanduser()))
    inv = {i: s for s, i in vocab_str2id(tok).items()}
    per_shard, cnt = [], Counter()
    for t in texts:
        ids = tok.encode(t)
        per_shard.append(len(ids))
        cnt.update(ids)
    fire = Counter({inv.get(i, f"<{i}>"): n for i, n in cnt.items()})
    print(f"[compare] encoded {name}: {sum(per_shard):,} tokens", file=sys.stderr)
    return {"name": name, "per_shard": per_shard, "fire": fire, "total": sum(per_shard)}


def report_compression(results):
    print("\n================  COMPRESSION (total tokens; fewer = better)  ================")
    base = results[0]
    for r in results:
        if r is base:
            print(f"  {r['name']:16} {r['total']:>14,}")
            continue
        d = r["total"] - base["total"]
        pct = 100 * d / base["total"] if base["total"] else 0
        wins = sum(1 for a, b in zip(r["per_shard"], base["per_shard"]) if a < b)
        print(f"  {r['name']:16} {r['total']:>14,}   vs {base['name']}: {d:+,} ({pct:+.3f}%), "
              f"{wins}/{len(r['per_shard'])} shards fewer")


def _is_whole_word(s, words):
    from tools.blocked_pairs_report import _is_word
    if not s.startswith(" "):
        return False
    w = s[1:].lower()
    return bool(w) and (w in words or _is_word(w, words))


def report_firing(by_name, ref, cmp, texts, standalone_min):
    from tools.blocked_pairs_report import load_wordlist
    from tools.word_realness import standalone_counts
    words = load_wordlist()
    print(f"\n================  FIRING — whole-word coverage  ({ref} -> {cmp})  ================")
    fr, fc = by_name[ref]["fire"], by_name[cmp]["fire"]
    ww_r = {s for s in fr if _is_whole_word(s, words)}
    ww_c = {s for s in fc if _is_whole_word(s, words)}
    print(f"  whole-word tokens that FIRE:  {ref} {len(ww_r):,}   {cmp} {len(ww_c):,}   "
          f"Δ {len(ww_c)-len(ww_r):+,}")
    born = {s: fc[s] for s in ww_c - ww_r}   # fire in cmp, not ref
    died = {s: fr[s] for s in ww_r - ww_c}   # fired in ref, lost in cmp
    stand, _ = standalone_counts(list(born) + list(died), texts)
    real = lambda s: stand.get(s[1:], 0) >= standalone_min
    born_real = {s: c for s, c in born.items() if real(s)}
    born_art = {s: c for s, c in born.items() if not real(s)}
    died_real = {s: c for s, c in died.items() if real(s)}
    print(f"  BORN ({cmp} gains): {len(born)} = real {len(born_real)} / artifact {len(born_art)}"
          f"   |   DIED ({cmp} loses): {len(died)}")
    print(f"  NET real whole-words: {len(born_real)-len(died_real):+}  "
          f"({sum(born_real.values()):,} firing events on real born vs {sum(died_real.values()):,} on real died)")

    def dump(title, d, sign):
        print(f"  -- {title} --")
        for s, c in sorted(d.items(), key=lambda x: -x[1])[:12]:
            print(f"     {sign}{s!r:14} fires {c:>6}  (standalone {stand.get(s[1:],0)})")
    dump("top real BORN", born_real, "+")
    dump(f"artifact BORN (standalone < {standalone_min})", born_art, "+")
    if died:
        dump("DIED", died, "-")


def report_morphemes(by_name, names, ref, cmp):
    from tools.find_mangled_morphemes import SUFFIXES, PREFIXES
    for label, inv in (("SUFFIX", SUFFIXES), ("PREFIX", PREFIXES)):
        print(f"\n================  MORPHEME ATOMS — {label} firing  ================")
        present = [m for m in inv if any(m in by_name[n]["fire"] for n in names)]
        tot = {n: sum(by_name[n]["fire"].get(m, 0) for m in present) for n in names}
        print(f"  total over {len(present)} atoms:  " + "   ".join(f"{n} {tot[n]:>12,}" for n in names))
        d = tot[cmp] - tot[ref]
        pct = 100 * d / tot[ref] if tot[ref] else 0
        print(f"     {ref} -> {cmp}: {d:+,} ({pct:+.3f}%)  [isolates the blocks]")
        rows = [(m, by_name[ref]["fire"].get(m, 0), by_name[cmp]["fire"].get(m, 0)) for m in present]
        rows = [(m, a, b, b - a) for m, a, b in rows]
        print(f"  -- top {ref}->{cmp} GAIN --")
        for m, a, b, dd in sorted(rows, key=lambda r: -r[3])[:8]:
            print(f"     {m!r:12} {a:>10,} -> {b:>10,}   {dd:+,}")
        print(f"  -- top LOSS --")
        for m, a, b, dd in sorted(rows, key=lambda r: r[3])[:8]:
            print(f"     {m!r:12} {a:>10,} -> {b:>10,}   {dd:+,}")


def main():
    p = argparse.ArgumentParser(
        description="Three-way tokenizer comparison (baseline / force_merges / combined).",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    p.add_argument("--shards", type=int, default=None, help="limit to first N held-out shards (default: all)")
    p.add_argument("--analyses", default="all", help="comma list of compression,firing,morphemes (default all)")
    p.add_argument("--ref", default="force_merges", help="reference tokenizer for firing/morpheme deltas")
    p.add_argument("--cmp", default="combined", help="compared tokenizer for firing/morpheme deltas")
    p.add_argument("--standalone-min", type=int, default=5,
                   help="standalone count for a born whole-word to count as real, not artifact (default 5)")
    p.add_argument("--tokenizer", action="append", default=[], metavar="NAME=DIR",
                   help="override/add a tokenizer dir (repeatable); default baseline/force_merges/combined")
    args = p.parse_args()

    toks = dict(DEFAULT_TOKENIZERS)
    for spec in args.tokenizer:
        name, _, d = spec.partition("=")
        if not d:
            raise SystemExit(f"--tokenizer expects NAME=DIR, got {spec!r}")
        toks[name] = d
    which = ({"compression", "firing", "morphemes"} if args.analyses == "all"
             else {a.strip() for a in args.analyses.split(",")})
    for name in (args.ref, args.cmp):
        if name not in toks:
            raise SystemExit(f"--ref/--cmp {name!r} not in tokenizers {list(toks)}")

    texts = load_texts(args.shards)
    results = [encode(n, d, texts) for n, d in toks.items()]
    by_name = {r["name"]: r for r in results}
    names = list(toks)

    if "compression" in which:
        report_compression(results)
    if "firing" in which:
        report_firing(by_name, args.ref, args.cmp, texts, args.standalone_min)
    if "morphemes" in which:
        report_morphemes(by_name, names, args.ref, args.cmp)


if __name__ == "__main__":
    main()
