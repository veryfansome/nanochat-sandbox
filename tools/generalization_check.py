"""Generalization check for the auto_tune block list.

The build-up audit selects pairs by their effect on a SINGLE eval shard (val =
ClimbMix shard_06542), so the kept list could be overfit to that one corpus.
This trains the kept-list tokenizer vs the empty baseline and measures BOTH on
several shards — the audit's eval shard plus held-out shards the tokenizer never
trained on — to see whether the gains hold.

Corpus-dependent metrics (the generalization signal): total_tokens (compression)
and mangled. Coverage (whole-word vocab count) is a vocab property — shard-
independent — so it's reported once.

    uv run python -m tools.generalization_check
    uv run python -m tools.generalization_check --checkpoint <ckpt> --shards a.parquet b.parquet
"""
import argparse
import json
import os
from pathlib import Path


def iter_capped_from_path(parquet_path, max_chars):
    """Doc-trimmed capped batches from an explicit parquet path (mirrors
    tools._corpus_iter.iter_capped_batches, but for an arbitrary file rather
    than the base_data_climbmix train/val split)."""
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(parquet_path)
    total = 0
    for rg_idx in range(pf.num_row_groups):
        if total >= max_chars:
            return
        texts = pf.read_row_group(rg_idx).column('text').to_pylist()
        docs = []
        for doc in texts:
            docs.append(doc)
            total += len(doc)
            if total >= max_chars:
                break
        if docs:
            yield docs
        if total >= max_chars:
            return


def main():
    home = os.path.expanduser("~")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", default="rustbpe_variants/auto_tune/audit/mined_pass1/checkpoint.json",
                   help="audit checkpoint to read the kept block list from")
    p.add_argument("--baseline", default=f"{home}/.cache/nanochat/tokenizer",
                   help="empty-block baseline tokenizer dir (= stock nanochat tokenizer)")
    p.add_argument("--work-base", default=f"{home}/.cache/nanochat-variants/auto_tune_gencheck")
    p.add_argument("--max-chars", type=int, default=20_000_000)
    p.add_argument("--bound-threshold", type=float, default=0.05)
    p.add_argument("--reuse", action="store_true",
                   help="reuse an already-trained kept tokenizer in --work-base (skip the retrain)")
    p.add_argument("--shards", nargs="+", default=None,
                   help="parquet shards to eval on (default: tune shard 06542 + every "
                        "held-out shard in base_data_climbmix_holdout/)")
    args = p.parse_args()
    if args.shards is None:
        import glob
        tune = f"{home}/.cache/nanochat/base_data_climbmix/shard_06542.parquet"   # the shard selection tuned on
        held = sorted(glob.glob(f"{home}/.cache/nanochat/base_data_climbmix_holdout/*.parquet"))
        args.shards = [tune] + held

    from tools.audit_blocked_pairs import train, measure
    from tools.blocked_pairs_report import load_wordlist
    words = load_wordlist()
    if words is None:
        raise SystemExit("no wordlist; whole-word metric unavailable")

    kept = [tuple(x) for x in json.loads(Path(args.checkpoint).read_text())["kept"]]
    print(f"[gencheck] kept block list: {len(kept)} pairs (from {args.checkpoint})", flush=True)

    # Train the kept-list tokenizer (full corpus, single train = all cores).
    wb = Path(args.work_base); wb.mkdir(parents=True, exist_ok=True)
    bdir = wb / "tokenizer"
    if args.reuse and bdir.exists():
        print(f"[gencheck] reusing existing kept tokenizer at {bdir}", flush=True)
    else:
        print("[gencheck] training kept-list tokenizer (~4 min)...", flush=True)
        bdir = train(kept, wb, None, wb / "train.log")
        if bdir is None:
            raise SystemExit(f"kept train failed; see {wb/'train.log'}")

    print(f"\n{'shard':16} {'chars':>11} {'empty_tok':>12} {'kept_tok':>12} {'Δcomp':>8} {'Δmangled':>9}", flush=True)
    print("-" * 74, flush=True)
    cov = None
    rows = []
    for s in args.shards:
        name = Path(s).stem
        corpus = list(iter_capped_from_path(s, args.max_chars))
        nchars = sum(len(d) for batch in corpus for d in batch)
        a = measure(args.baseline, corpus, words, args.bound_threshold)
        b = measure(str(bdir), corpus, words, args.bound_threshold)
        dcomp = b["total_tokens"] - a["total_tokens"]
        dmang = b["mangled"] - a["mangled"]
        rows.append((name, dcomp, dmang))
        print(f"{name:16} {nchars:>11,} {a['total_tokens']:>12,} "
              f"{b['total_tokens']:>12,} {dcomp:>+8} {dmang:>+9}", flush=True)
        if cov is None:
            cov = (a["total_words"], b["total_words"])

    print("-" * 74, flush=True)
    import statistics
    tune = [(n, dc, dm) for (n, dc, dm) in rows if n == "shard_06542"]
    held = [(n, dc, dm) for (n, dc, dm) in rows if n != "shard_06542"]
    print(f"coverage (vocab, shard-independent): empty={cov[0]:,} → kept={cov[1]:,} (Δ{cov[1]-cov[0]:+})", flush=True)
    if tune:
        print(f"tune shard (06542, what selection optimized): Δcomp {tune[0][1]:+}, Δmangled {tune[0][2]:+}", flush=True)
    if held:
        comps = [dc for _, dc, _ in held]
        mangs = [dm for _, _, dm in held]
        print(f"\nHELD-OUT generalization ({len(held)} shards; kept − empty, negative = kept helps):", flush=True)
        print(f"  compression Δ: mean {statistics.mean(comps):+.1f}  range [{min(comps):+}..{max(comps):+}]  "
              f"{sum(1 for c in comps if c < 0)}/{len(comps)} shards improved", flush=True)
        print(f"  mangled Δ:     mean {statistics.mean(mangs):+.1f}  range [{min(mangs):+}..{max(mangs):+}]  "
              f"{sum(1 for m in mangs if m < 0)}/{len(mangs)} shards improved", flush=True)
        print(f"  coverage Δ:    {cov[1]-cov[0]:+} (vocab-wide — applies to every shard)", flush=True)


if __name__ == "__main__":
    main()
