"""Held-out-gated Pareto build-up for the auto_tune block list.

SUPERSEDED by `tools.heldout_fixpoint_cached` (greedy fixpoint + corpus-cached
trainer, ~6x faster). Retained as the single-forward-pass baseline and because it
exports `iter_capped_from_path` / `vocab_str2id`, which the fixpoint driver and
`tools.token_lineage` import.

Rebuilds BLOCKED_PAIRS from a candidate list, keeping ONLY pairs that are robust
Pareto improvements *across the held-out shards* — so generalization is baked into
selection rather than vetted afterward. Build-up: each candidate is trained on top
of the growing kept set, then measured on N held-out shards (fast: compression +
dead) plus coverage (vocab). A candidate is KEPT iff, vs the current reference
(averaged over the held-out shards):

  - whole-word coverage (total dict words) INCREASES, AND
  - compression (avg total_tokens) does NOT worsen, AND
  - dead (avg) does NOT worsen.

Otherwise it's LOGGED (full per-shard data) for later human review — never silently
dropped. Mangled is logged on the tune shard for KEEPs only (it's the weak
heuristic; not gated, see audit/README.md). Per keep, lineage goes to
lineage.jsonl: born/died tokens (vocab diff vs the previous keep) + top firing-mass
gainers/losers. Keep-tokenizers are retained for post-hoc token_lineage.

    uv run python -m tools.heldout_buildup
    uv run python -m tools.heldout_buildup --resume
    uv run python -m tools.heldout_buildup --limit 2 --smoke-train-chars 3000000 \\
        --state-dir /tmp/hb_smoke --work-base /tmp/hb_work --max-chars 2000000   # plumbing smoke
"""
import argparse
import glob
import importlib
import json
import os
import shutil
import time
from collections import Counter
from pathlib import Path


def iter_capped_from_path(parquet_path, max_chars):
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(parquet_path)
    total = 0
    for rg in range(pf.num_row_groups):
        if total >= max_chars:
            return
        texts = pf.read_row_group(rg).column("text").to_pylist()
        docs = []
        for d in texts:
            docs.append(d)
            total += len(d)
            if total >= max_chars:
                break
        if docs:
            yield docs
        if total >= max_chars:
            return


def vocab_str2id(tok):
    """{decoded_string: id} for non-special tokens (first id wins)."""
    n = tok.get_vocab_size() - len(tok.get_special_tokens())
    out = {}
    for tid in range(n):
        try:
            s = tok.decode([tid])
        except Exception:
            continue
        out.setdefault(s, tid)
    return out


def main():
    home = os.path.expanduser("~")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pairs-module", default="rustbpe_variants.auto_tune.pairs")
    p.add_argument("--candidates-attr", default="CANDIDATES_PASS2")
    p.add_argument("--holdout-glob", default=f"{home}/.cache/nanochat/base_data_climbmix_holdout/*.parquet")
    p.add_argument("--tune-shard", default=f"{home}/.cache/nanochat/base_data_climbmix/shard_06542.parquet")
    p.add_argument("--max-chars", type=int, default=10_000_000, help="char cap per held-out shard")
    p.add_argument("--work-base", default=f"{home}/.cache/nanochat-variants/auto_tune_heldout")
    p.add_argument("--state-dir", default="rustbpe_variants/auto_tune/audit/heldout_pass")
    p.add_argument("--train-rayon-threads", type=int, default=None)
    p.add_argument("--bound-threshold", type=float, default=0.05)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--smoke-train-chars", type=int, default=None)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()

    from tools.audit_blocked_pairs import train, whole_word_counts
    from tools.pass_metrics import fire_counts, mergeable_vocab_size, compute_metrics
    from tools.blocked_pairs_report import load_wordlist
    from nanochat.tokenizer import RustBPETokenizer

    mod = importlib.import_module(args.pairs_module)
    candidates = [tuple(c) for c in getattr(mod, args.candidates_attr)]
    words = load_wordlist()
    if words is None:
        raise SystemExit("no wordlist at /usr/share/dict/words")

    work_base = Path(args.work_base); work_base.mkdir(parents=True, exist_ok=True)
    train_log = work_base / "last_train.log"
    state_dir = Path(args.state_dir); state_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = state_dir / "checkpoint.json"
    ledger_path = state_dir / "ledger.md"
    lineage_path = state_dir / "lineage.jsonl"
    retain_dir = work_base / "kept_tokenizers" / state_dir.name
    retain_dir.mkdir(parents=True, exist_ok=True)

    shards = sorted(glob.glob(args.holdout_glob))
    if not shards:
        raise SystemExit(f"no held-out shards match {args.holdout_glob}")
    print(f"[heldout] loading {len(shards)} held-out shards ({args.max_chars:,} cap each)...", flush=True)
    holdout = {Path(s).stem: list(iter_capped_from_path(s, args.max_chars)) for s in shards}
    tune_corpus = list(iter_capped_from_path(args.tune_shard, 20_000_000))

    def measure_heldout(tdir):
        """(metrics dict, aggregate fire Counter, str->id vocab) for a tokenizer dir."""
        tok = RustBPETokenizer.from_directory(str(tdir))
        base, infl = whole_word_counts(tok, words)
        n_merge = mergeable_vocab_size(tok)
        per_shard = {}
        agg = Counter()
        for name, corp in holdout.items():
            f = fire_counts(tok, corp)
            agg.update(f)
            per_shard[name] = {"tokens": sum(f.values()),
                               "dead": sum(1 for t in range(n_merge) if f.get(t, 0) == 0)}
        nsh = len(per_shard)
        m = {"coverage": base + infl, "base": base, "infl": infl,
             "avg_comp": sum(s["tokens"] for s in per_shard.values()) / nsh,
             "avg_dead": sum(s["dead"] for s in per_shard.values()) / nsh,
             "per_shard": per_shard}
        return m, agg, vocab_str2id(tok)

    def retain(tdir, n):
        dst = retain_dir / f"keep_{n:04d}"
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(tdir, dst)

    def write_ledger(state):
        kept = state["kept"]
        keeps = [r for r in state["rows"] if r["verdict"] == "KEEP"]
        lines = [
            "# auto_tune held-out-Pareto build-up ledger", "",
            f"- held-out shards: {len(holdout)} ({args.max_chars:,} chars each), avg-gated",
            f"- gate: coverage↑ AND avg-compression≤0 AND avg-dead≤0 (mangled logged, not gated)",
            f"- processed: {len(state['rows'])}/{len(candidates)}  |  kept: {len(kept)}  logged: {len(state['rows'])-len(keeps)}",
            f"- reference: {fmt_ref(state.get('ref'))}", "",
            f"| # | pair | verdict | Δcov | Δavg-comp | Δavg-dead | kept |",
            "|---|------|---------|------|-----------|-----------|------|",
        ]
        for i, r in enumerate(state["rows"]):
            L, R = r["pair"]
            lines.append(f"| {i} | `{L!r}+{R!r}` | {r['verdict']} | "
                         f"{r.get('d_cov','-'):+} | {r.get('d_comp','-'):+} | {r.get('d_dead','-'):+} | {r.get('kept_size','-')} |"
                         if r["verdict"] != "ERROR" else
                         f"| {i} | `{L!r}+{R!r}` | ERROR | - | - | - | - |")
        lines += ["", "## kept (held-out Pareto winners → promote into BLOCKED_PAIRS)", "```python"]
        lines += [f"    {tuple(p)!r}," for p in kept]
        lines += ["```"]
        ledger_path.write_text("\n".join(lines) + "\n")

    # ---- resume or init ----
    if args.resume and ckpt_path.exists():
        state = json.loads(ckpt_path.read_text())
        state["kept"] = [tuple(p) for p in state["kept"]]
        print(f"[heldout] resumed: {len(state['rows'])}/{len(candidates)} done, kept={len(state['kept'])}", flush=True)
        ref_tdir = retain_dir / f"keep_{len(state['kept']):04d}"
        if ref_tdir.exists():
            _, ref_fire, ref_vocab = measure_heldout(ref_tdir)
        else:
            print("[heldout] WARN: reference tokenizer not retained; lineage diffs restart here", flush=True)
            ref_fire, ref_vocab = Counter(), {}
    else:
        state = {"kept": [], "ref": None, "rows": [], "n_candidates": len(candidates)}
        ref_fire, ref_vocab = Counter(), {}

    # ---- reference (seed) ----
    if state["ref"] is None:
        print(f"[heldout] training reference (kept={len(state['kept'])})...", flush=True)
        tdir = train([list(p) for p in state["kept"]], work_base, args.smoke_train_chars, train_log,
                     rayon_threads=args.train_rayon_threads)
        if tdir is None:
            raise SystemExit(f"reference train failed; see {train_log}")
        state["ref"], ref_fire, ref_vocab = measure_heldout(tdir)
        retain(tdir, len(state["kept"]))
        ckpt_path.write_text(json.dumps(state))
        print(f"[heldout] reference: {fmt_ref(state['ref'])}", flush=True)

    # ---- build-up loop ----
    start = len(state["rows"])
    end = len(candidates) if args.limit is None else min(start + args.limit, len(candidates))
    for i in range(start, end):
        cand = candidates[i]
        t0 = time.time()
        tdir = train([list(p) for p in state["kept"]] + [list(cand)], work_base,
                     args.smoke_train_chars, train_log, rayon_threads=args.train_rayon_threads)
        if tdir is None:
            print(f"[heldout] {i+1}/{len(candidates)} {cand!r}  TRAIN FAILED", flush=True)
            state["rows"].append({"pair": list(cand), "verdict": "ERROR"})
            ckpt_path.write_text(json.dumps(state)); continue
        m, fire, new_vocab = measure_heldout(tdir)
        ref = state["ref"]
        d_cov = m["coverage"] - ref["coverage"]
        d_comp = round(m["avg_comp"] - ref["avg_comp"], 1)
        d_dead = round(m["avg_dead"] - ref["avg_dead"], 1)
        keep = d_cov > 0 and d_comp <= 0 and d_dead <= 0
        verdict = "KEEP" if keep else "LOG"
        print(f"[heldout] {i+1}/{len(candidates)} {cand!r}  {verdict}  "
              f"Δcov{d_cov:+} Δcomp{d_comp:+} Δdead{d_dead:+}  [{time.time()-t0:.0f}s]", flush=True)
        row = {"pair": list(cand), "verdict": verdict, "d_cov": d_cov, "d_comp": d_comp, "d_dead": d_dead,
               "coverage": m["coverage"], "base": m["base"], "infl": m["infl"],
               "avg_comp": round(m["avg_comp"], 1), "avg_dead": round(m["avg_dead"], 1),
               "per_shard": m["per_shard"], "kept_size": len(state["kept"]) + (1 if keep else 0)}
        if keep:
            born = sorted(s for s in new_vocab if s not in ref_vocab)
            died = sorted(s for s in ref_vocab if s not in new_vocab)
            new_fbs = {s: fire.get(i_, 0) for s, i_ in new_vocab.items()}
            ref_fbs = {s: ref_fire.get(i_, 0) for s, i_ in ref_vocab.items()}
            shifts = {s: new_fbs.get(s, 0) - ref_fbs.get(s, 0) for s in set(new_fbs) | set(ref_fbs)}
            gainers = sorted(shifts.items(), key=lambda kv: -kv[1])[:15]
            losers = sorted(shifts.items(), key=lambda kv: kv[1])[:15]
            tok = RustBPETokenizer.from_directory(str(tdir))
            mng = compute_metrics(tok, tune_corpus, count_mangled=True, bound_threshold=args.bound_threshold)["mangled_count"]
            row["mangled_tune"] = mng
            with open(lineage_path, "a") as lf:
                lf.write(json.dumps({
                    "keep": len(state["kept"]) + 1, "pair": list(cand),
                    "n_born": len(born), "n_died": len(died),
                    "born": born[:60], "died": died[:60],
                    "top_gainers": gainers, "top_losers": losers,
                }) + "\n")
            state["kept"].append(list(cand)); state["ref"] = m
            ref_fire, ref_vocab = fire, new_vocab
            retain(tdir, len(state["kept"]))
        state["rows"].append(row)
        ckpt_path.write_text(json.dumps(state)); write_ledger(state)

    keeps = [r for r in state["rows"] if r["verdict"] == "KEEP"]
    print(f"\n[heldout] DONE {len(state['rows'])}/{len(candidates)}.  kept={len(keeps)} "
          f"logged={len(state['rows'])-len(keeps)}", flush=True)
    print(f"[heldout] kept pairs:")
    for kp in state["kept"]:
        print(f"    {tuple(kp)!r},")


def fmt_ref(m):
    if not m:
        return "(pending)"
    return f"cov={m['coverage']} (base {m['base']}/infl {m['infl']}) avg_comp={m['avg_comp']:.0f} avg_dead={m['avg_dead']:.1f}"


if __name__ == "__main__":
    main()
