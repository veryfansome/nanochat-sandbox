"""Fresh-shard generalization check for the held-out-Pareto FIXPOINT.

The fixpoint gates every keep on 10 held-out shards, so the kept list could in
principle be overfit to that particular shard set. This evaluates the CURRENT
kept set (read live from a fixpoint checkpoint) and the empty baseline on a
DIFFERENT set of shards, using the exact measurement stack of
tools.heldout_fixpoint_cached — cached corpus + train_from_cached + the fused
Rust evaluate_many (comp = total tokens on the capped shard, dead = merge-vocab
tokens with zero firings) — so the deltas are directly comparable with the
run's gate-shard tally. Coverage is a vocab property (shard-independent); it's
reported as a sanity check that the right kept set was evaluated.

Replaces the earlier non-cached generalization_check (single-shard era:
subprocess trainer, no dead metric; removed in the 2026-06 cleanup).

    uv run python -m tools.generalization_check_cached
    uv run python -m tools.generalization_check_cached --shards-glob '.../base_data_climbmix_holdout2/*.parquet'
"""
import argparse
import glob
import json
import os
import shutil
import tempfile
import time
from pathlib import Path


def main():
    home = os.path.expanduser("~")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint",
                   default="rustbpe_variants/auto_tune/audit/checkpoint.json",
                   help="fixpoint checkpoint to read the kept block list from (snapshot-read, so a "
                        "live run rewriting it is fine)")
    p.add_argument("--shards-glob",
                   default=f"{home}/.cache/nanochat/base_data_climbmix_holdout2/*.parquet",
                   help="fresh shards to evaluate on (must differ from the gate's shards for a "
                        "true generalization check)")
    p.add_argument("--max-chars", type=int, default=10_000_000,
                   help="char cap per shard — keep equal to the fixpoint's --max-chars so "
                        "comp/dead are on the same scale as the gate tally")
    p.add_argument("--vocab-size", type=int, default=32768)
    p.add_argument("--corpus-max-chars", type=int, default=2_000_000_000)
    p.add_argument("--doc-cap", type=int, default=10_000)
    p.add_argument("--threads", type=int, default=2,
                   help="evaluate_many threads; few sets are evaluated, and a low count "
                        "stays polite to a concurrently running fixpoint")
    p.add_argument("--trajectory", type=int, default=0, metavar="N",
                   help="instead of just the current kept set, replay the journal and evaluate "
                        "the kept-set prefix at every N rounds (plus the final set) on the fresh "
                        "shards — charts how fresh-shard comp/dead track coverage across the "
                        "run's history (e.g. does the comp cost grow as coverage purchases "
                        "accumulate?). 0 = off")
    p.add_argument("--upto", type=int, default=0, metavar="R",
                   help="pin the evaluation to the kept set as of round R (journal replay) instead "
                        "of the checkpoint's live kept set — results don't drift while a run "
                        "advances. 0 = live")
    args = p.parse_args()

    import rustbpe_auto_tune
    from nanochat.tokenizer import SPLIT_PATTERN, SPECIAL_TOKENS
    from nanochat.dataset import parquets_iter_batched
    from tools.blocked_pairs_report import load_wordlist
    from tools.heldout_buildup import iter_capped_from_path

    # snapshot-read the checkpoint: the live run's write_text() is not atomic, so
    # copy first and retry once if we caught a partial write
    for attempt in (1, 2):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
            snap = tf.name
        shutil.copy(args.checkpoint, snap)
        try:
            cp = json.loads(Path(snap).read_text())
            break
        except json.JSONDecodeError:
            if attempt == 2:
                raise
            time.sleep(2)
        finally:
            os.unlink(snap)
    def replay_kept(upto):
        """The kept set after round `upto` (commits + cleanups, same replay rule as
        the driver's --rerun)."""
        ks = []
        for r in cp["rounds"][:upto]:
            if r.get("committed"):
                ks.append(tuple(r["committed"]))
            for rm in r.get("cleanup", []):
                t = tuple(rm)
                if t in ks:
                    ks.remove(t)
        return ks

    rounds = len(cp.get("rounds", []))
    if args.upto:
        if args.upto > rounds:
            raise SystemExit(f"--upto {args.upto} exceeds journal length {rounds}")
        rounds = args.upto
        kept = replay_kept(rounds)
    else:
        kept = [tuple(k) for k in cp["kept"]]
    gate_base, gate_ref = cp.get("baseline"), cp.get("ref")
    print(f"[genc] kept set: {len(kept)} pairs @ round {rounds}"
          f"{' (pinned via --upto)' if args.upto else ''} (from {args.checkpoint})", flush=True)

    shards = sorted(glob.glob(args.shards_glob))
    if not shards:
        raise SystemExit(f"no shards match {args.shards_glob}")
    words = load_wordlist()
    if words is None:
        raise SystemExit("no wordlist at /usr/share/dict/words")
    V = args.vocab_size - len(SPECIAL_TOKENS)

    def corpus_iter():
        nchars = 0
        for batch in parquets_iter_batched(split="train"):
            for doc in batch:
                d = doc[:args.doc_cap] if len(doc) > args.doc_cap else doc
                nchars += len(d)
                yield d
                if nchars > args.corpus_max_chars:
                    return

    print(f"[genc] caching training corpus (doc_cap {args.doc_cap:,}, max {args.corpus_max_chars:,})...", flush=True)
    t0 = time.time()
    trainer = rustbpe_auto_tune.Tokenizer()
    trainer.load_corpus(corpus_iter(), pattern=SPLIT_PATTERN)
    print(f"[genc] corpus cached [{time.time()-t0:.0f}s]", flush=True)

    print(f"[genc] staging {len(shards)} fresh shards ({args.max_chars:,} cap each)...", flush=True)
    for s in shards:
        batches = list(iter_capped_from_path(s, args.max_chars))
        trainer.add_holdout_shard(Path(s).stem, (doc for batch in batches for doc in batch),
                                  pattern=SPLIT_PATTERN)
    trainer.finalize_holdout()
    trainer.set_wordlist(list(words))
    names = trainer.get_holdout_names()

    if args.trajectory:
        points = list(range(args.trajectory, rounds, args.trajectory)) + [rounds]
        sets, labels = [[]], ["baseline"]
        for pt in points:
            ks = replay_kept(pt)
            if ks != sets[-1]:  # skip rounds where nothing changed
                sets.append(ks)
                labels.append(f"r{pt}")
        print(f"[genc] trajectory: evaluating {len(sets)} kept-set prefixes "
              f"(every {args.trajectory} rounds) on {len(names)} fresh shards...", flush=True)
        t0 = time.time()
        res = trainer.evaluate_many(V, [[tuple(k) for k in s] for s in sets],
                                    num_threads=args.threads, forced_pairs=[])
        print(f"[genc] evaluated [{time.time()-t0:.0f}s]", flush=True)
        n = len(names)
        b_base, b_infl, b_ps = res[0]
        b_cov = b_base + b_infl
        b_comp = sum(t for t, _ in b_ps) / n
        b_dead = sum(d for _, d in b_ps) / n
        print(f"\nFRESH-shard trajectory (deltas vs baseline; comp/dead = avg over {n} shards):", flush=True)
        print(f"{'point':>9} {'kept':>5} {'Δcov':>6} {'Δcomp':>8} {'Δdead':>7}", flush=True)
        print("-" * 40, flush=True)
        for lbl, s, (cb, ci, ps) in zip(labels, sets, res):
            cov = cb + ci
            comp = sum(t for t, _ in ps) / n
            dead = sum(d for _, d in ps) / n
            print(f"{lbl:>9} {len(s):>5} {cov - b_cov:>+6} {comp - b_comp:>+8.1f} {dead - b_dead:>+7.1f}", flush=True)
        return

    print(f"[genc] training + measuring baseline (0 pairs) and kept ({len(kept)} pairs)...", flush=True)
    t0 = time.time()
    res = trainer.evaluate_many(V, [[], [tuple(k) for k in kept]],
                                num_threads=args.threads, forced_pairs=[])
    print(f"[genc] evaluated [{time.time()-t0:.0f}s]", flush=True)
    (b_base, b_infl, b_ps), (k_base, k_infl, k_ps) = res

    print(f"\n{'shard':16} {'base_tok':>12} {'kept_tok':>12} {'Δcomp':>8} {'base_dead':>10} {'kept_dead':>10} {'Δdead':>7}", flush=True)
    print("-" * 82, flush=True)
    for i, nm in enumerate(names):
        bt, bd = b_ps[i]
        kt, kd = k_ps[i]
        print(f"{nm:16} {bt:>12,} {kt:>12,} {kt-bt:>+8} {bd:>10} {kd:>10} {kd-bd:>+7}", flush=True)
    print("-" * 82, flush=True)
    n = len(names)
    b_comp, k_comp = sum(t for t, _ in b_ps) / n, sum(t for t, _ in k_ps) / n
    b_dead, k_dead = sum(d for _, d in b_ps) / n, sum(d for _, d in k_ps) / n
    comp_wins = sum(1 for (bt, _), (kt, _) in zip(b_ps, k_ps) if kt < bt)
    dead_wins = sum(1 for (_, bd), (_, kd) in zip(b_ps, k_ps) if kd < bd)
    print(f"coverage (vocab, shard-independent): {b_base + b_infl:,} → {k_base + k_infl:,} "
          f"(Δ{k_base + k_infl - b_base - b_infl:+}; base Δ{k_base - b_base:+} / infl Δ{k_infl - b_infl:+})", flush=True)
    print(f"\nFRESH shards ({n}; kept − baseline, negative = kept helps):", flush=True)
    print(f"  avg compression Δ: {k_comp - b_comp:+.1f}  ({comp_wins}/{n} shards improved)", flush=True)
    print(f"  avg dead Δ:        {k_dead - b_dead:+.1f}  ({dead_wins}/{n} shards improved)", flush=True)
    if gate_base and gate_ref:
        print(f"\nGATE shards (the run's own tally at round {rounds}, for comparison):", flush=True)
        print(f"  avg compression Δ: {gate_ref['avg_comp'] - gate_base['avg_comp']:+.1f}", flush=True)
        print(f"  avg dead Δ:        {gate_ref['avg_dead'] - gate_base['avg_dead']:+.1f}", flush=True)
        print(f"  coverage Δ:        {gate_ref['coverage'] - gate_base['coverage']:+}", flush=True)


if __name__ == "__main__":
    main()
