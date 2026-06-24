"""Held-out-Pareto block-list FIXPOINT — in-process, corpus-cached.

Greedy fixpoint over the candidate pool: recheck every candidate after every keep;
commit the single best (lexicographic: max Δcov, then min Δcomp, then min Δdead);
converge when a full round finds no keep; then a leave-one-out redundancy pass.
Trains via the auto_tune crate's `load_corpus` + `train_from_cached`: the training
corpus is read/tokenized ONCE (~85% of a train's wall-time) and reused for every
candidate, so a per-candidate train drops from ~174s (a subprocess that re-reads
the corpus) to ~22s (merge loop only) + measure — which makes the maximally-
thorough "recheck everything after every keep" fixpoint feasible (~1 day vs ~5).

Everything is in-process and SERIAL: one `Tokenizer` holds the cached corpus and
runs merge loops back-to-back (each already uses all cores via rayon — concurrent
trains would only oversubscribe). `train_from_cached` is proven bit-identical to
the canonical `train_from_iterator` (and to force_merges), so results match the
subprocess path exactly.

    uv run python -m tools.heldout_fixpoint_cached
    uv run python -m tools.heldout_fixpoint_cached --resume
    # plumbing smoke (tiny corpus + few candidates + small held-out):
    uv run python -m tools.heldout_fixpoint_cached --limit 8 --corpus-max-chars 10_000_000 \\
        --max-chars 1_000_000 --state-dir /tmp/fxc_smoke --work-base /tmp/fxc_work
"""
import argparse
import glob
import json
import os
import shutil
import time
from collections import Counter
from pathlib import Path


def main():
    home = os.path.expanduser("~")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--holdout-glob", default=f"{home}/.cache/nanochat/base_data_climbmix_holdout/*.parquet")
    p.add_argument("--tune-shard", default=f"{home}/.cache/nanochat/base_data_climbmix/shard_06542.parquet")
    p.add_argument("--max-chars", type=int, default=10_000_000, help="char cap per held-out shard")
    p.add_argument("--corpus-max-chars", type=int, default=2_000_000_000, help="training-corpus char cap (tok_train default 2B)")
    p.add_argument("--doc-cap", type=int, default=10_000, help="per-doc char cap (tok_train default 10k)")
    p.add_argument("--vocab-size", type=int, default=32768)
    p.add_argument("--fm-context", action=argparse.BooleanOptionalAction, default=False,
                   help="tune blocks INSIDE force_merges' pretokenization: carve-out SPLIT_PATTERN "
                        "+ 122 forced phrases + block_trailing_space, vocab 32890 (32768+122); selects "
                        "blocks optimal for force_merges. Default off = pristine audit.")
    p.add_argument("--work-base", default=f"{home}/.cache/nanochat-variants/auto_tune_fixpoint")
    p.add_argument("--state-dir", default="rustbpe_variants/auto_tune/audit/heldout_fixpoint")
    p.add_argument("--bound-threshold", type=float, default=0.05)
    p.add_argument("--limit", type=int, default=None, help="cap pool size (smoke)")
    p.add_argument("--no-loo", action="store_true")
    p.add_argument("--resume", action="store_true")
    # --- orchestrated mode: a human/Claude judge picks the commit each round ---
    p.add_argument("--propose", action="store_true",
                   help="orchestrated mode: run ONE round (cleanup → cooldown → eval → gate), compute "
                        "net-firing-coverage for the top keeps, print them ranked, then STOP before "
                        "committing — for an external judge to pick. Resumes/inits the checkpoint like a "
                        "normal run; pairs with --commit. Run it in the background (the eval is the slow part).")
    p.add_argument("--commit", default=None, metavar='\'["L","R"]\'',
                   help="orchestrated mode: net-commit the chosen keep into a round already scored by "
                        "--propose, then re-mine. Value is a JSON 2-list (e.g. '[\" ev\",\"ol\"]'), or "
                        "'none' to advance the round with no commit. A net-negative pick is vetoed+jailed "
                        "(then --propose the next round and pick another).")
    p.add_argument("--propose-top-n", type=int, default=12,
                   help="how many top keeps (by the heuristic order) to compute net-firing-cov for + "
                        "present in --propose (each costs one retrain + a held-out fire pass)")
    p.add_argument("--rerun", default=None,
                   help="re-run IN PLACE from a prior round: 'last' (re-do the final round) or an int N "
                        "(roll back to before round N). Discards rounds >= N, re-derives kept from rounds "
                        "1..N-1, re-mines with the CURRENT flags (e.g. a changed --mine-firing-percentile), "
                        "and continues to re-convergence. Backs up the prior checkpoint/ledger to *.prerun.* first.")
    p.add_argument("--inspect-round", type=int, default=None,
                   help="READ-ONLY: reconstruct a past round's reference (keeps 1..N-1), re-mine its "
                        "pool, and show the ACTUAL born/died whole-words + held-out firing counts for "
                        "that round's top gate-passers (by Δcov) — the same view --propose gives, so a "
                        "past pick can be re-judged against its alternatives. Writes inspect_r{N}.md. "
                        "Mutates NO audit state (no checkpoint/ledger/lineage write).")
    p.add_argument("--inspect-top", type=int, default=15,
                   help="how many of the round's gate-passers to retrain for --inspect-round")
    p.add_argument("--pool-override", default=None, metavar='\'[["L","R"],...]\'',
                   help="replace the mined pool with this exact JSON list of pairs for the next round "
                        "(with --rerun/--resume). Commits a KNOWN pick WITHOUT scoring the whole pool: "
                        "`--rerun N --pool-override '[[\"L\",\"R\"]]' --propose` then `--commit '[\"L\",\"R\"]'` "
                        "— scores one candidate (~minutes) instead of the ~18K pool (~a full round). The "
                        "net-commit gate still runs on the pick; round N+1 re-mines the full pool normally.")
    p.add_argument("--bulk-pairs", default=None, metavar='\'[["L","R"],...]\'',
                   help="BULK-HARVEST a large vetted free-win list IN-PROCESS (corpus loaded once): "
                        "eval candidates in parallel chunks vs the current reference, commit the "
                        "gate-passers per chunk with ONE reference-retrain, periodic LOO ejects "
                        "non-earners. ~2h for ~800 vs ~40h one-at-a-time. Journals each commit "
                        "(committed=pair) so --rerun/--trajectory work. Checkpointed/resumable "
                        "(re-pass the same list; already-kept pairs are skipped). Skips per-pair "
                        "lineage/net-commit settle — run a final --loo-words + generalization check.")
    p.add_argument("--bulk-chunk", type=int, default=12, help="candidates per parallel eval chunk")
    p.add_argument("--bulk-loo-every", type=int, default=50, help="run a retention LOO every N commits")
    p.add_argument("--bulk-max-cov", type=int, default=None,
                   help="only commit free wins with Δcov <= this (use 0 for veto-safe pure-compression "
                        "bulk; cov-bearing candidates need individual judgment). Pass 'mine' as "
                        "--bulk-pairs to scan the freshly-mined pool instead of an explicit list.")
    p.add_argument("--loo-words", action="store_true",
                   help="READ-ONLY: for the current kept set, leave-one-out each pair (vocab WITH vs "
                        "WITHOUT it) and show the actual whole-words it marginally contributes/suppresses, "
                        "realness-annotated (tools.word_realness) + its LOO Δcov/comp/dead. Writes "
                        "loo_words.md. Catches keeps that pass the numeric gate but whose words have "
                        "degraded to artifacts/duplicates as the kept set grew. Mutates no audit state.")
    p.add_argument("--threads", type=int, default=10,
                   help="rayon threads for the fused parallel evaluate_many (M4 Pro has 10 P-cores; "
                        "throughput peaks ~10-11 and drops at 14)")
    p.add_argument("--eval-batch", type=int, default=30,
                   help="candidates per evaluate_many call = checkpoint/progress granularity")
    p.add_argument("--gate-dead-tolerance", type=float, default=0.0,
                   help="relax the gate: also KEEP a candidate that improves compression without losing coverage "
                        "(Δcov>=0, Δcomp<0) at up to this dead cost (Δdead<=T) — a comp win may pay only the "
                        "lower-priority dead axis. Such 'costed' keeps are committed only after all free "
                        "(Δdead<=0) keeps. 0 = strict Pareto (default).")
    p.add_argument("--gate-comp-tolerance", type=float, default=0.0,
                   help="relax the gate: also KEEP a candidate that gains coverage (Δcov>0) at up to this "
                        "COMPRESSION cost (Δcomp<=T), within the dead tolerance (Δdead<=--gate-dead-tolerance). "
                        "Buys whole-word coverage at a small compression price (coverage>compression>dead). "
                        "Such costed keeps are committed only after all free (Δcomp<=0,Δdead<=0) keeps. "
                        "0 = no compression tolerance (default).")
    p.add_argument("--cooldown-base", type=int, default=0,
                   help="anti-churn exponential backoff: after its k-th CLEANUP ejection a pair is "
                        "benched (excluded from the pool) for base*2^(k-1) rounds, so chronic "
                        "add/eject cyclers re-audition geometrically less often and fresh candidates "
                        "get the compute. Derived from the rounds journal, so --resume/--rerun replay "
                        "it exactly. On a 0-keep round the earliest-expiring benched pair is released "
                        "early (stall fast-forward) — fixpoint still requires an empty bench. 0 = off.")
    p.add_argument("--net-commit", action=argparse.BooleanOptionalAction, default=True,
                   help="transactional commits: trial-commit the best keep, settle the LOO ejections "
                        "it triggers (cascades included), then judge the SETTLED state vs the "
                        "pre-commit reference with the same gate. Net passes → commit + ejections "
                        "land as one transaction (the flipped pairs are benched). Net fails → revert, "
                        "veto + jail the CANDIDATE instead (same exponential bench; pair with "
                        "--cooldown-base>0), and trial the next-best keep. Per-pair deltas can't "
                        "assign blame for an interaction, so the net transaction is what's gated; "
                        "accepted transactions are strict net improvements, restoring monotonic "
                        "progress (no add/eject limit cycles through commits).")
    p.add_argument("--net-retries", type=int, default=6,
                   help="max net-commit vetoes per round before deferring the remaining keeps to the "
                        "next round (cheap: the reference is unchanged, so the round's scores carry over)")
    # --- adaptive candidate mining (auto-mine): pool is re-derived from the reference
    #     vocab each round, so new mangled tokens that arise enter and ones that vanish drop ---
    p.add_argument("--mine", action=argparse.BooleanOptionalAction, default=True,
                   help="adaptive mining (default, and the only candidate source): mine the reference "
                        "vocab each round — add newly-arisen mangled/low-firing tokens, drop vanished "
                        "ones. --no-mine yields an empty pool (trivial no-op; the old static candidate "
                        "queues were retired in the 2026-06 cleanup).")
    p.add_argument("--mine-bound-threshold", type=float, default=0.15)
    p.add_argument("--mine-sibling", type=int, default=3, help="sibling_min_stem_len for mining")
    p.add_argument("--mine-mode", choices=["suffix", "prefix", "both", "none"], default="suffix",
                   help="affix-based mangled mining: suffix / prefix / both, or 'none' to disable it "
                        "entirely and rely solely on --mine-firing-percentile. (--mine-bound-threshold "
                        "and --mine-sibling feed only the affix detectors, so they are INERT when "
                        "mode=none.)")
    p.add_argument("--mine-all-routes", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--mine-min-firings", type=int, default=1)
    p.add_argument("--mine-max-chars", type=int, default=20_000_000, help="discovery-corpus (val) char cap")
    p.add_argument("--mine-max-pool", type=int, default=0, help="cap pool size (0 = uncapped)")
    p.add_argument("--mine-firing-percentile", type=float, default=1.0,
                   help="ALSO mine every fireable merge token (specials + base bytes excluded) "
                        "whose firing is at or below this percentile of the fireable-merge "
                        "firing distribution, emitting its formation routes — a raw firing cut "
                        "with no morpheme/word filter (the held-out gate is the arbiter). Unions "
                        "with the mangled --mine pool. 0 disables; percentile 1 (the default) "
                        "captures the 0-firing dead floor.")
    p.add_argument("--mine-firing-max-chars", type=int, default=100_000_000,
                   help="corpus size for the firing-percentile fire count (big, so the percentile "
                        "reflects real rarity, not a small-slice artifact)")
    args = p.parse_args()

    import rustbpe_auto_tune
    import tiktoken
    from nanochat.tokenizer import RustBPETokenizer, SPLIT_PATTERN, SPECIAL_TOKENS
    from nanochat.dataset import parquets_iter_batched
    from tools.pass_metrics import whole_word_counts, fire_counts, mergeable_vocab_size, compute_metrics
    from tools.blocked_pairs_report import load_wordlist
    from tools.heldout_buildup import iter_capped_from_path, vocab_str2id
    from tools.find_mangled_morphemes import mine_candidates

    # --- force_merges pretokenization context (opt-in): select blocks where they'll be deployed ---
    if args.fm_context:
        from rustbpe_variants.force_merges.pairs import (
            SPLIT_PATTERN as PATTERN, FORCED_PAIRS as FORCED, BLOCKED_PAIRS as _BLK)
        BTS = True
        BLS = True  # block_leading_space: bans natural multi-word bigrams in the base loop, so no
        #             ' It'+' is' can intercept the injected forced '. It is' (the cannibalization fix).
        # force_merges' OWN explicit blocks (now [] — the predicate subsumes the old inner-bigram list);
        # still prepended as base blocks so the audit tunes ON TOP of the corrected force_merges.
        BASE_BLOCKS = [tuple(b) for b in _BLK]
        if args.vocab_size == 32768:  # default not overridden -> make the 122 forced phrases additive
            args.vocab_size = 32890
        print(f"[fxc] --fm-context: carve-out regex + {len(FORCED)} forced phrases + block_trailing_space "
              f"+ block_leading_space + {len(BASE_BLOCKS)} extra base blocks, vocab={args.vocab_size}", flush=True)
    else:
        PATTERN, FORCED, BTS, BLS, BASE_BLOCKS = SPLIT_PATTERN, [], False, False, []

    # The pool is mined from the reference vocab each round (see --mine); there is no
    # static candidate list anymore. `candidates` stays empty so the residual --no-mine
    # branches below are harmless no-ops (empty pool → trivial convergence).
    candidates = []
    words = load_wordlist()
    if words is None:
        raise SystemExit("no wordlist at /usr/share/dict/words")
    V = args.vocab_size - len(SPECIAL_TOKENS)

    state_dir = Path(args.state_dir); state_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = state_dir / "checkpoint.json"
    ledger_path = state_dir / "ledger.md"
    lineage_path = state_dir / "lineage.jsonl"
    work_base = Path(args.work_base); work_base.mkdir(parents=True, exist_ok=True)
    retain_dir = work_base / "kept_tokenizers" / state_dir.name
    retain_dir.mkdir(parents=True, exist_ok=True)

    # ---- held-out + tune corpora (in-memory, once) ----
    shards = sorted(glob.glob(args.holdout_glob))
    if not shards:
        raise SystemExit(f"no held-out shards match {args.holdout_glob}")
    print(f"[fxc] loading {len(shards)} held-out shards ({args.max_chars:,} cap each)...", flush=True)
    holdout = {Path(s).stem: list(iter_capped_from_path(s, args.max_chars)) for s in shards}
    tune_corpus = list(iter_capped_from_path(args.tune_shard, 20_000_000))

    # ---- load the training corpus ONCE into the cached trainer ----
    def corpus_iter():
        nchars = 0
        for batch in parquets_iter_batched(split="train"):
            for doc in batch:
                d = doc[:args.doc_cap] if len(doc) > args.doc_cap else doc
                nchars += len(d)
                yield d
                if nchars > args.corpus_max_chars:
                    return
    print(f"[fxc] caching training corpus (doc_cap {args.doc_cap:,}, max {args.corpus_max_chars:,})...", flush=True)
    t0 = time.time()
    trainer = rustbpe_auto_tune.Tokenizer()
    trainer.load_corpus(corpus_iter(), pattern=PATTERN)
    print(f"[fxc] corpus cached [{time.time()-t0:.0f}s]", flush=True)

    # Stage held-out + wordlist into the cached trainer so train+measure FUSE into one
    # GIL-released, rayon-parallel pass (evaluate_many): each candidate trains its merge
    # loop and measures coverage/compression/dead in Rust, encoding each held-out chunk
    # once (global dedup). Reuses the in-memory `holdout`/`words` (no re-read).
    for nm, batches in holdout.items():
        trainer.add_holdout_shard(nm, (doc for batch in batches for doc in batch), pattern=PATTERN)
    trainer.finalize_holdout()
    trainer.set_wordlist(list(words))
    holdout_names = trainer.get_holdout_names()
    print(f"[fxc] held-out staged for evaluate_many ({len(holdout_names)} shards, threads={args.threads})", flush=True)

    def build_tok(blocked):
        """Train the merge loop on the cached corpus with `blocked` pairs; return an
        in-process RustBPETokenizer (bit-identical to the canonical wrapper path)."""
        trainer.train_from_cached(V, blocked_pairs=[tuple(b) for b in blocked] + BASE_BLOCKS, forced_pairs=FORCED, block_trailing_space=BTS, block_leading_space=BLS)
        ranks = {bytes(k): v for k, v in trainer.get_mergeable_ranks()}
        offset = len(ranks)
        specials = {name: offset + i for i, name in enumerate(SPECIAL_TOKENS)}
        enc = tiktoken.Encoding(name="auto_tune_fixpoint", pat_str=trainer.get_pattern(),
                                mergeable_ranks=ranks, special_tokens=specials)
        return RustBPETokenizer(enc, "<|bos|>")

    def _eval(block_lists):
        """evaluate_many with the fm-context BASE_BLOCKS (force_merges' inner-bigram
        blocks) prepended to every candidate block set — so the reference and every
        candidate share the corrected force_merges base — plus the carve-out forced/bts
        context. Keeps the trained inputs identical to build_tok's."""
        return trainer.evaluate_many(
            V, [[tuple(p) for p in bl] + BASE_BLOCKS for bl in block_lists],
            num_threads=args.threads, forced_pairs=FORCED, block_trailing_space=BTS, block_leading_space=BLS)

    def measure(tok, with_lineage=False):
        base, infl = whole_word_counts(tok, words)
        n_merge = mergeable_vocab_size(tok)
        per_shard, agg = {}, (Counter() if with_lineage else None)
        for name, corp in holdout.items():
            f = fire_counts(tok, corp)
            if with_lineage:
                agg.update(f)
            per_shard[name] = {"tokens": sum(f.values()),
                               "dead": sum(1 for t in range(n_merge) if f.get(t, 0) == 0)}
        nsh = len(per_shard)
        m = {"coverage": base + infl, "base": base, "infl": infl,
             "avg_comp": sum(s["tokens"] for s in per_shard.values()) / nsh,
             "avg_dead": sum(s["dead"] for s in per_shard.values()) / nsh,
             "per_shard": per_shard}
        if with_lineage:
            return m, agg, vocab_str2id(tok)
        return m

    def metrics_from_eval(base, infl, ps_list):
        """Same shape as measure(), assembled from evaluate_many's per-candidate
        (coverage_base, coverage_inflected, [(tokens, dead) per shard])."""
        per_shard = {holdout_names[i]: {"tokens": t, "dead": d} for i, (t, d) in enumerate(ps_list)}
        nsh = len(per_shard)
        return {"coverage": base + infl, "base": base, "infl": infl,
                "avg_comp": sum(s["tokens"] for s in per_shard.values()) / nsh,
                "avg_dead": sum(s["dead"] for s in per_shard.values()) / nsh,
                "per_shard": per_shard}

    def deltas(ref, cur):
        dc = cur["coverage"] - ref["coverage"]
        dco = round(cur["avg_comp"] - ref["avg_comp"], 1)
        dd = round(cur["avg_dead"] - ref["avg_dead"], 1)
        return dc, dco, dd

    def gate(dc, dco, dd):
        """Held-out Pareto KEEP: no axis regresses (cov>=0, comp<=0, dead<=0) AND at
        least one strictly improves. Subsumes coverage-up keeps AND coverage-flat
        compression/dead wins (e.g. Δcov0 Δcomp-11 Δdead-0.2 — a real win even with no
        new vocab word).

        Optional dead-tolerance relaxation (--gate-dead-tolerance T>0): ALSO keep a
        candidate that improves compression without losing coverage but costs up to T dead
        (Δcov>=0 AND Δcomp<0 AND Δdead<=T) — a negligible-dead-cost compression win,
        coverage-flat included (lexicographic: a comp gain may pay only the lower-priority
        dead axis; requiring Δcov>0 here made Δcov0 comp-winners flap on noise-scale
        Δdead drift). These 'costed' keeps are committed only AFTER the free ones: the
        best-selection key sorts free-before-costed, then by (-Δcov, Δcomp, Δdead). So
        coverage is bought at zero dead cost first; tolerance keeps are a lower tier.

        Optional compression-tolerance relaxation (--gate-comp-tolerance T>0): ALSO keep a
        candidate that gains coverage (Δcov>0) at up to T compression cost, within the dead
        tolerance (Δcov>0 AND Δcomp<=T AND Δdead<=--gate-dead-tolerance) — buys whole-word
        coverage at a small compression price (priority coverage>compression>dead). Any keep
        with Δcomp>0 or Δdead>0 is 'costed' and committed after the free ones.
        Computed from stored deltas, so the gate/tolerance can change and a run can
        resume (or --rerun) onto it without re-training."""
        strict = dc >= 0 and dco <= 0 and dd <= 0 and (dc > 0 or dco < 0 or dd < 0)
        # lexicographic tolerance: each axis's gain may pay costs only on LOWER-priority axes.
        # dead tolerance: a compression win (Δcomp<0, Δcov>=0 — coverage-flat included) may
        # pay up to T_dead dead. Without the Δcov>=0-not->0 form, Δcov0 comp-winners flap:
        # their marginal Δdead drifts ±0.2 (noise-scale) as the set evolves, and a strict
        # dd<=0 retention check recycles pairs the commit gate admitted (r94/104/106 churn).
        relaxed_dead = args.gate_dead_tolerance > 0 and dc >= 0 and dco < 0 and dd <= args.gate_dead_tolerance
        # compression tolerance: only a coverage gain (Δcov>0) may pay up to T_comp compression,
        # within the dead tolerance. Coverage never regresses; nothing pays a coverage price.
        relaxed_comp = (args.gate_comp_tolerance > 0 and dc > 0
                        and dco <= args.gate_comp_tolerance and dd <= args.gate_dead_tolerance)
        return strict or relaxed_dead or relaxed_comp

    def commit(kept_list, n, with_lineage=True):
        """Build + measure the kept set, retain its tokenizer as keep_{n}."""
        tok = build_tok(kept_list)
        # Gate metrics come from evaluate_many — the SAME path candidates use — so comp/dead
        # use identical held-out chunking. Python measure() (live tiktoken encode) and Rust
        # evaluate_many (held-out staged via the Rust split regex) diverge on comp/dead under
        # the carve-out pattern by a constant per-shard offset; mixing ref(Python) with
        # candidates(Rust) sank every Δcomp through the tight comp tolerance → spurious 0
        # keeps. eval-to-eval cancels the offset. (Coverage is vocab-based and agrees, but we
        # take the whole metric dict from eval for consistency.) Python fire/vocab kept for lineage.
        gm = metrics_from_eval(*_eval([kept_list])[0])
        if with_lineage:
            _, agg, voc = measure(tok, with_lineage=True)
            res = (gm, agg, voc)
        else:
            res = gm
        dst = retain_dir / f"keep_{n:04d}"
        if dst.exists():
            shutil.rmtree(dst)
        tok.save(str(dst))
        return tok, res

    def cooldowns():
        """Anti-churn bench (--cooldown-base), derived purely from the rounds journal
        so --resume and --rerun replay it exactly: a pair's k-th jail event (cleanup
        ejection or net-commit veto) benches it for base*2^(k-1) rounds (exponential
        backoff — chronic add/eject cyclers re-audition geometrically less often); a
        'released' journal entry (stall fast-forward on a 0-keep round) ends its bench
        early. Returns {pair: (ejects, benched_until_round)} — compare `until` against
        the NEXT round number (len(rounds)+1), the round any pool built now is
        evaluated in."""
        cd = {}
        if not args.cooldown_base:
            return cd
        for i, r in enumerate(state["rounds"], 1):
            for rm in list(r.get("cleanup", [])) + list(r.get("vetoed", [])):
                k = tuple(rm)
                e = cd.get(k, (0, 0))[0] + 1
                cd[k] = (e, i + args.cooldown_base * (2 ** min(e - 1, 10)))
            for rl in r.get("released", []):
                k = tuple(rl)
                if k in cd:
                    cd[k] = (cd[k][0], i)
        return cd

    def cooling_set():
        nxt = len(state["rounds"]) + 1
        return {k for k, (e, u) in cooldowns().items() if u >= nxt}

    def mine_pool(ref_tok):
        """Adaptive pool: mine the reference tokenizer's CURRENT vocab for mangled
        tokens → sorted (L,R) candidates, dropping any already committed to `kept`
        and any pair benched by the eject-cooldown (capped at --mine-max-pool).
        Re-run each round so the pool tracks the vocab as blocking reallocates
        merge slots — new mangles enter, vanished ones leave."""
        cands = mine_candidates(
            ref_tok, max_chars=args.mine_max_chars, bound_threshold=args.mine_bound_threshold,
            sibling_min_stem_len=args.mine_sibling, min_firings=args.mine_min_firings,
            mode=args.mine_mode, all_routes=args.mine_all_routes,
            firing_percentile=args.mine_firing_percentile, firing_max_chars=args.mine_firing_max_chars)
        skip = {tuple(k) for k in state["kept"]} | cooling_set()
        pool = [c for c in sorted(cands) if c not in skip]
        if args.mine_max_pool and len(pool) > args.mine_max_pool:
            pool = pool[:args.mine_max_pool]
        if args.limit:  # smoke: cap the mined pool
            pool = pool[:args.limit]
        return pool

    def loo_deltas(kept, full_ref):
        """Leave-one-out over the kept set (parallel, via evaluate_many): each keep's
        CURRENT marginal contribution vs the others = (Δcov, Δcomp, Δdead). Returns
        {(L,R): (dc, dco, dd)}. Two consumers, one pass:
          - cleanup recycles EVERY non-earner (fails the gate vs the others), so
            retention matches selection — a kept pair we wouldn't currently commit
            doesn't stay. (Recycling a coverage-positive non-earner is a coverage
            regression, so monotonic termination no longer holds — run without a cap
            by choice, watch the ledger for churn.);
          - the ledger uses ALL of them as the live 'what does this pair contribute
            NOW' figures — the rounds table's deltas are commit-time history and drift."""
        if len(kept) < 2:
            return {}
        loo_sets = [[x for x in kept if x != k] for k in kept]
        try:
            results = _eval(loo_sets)
        except Exception as e:
            print(f"[fxc]   LOO error {e!r} — skipping cleanup/recompute this pass", flush=True)
            return {}
        out = {}
        for k, res in zip(kept, results):
            out[k] = deltas(metrics_from_eval(*res), full_ref)  # full(others+k) vs others = k's marginal
        return out

    def write_lineage(n, pair, tok, new_vocab, new_fire, ref_vocab, ref_fire):
        born = sorted(s for s in new_vocab if s not in ref_vocab)
        died = sorted(s for s in ref_vocab if s not in new_vocab)
        nf = {s: new_fire.get(i_, 0) for s, i_ in new_vocab.items()}
        rf = {s: ref_fire.get(i_, 0) for s, i_ in ref_vocab.items()}
        shifts = {s: nf.get(s, 0) - rf.get(s, 0) for s in set(nf) | set(rf)}
        gainers = sorted(shifts.items(), key=lambda kv: -kv[1])[:15]
        losers = sorted(shifts.items(), key=lambda kv: kv[1])[:15]
        mng = compute_metrics(tok, tune_corpus, count_mangled=True,
                              bound_threshold=args.bound_threshold)["mangled_count"]
        with open(lineage_path, "a") as lf:
            lf.write(json.dumps({"keep": n, "pair": list(pair), "n_born": len(born), "n_died": len(died),
                                 "born": born[:60], "died": died[:60],
                                 "top_gainers": gainers, "top_losers": losers}) + "\n")
        return mng

    def write_ledger(state):
        kept = state["kept"]
        ref = state.get("ref") or {}
        bl = state.get("baseline") or {}
        nc = sum(1 for r in state["rounds"] if r.get("committed"))
        ncl = sum(1 for r in state["rounds"] if r.get("cleanup"))
        nrec = sum(len(r.get("cleanup", [])) for r in state["rounds"])
        nv = sum(len(r.get("vetoed", [])) for r in state["rounds"])
        def _vs(k, intfmt=False):
            if k not in ref or k not in bl:
                return "?"
            d = ref[k] - bl[k]
            return f"{d:+d}" if intfmt else f"{d:+.1f}"
        ac, ad = ref.get("avg_comp"), ref.get("avg_dead")
        if args.mine:
            _affix = "" if args.mine_mode == "none" else (
                f"{args.mine_mode}, thr {args.mine_bound_threshold}, sib {args.mine_sibling}, "
                f"all-routes {args.mine_all_routes}")
            _fpct = f"firing-pct {args.mine_firing_percentile}" if args.mine_firing_percentile > 0 else ""
            cand_desc = "adaptive mine (" + ", ".join(x for x in (_affix, _fpct) if x) + ")"
        else:
            cand_desc = str(len(candidates))
        L = ["# auto_tune held-out-Pareto FIXPOINT ledger (corpus-cached)", "",
             f"- held-out shards: {len(holdout)} ({args.max_chars:,} chars each), avg-gated",
             f"- gate: held-out Pareto — cov≥0 AND comp≤0 AND dead≤0 AND ≥1 strict improvement "
             f"(coverage-up OR coverage-flat comp/dead win)"
             + (f" + dead-tolerance {args.gate_dead_tolerance} (Δcov≥0,Δcomp<0,Δdead≤{args.gate_dead_tolerance}, committed after free keeps)"
                if args.gate_dead_tolerance > 0 else "")
             + (f" + comp-tolerance {args.gate_comp_tolerance} (Δcov>0,Δcomp≤{args.gate_comp_tolerance},Δdead≤{args.gate_dead_tolerance}, committed after free keeps)"
                if args.gate_comp_tolerance > 0 else "")
             + "; coverage prioritized; recheck ALL + recycle non-earners every round"
             + (f"; eject-cooldown base {args.cooldown_base} (k-th jail event benches a pair "
                f"{args.cooldown_base}*2^(k-1) rounds; early release on a 0-keep round)"
                if args.cooldown_base else "")
             + ("; transactional net-commit (trial → settle ejections → net gate; "
                "net-negative trials vetoed + jailed)" if args.net_commit else ""),
             f"- candidates: {cand_desc}",
             "",
             "## running tally",
             f"- status **{state['status']}** | rounds {len(state['rounds'])} (commits {nc}, cleanups {ncl}, "
             f"pairs recycled {nrec}, vetoes {nv}) | pool {len(state['pool'])} | kept **{len(kept)}**",
             f"- coverage **{ref.get('coverage', '?')}** (base {ref.get('base', '?')} / infl {ref.get('infl', '?')})  ·  Δ vs baseline {_vs('coverage', True)}",
             (f"- compression avg/shard **{ac:.0f}**  ·  Δ vs baseline {_vs('avg_comp')}" if ac is not None else "- compression: (pending)"),
             (f"- dead avg/shard **{ad:.1f}**  ·  Δ vs baseline {_vs('avg_dead')}" if ad is not None else "- dead: (pending)"),
             "",
             "## rounds (one commit each; pool re-evaluated vs the new reference)",
             "| round | pool | keeps | committed | Δcov | Δcomp | Δdead |",
             "|---|---|---|---|---|---|---|"]
        for r in state["rounds"]:
            c, cd = r.get("committed"), r.get("committed_deltas") or ["-", "-", "-"]
            if r.get("cleanup") and not c:
                cs = f"— cleanup −{len(r['cleanup'])} recycled"
            elif r.get("released"):
                rl = r["released"][0]
                cs = f"— 0 keeps → released `{rl[0]!r}+{rl[1]!r}` from cooldown"
            elif c:
                cs = f"`{c[0]!r}+{c[1]!r}`"
                notes = []
                if r.get("cleanup"):
                    nd = r.get("net_deltas") or ["?", "?", "?"]
                    notes.append(f"−{len(r['cleanup'])} ejected, net {nd[0]}/{nd[1]}/{nd[2]}")
                if r.get("vetoed"):
                    vp = ", ".join(f"{v[0]}+{v[1]}" for v in r["vetoed"][:6])
                    notes.append(f"vetoed {vp}" + ("…" if len(r["vetoed"]) > 6 else ""))
                if notes:
                    cs += " — " + "; ".join(notes)
            elif r.get("vetoed"):
                vp = ", ".join(f"{v[0]}+{v[1]}" for v in r["vetoed"][:8])
                cs = (f"— no commit: vetoed {vp}" + ("…" if len(r["vetoed"]) > 8 else "")
                      + " (net-negative)")
            else:
                cs = "— (fixpoint)"
            L.append(f"| {r['round']} | {r['n_pool']} | {r['n_keeps']} | {cs} | {cd[0]} | {cd[1]} | {cd[2]} |")
        cur = state.get("cur_round")
        if cur and cur.get("done"):
            done = cur["done"]
            ks = sorted(((k, v["d"]) for k, v in done.items() if gate(v["d"][0], v["d"][1], v["d"][2])),
                        key=lambda kv: (-kv[1][0], kv[1][1], kv[1][2]))
            L += ["", f"## round {cur['round']} IN PROGRESS — {len(done)}/{len(state['pool'])} evaluated, "
                  f"{len(ks)} keep(s) so far (best committed at round end)"]
            if ks:
                L += ["| pair | Δcov | Δcomp | Δdead |", "|---|---|---|---|"]
                L += [f"| `{k!r}` | {d[0]} | {d[1]} | {d[2]} |" for k, d in ks]
        kd = state.get("kept_deltas") or {}
        L += ["", f"## kept ({len(kept)}) — LIVE contribution (LOO vs the rest, recomputed every round; "
              "rounds table above is commit-time history and drifts). Promote these into BLOCKED_PAIRS.",
              "| pair | Δcov | Δcomp | Δdead | earns now? |", "|---|---|---|---|---|"]
        for p in kept:
            d = kd.get(f"{p[0]}+{p[1]}")
            if d is None:
                L.append(f"| `{p[0]!r}+{p[1]!r}` | - | - | - | - |")
            else:
                L.append(f"| `{p[0]!r}+{p[1]!r}` | {d[0]} | {d[1]} | {d[2]} | "
                         f"{'yes' if gate(d[0], d[1], d[2]) else '**NO**'} |")
        if args.cooldown_base:
            nxt = len(state["rounds"]) + 1
            kept_now = {tuple(k) for k in kept}
            bench = sorted([(u, e, k) for k, (e, u) in cooldowns().items()
                            if u >= nxt and k not in kept_now])
            if bench:
                L += ["", f"## cooling ({len(bench)}) — benched by the eject-cooldown (exponential "
                      "backoff; re-enters the pool when the bench expires, or earlier via a "
                      "0-keep-round release)",
                      "| pair | ejects | benched until round |", "|---|---|---|"]
                L += [f"| `{k[0]!r}+{k[1]!r}` | {e} | {u} |" for u, e, k in bench]
        if state.get("loo"):
            L += ["", "## leave-one-out (does each keep still earn its slot vs the others?)",
                  "| pair | Δcov | Δcomp | Δdead | earns? |", "|---|---|---|---|---|"]
            for e in state["loo"]:
                L.append(f"| `{e['pair'][0]!r}+{e['pair'][1]!r}` | {e['d_cov']} | {e['d_comp']} | "
                         f"{e['d_dead']} | {'yes' if e['earns'] else '**REDUNDANT**'} |")
            red = sum(1 for e in state["loo"] if not e["earns"])
            L += ["", f"redundant keeps flagged: {red}"]
        ledger_path.write_text("\n".join(L) + "\n")

    def present_round(keeps_order, ref_fire, ref_vocab, top_n, out_name="propose.md"):
        """Orchestrated --propose: for the top-N keeps (heuristic order), retrain each,
        diff its vocab vs the reference, and report NET firing whole-words added
        (born-WW-that-fire − died-WW-that-fired) + the surface token each block prevents,
        so a judge can rank by firing-coverage AND veto blocks that would shatter a useful
        token. Pure presentation: nothing here is persisted (only the committed pick is)."""
        from tools.blocked_pairs_report import _is_word
        from tools.word_realness import standalone_counts, annotate, ambient_surfaces

        def is_ww(s):
            if not s.startswith(" "):
                return False
            w = s[1:].lower()
            return bool(w) and (w in words or _is_word(w, words))

        rows = []
        for (cand, dcv, dco, ddd, _kp, _ps) in keeps_order[:top_n]:
            tok = build_tok(list(state["kept"]) + [cand])
            new_vocab = vocab_str2id(tok)
            new_fire = Counter()
            for corp in holdout.values():
                new_fire.update(fire_counts(tok, corp))
            born_ww = [s for s in new_vocab if s not in ref_vocab and is_ww(s)]
            died_ww = [s for s in ref_vocab if s not in new_vocab and is_ww(s)]
            born_counts = sorted(((s, new_fire.get(new_vocab[s], 0)) for s in born_ww), key=lambda x: -x[1])
            died_counts = sorted(((s, ref_fire.get(ref_vocab[s], 0)) for s in died_ww), key=lambda x: -x[1])
            born_fire = [s for s, c in born_counts if c > 0]
            died_fire = [s for s, c in died_counts if c > 0]
            surface = cand[0] + cand[1]
            sfire = ref_fire.get(ref_vocab.get(surface), 0)
            rows.append({"cand": cand, "surface": surface, "sfire": sfire,
                         "net_fc": len(born_fire) - len(died_fire), "dcv": dcv, "dco": dco, "ddd": ddd,
                         "born_counts": born_counts, "died_counts": died_counts})
        rows.sort(key=lambda r: (-r["net_fc"], -r["dcv"], r["dco"], r["ddd"]))

        # realness lens (tools.word_realness): standalone-word counts on held-out (one pass) +
        # ambient slot-fillers, so each firing count reads as REAL coverage vs reroute ARTIFACT
        # vs AMBIENT. DATA for the per-round hand judgment — it does NOT re-rank or pick.
        shown = {s for r in rows for s, _ in (r["born_counts"] + r["died_counts"])}
        ho_docs = [doc for corp in holdout.values() for batch in corp for doc in batch]  # holdout = list of batches
        stand_map, _ = standalone_counts(shown, ho_docs)
        ambient = ambient_surfaces([[s for s, _ in r["born_counts"]] for r in rows])

        def fmt_ww(counts):
            # each WW as fire[sStandalone·DICT/INFL·flags]: DICT w/ standalone usage = real
            # coverage; INFL/ART? (fires but ~0 standalone) = reroute fragment; AMBIENT = a
            # slot-filler any block gets. JUDGE realness + firing BY HAND, never the Δfc headline.
            return "  ".join(annotate(s, c, words, stand_map, ambient) for s, c in counts[:12]) or "—"
        out = [f"# round {len(state['rounds']) + 1} — AWAITING COMMIT  (kept so far: {len(state['kept'])})",
               f"# JUDGE the actual added words BY HAND (realness + standalone usage + firing), NOT the Δfc headline.",
               f"# WW annotated fire[sStandalone·dict·flags] — DICT+standalone=coverage · INFL/ART?=reroute fragment · AMBIENT=slot-filler",
               f"{'#':>2}  {'(L, R) blocked':<26} {'prevents':<16} {'fS':>5} {'Δfc':>4} {'Δcov':>5} {'Δcomp':>7} {'Δdead':>6}"]
        for i, r in enumerate(rows, 1):
            out.append(f"{i:>2}  {repr(r['cand']):<26} {repr(r['surface'])[1:-1]:<16} {r['sfire']:>5} "
                       f"{r['net_fc']:>+4} {r['dcv']:>+5} {r['dco']:>+7.1f} {r['ddd']:>+6.1f}   "
                       f"commit={json.dumps(list(r['cand']))}")
            out.append(f"      adds WW: {fmt_ww(r['born_counts'])}")
            if r["died_counts"]:
                out.append(f"      loses WW: {fmt_ww(r['died_counts'])}")
        out.append("# commit:  --commit '[\"L\",\"R\"]'  (use the exact JSON above)   |   skip:  --commit none")
        text = "\n".join(out)
        print("\n[propose]\n" + text, flush=True)
        (state_dir / out_name).write_text(text + "\n")  # clean, Read-able (no grep needed)
        return rows

    # ---- READ-ONLY: inspect a past round's alternatives (mutates no audit state) ----
    if args.inspect_round is not None:
        if not ckpt_path.exists():
            raise SystemExit(f"--inspect-round needs an existing checkpoint at {ckpt_path}")
        prev = json.loads(ckpt_path.read_text())
        all_rounds = prev.get("rounds", [])
        N = args.inspect_round
        rec = next((r for r in all_rounds if r["round"] == N), None)
        if rec is None:
            raise SystemExit(f"round {N} not in checkpoint ({len(all_rounds)} rounds)")
        # reconstruct kept = rounds 1..N-1 (replay commits + cleanups), same as --rerun
        kept = []
        for r in all_rounds[:N - 1]:
            if r.get("committed"):
                kept.append(tuple(r["committed"]))
            for rm in r.get("cleanup", []):
                t = tuple(rm)
                if t in kept:
                    kept.remove(t)
        state = {"kept": kept, "rounds": all_rounds[:N - 1]}  # for cooling_set() + present_round header
        print(f"[fxc] INSPECT round {N}: reference = {len(kept)} keeps (rounds 1..{N-1}); "
              f"round {N} actually committed {rec.get('committed')!r}", flush=True)
        ref_tok = build_tok(kept)                                  # build_tok + measure, NOT commit
        _, ref_fire, ref_vocab = measure(ref_tok, with_lineage=True)   # (skips commit's tokenizer-save)
        # re-mine the pool at this reference to recover candidate (L,R) tuples (verdict keys are
        # the ambiguous "L+R" join — re-mining gives exact tuples and reproduces round N's pool)
        pool = mine_pool(ref_tok)
        by_key = {f"{c[0]}+{c[1]}": tuple(c) for c in pool}
        verd = rec.get("verdicts", {})
        gp = [(k, d) for k, d in verd.items() if len(d) >= 4 and d[3]]  # gate-passers
        gp.sort(key=lambda kd: (-kd[1][0], kd[1][1]))                   # Δcov desc, then Δcomp asc
        sel = [(by_key[k], d) for k, d in gp if k in by_key][:args.inspect_top]
        missing = [k for k, d in gp if k not in by_key]
        if missing:
            print(f"[fxc]   note: {len(missing)} gate-passer(s) absent from re-mined pool "
                  f"(cooldown/flag drift), skipped: {missing[:8]}", flush=True)
        com = rec.get("committed")                                     # always include the actual pick
        if com:
            ckey = f"{com[0]}+{com[1]}"
            if ckey in by_key and not any(f"{c[0]}+{c[1]}" == ckey for c, _ in sel) and ckey in verd:
                sel.append((by_key[ckey], verd[ckey]))
        keeps_order = [(c, d[0], d[1], d[2], None, None) for c, d in sel]
        present_round(keeps_order, ref_fire, ref_vocab, len(keeps_order), out_name=f"inspect_r{N}.md")
        print(f"[fxc] INSPECT done ({len(keeps_order)} candidates retrained) — "
              f"committed pick was {com!r}; see {state_dir / f'inspect_r{N}.md'}", flush=True)
        return

    # ---- READ-ONLY: word-level leave-one-out of the current kept set (mutates no state) ----
    if args.loo_words:
        if not ckpt_path.exists():
            raise SystemExit(f"--loo-words needs an existing checkpoint at {ckpt_path}")
        from tools.blocked_pairs_report import _is_word
        from tools.word_realness import standalone_counts, annotate, ambient_surfaces
        prev = json.loads(ckpt_path.read_text())
        kept = [tuple(p) for p in prev["kept"]]
        if not kept:
            raise SystemExit("kept set is empty — nothing to LOO")
        state = {"kept": kept, "rounds": prev.get("rounds", [])}

        def is_ww(s):
            if not s.startswith(" "):
                return False
            w = s[1:].lower()
            return bool(w) and (w in words or _is_word(w, words))

        def fire_of(tok):
            f = Counter()
            for corp in holdout.values():
                f.update(fire_counts(tok, corp))
            return f

        print(f"[fxc] LOO-WORDS over {len(kept)} kept pairs (build_tok + eval each)...", flush=True)
        full_tok = build_tok(kept)
        full_vocab, full_fire = vocab_str2id(full_tok), fire_of(full_tok)
        ref_full = metrics_from_eval(*_eval([kept])[0])
        rows = []
        for k in kept:
            rest = [x for x in kept if x != k]
            minus_tok = build_tok(rest)
            minus_vocab, minus_fire = vocab_str2id(minus_tok), fire_of(minus_tok)
            born = sorted(((s, full_fire.get(full_vocab[s], 0)) for s in full_vocab          # k's UNIQUE WW
                           if s not in minus_vocab and is_ww(s)), key=lambda x: -x[1])
            died = sorted(((s, minus_fire.get(minus_vocab[s], 0)) for s in minus_vocab        # restored WITHOUT k
                           if s not in full_vocab and is_ww(s)), key=lambda x: -x[1])
            dc, dco, dd = deltas(metrics_from_eval(*_eval([rest])[0]), ref_full)              # k's marginal Δ (cov+ = adds)
            rows.append({"pair": k, "born": born, "died": died, "dcov": dc, "dcomp": dco,
                         "ddead": dd, "earns": gate(dc, dco, dd)})

        # realness annotations: standalone on held-out (one pass) + ambient across pairs' born sets
        shown = {s for r in rows for s, _ in (r["born"] + r["died"])}
        ho_docs = [doc for corp in holdout.values() for batch in corp for doc in batch]
        stand_map, _ = standalone_counts(shown, ho_docs)
        shared = ambient_surfaces([[s for s, _ in r["born"]] for r in rows], min_n=2)  # co-held by >=2 keeps

        def fmt(counts):
            return "  ".join(annotate(s, c, words, stand_map, shared, ambient_tag="SHARED") for s, c in counts[:12]) or "—"
        rows.sort(key=lambda r: (r["earns"], sum(1 for _, c in r["born"] if c > 0)))  # weakest first
        out = [f"# LOO-WORDS — current kept set ({len(kept)} pairs); each pair's MARGINAL whole-words",
               f"# provides = WW present ONLY with the pair · restores = WW that reappear WITHOUT it",
               f"# fire[sStandalone·dict·flags]: DICT+standalone=real coverage · INFL/ART?=reroute fragment",
               f"#   SHARED=co-held by >=2 keeps (removing any one drops it) — real but slot-marginal, NOT low-value",
               f"# POP (hand call, like commits) a pair whose UNIQUE 'provides' is now only artifacts/duplicates"]
        for r in rows:
            mark = "" if r["earns"] else "   <== fails numeric gate"
            out.append(f"\n{r['pair'][0]!r}+{r['pair'][1]!r}  LOO Δcov{r['dcov']:+} Δcomp{r['dcomp']:+.1f} "
                       f"Δdead{r['ddead']:+.1f}{mark}")
            out.append(f"   provides: {fmt(r['born'])}")
            if r["died"]:
                out.append(f"   restores: {fmt(r['died'])}")
        text = "\n".join(out)
        print("\n[loo-words]\n" + text, flush=True)
        (state_dir / "loo_words.md").write_text(text + "\n")
        print(f"[fxc] LOO-WORDS done — see {state_dir / 'loo_words.md'}", flush=True)
        return

    # ---- BULK-HARVEST a vetted free-win list in-process (chunked eval + commit) ----
    if args.bulk_pairs is not None:
        if not ckpt_path.exists():
            raise SystemExit("--bulk-pairs needs an existing checkpoint")
        state = json.loads(ckpt_path.read_text())
        state["kept"] = [tuple(p) for p in state["kept"]]
        state["loo"] = None
        keptset = set(state["kept"])
        if args.bulk_pairs.strip() == "mine":   # scan the freshly-mined pool (the free-win list is
            cands = mine_pool(build_tok(state["kept"]))   # rediscovered, not re-typed)
        else:
            cands = [tuple(p) for p in json.loads(args.bulk_pairs)]
        pairs = [tuple(p) for p in cands if tuple(p) not in keptset]
        rnd = len(state["rounds"])
        state["ref"] = metrics_from_eval(*_eval([state["kept"]])[0])
        print(f"[bulk] kept={len(state['kept'])} (round {rnd}); {len(pairs)} candidate(s) to grind "
              f"(chunk={args.bulk_chunk}, loo every {args.bulk_loo_every})", flush=True)

        def bulk_ref_retrain():
            tok = build_tok(state["kept"])
            state["ref"] = metrics_from_eval(*_eval([state["kept"]])[0])
            tok.save(str(retain_dir / f"keep_{len(state['kept']):04d}"))

        def bulk_loo():
            if len(state["kept"]) < 2:
                return 0
            ld = loo_deltas(state["kept"], state["ref"])
            ne = [k for k in state["kept"] if k in ld and not gate(*ld[k])]
            if ne:
                nes = set(ne)
                nonlocal rnd
                rnd += 1
                state["kept"] = [k for k in state["kept"] if k not in nes]
                state["rounds"].append({"round": rnd, "n_pool": 0, "n_keeps": 0, "committed": None,
                                        "committed_deltas": None, "verdicts": {},
                                        "cleanup": [list(k) for k in ne], "bulk": True})
                state["kept_deltas"] = {f"{a}+{b}": list(v) for (a, b), v in ld.items() if (a, b) not in nes}
                bulk_ref_retrain()
                print(f"[bulk]   LOO ejected {len(ne)}: {[f'{a}+{b}' for a,b in ne][:8]}", flush=True)
            else:
                state["kept_deltas"] = {f"{a}+{b}": list(v) for (a, b), v in ld.items()}
            return len(ne)

        t0 = time.time()
        committed_total, since_loo = 0, 0
        for ci in range(0, len(pairs), args.bulk_chunk):
            chunk = pairs[ci:ci + args.bulk_chunk]
            try:
                results = _eval([list(state["kept"]) + [c] for c in chunk])
            except Exception as e:
                print(f"[bulk]   chunk eval error {e!r} — per-candidate retry", flush=True)
                results = []
                for c in chunk:
                    try:
                        results.append(_eval([list(state["kept"]) + [c]])[0])
                    except Exception:
                        results.append(None)
            passers = []
            for c, res in zip(chunk, results):
                if res is None:
                    continue
                dc, dco, dd = deltas(state["ref"], metrics_from_eval(*res))
                # --bulk-max-cov keeps the bulk grind veto-safe: only PURE-compression free wins
                # (cov<=0) can't be a shatter-fragment-as-coverage (ob+by-style), so cov-bearing
                # candidates are deferred to individual propose-board judgment.
                if gate(dc, dco, dd) and (args.bulk_max_cov is None or dc <= args.bulk_max_cov):
                    passers.append((c, dc, dco, dd))
            passers.sort(key=lambda x: x[2])  # most-compressive first
            nc = 0
            for c, *_ in passers:
                # RE-EVAL vs the LIVE ref (advanced by earlier commits this chunk). The pre-filter
                # above measured each candidate vs the PRE-CHUNK ref, but cov=0-alone blocks INTERACT
                # and can shatter whole-words together (the bug that lost cov +45→+7). Re-validating
                # each vs the current state restores telescoping: every commit is cov>=0 vs the live
                # ref, so cumulative coverage cannot decrease. The re-eval's metrics ARE the new ref.
                m = metrics_from_eval(*_eval([list(state["kept"]) + [c]])[0])
                dc, dco, dd = deltas(state["ref"], m)
                if not (gate(dc, dco, dd) and (args.bulk_max_cov is None or dc <= args.bulk_max_cov)):
                    continue
                rnd += 1
                state["kept"].append(c)
                state["ref"] = m
                state["rounds"].append({"round": rnd, "n_pool": len(pairs), "n_keeps": 1,
                                        "committed": list(c), "committed_deltas": [dc, dco, dd],
                                        "verdicts": {}, "bulk": True})
                committed_total += 1; since_loo += 1; nc += 1
            if nc:
                build_tok(state["kept"]).save(str(retain_dir / f"keep_{len(state['kept']):04d}"))
            done = min(ci + args.bulk_chunk, len(pairs))
            print(f"[bulk] {done}/{len(pairs)} scanned, +{nc} (kept={len(state['kept'])}) "
                  f"covΔ{state['ref']['coverage']-state['baseline']['coverage']:+} [{time.time()-t0:.0f}s]", flush=True)
            ckpt_path.write_text(json.dumps(state)); write_ledger(state)
            if since_loo >= args.bulk_loo_every:
                bulk_loo()
                since_loo = 0
                ckpt_path.write_text(json.dumps(state)); write_ledger(state)
        n_ej = bulk_loo()  # final retention pass
        ckpt_path.write_text(json.dumps(state)); write_ledger(state)
        print(f"[bulk] DONE: committed {committed_total}, final LOO ejected {n_ej}, "
              f"kept={len(state['kept'])} [{time.time()-t0:.0f}s]", flush=True)
        return

    # ---- rerun (in place) / resume / init ----
    if args.rerun:
        if not ckpt_path.exists():
            raise SystemExit(f"--rerun needs an existing checkpoint at {ckpt_path}")
        prev = json.loads(ckpt_path.read_text())
        rounds = prev.get("rounds", [])
        N = len(rounds) if args.rerun == "last" else int(args.rerun)
        shutil.copy(ckpt_path, state_dir / "checkpoint.prerun.json")     # in-place rerun overwrites — keep a backup
        if ledger_path.exists():
            shutil.copy(ledger_path, state_dir / "ledger.prerun.md")
        keep_rounds = rounds[:max(0, N - 1)]                              # rounds 1..N-1 survive
        kept = []                                                         # replay commits + cleanups
        for r in keep_rounds:
            if r.get("committed"):
                kept.append(tuple(r["committed"]))
            for rm in r.get("cleanup", []):
                t = tuple(rm)
                if t in kept:
                    kept.remove(t)
        state = {"kept": kept, "pool": [] if args.mine else list(candidates), "ref": None,
                 "rounds": keep_rounds, "loo": None, "status": "running", "cur_round": None,
                 "n_candidates": len(candidates)}
        ref_fire, ref_vocab = Counter(), {}
        print(f"[fxc] RERUN from round {N}: rolled back to {len(kept)} kept (rounds 1..{N-1}); "
              f"re-mining with current flags (backup: checkpoint.prerun.json)", flush=True)
    elif (args.resume or args.commit is not None or args.propose) and ckpt_path.exists():
        state = json.loads(ckpt_path.read_text())
        state["kept"] = [tuple(p) for p in state["kept"]]
        state["pool"] = [tuple(p) for p in state["pool"]]
        print(f"[fxc] resumed: round {len(state['rounds'])}, kept={len(state['kept'])}, pool={len(state['pool'])}, status={state['status']}", flush=True)
        ref_tok = build_tok(state["kept"])
        state["ref"] = metrics_from_eval(*_eval([state["kept"]])[0])   # gate metric from eval (match candidates)
        _, ref_fire, ref_vocab = measure(ref_tok, with_lineage=True)   # Python fire/vocab for lineage only
        # A plain --propose (no --pool-override) onto an EMPTY/stunted pool means the persisted
        # pool was left small by prior --pool-override commits (each overwrites state["pool"], and
        # a final veto empties it). Re-mine the full pool for a real board instead of scanning the
        # leftover; reset the stale cur_round so its scores don't poison the fresh round.
        if args.propose and args.pool_override is None and args.mine and not state["pool"]:
            state["pool"] = mine_pool(ref_tok)
            state["cur_round"] = None
            print(f"[fxc] --propose onto empty pool → re-mined {len(state['pool'])} candidates", flush=True)
    else:
        state = {"kept": [], "pool": list(candidates), "ref": None, "rounds": [],
                 "loo": None, "status": "running", "cur_round": None, "n_candidates": len(candidates)}
        ref_fire, ref_vocab = Counter(), {}

    # ---- reference (kept set so far: empty for a fresh run, N-1 keeps for a rerun) ----
    if state["ref"] is None:
        n0 = len(state["kept"])
        print(f"[fxc] training reference (kept={n0})...", flush=True)
        ref_tok, (state["ref"], ref_fire, ref_vocab) = commit(state["kept"], n0, with_lineage=True)
        if args.mine:
            state["pool"] = mine_pool(ref_tok)
            print(f"[fxc] initial mined pool: {len(state['pool'])} candidates "
                  f"(thr={args.mine_bound_threshold} sib={args.mine_sibling} "
                  f"mode={args.mine_mode} all_routes={args.mine_all_routes})", flush=True)
        ckpt_path.write_text(json.dumps(state))
        r = state["ref"]
        print(f"[fxc] reference: cov={r['coverage']} (base {r['base']}/infl {r['infl']}) "
              f"avg_comp={r['avg_comp']:.0f} avg_dead={r['avg_dead']:.1f}", flush=True)

    # pool override (commit a known pick without scoring the whole pool): replace the mined
    # pool with the given pair(s) so the round scores just them. Persisted so the follow-up
    # --commit (a separate invocation) resumes onto the single-candidate pool.
    if args.pool_override is not None:
        state["pool"] = [tuple(p) for p in json.loads(args.pool_override)]
        print(f"[fxc] pool OVERRIDE: {len(state['pool'])} candidate(s) "
              f"{[list(c) for c in state['pool']]} — skips full-pool scoring this round", flush=True)
        ckpt_path.write_text(json.dumps(state))

    # baseline (kept=0) metrics, once — drives the running-tally's Δ-vs-baseline
    if "baseline" not in state:
        # eval-path (same as commit/candidates) so the ledger's Δ-vs-baseline isn't offset
        bm = metrics_from_eval(*_eval([[]])[0])
        state["baseline"] = {k: bm[k] for k in ("coverage", "base", "infl", "avg_comp", "avg_dead")}
        print(f"[fxc] baseline (kept=0): cov={bm['coverage']} avg_comp={bm['avg_comp']:.0f} avg_dead={bm['avg_dead']:.1f}", flush=True)
        ckpt_path.write_text(json.dumps(state))

    # --- orchestrated --commit needs a round already scored by --propose, EXCEPT when
    # --pool-override supplies the candidates: then the loop scores that small pool itself,
    # so `--pool-override '[["L","R"]]' --commit '["L","R"]'` commits in one invocation. ---
    if args.commit is not None and args.commit.strip().lower() != "none" and args.pool_override is None:
        cr = state.get("cur_round")
        if not cr or not cr.get("done"):
            raise SystemExit("--commit needs a round already scored by --propose "
                             "(no cur_round.done in the checkpoint). Run --propose first.")

    # ---- fixpoint loop ----
    while state["status"] == "running" and state["pool"]:
        rnd = len(state["rounds"]) + 1
        # The leave-one-out redundancy snapshot (state["loo"]) is only valid for the kept set
        # it was computed on (at a fixpoint). Any new round means the set may change, so drop a
        # stale snapshot now — it's recomputed by the post-loop pass when we actually converge.
        # (Guards against a snapshot outliving its set, e.g. a false fixpoint that was repaired.)
        if state.get("loo") is not None:
            state["loo"] = None
        # Resume a partially-evaluated round if present (the reference is fixed
        # within a round, so already-scored candidates are still valid).
        cur = state.get("cur_round")
        if not cur or cur.get("round") != rnd:
            cur = {"round": rnd, "done": {}}
            state["cur_round"] = cur
        # --- eager non-earner cleanup, top of every fresh round ---
        # Recompute each kept pair's CURRENT contribution (LOO) and recycle EVERY pair
        # that no longer earns its slot (fails the gate vs the others) BEFORE evaluating
        # this round — not just coverage-harmful ones. This makes RETENTION match
        # SELECTION: a kept pair we wouldn't currently commit doesn't get to stay. The
        # coverage a recycled pair provided only at a compression/dead cost is either
        # forgone (strict Pareto) or re-found more cleanly via re-mining/slosh.
        # NOTE: recycling a coverage-POSITIVE non-earner is a coverage regression, so the
        # monotonic-termination guarantee no longer holds. Running WITHOUT a cap on
        # purpose — if churn happens it'll show in the ledger (rounds cycling the same
        # pairs); add a dry-round cap only if we actually see it.
        # The same LOO refreshes state["kept_deltas"] (the ledger's live-contribution view).
        if not cur["done"] and not args.no_loo and len(state["kept"]) > 1:
            ld = loo_deltas(state["kept"], state["ref"])
            if ld:
                state["kept_deltas"] = {f"{a}+{b}": list(v) for (a, b), v in ld.items()}
            nonearners = [k for k in state["kept"] if k in ld and not gate(ld[k][0], ld[k][1], ld[k][2])]
            if nonearners:
                ne = set(nonearners)
                n_pool_pre = len(state["pool"])
                state["kept"] = [k for k in state["kept"] if k not in ne]
                # journal the ejections BEFORE rebuilding the pool — the cooldown bench
                # is derived from the journal, so this is what benches the ejected pairs
                state["rounds"].append({"round": rnd, "n_pool": n_pool_pre, "n_keeps": 0,
                                        "committed": None, "committed_deltas": None, "verdicts": {},
                                        "cleanup": [list(k) for k in nonearners]})
                tok = build_tok(state["kept"])
                state["ref"] = metrics_from_eval(*_eval([state["kept"]])[0])   # gate metric from eval (match candidates)
                _, ref_fire, ref_vocab = measure(tok, with_lineage=True)       # Python fire/vocab for lineage only
                cooling = cooling_set()
                state["pool"] = mine_pool(tok) if args.mine else \
                    [c for c in candidates if c not in set(state["kept"]) and c not in cooling]
                for k in ne:
                    if k not in state["pool"] and k not in cooling:
                        state["pool"].append(k)  # recycle: re-judge it as a candidate
                state["cur_round"] = None
                shown = ', '.join(f'{a}+{b} (Δcov{ld[(a, b)][0]:+},Δcomp{ld[(a, b)][1]:+},Δdead{ld[(a, b)][2]:+})'
                                  for a, b in nonearners[:8])
                print(f"[fxc] round {rnd}: CLEANUP recycled {len(ne)} non-earning keep(s) "
                      f"[{shown}{'...' if len(nonearners) > 8 else ''}]; kept now {len(state['kept'])}", flush=True)
                ckpt_path.write_text(json.dumps(state)); write_ledger(state)
                continue
        # benched pairs never sit in the pool: rebuilds filter them via mine_pool, and
        # this prune covers pools built before a bench existed (e.g. --resume onto a
        # checkpoint from a pre-cooldown run, or a flag change). Done entries for
        # pruned pairs are ignored at round end (scored iterates the pool).
        if args.cooldown_base:
            cooling = cooling_set()
            n_pre = len(state["pool"])
            state["pool"] = [c for c in state["pool"] if tuple(c) not in cooling]
            if len(state["pool"]) != n_pre:
                print(f"[fxc] round {rnd}: pruned {n_pre - len(state['pool'])} benched pair(s) "
                      f"from the pool ({len(cooling)} cooling)", flush=True)
        t0 = time.time()
        # Score the round's still-pending candidates in parallel batches: each
        # evaluate_many call fuses train+measure across `args.threads` cores. The
        # reference is fixed within a round, so a resumed round just skips the
        # already-scored candidates and finishes the rest.
        pending = [c for c in state["pool"] if f"{c[0]}+{c[1]}" not in cur["done"]]
        for bstart in range(0, len(pending), args.eval_batch):
            chunk = pending[bstart:bstart + args.eval_batch]
            batch = [list(state["kept"]) + [c] for c in chunk]
            try:
                results = _eval(batch)
            except Exception as e:  # a bad candidate must not kill the run — isolate it
                print(f"[fxc]   r{rnd} batch ERROR {e!r} — retrying chunk per-candidate", flush=True)
                results = []
                for c in chunk:
                    try:
                        results.append(_eval([list(state["kept"]) + [c]])[0])
                    except Exception as e2:
                        print(f"[fxc]   r{rnd} ERROR on {c[0]}+{c[1]}: {e2!r} — skip (non-keep)", flush=True)
                        results.append(None)
            for c, res in zip(chunk, results):
                key = f"{c[0]}+{c[1]}"
                if res is None:
                    cur["done"][key] = {"d": [0, 0.0, 0.0], "ps": {}, "error": "eval failed"}
                    continue
                base, infl, ps = res
                m = metrics_from_eval(base, infl, ps)
                dc, dco, dd = deltas(state["ref"], m)
                cur["done"][key] = {"d": [dc, dco, dd], "ps": m["per_shard"]}
            ndone = len(cur["done"])
            nk = sum(1 for v in cur["done"].values() if gate(*v["d"][:3]))
            print(f"[fxc]   r{rnd} {ndone}/{len(state['pool'])} keeps={nk} [{time.time()-t0:.0f}s]", flush=True)
            ckpt_path.write_text(json.dumps(state)); write_ledger(state)  # incremental persist
        # round complete — assemble verdicts (pool order) from this round's evals
        scored = []
        for cand in state["pool"]:
            d = cur["done"][f"{cand[0]}+{cand[1]}"]
            dcv, dco, ddd = d["d"][0], d["d"][1], d["d"][2]
            scored.append((cand, dcv, dco, ddd, gate(dcv, dco, ddd), d["ps"]))
        keeps = [s for s in scored if s[4]]
        rec = {"round": rnd, "n_pool": len(state["pool"]), "n_keeps": len(keeps),
               "committed": None, "committed_deltas": None,
               "verdicts": {f"{c[0]}+{c[1]}": [dc, dco, dd, int(kp)] for (c, dc, dco, dd, kp, _ps) in scored}}  # kp re-derived via gate()
        if not keeps:
            if args.pool_override is not None:
                # 0 keeps from an ARTIFICIAL override pool = the override candidate(s) don't pass
                # vs the CURRENT kept set (e.g. a redundant pick mid-harvest), NOT global
                # convergence. Do NOT journal a phantom round or declare fixpoint (that would
                # poison `status` and stall the audit). Re-mine the real pool so the checkpoint
                # is left clean, then return.
                if args.mine:
                    state["pool"] = mine_pool(build_tok(state["kept"]))
                state["cur_round"] = None
                msg = f"override candidate(s) evaporated vs kept={len(state['kept'])} (NOT convergence)"
                print(f"[fxc] round {rnd}: override → 0 keeps [{time.time()-t0:.0f}s] — {msg}; "
                      f"pool re-mined, state unchanged.", flush=True)
                if args.commit is not None or args.propose:
                    (state_dir / "propose.md").write_text(
                        f"# OVERRIDE EVAPORATED at round {rnd}: {msg}. No commit.\n")
                ckpt_path.write_text(json.dumps(state)); write_ledger(state)
                return
            state["rounds"].append(rec)
            nxt = len(state["rounds"]) + 1
            kept_now = {tuple(k) for k in state["kept"]}
            benched = sorted([(u, e, k) for k, (e, u) in cooldowns().items()
                              if u >= nxt and k not in kept_now])  # a kept pair's stale bench is moot
            if benched:
                # 0 keeps but pairs are still benched: with an unchanged reference nothing
                # can change until one re-auditions, so release the earliest-expiring pair
                # NOW (stall fast-forward) instead of idling out its bench. This round's
                # scores are carried into the next round (reference unchanged), so only
                # the released pair gets evaluated — seconds, not a full round.
                u, e, k = benched[0]
                rec["released"] = [list(k)]
                state["pool"] = list(state["pool"]) + [k]
                state["cur_round"] = {"round": nxt, "done": dict(cur["done"])}
                print(f"[fxc] round {rnd}: pool={rec['n_pool']} keeps=0 [{time.time()-t0:.0f}s] "
                      f"→ RELEASE {k[0]}+{k[1]} from cooldown (eject #{e}, was benched to r{u}; "
                      f"{len(benched) - 1} still benched)", flush=True)
                ckpt_path.write_text(json.dumps(state)); write_ledger(state)
                continue
            # The set was cleaned at the top of this round AND the bench is empty, so
            # 0 keeps here = genuine convergence (nothing passes AND every kept pair
            # earns its slot AND no pair is excluded by a cooldown timing artifact).
            state["status"] = "fixpoint"
            state["final_logs"] = [{"pair": list(c), "d_cov": dc, "d_comp": dco, "d_dead": dd, "per_shard": ps}
                                   for (c, dc, dco, dd, kp, ps) in scored]
            state["cur_round"] = None
            print(f"[fxc] round {rnd}: pool={len(state['pool'])} keeps=0 [{time.time()-t0:.0f}s] "
                  f"→ FIXPOINT ({len(state['pool'])} final LOGs)", flush=True)
            if args.propose or args.commit is not None:
                (state_dir / "propose.md").write_text(
                    f"# CONVERGED — fixpoint at round {rnd}: 0 viable candidates, kept {len(state['kept'])}.\n"
                    f"# The orchestrated loop is DONE. Promote/inspect the kept set (see ledger.md).\n")
            ckpt_path.write_text(json.dumps(state)); write_ledger(state)
            break
        # free keeps (Δcomp<=0 AND Δdead<=0) before costed ones (comp- or dead-tolerance);
        # then max Δcov, min Δcomp, min Δdead
        order = sorted(keeps, key=lambda s: (1 if (s[2] > 0 or s[3] > 0) else 0, -s[1], s[2], s[3]))
        if args.propose:
            present_round(order, ref_fire, ref_vocab, args.propose_top_n)
            ckpt_path.write_text(json.dumps(state)); write_ledger(state)
            print(f"[propose] round {rnd}: {len(keeps)} gate-passing keep(s); awaiting --commit", flush=True)
            return
        if args.commit is not None:
            if args.commit.strip().lower() == "none":
                state["rounds"].append(rec)
                state["cur_round"] = {"round": rnd + 1, "done": dict(cur["done"])}
                print(f"[commit] round {rnd}: NO COMMIT (judge skipped all keeps); "
                      f"scores carried to round {rnd + 1}", flush=True)
                ckpt_path.write_text(json.dumps(state)); write_ledger(state)
                return
            chosen = tuple(json.loads(args.commit))
            sel = next((s for s in keeps if s[0] == chosen), None)
            if sel is None:
                raise SystemExit(f"--commit {chosen!r} is not a gate-passing keep in round {rnd}; "
                                 f"re-run --propose and pick from its list")
            order = [sel]  # trial ONLY the chosen pair; the net-commit logic below runs on it
        # --- transactional commit (--net-commit): trial the best keep, settle the LOO
        # ejections it triggers (cascades included), and judge the SETTLED state against
        # the pre-commit reference with the same gate. Per-pair deltas can't assign blame
        # for an interaction — the trial's own (passing) delta and the incumbents' LOO
        # failures describe the same conflict — so the net transaction is what's gated:
        # net passes → commit + ejections land together (flipped pairs benched); net
        # fails → revert, veto + jail the candidate, trial the next-best keep. Vetoed
        # incumbent flips are NOT ejected (the whole trial is reverted).
        accepted, vetoed = None, []
        for best in order:
            if not args.net_commit:
                accepted = (best, list(state["kept"]) + [best[0]], [], None, None)
                break
            if len(vetoed) >= max(1, args.net_retries):
                print(f"[fxc]   r{rnd} net-commit: veto cap ({args.net_retries}) hit — deferring "
                      f"{len(order) - len(vetoed)} remaining keep(s) to the next round", flush=True)
                break
            cand = best[0]
            settled, flipped = list(state["kept"]) + [cand], []
            m = metrics_from_eval(*_eval([settled])[0])
            last_ld = None
            while not args.no_loo and len(settled) > 1:  # settle: eject non-earners until stable
                ld = loo_deltas(settled, m)
                if not ld:
                    break
                ne = [k for k in settled if k in ld and not gate(*ld[k])]
                if not ne:
                    last_ld = ld
                    break
                flipped += ne
                nes = set(ne)
                settled = [k for k in settled if k not in nes]
                m = metrics_from_eval(*_eval([settled])[0])
            net = deltas(state["ref"], m)
            if cand in settled and gate(*net):
                accepted = (best, settled, flipped, net, last_ld)
                break
            vetoed.append((cand, list(net)))
            fl = ", ".join(f"{a}+{b}" for a, b in flipped[:6])
            print(f"[fxc]   r{rnd} net-commit VETO {cand[0]}+{cand[1]} "
                  f"Δcov{best[1]:+} Δcomp{best[2]:+} Δdead{best[3]:+}: flips {len(flipped)} keep(s) "
                  f"[{fl}{'...' if len(flipped) > 6 else ''}] → net Δcov{net[0]:+} Δcomp{net[1]:+} "
                  f"Δdead{net[2]:+} fails the gate — jailed, trying next-best", flush=True)
        if vetoed:
            rec["vetoed"] = [list(k) for k, _ in vetoed]
            rec["vetoed_nets"] = [nd for _, nd in vetoed]
        if accepted is None:
            # every trialed keep was net-negative (now jailed), or the veto cap hit: no
            # commit this round. The reference is unchanged, so carry this round's scores
            # into the next round — only the not-yet-trialed leftovers get considered
            # (benched pairs are pruned at the top of the round; the explicit pool drop
            # below covers --cooldown-base 0, where a veto would otherwise loop forever).
            vset = {k for k, _ in vetoed}
            state["pool"] = [c for c in state["pool"] if c not in vset]
            state["rounds"].append(rec)
            state["cur_round"] = {"round": rnd + 1, "done": dict(cur["done"])}
            print(f"[fxc] round {rnd}: pool={rec['n_pool']} keeps={len(keeps)} [{time.time()-t0:.0f}s] "
                  f"→ NO COMMIT ({len(vetoed)} trial(s) vetoed net-negative)", flush=True)
            ckpt_path.write_text(json.dumps(state)); write_ledger(state)
            if args.commit is not None:
                print(f"[commit] your pick was vetoed (net-negative after LOO settling) and jailed; "
                      f"--propose round {rnd + 1} (scores carried) and pick another.", flush=True)
                return
            continue
        best, settled, flipped, net, last_ld = accepted
        cand = best[0]
        n = len(settled)
        tok, (new_ref, new_fire, new_vocab) = commit(settled, n, with_lineage=True)
        mng = write_lineage(n, cand, tok, new_vocab, new_fire, ref_vocab, ref_fire)
        rec["committed"] = list(cand); rec["committed_deltas"] = [best[1], best[2], best[3]]
        rec["committed_mangled_tune"] = mng
        if flipped:
            # part of the accepted transaction: journaling them as cleanup benches them
            # and lets --rerun replay re-eject them
            rec["cleanup"] = [list(k) for k in flipped]
            rec["net_deltas"] = list(net)
        state["rounds"].append(rec)
        state["kept"] = settled
        if last_ld:
            state["kept_deltas"] = {f"{a}+{b}": list(v) for (a, b), v in last_ld.items()}
        if args.mine:
            state["pool"] = mine_pool(tok)  # re-mine new ref: add arisen, drop vanished, minus kept/benched
            print(f"[fxc]   re-mined pool: {len(state['pool'])} candidates (kept {len(state['kept'])})", flush=True)
        else:
            cooling = cooling_set()
            state["pool"] = [c for c in state["pool"] if c != cand and c not in cooling]
        state["ref"], ref_fire, ref_vocab = new_ref, new_fire, new_vocab
        state["cur_round"] = None
        print(f"[fxc] round {rnd}: pool={rec['n_pool']} keeps={len(keeps)} [{time.time()-t0:.0f}s] "
              f"→ commit {cand!r} Δcov{best[1]:+} Δcomp{best[2]:+} Δdead{best[3]:+}"
              + (f" | settled −{len(flipped)} ejected, NET Δcov{net[0]:+} Δcomp{net[1]:+} Δdead{net[2]:+}" if flipped else "")
              + (f" | {len(vetoed)} vetoed" if vetoed else "")
              + f" (mangled_tune {mng})", flush=True)
        ckpt_path.write_text(json.dumps(state)); write_ledger(state)
        if args.commit is not None:
            return

    # ---- leave-one-out redundancy pass ----
    if state["status"] == "fixpoint" and not args.no_loo and state.get("loo") is None and state["kept"]:
        print(f"[fxc] leave-one-out over {len(state['kept'])} keeps...", flush=True)
        final_ref = state["ref"]
        loo = []
        for k in state["kept"]:
            try:
                m = metrics_from_eval(*_eval([[x for x in state["kept"] if x != k]])[0])
                dc, dco, dd = deltas(m, final_ref)  # full(=others+k) vs others: does k still earn its slot?
                loo.append({"pair": list(k), "d_cov": dc, "d_comp": dco, "d_dead": dd, "earns": gate(dc, dco, dd)})
            except Exception as e:  # conservatively keep k (don't flag redundant) on error
                print(f"[fxc]   loo ERROR on {k!r}: {e!r} — assuming earns", flush=True)
                loo.append({"pair": list(k), "d_cov": "-", "d_comp": "-", "d_dead": "-", "earns": True, "error": str(e)})
        state["loo"] = loo
        ckpt_path.write_text(json.dumps(state)); write_ledger(state)
        red = [e for e in loo if not e["earns"]]
        print(f"[fxc] leave-one-out: {len(red)} redundant"
              + ("" if not red else ": " + ", ".join(f"{e['pair'][0]}+{e['pair'][1]}" for e in red)), flush=True)

    print(f"\n[fxc] DONE. status={state['status']} rounds={len(state['rounds'])} kept={len(state['kept'])}", flush=True)
    for kp in state["kept"]:
        print(f"    {tuple(kp)!r},")


if __name__ == "__main__":
    main()
