"""
Build-up audit of an auto_tune BLOCKED_PAIRS candidate queue.

Premise ("nothing is golden"): start from an EMPTY block list and let each
candidate earn its slot. We walk `CANDIDATES` (from the pairs module) one at a
time; for each, we retrain the tokenizer with `kept + [candidate]`, measure the
priority metrics on a fixed held-out corpus, and compare to the current `kept`
reference. A candidate is COMMITTED into `kept` (shifting the reference) only if
it is a lexicographic win under the metric priority; otherwise it is FLAGGED
(priority-1 win that costs compression — a human call per the README "stop and
surface the expensive trade-offs" rule) or DROPPED.

Why build-up rather than leave-one-out: a candidate redundant with an
already-kept pair shows ~0 gain on top of `kept`, so it's dropped naturally —
no redundancy masking. Training is deterministic (fixed corpus + BPE), so one
retrain per candidate is sufficient; deltas are real, not sampling noise.

Priority metrics (see auto_tune/README.md "Tuning instructions"):
  1. whole words (higher better)   -- `--p1-metric total` (default) or `base`
  2. compression tokens (lower better)
  3. mangled tokens remaining (lower better)

Lexicographic commit rule (delta = kept+candidate  minus  kept):
  d_p1 < 0                       -> DROP   (hurts top priority)
  d_p1 > 0 and d_comp <= 0       -> KEEP   (P1 up, compression not worse)
  d_p1 > 0 and d_comp >  0       -> FLAG   (P1 up but pays compression -> defer)
  d_p1 == 0 and d_comp < 0       -> KEEP   (compression win at no P1 cost)
  d_p1 == 0 and d_comp > 0       -> DROP   (pays compression for no P1 gain)
  d_p1 == 0 and d_comp == 0:
      d_mangled < 0 -> KEEP      (free mangled cleanup) else DROP (inert)
Only KEEP shifts the reference; FLAG/DROP leave it unchanged.

Each candidate = one ~5 min retrain + ~1-2 min measure, serial (CPU runs hot;
no concurrency). ~70 candidates => several hours. Checkpoints after every
candidate to <work-base>/checkpoint.json; rerun with --resume to continue.

Usage:
    uv run python -m tools.audit_blocked_pairs                # full run (serial build-up)
    uv run python -m tools.audit_blocked_pairs --resume       # continue
    uv run python -m tools.audit_blocked_pairs --jobs 8       # parallel build-up (waves of 8)
    uv run python -m tools.audit_blocked_pairs --jobs 8 --reference fixed   # parallel, all vs
                                                              # the fixed reference (combos / screen)
    uv run python -m tools.audit_blocked_pairs --limit 1 \\
        --smoke-train-chars 5000000                           # plumbing smoke
"""

import argparse
import importlib
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from queue import Queue


# ---- metric measurement -------------------------------------------------

def whole_word_counts(tok, words):
    """(base, inflected) space-prefixed dictionary-word tokens in the vocab.
    Mirrors blocked_pairs_report.whole_word_count but reads token strings
    straight from the tokenizer (no dump file needed)."""
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


def measure(tok_dir, corpus, words, bound_threshold):
    """Priority metrics for a trained tokenizer dir."""
    from nanochat.tokenizer import RustBPETokenizer
    from tools.pass_metrics import compute_metrics
    tok = RustBPETokenizer.from_directory(str(tok_dir))
    m = compute_metrics(tok, corpus, count_mangled=True, bound_threshold=bound_threshold)
    base, infl = whole_word_counts(tok, words)
    return {
        "total_tokens": m["total_tokens"],
        "mangled": m["mangled_count"],
        "dead": m["dead"],
        "base_words": base,
        "inflected_words": infl,
        "total_words": base + infl,
    }


# ---- training -----------------------------------------------------------

def train(pairs, work_base, smoke_train_chars, log_path, rayon_threads=None):
    """Retrain auto_tune with `pairs` as the block list (via env override).
    Returns the tokenizer dir on success, else None. rayon_threads caps
    RAYON_NUM_THREADS for THIS train subprocess only (so concurrent trains under
    --jobs don't oversubscribe cores); the parent/measure threading is untouched."""
    env = dict(os.environ)
    env["AUTO_TUNE_BLOCKED_PAIRS_JSON"] = json.dumps([list(p) for p in pairs])
    env["VARIANT_BASE"] = str(work_base)
    if rayon_threads:
        env["RAYON_NUM_THREADS"] = str(rayon_threads)
    cmd = [sys.executable, "-m", "wrappers.tok_train_auto_tune"]
    if smoke_train_chars:
        cmd += ["--max-chars", str(smoke_train_chars)]
    with open(log_path, "w") as logf:
        proc = subprocess.run(cmd, env=env, stdout=logf, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        return None
    return Path(work_base) / "tokenizer"


# ---- classification -----------------------------------------------------

def classify(ref, cur, p1_metric, keep_rule="lexicographic"):
    """Return (verdict, deltas) where verdict in {KEEP, FLAG, DROP}."""
    p1_key = "total_words" if p1_metric == "total" else "base_words"
    d_p1 = cur[p1_key] - ref[p1_key]           # higher better
    d_comp = cur["total_tokens"] - ref["total_tokens"]  # lower better
    d_mangled = cur["mangled"] - ref["mangled"]         # lower better
    deltas = {"p1": d_p1, "comp": d_comp, "mangled": d_mangled}
    if keep_rule == "cov-or-mangled":
        # Stage-A selection on the GENERALIZING signals only: keep if it adds
        # whole-word coverage OR reduces mangled tokens. Compression is NOT gated
        # here (single-shard compression overfits — vet it separately as an
        # average over held-out shards). Build-up still applies, so a flag
        # redundant with the growing kept set shows d_p1<=0 & d_mangled>=0 → DROP.
        v = "KEEP" if (d_p1 > 0 or d_mangled < 0) else "DROP"
        return v, deltas
    if d_p1 < 0:
        v = "DROP"
    elif d_p1 > 0:
        v = "KEEP" if d_comp <= 0 else "FLAG"
    else:  # d_p1 == 0
        if d_comp < 0:
            v = "KEEP"
        elif d_comp > 0:
            v = "DROP"
        else:
            v = "KEEP" if d_mangled < 0 else "DROP"
    return v, deltas


# ---- ledger / checkpoint ------------------------------------------------

def _unit_of(row):
    """The list of (L, R) pairs a row tested. Backward-compatible: old single-pair
    rows store `pair` (one [L, R]); group rows store `pairs` (a list of [L, R])."""
    if "pairs" in row:
        return [tuple(p) for p in row["pairs"]]
    return [tuple(row["pair"])]


def fmt_unit(unit):
    """Markdown label for a tested unit (one pair, or a co-requisite route-set)."""
    routes = ",".join(f"`{a!r}+{b!r}`" for a, b in unit)
    if len(unit) == 1:
        return routes
    surfaces = {a + b for a, b in unit}
    return f"`{next(iter(surfaces))}`: {routes}" if len(surfaces) == 1 else routes


def write_ledger_md(path, state, p1_metric):
    p1_label = "Δ total words" if p1_metric == "total" else "Δ base words"
    n_total = state.get("n_units", state.get("n_candidates"))
    unit_word = "groups" if state.get("mode") == "groups" else "candidates"
    lines = [
        "# auto_tune build-up audit ledger",
        "",
        f"- mode: **{state.get('mode', 'single')}** ({unit_word})",
        f"- p1 metric: **{p1_metric}** ({'total whole words' if p1_metric=='total' else 'base coverage'})",
        f"- current reference (after {len(state['kept'])} keeps): {fmt_metrics(state.get('ref'))}",
        f"- processed: {len(state['rows'])}/{n_total}",
        f"- kept: {len(state['kept'])}  flagged: {sum(1 for r in state['rows'] if r['verdict']=='FLAG')}  "
        f"dropped: {sum(1 for r in state['rows'] if r['verdict']=='DROP')}  "
        f"errors: {sum(1 for r in state['rows'] if r['verdict']=='ERROR')}",
        "",
        f"| # | {'route-set' if unit_word=='groups' else 'pair'} | verdict | {p1_label} | Δ comp (tok) | Δ mangled | committed kept size |",
        "|---|------|---------|----------|--------------|-----------|---------------------|",
    ]
    for i, r in enumerate(state["rows"]):
        d = r.get("deltas") or {}
        unit = fmt_unit(_unit_of(r))
        lines.append(
            f"| {i} | {unit} | {r['verdict']} | "
            f"{d.get('p1','-'):+} | {d.get('comp','-'):+} | {d.get('mangled','-'):+} | "
            f"{r.get('kept_size','-')} |"
            if d else
            f"| {i} | {unit} | {r['verdict']} | - | - | - | {r.get('kept_size','-')} |"
        )
    lines += ["", "## current kept (promote into BLOCKED_PAIRS)", "```python"]
    lines += [f"    {tuple(p)!r}," for p in state["kept"]]
    lines += ["```", "", "## flagged (expensive: P1 win that costs compression — human call)"]
    for r in state["rows"]:
        if r["verdict"] == "FLAG":
            d = r["deltas"]
            lines.append(f"- {fmt_unit(_unit_of(r))}  "
                         f"p1{d['p1']:+}  comp{d['comp']:+}  mangled{d['mangled']:+}")
    Path(path).write_text("\n".join(lines) + "\n")


def fmt_metrics(m):
    if not m:
        return "(pending)"
    return (f"tokens={m['total_tokens']:,} words={m['total_words']:,} "
            f"(base {m['base_words']:,}/infl {m['inflected_words']:,}) "
            f"mangled={m['mangled']} dead={m['dead']}")


def main(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pairs-module", default="rustbpe_variants.auto_tune.pairs")
    p.add_argument("--mode", choices=["single", "groups"], default="single",
                   help="single: walk CANDIDATES one pair at a time (pass 1). "
                        "groups: walk CANDIDATE_GROUPS, training/committing each "
                        "route-set JOINTLY as one unit (multi-pair follow-up).")
    p.add_argument("--candidates-attr", default="CANDIDATES",
                   help="module attribute holding the single-pair candidate list (single mode)")
    p.add_argument("--groups-attr", default="CANDIDATE_GROUPS",
                   help="module attribute holding the list of route-sets (groups mode)")
    p.add_argument("--work-base", default=os.path.expanduser("~/.cache/nanochat-variants/auto_tune_audit"),
                   help="scratch dir for throwaway per-candidate tokenizer training "
                        "output (cache, gitignored — NOT the ledger)")
    p.add_argument("--state-dir", default=None,
                   help="dir for the portable, checked-in ledger.md + checkpoint.json "
                        "(default: <pairs-module dir>/audit) so the run can be resumed "
                        "on another machine")
    p.add_argument("--max-chars", type=int, default=20_000_000,
                   help="held-out corpus cap for measurement (default 20M)")
    p.add_argument("--bound-threshold", type=float, default=0.05)
    p.add_argument("--p1-metric", choices=["total", "base"], default="total",
                   help="priority-1 whole-word metric (default total)")
    p.add_argument("--keep-rule", choices=["lexicographic", "cov-or-mangled"], default="lexicographic",
                   help="lexicographic (default): coverage>compression>mangled; FLAG when "
                        "coverage-up costs compression. cov-or-mangled: KEEP if coverage up OR "
                        "mangled down (compression NOT gated — vet it separately on held-out "
                        "shards). Use with build-up for the flag-revisit / generalizing-signal selection.")
    p.add_argument("--limit", type=int, default=None,
                   help="only process the first N candidates (smoke)")
    p.add_argument("--smoke-train-chars", type=int, default=None,
                   help="pass --max-chars to the trainer for a fast plumbing smoke "
                        "(produces a junk vocab — do NOT use for real verdicts)")
    p.add_argument("--jobs", type=int, default=1,
                   help="max concurrent retrains (parallel runner). 1 = serial. Each "
                        "job trains in its own --work-base slot dir (no clobbering).")
    p.add_argument("--keep-tokenizers", action=argparse.BooleanOptionalAction, default=True,
                   help="retain each KEPT candidate's tokenizer under "
                        "<work-base>/kept_tokenizers/<pass>/keep_<N>/ (~0.5 MB each) so held-out / "
                        "new-shard re-evaluation is measure-only (no retrain). Serial mode only "
                        "(parallel reuses slot dirs). Default on.")
    p.add_argument("--train-rayon-threads", type=int, default=None,
                   help="cap RAYON_NUM_THREADS for each train subprocess. rustbpe training "
                        "is internally multi-threaded (~3 cores/train), so with many --jobs "
                        "set this to 1-2 so concurrent trains don't oversubscribe the box "
                        "(calibrated best: --jobs ≈ cores, --train-rayon-threads 1). Leaves "
                        "the parent/measure threading untouched.")
    p.add_argument("--reference", choices=["buildup", "fixed"], default="buildup",
                   help="buildup (default): reference shifts as KEEPs accumulate. With "
                        "--jobs>1 this runs in waves (within-wave redundancy not "
                        "eliminated; ref refreshed after multi-keep waves). "
                        "fixed: reference never shifts — every unit is measured against "
                        "the starting reference (exact + fully parallel; for combination "
                        "tests / fast screening; does NOT promote keeps).")
    p.add_argument("--resume", action="store_true")
    args = p.parse_args(argv)

    work_base = Path(args.work_base)
    work_base.mkdir(parents=True, exist_ok=True)
    train_log = work_base / "last_train.log"

    mod = importlib.import_module(args.pairs_module)
    seed_kept = [tuple(c) for c in mod.BLOCKED_PAIRS]  # single: usually []; groups: pass-1 keeps
    # A "unit" is the list of pairs committed together. single mode: one pair per
    # unit; groups mode: a whole route-set per unit (co-requisite blocking).
    if args.mode == "groups":
        units = [[tuple(p) for p in group] for group in getattr(mod, args.groups_attr)]
    else:
        units = [[tuple(c)] for c in getattr(mod, args.candidates_attr)]

    # Portable, checked-in state lives WITH the variant (not in the cache) so the
    # ledger + resume checkpoint travel with the repo to another machine. Scratch
    # per-candidate training output stays under --work-base (cache, gitignored).
    state_dir = Path(args.state_dir) if args.state_dir else Path(mod.__file__).resolve().parent / "audit"
    state_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = state_dir / "checkpoint.json"
    ledger_path = state_dir / "ledger.md"

    # Keep-tokenizer retention: copy each KEPT candidate's tokenizer to a stable,
    # keep-numbered dir under the (gitignored) work-base, so re-measuring the
    # keep-trajectory on held-out shards later is measure-only, no retrain.
    retention_dir = Path(args.work_base) / "kept_tokenizers" / state_dir.name
    if args.keep_tokenizers:
        retention_dir.mkdir(parents=True, exist_ok=True)
        print(f"[audit] retaining keep tokenizers under {retention_dir}", flush=True)

    def retain_keep(src_tdir, n):
        if not args.keep_tokenizers or src_tdir is None:
            return
        dst = retention_dir / f"keep_{n:04d}"
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src_tdir, dst)

    from tools._corpus_iter import iter_capped_batches
    from tools.blocked_pairs_report import load_wordlist
    words = load_wordlist()
    if words is None:
        sys.exit("No wordlist at /usr/share/dict/words; whole-word metric unavailable.")
    print(f"[audit] loading corpus (val, {args.max_chars:,} cap)...", flush=True)
    corpus = list(iter_capped_batches("val", args.max_chars))

    # ---- resume or init ----
    if args.resume and ckpt_path.exists():
        state = json.loads(ckpt_path.read_text())
        state["kept"] = [tuple(p) for p in state["kept"]]
        print(f"[audit] resumed ({state.get('mode','single')}): "
              f"{len(state['rows'])}/{len(units)} done, kept={len(state['kept'])}", flush=True)
    else:
        state = {"kept": list(seed_kept), "ref": None, "rows": [],
                 "n_units": len(units), "mode": args.mode, "p1_metric": args.p1_metric,
                 "jobs": args.jobs, "reference": args.reference}

    # ---- reference (empty/kept list) ----
    if state["ref"] is None:
        print(f"[audit] training reference (kept={len(state['kept'])})...", flush=True)
        t0 = time.time()
        tdir = train(state["kept"], work_base, args.smoke_train_chars, train_log,
                     rayon_threads=args.train_rayon_threads)
        if tdir is None:
            sys.exit(f"reference train failed; see {train_log}")
        state["ref"] = measure(tdir, corpus, words, args.bound_threshold)
        print(f"[audit] reference [{time.time()-t0:.0f}s]: {fmt_metrics(state['ref'])}", flush=True)
        retain_keep(tdir, len(state["kept"]))  # seed-point of the keep trajectory
        ckpt_path.write_text(json.dumps(state))

    # ---- evaluation loop ----
    start = len(state["rows"])
    end = len(units) if args.limit is None else min(start + args.limit, len(units))

    def checkpoint():
        # atomic: write to a temp file then rename, so a crash mid-write can never
        # truncate checkpoint.json (which would break --resume). os.replace is atomic
        # on the same filesystem. The ledger is regenerable from the checkpoint, so a
        # plain write there is fine.
        tmp = ckpt_path.parent / (ckpt_path.name + ".tmp")
        tmp.write_text(json.dumps(state))
        os.replace(tmp, ckpt_path)
        write_ledger_md(ledger_path, state, args.p1_metric)

    if args.jobs == 1 and args.reference == "buildup":
        # serial build-up (unchanged): each candidate vs the running reference.
        for i in range(start, end):
            unit = units[i]
            trial = state["kept"] + list(unit)
            print(f"\n[audit] {i+1}/{len(units)} {fmt_unit(unit)}  "
                  f"(kept={len(state['kept'])})", flush=True)
            t0 = time.time()
            tdir = train(trial, work_base, args.smoke_train_chars, train_log,
                         rayon_threads=args.train_rayon_threads)
            if tdir is None:
                print(f"[audit]   TRAIN FAILED; see {train_log}", flush=True)
                state["rows"].append({"pairs": [list(p) for p in unit], "verdict": "ERROR",
                                      "deltas": None, "kept_size": len(state["kept"])})
                checkpoint(); continue
            cur = measure(tdir, corpus, words, args.bound_threshold)
            verdict, deltas = classify(state["ref"], cur, args.p1_metric, args.keep_rule)
            print(f"[audit]   [{time.time()-t0:.0f}s] {verdict}  "
                  f"p1{deltas['p1']:+} comp{deltas['comp']:+} mangled{deltas['mangled']:+}  "
                  f"| {fmt_metrics(cur)}", flush=True)
            if verdict == "KEEP":
                state["kept"].extend(list(p) for p in unit)
                state["ref"] = cur
                retain_keep(tdir, len(state["kept"]))
            state["rows"].append({"pairs": [list(p) for p in unit], "verdict": verdict,
                                  "deltas": deltas, "kept_size": len(state["kept"]),
                                  "metrics": cur})
            checkpoint()
    else:
        # ---- parallel runner ----
        # fixed: every unit vs the starting reference (exact, fully parallel; no promotion).
        # buildup: waves of size --jobs vs the wave-start reference; KEEPs applied in index
        # order; reference refreshed after any wave with >=2 keeps so the next wave's deltas
        # stay exact. Caveat: redundancy between two candidates in the SAME wave isn't caught.
        fixed = args.reference == "fixed"
        if args.keep_tokenizers and not fixed:
            print("[audit] note: keep-tokenizer retention is serial-only (parallel reuses slot "
                  "dirs); not retaining for this parallel run — use serial for the keep-trajectory.", flush=True)
        slot_q = Queue()
        for s in range(args.jobs):
            sd = work_base / f"slot{s}"; sd.mkdir(parents=True, exist_ok=True); slot_q.put(sd)
        refresh_dir = work_base / "refresh"; refresh_dir.mkdir(parents=True, exist_ok=True)
        print(f"[audit] parallel runner: jobs={args.jobs} reference={args.reference}", flush=True)

        def eval_unit(idx, unit, kept_snapshot):
            sd = slot_q.get()
            try:
                tdir = train(kept_snapshot + [list(p) for p in unit], sd,
                             args.smoke_train_chars, sd / "train.log",
                             rayon_threads=args.train_rayon_threads)
                if tdir is None:
                    return idx, None
                return idx, measure(tdir, corpus, words, args.bound_threshold)
            except Exception as e:  # keep one bad unit from killing the wave
                print(f"[audit]   worker error on unit {idx+1}: {e}", flush=True)
                return idx, None
            finally:
                slot_q.put(sd)

        with ThreadPoolExecutor(max_workers=args.jobs) as ex:
            i = start
            while i < end:
                wave = list(range(i, min(i + args.jobs, end)))
                kept_snap = [tuple(p) for p in state["kept"]]
                ref_snap = state["ref"]
                t0 = time.time()
                futs = {ex.submit(eval_unit, j, units[j], kept_snap): j for j in wave}
                res = {}
                for fut in as_completed(futs):
                    idx, cur = fut.result(); res[idx] = cur
                wave_keeps = 0
                for j in wave:
                    unit = units[j]; cur = res[j]
                    if cur is None:
                        print(f"[audit] {j+1}/{len(units)} {fmt_unit(unit)}  TRAIN FAILED", flush=True)
                        state["rows"].append({"pairs": [list(p) for p in unit], "verdict": "ERROR",
                                              "deltas": None, "kept_size": len(state["kept"])})
                        continue
                    verdict, deltas = classify(ref_snap, cur, args.p1_metric, args.keep_rule)
                    print(f"[audit] {j+1}/{len(units)} {fmt_unit(unit)}  {verdict}  "
                          f"p1{deltas['p1']:+} comp{deltas['comp']:+} mangled{deltas['mangled']:+}  "
                          f"| {fmt_metrics(cur)}", flush=True)
                    if verdict == "KEEP" and not fixed:
                        state["kept"].extend(list(p) for p in unit)
                        state["ref"] = cur
                        wave_keeps += 1
                    state["rows"].append({"pairs": [list(p) for p in unit], "verdict": verdict,
                                          "deltas": deltas, "kept_size": len(state["kept"]),
                                          "metrics": cur})
                if not fixed and wave_keeps >= 2:  # ref no longer matches the multi-keep kept set
                    rt = train([list(p) for p in state["kept"]], refresh_dir,
                               args.smoke_train_chars, refresh_dir / "train.log",
                               rayon_threads=args.train_rayon_threads)
                    if rt is not None:
                        state["ref"] = measure(rt, corpus, words, args.bound_threshold)
                        print(f"[audit]   (reference refreshed after {wave_keeps} keeps: "
                              f"{fmt_metrics(state['ref'])})", flush=True)
                print(f"[audit] wave {wave[0]+1}-{wave[-1]+1} done [{time.time()-t0:.0f}s] "
                      f"kept={len(state['kept'])}", flush=True)
                checkpoint()
                i += len(wave)

    # ---- summary ----
    kept = state["kept"]
    flagged = [r for r in state["rows"] if r["verdict"] == "FLAG"]
    print(f"\n[audit] DONE {len(state['rows'])}/{len(units)}.  "
          f"kept={len(kept)} flagged={len(flagged)} "
          f"dropped={sum(1 for r in state['rows'] if r['verdict']=='DROP')} "
          f"errors={sum(1 for r in state['rows'] if r['verdict']=='ERROR')}", flush=True)
    print(f"[audit] ledger: {ledger_path}", flush=True)
    print(f"[audit] kept pairs:")
    for kp in kept:
        print(f"    {tuple(kp)!r},")


if __name__ == "__main__":
    main()
