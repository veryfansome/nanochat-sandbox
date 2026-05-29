"""Classify the held-out-fixpoint's LOG (non-keep) candidates by WHY they fail the
gate, from an `audit/heldout_fixpoint/checkpoint.json`.

The gate (in `tools.heldout_fixpoint_cached`) is: KEEP iff, averaged over the 10
held-out shards, Δcoverage > 0 AND Δcompression <= 0 AND Δdead <= 0. A LOG fails at
least one, which sorts every rejection into one of three modes:

  A  no coverage gain        Δcov <= 0                     (blocking yields no new
                                                            whole-word token; Δcov<0
                                                            is *harmful* — shatters one)
  B  coverage-for-compression Δcov > 0 AND Δcomp > 0       (+word, but costs tokens)
  C  coverage, comp-ok, dead  Δcov > 0, Δcomp<=0, Δdead>0  (+word, frees a dead slot)

For each it reports counts, the Δ distributions, the class-A non-harmful split
(coverage-flat candidates: comp-win-but-dead-regress / inert / flat+cost — note clean
coverage-flat comp/dead wins now PASS the gate), the class-B split into near-misses
(small comp cost — the "is +1 word worth +x tokens?" set) vs clear rejects, and the
suffix patterns.

Source selection (most authoritative available, unless overridden):
  - `final_logs` if the run has converged (status=fixpoint),
  - else the latest *complete* round's verdicts,
  - `--cur` to use the in-progress round instead (freshest, partial).

    uv run python -m tools.analyze_logs
    uv run python -m tools.analyze_logs --near-miss-comp 1.0 --examples 15
    uv run python -m tools.analyze_logs --checkpoint <path> --cur
"""
import argparse
import json
from collections import Counter
from pathlib import Path


def pick_source(ck, use_cur):
    """Return (label, {‘L+R’: (dcov, dcomp, ddead)}) for the LOG candidates to analyze."""
    if use_cur and ck.get("cur_round", {}).get("done"):
        cur = ck["cur_round"]
        return (f"round {cur['round']} (in progress, {len(cur['done'])} evaluated)",
                {k: (v["d"][0], v["d"][1], v["d"][2]) for k, v in cur["done"].items()})
    if ck.get("status") == "fixpoint" and ck.get("final_logs"):
        return ("final (converged)",
                {f"{r['pair'][0]}+{r['pair'][1]}": (r["d_cov"], r["d_comp"], r["d_dead"])
                 for r in ck["final_logs"]})
    if ck.get("rounds"):
        last = ck["rounds"][-1]
        return (f"round {last['round']} (vs {last['round']-1}-keep ref)",
                {k: (v[0], v[1], v[2]) for k, v in last["verdicts"].items()})
    raise SystemExit("checkpoint has no rounds / final_logs / cur_round to analyze")


def mode(dcov, dcomp, ddead):
    # matches tools.heldout_fixpoint_cached.gate: held-out Pareto improvement —
    # no axis regresses and at least one strictly improves (incl. coverage-flat
    # compression/dead wins).
    if dcov >= 0 and dcomp <= 0 and ddead <= 0 and (dcov > 0 or dcomp < 0 or ddead < 0):
        return "KEEP"
    if dcov <= 0:
        return "A"
    if dcomp > 0:
        return "B"
    return "C"


def suffix(pair_str):
    return pair_str.split("+", 1)[1] if "+" in pair_str else pair_str


def main():
    home = str(Path.home())
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", default="rustbpe_variants/auto_tune/audit/heldout_fixpoint/checkpoint.json")
    p.add_argument("--cur", action="store_true", help="use the in-progress round instead of the latest complete one")
    p.add_argument("--near-miss-comp", type=float, default=1.0,
                   help="class-B comp-cost cutoff for the 'near-miss' (exchange-rate) set (default 1.0)")
    p.add_argument("--examples", type=int, default=12)
    args = p.parse_args()

    ck = json.loads(Path(args.checkpoint).read_text())
    label, src = pick_source(ck, args.cur)
    kept = [f"{a}+{b}" for a, b in (tuple(x) for x in ck.get("kept", []))]

    by = {"KEEP": {}, "A": {}, "B": {}, "C": {}}
    for k, (dcov, dcomp, ddead) in src.items():
        by[mode(dcov, dcomp, ddead)][k] = (dcov, dcomp, ddead)
    logs = {**by["A"], **by["B"], **by["C"]}
    nlog = max(1, len(logs))

    print(f"source: {label}  |  candidates {len(src)}  →  {len(by['KEEP'])} pass-gate / {len(logs)} LOG"
          f"   (committed keeps so far: {len(kept)})\n")
    print("LOG failure modes:")
    print(f"  A  no coverage gain  (Δcov<=0):                 {len(by['A']):3d}  ({100*len(by['A'])/nlog:.0f}%)")
    print(f"  B  coverage+ but compression cost (Δcomp>0):    {len(by['B']):3d}  ({100*len(by['B'])/nlog:.0f}%)")
    print(f"  C  coverage+, comp-ok, dead cost (Δdead>0):     {len(by['C']):3d}  ({100*len(by['C'])/nlog:.0f}%)")

    cov_dist = Counter(v[0] for v in by["A"].values())
    harmful = {k: v for k, v in by["A"].items() if v[0] < 0}
    flat = {k: v for k, v in by["A"].items() if v[0] == 0}  # non-harmful: coverage-flat
    print(f"\n— A (no coverage): Δcov dist {dict(sorted(cov_dist.items(), reverse=True))}; "
          f"{len(harmful)} HARMFUL (Δcov<0 — blocking shatters a whole word), {len(flat)} non-harmful (Δcov=0)")
    # non-harmful (Δcov=0): sub-classify by what they DO despite adding no coverage.
    # Clean coverage-flat comp/dead wins now PASS the gate (counted in pass-gate, not A).
    # What's left here: candidates that compress but regress dead, plus inert / flat+cost.
    comp_dr = sorted(((k, v) for k, v in flat.items() if v[1] < 0), key=lambda kv: kv[1][1])  # Δcomp<0 but Δdead>0
    inert = [(k, v) for k, v in flat.items() if v[1] == 0 and v[2] == 0]                       # zero effect
    costly = [(k, v) for k, v in flat.items() if v[1] > 0 or (v[1] == 0 and v[2] > 0)]         # flat + a regression
    print(f"  non-harmful breakdown: {len(comp_dr)} comp-win-but-dead-regress (Δcomp<0,Δdead>0), "
          f"{len(inert)} fully inert, {len(costly)} flat+cost   "
          f"(clean coverage-flat comp/dead wins now pass the gate)")
    if comp_dr:
        print(f"  comp-win-but-dead-regress (compresses but Δdead>0, so rejected):")
        for k, v in comp_dr[:args.examples]:
            print(f"    {k:<20} Δcov{v[0]:+} Δcomp{v[1]:+} Δdead{v[2]:+}")

    near = sorted(((k, v) for k, v in by["B"].items() if v[1] <= args.near_miss_comp), key=lambda kv: kv[1][1])
    print(f"\n— B near-misses (Δcomp <= {args.near_miss_comp}; the exchange-rate set, {len(near)} pairs):")
    for k, v in near[:args.examples]:
        print(f"    {k:<20} Δcov{v[0]:+} Δcomp{v[1]:+5} Δdead{v[2]:+}")
    rejects = sorted(by["B"].items(), key=lambda kv: -kv[1][1])
    print(f"  B clear rejects (largest comp cost):")
    for k, v in rejects[:5]:
        print(f"    {k:<20} Δcov{v[0]:+} Δcomp{v[1]:+}")

    if by["C"]:
        print(f"\n— C (coverage + comp-ok, dead cost):")
        for k, v in sorted(by["C"].items(), key=lambda kv: -kv[1][2]):
            print(f"    {k:<20} Δcov{v[0]:+} Δcomp{v[1]:+} Δdead{v[2]:+}")

    print(f"\n— suffix (R) patterns —")
    print(f"  committed keeps R: {Counter(suffix(k) for k in kept).most_common(8)}")
    print(f"  A no-cov       R: {Counter(suffix(k) for k in by['A']).most_common(8)}")
    print(f"  B comp-cost    R: {Counter(suffix(k) for k in by['B']).most_common(8)}")


if __name__ == "__main__":
    main()
