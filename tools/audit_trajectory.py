"""Reference-metric trajectory as keeps accumulate — replaces the hand-kept tally.

Walks the KEEP rows across the audit checkpoint chain (round 1 → 2 → 3 →
flag-revisit) and shows, after each keep: tokens / whole-words (base+infl) /
mangled / dead, with deltas vs the empty baseline. ASCII sparklines summarize
each series; `--csv` dumps the full series for external graphing (no matplotlib
in this env).

    uv run python -m tools.audit_trajectory                 # full table + sparklines
    uv run python -m tools.audit_trajectory --every 5       # sample every 5th keep
    uv run python -m tools.audit_trajectory --csv traj.csv   # dump for graphing
"""
import argparse
import json
from pathlib import Path

DEFAULT_CHAIN = [
    ("R1", "rustbpe_variants/auto_tune/audit/checkpoint.json"),
    ("R2", "rustbpe_variants/auto_tune/audit/manual_pass1/checkpoint.json"),
    ("R3", "rustbpe_variants/auto_tune/audit/mined_pass1/checkpoint.json"),
    ("FLAG", "rustbpe_variants/auto_tune/audit/flag_revisit/checkpoint.json"),
]
_SPARK = "▁▂▃▄▅▆▇█"


def _unit(r):
    return tuple(r["pairs"][0]) if "pairs" in r else tuple(r["pair"])


def sparkline(vals):
    lo, hi = min(vals), max(vals)
    if hi == lo:
        return _SPARK[0] * len(vals)
    return "".join(_SPARK[min(7, int((v - lo) / (hi - lo) * 7.999))] for v in vals)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoints", nargs="+", default=None,
                   help="checkpoint.json paths in keep-order (default: the R1→R2→R3→FLAG chain)")
    p.add_argument("--every", type=int, default=1, help="print every Nth keep (default 1 = all)")
    p.add_argument("--csv", default=None, help="write full per-keep series to this CSV path")
    p.add_argument("--graph", nargs="?", const="results/audit_trajectory.png", default=None,
                   metavar="PNG", help="render a matplotlib graph to PNG (default results/audit_trajectory.png)")
    args = p.parse_args()

    chain = [(None, c) for c in args.checkpoints] if args.checkpoints else DEFAULT_CHAIN
    rows = []
    for tag, cp in chain:
        path = Path(cp)
        if not path.exists():
            continue
        for r in json.loads(path.read_text())["rows"]:
            if r["verdict"] == "KEEP" and r.get("metrics"):
                rows.append((tag, r["kept_size"], _unit(r), r["metrics"], r["deltas"]))
    if not rows:
        raise SystemExit("no KEEP rows found in the checkpoint chain")

    # Derive the empty baseline from keep #1's deltas (deltas track p1/comp/mangled).
    _, _, _, m0, d0 = rows[0]
    base = {"tokens": m0["total_tokens"] - d0["comp"],
            "words": m0["total_words"] - d0["p1"],
            "mangled": m0["mangled"] - d0["mangled"]}

    print(f"baseline (empty): tokens={base['tokens']:,}  words={base['words']:,}  mangled={base['mangled']}\n", flush=True)
    print(f"{'keep':>4} {'src':>4} {'pair':20} {'tokens':>11} {'Δtok':>6} "
          f"{'words':>6} {'base':>6} {'infl':>5} {'mangled':>7} {'dead':>5}", flush=True)
    prev_tag = None
    for tag, k, (L, R), m, d in rows:
        if tag != prev_tag and tag is not None:
            print(f"--- {tag} ---", flush=True)
            prev_tag = tag
        if args.every > 1 and k % args.every != 0 and (tag, k) != (rows[-1][0], rows[-1][1]):
            continue
        print(f"{k:>4} {tag or '':>4} {L+'+'+R:20} {m['total_tokens']:>11,} "
              f"{m['total_tokens']-base['tokens']:>+6} {m['total_words']:>6} "
              f"{m['base_words']:>6} {m['inflected_words']:>5} {m['mangled']:>7} {m['dead']:>5}", flush=True)

    # sparklines over ALL keeps (not sampled)
    ks = [r[1] for r in rows]
    series = {
        "Δtokens vs base": [r[3]["total_tokens"] - base["tokens"] for r in rows],
        "words":           [r[3]["total_words"] for r in rows],
        "mangled":         [r[3]["mangled"] for r in rows],
        "dead":            [r[3]["dead"] for r in rows],
    }
    print(f"\nsparklines over keeps {ks[0]}..{ks[-1]} ({len(rows)} keeps):", flush=True)
    for name, vals in series.items():
        print(f"  {name:16} {sparkline(vals)}  [{min(vals):+}..{max(vals):+}]" if name.startswith('Δ')
              else f"  {name:16} {sparkline(vals)}  [{min(vals)}..{max(vals)}]", flush=True)

    if args.graph:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        dtok = [r[3]["total_tokens"] - base["tokens"] for r in rows]
        wtot = [r[3]["total_words"] for r in rows]
        wbase = [r[3]["base_words"] for r in rows]
        mang = [r[3]["mangled"] for r in rows]
        dead = [r[3]["dead"] for r in rows]
        bounds, prev = [], object()
        for tag, k, *_ in rows:
            if tag != prev:
                bounds.append((k, tag)); prev = tag
        fig, ax = plt.subplots(4, 1, figsize=(11, 11), sharex=True)

        def deco(a):
            for bk, _ in bounds:
                a.axvline(bk - 0.5, color="gray", ls=":", lw=0.7)
            a.grid(alpha=0.3)

        ax[0].plot(ks, dtok, color="tab:red"); ax[0].axhline(0, color="k", lw=0.6)
        ax[0].set_ylabel("Δtokens vs empty\n(+ = compression cost)"); deco(ax[0])
        for bk, tag in bounds:
            if tag:
                ax[0].text(bk, ax[0].get_ylim()[1], f" {tag}", fontsize=8, va="top", color="gray")
        ax[1].plot(ks, [w - wtot[0] for w in wtot], color="tab:green", label="total")
        ax[1].plot(ks, [w - wbase[0] for w in wbase], color="tab:olive", ls="--", label="base")
        ax[1].axhline(0, color="k", lw=0.6)
        ax[1].set_ylabel("Δwhole-words vs keep 1\n(+ = coverage gain)"); ax[1].legend(fontsize=8, loc="upper left"); deco(ax[1])
        ax[2].plot(ks, mang, color="tab:blue"); ax[2].set_ylabel("mangled (heuristic)"); deco(ax[2])
        ax[3].plot(ks, dead, color="tab:purple"); ax[3].set_ylabel("dead"); deco(ax[3])
        ax[3].set_xlabel("# keeps")
        fig.suptitle("auto_tune block-list audit — reference-metric trajectory (tune shard 06542)")
        fig.tight_layout()
        Path(args.graph).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.graph, dpi=120)
        print(f"\nwrote graph {args.graph}", flush=True)

    if args.csv:
        import csv
        with open(args.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["keep", "src", "L", "R", "tokens", "dtok_vs_base", "words",
                        "base_words", "infl_words", "mangled", "dead"])
            for tag, k, (L, R), m, d in rows:
                w.writerow([k, tag or "", L, R, m["total_tokens"], m["total_tokens"] - base["tokens"],
                            m["total_words"], m["base_words"], m["inflected_words"], m["mangled"], m["dead"]])
        print(f"\nwrote {args.csv} ({len(rows)} rows)", flush=True)


if __name__ == "__main__":
    main()
