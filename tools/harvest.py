"""Reusable free-win harvest executor for the auto_tune firing-33 orchestrated audit.

Replaces the per-batch shell scripts (hand-retyping pairs into fragile bash quoting is the
error-prone step) with one tested tool that takes a vetted pair-list as DATA.

EXECUTOR, NOT A JUDGE. The selection of *which* pairs to commit stays a by-hand judgment —
semantic vetoes (don't shatter a real word/morpheme), cascade holds (e.g. the ` ,00` −500
dead-token slot reshuffle), and real-coverage assessment (tools.word_realness) — per the
`autotune-firing33-orchestrated-judge` memory. This tool only *runs* a list I've already
vetted; it never picks.

Two modes:
  # 1. SUGGEST (read-only): print the current board's numeric free-win candidates, ranked by
  #    Δcomp, recovered as exact (L,R) tuples — the shortlist to then realness-review by hand.
  uv run python -m tools.harvest --suggest --state-dir rustbpe_variants/auto_tune/audit/heldout_fixpoint_fm

  # 2. COMMIT a vetted list: one-invocation --pool-override + --commit per pair, in the order
  #    given (use descending Δcomp so overlapping picks evaporate). Each is net-commit
  #    re-validated against the growing kept set; redundant/vetoed ones are skipped, not fatal.
  uv run python -m tools.harvest --pairs '[[" m","ov"],["·","·"],[" inc","lud"]]' \
      --state-dir rustbpe_variants/auto_tune/audit/heldout_fixpoint_fm

Always pass pairs as JSON (a list of 2-element [L,R] lists) — never hand-built shell tokens.
"""
import sys
import json
import argparse
import subprocess
from pathlib import Path

PROJECT_DIR = str(Path(__file__).resolve().parent.parent)

# the audit's standard firing-33 fm-context config — one place to change if the run's flags change
DRIVER_FLAGS = [
    "--fm-context", "--gate-comp-tolerance", "3", "--gate-dead-tolerance", "1",
    "--cooldown-base", "4", "--net-commit", "--net-retries", "99",
    "--mine-mode", "none", "--mine-firing-percentile", "33",
    "--eval-batch", "200", "--propose-top-n", "20",
    # --no-mine: committing an EXPLICIT vetted list doesn't need the pool re-mined after each
    # commit (the driver re-mines by default — a full ~18K corpus scan per commit that made the
    # harvest tool ~5min/commit at large kept; the eventual --propose re-mines via the empty-pool
    # fix). Keeps the (cascade-safe) net-commit settle; drops the pointless per-commit re-scan.
    "--no-mine",
]
TOLC, TOLD = 3.0, 1.0


def gate(dc, dco, dd):
    """Held-out Pareto gate (matches the driver): no axis regresses past tolerance AND ≥1
    strict improvement."""
    if dc < 0 or dco > TOLC or dd > TOLD:
        return False
    return dc > 0 or dco < 0 or dd < 0


def suggest(state_dir, floor):
    """Print gate-passing free-win candidates from the checkpoint's current round (partial or
    complete), ranked by Δcomp. Recovers exact (L,R) tuples via the pool join-key map (verdict
    keys are the ambiguous 'L+R' join). Read-only — no corpus, no training."""
    ckpt = json.loads((Path(state_dir) / "checkpoint.json").read_text())
    pool = [tuple(p) for p in ckpt.get("pool", [])]
    by_key = {f"{c[0]}+{c[1]}": list(c) for c in pool}
    done = (ckpt.get("cur_round") or {}).get("done", {})
    if not done:
        print(f"[harvest] no scored cur_round in {state_dir} — run --propose (or --pool-override "
              f"--propose) first.", file=sys.stderr)
        return
    rows = []
    for k, v in done.items():
        dc, dco, dd = v["d"][:3]
        if k in by_key and gate(dc, dco, dd) and dco <= floor:
            rows.append((by_key[k], dc, dco, dd))
    rows.sort(key=lambda r: r[2])  # most-compressive first
    print(f"# harvest --suggest: {len(rows)} free-win candidate(s) (Δcomp ≤ {floor}) from "
          f"{len(done)} scored / {len(pool)} pool")
    print(f"# REVIEW BY HAND before committing — exclude vetoes/cascades/swaps, realness-check coverage.")
    print(f"# {'(L, R)':<26} {'Δcov':>5} {'Δcomp':>8} {'Δdead':>6}")
    for c, dc, dco, dd in rows:
        print(f"  {str(tuple(c)):<26} {dc:>+5} {dco:>+8.1f} {dd:>+6.1f}")
    print("\n# vetted JSON (EDIT before use — this is every numeric free win, not a judged list):")
    print(json.dumps([c for c, *_ in rows]))


def commit_one(pair, state_dir):
    """Run the driver's one-invocation --pool-override + --commit for a single pair. Returns
    ('committed'|'evaporated'|'vetoed'|'error', detail). Non-zero exit on evaporation is
    expected (the driver SystemExits when the pick no longer gate-passes) — not fatal."""
    cmd = [sys.executable, "-m", "tools.heldout_fixpoint_cached",
           "--pool-override", json.dumps([pair]), "--commit", json.dumps(pair),
           "--state-dir", state_dir] + DRIVER_FLAGS
    r = subprocess.run(cmd, cwd=PROJECT_DIR, capture_output=True, text=True)
    out = r.stdout + r.stderr
    for line in out.splitlines():
        if "→ commit (" in line:
            return "committed", line.split("→ commit", 1)[1].strip()
        if "is not a gate-passing keep" in line or "OVERRIDE EVAPORATED" in line or "→ 0 keeps" in line:
            return "evaporated", "no longer gate-passes vs the grown set"
        if "was vetoed" in line or "net-commit VETO" in line:
            return "vetoed", "net-negative after LOO settling"
    return "error", (out.strip().splitlines() or ["(no output)"])[-1]


def main():
    p = argparse.ArgumentParser(description="Free-win harvest executor for the auto_tune firing-33 audit.")
    p.add_argument("--state-dir", required=True)
    p.add_argument("--suggest", action="store_true", help="read-only: print the board's free-win shortlist")
    p.add_argument("--floor", type=float, default=-2.0, help="--suggest: only Δcomp ≤ this (default -2.0)")
    p.add_argument("--pairs", help='JSON list of vetted [L,R] pairs to commit in order, e.g. \'[[" m","ov"]]\'')
    p.add_argument("--pairs-file", help="path to a file holding the same JSON (alternative to --pairs)")
    args = p.parse_args()

    if args.suggest:
        suggest(args.state_dir, args.floor)
        return

    raw = args.pairs or (Path(args.pairs_file).read_text() if args.pairs_file else None)
    if not raw:
        raise SystemExit("pass --suggest, or --pairs / --pairs-file with a vetted JSON pair-list")
    pairs = [list(pr) for pr in json.loads(raw)]
    if not all(len(pr) == 2 for pr in pairs):
        raise SystemExit("each pair must be a 2-element [L, R] list")

    print(f"[harvest] committing {len(pairs)} vetted pair(s) into {args.state_dir} (net-commit "
          f"re-validated each; overlapping picks evaporate)", flush=True)
    tally = {"committed": [], "evaporated": [], "vetoed": [], "error": []}
    for i, pair in enumerate(pairs, 1):
        status, detail = commit_one(pair, args.state_dir)
        tally[status].append(pair)
        mark = {"committed": "✓", "evaporated": "·", "vetoed": "✗", "error": "!"}[status]
        print(f"[harvest] {i}/{len(pairs)} {mark} {tuple(pair)!s:28} {status}: {detail}", flush=True)
    print(f"\n[harvest] done — committed {len(tally['committed'])}, evaporated {len(tally['evaporated'])}, "
          f"vetoed {len(tally['vetoed'])}, error {len(tally['error'])}")
    if tally["error"]:
        print(f"[harvest] ERRORS on: {[tuple(p) for p in tally['error']]} — inspect manually", file=sys.stderr)


if __name__ == "__main__":
    main()
