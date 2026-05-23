"""Pull metrics from wandb and compare two or more runs side by side.

Uses the credentials already in ~/.netrc from `wandb login` (no extra setup).
The first positional run is the reference; deltas are computed against it.

Usage:
    # Basic A/B (zloss vs baseline)
    uv run python -m tools.compare_runs d6_baseline d6_zloss

    # Three-way comparison (e.g., a coefficient sweep)
    uv run python -m tools.compare_runs d24_baseline d24_zloss_1e4 d24_zloss_1e3

    # Different project / entity (rare — we default to base_train.py's project="nanochat")
    uv run python -m tools.compare_runs --project nanochat --entity my_team d6_baseline d6_zloss

    # JSON output for piping into other tools / spreadsheets
    uv run python -m tools.compare_runs --json d6_baseline d6_zloss

What it pulls (per run):
  - Summary scalars: val/bpb, train/loss, train/tok_per_sec, train/mfu,
    total_training_time, total_training_flops, core_metric (if logged)
  - Trajectory: val/bpb at every eval step (shared steps shown side by side)
  - train/loss summary: count, max, min, last-5 mean

Extend by adding to SUMMARY_FIELDS / TRAJECTORY_FIELDS below.

What gets logged to wandb is defined by `scripts/base_train.py` (lines ~430,
~447, ~580 in upstream). CORE is only logged if `--core-metric-every > 0`
(disabled by default in runs/runcpu.sh).
"""
from __future__ import annotations

import argparse
import json
import sys
from statistics import mean
from typing import Any


# (wandb summary key, display name, format spec). Extend freely.
SUMMARY_FIELDS: list[tuple[str, str, str]] = [
    ("val/bpb",              "val/bpb (final)",      "{:.6f}"),
    ("train/loss",           "train/loss (final)",   "{:.6f}"),
    ("core_metric",          "CORE (if logged)",     "{:.6f}"),
    ("train/tok_per_sec",    "train tok/sec",        "{:.1f}"),
    ("train/mfu",            "train MFU",            "{:.4f}"),
    ("total_training_time",  "total time (s)",       "{:.1f}"),
    ("total_training_flops", "total FLOPs",          "{:.3e}"),
]

# Keys to fetch full step-by-step history for (for trajectory analysis).
TRAJECTORY_FIELDS: list[str] = ["val/bpb", "train/loss"]


def _hist(run: Any, key: str) -> list[tuple[int, float]]:
    """Return [(step, value), ...] in step order, skipping rows with None."""
    rows = run.history(keys=["step", key], pandas=False)
    return [(r["step"], r[key]) for r in rows if r.get(key) is not None]


def _fmt(value: Any, spec: str) -> str:
    if value is None:
        return "—"
    try:
        return spec.format(value)
    except Exception:
        return str(value)


def _fmt_delta(value: float, ref: float, spec: str) -> str:
    delta = value - ref
    pct = (delta / ref * 100) if ref else 0
    sign = "+" if delta >= 0 else ""
    return f"{sign}{spec.format(delta)} ({sign}{pct:.2f}%)"


def print_summary_table(runs: dict[str, Any], ref_name: str) -> None:
    ordered = [ref_name] + [n for n in runs if n != ref_name]
    col_w = max(20, max(len(n) for n in ordered) + 2)
    headers = ["metric"] + ordered + [f"Δ vs {ref_name}" for _ in ordered[1:]]
    widths = [26] + [col_w] * len(ordered) + [col_w] * (len(ordered) - 1)

    sep = "  "
    def row(cells: list[str]) -> str:
        return sep.join(c.rjust(w) if i > 0 else c.ljust(w) for i, (c, w) in enumerate(zip(cells, widths)))

    total_w = sum(widths) + len(sep) * (len(widths) - 1)
    print("=" * total_w)
    print(row(headers))
    print("-" * total_w)

    for key, label, spec in SUMMARY_FIELDS:
        cells: list[str] = [label]
        ref_val = runs[ref_name].summary.get(key)
        for name in ordered:
            v = runs[name].summary.get(key)
            cells.append(_fmt(v, spec))
        for name in ordered[1:]:
            v = runs[name].summary.get(key)
            cells.append(_fmt_delta(v, ref_val, spec) if (v is not None and ref_val is not None) else "—")
        print(row(cells))
    print("=" * total_w)


def print_trajectory(runs: dict[str, Any], ref_name: str, key: str = "val/bpb") -> None:
    histories = {name: dict(_hist(run, key)) for name, run in runs.items()}
    common = sorted(set.intersection(*(set(h) for h in histories.values()))) if histories else []
    if not common:
        print(f"\nno common steps for {key} across runs — skipping trajectory")
        return

    ordered = [ref_name] + [n for n in runs if n != ref_name]
    print(f"\n{key} trajectory (steps where all runs have a value):")
    header = f"  {'step':>6}" + "".join(f"{n:>14}" for n in ordered) + "".join(f"  Δ {n} vs {ref_name}" for n in ordered[1:] if len(ordered) > 1)
    print(header)
    for s in common:
        ref_v = histories[ref_name][s]
        line = f"  {s:>6}" + "".join(f"{histories[n][s]:>14.6f}" for n in ordered)
        for n in ordered[1:]:
            line += f"  {histories[n][s] - ref_v:+13.6f}"
        print(line)


def print_train_loss_stats(runs: dict[str, Any]) -> None:
    print("\ntrain/loss summary (smoothed series logged every 100 steps):")
    for name, run in runs.items():
        losses = [v for _, v in _hist(run, "train/loss")]
        if not losses:
            print(f"  {name:<24} (no train/loss in history)")
            continue
        print(f"  {name:<24} count={len(losses):4d}  max={max(losses):.4f}  min={min(losses):.4f}  final_mean(last5)={mean(losses[-min(5, len(losses)):]):.4f}")


def as_json(runs: dict[str, Any]) -> dict:
    out: dict[str, dict] = {}
    for name, run in runs.items():
        rec: dict[str, Any] = {
            "id": run.id,
            "state": run.state,
            "created_at": str(run.created_at),
            "url": run.url,
            "summary": {k: run.summary.get(k) for k, _, _ in SUMMARY_FIELDS},
            "trajectory": {key: _hist(run, key) for key in TRAJECTORY_FIELDS},
        }
        out[name] = rec
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Compare metrics across wandb runs.")
    p.add_argument("runs", nargs="+", help="Run names to compare. First is the reference for deltas.")
    p.add_argument("--project", default="nanochat", help="wandb project (default: nanochat, matching base_train.py)")
    p.add_argument("--entity", default=None, help="wandb entity (default: auto via api.default_entity)")
    p.add_argument("--json", action="store_true", help="Dump structured JSON instead of the human table")
    args = p.parse_args(argv)

    # Import wandb lazily so --help works without it installed.
    import wandb
    api = wandb.Api()
    entity = args.entity or api.default_entity
    project_path = f"{entity}/{args.project}" if entity else args.project

    found = {r.name: r for r in api.runs(project_path) if r.name in set(args.runs)}
    missing = [n for n in args.runs if n not in found]
    if missing:
        print(f"runs not found in {project_path}: {missing}", file=sys.stderr)
        print(f"available runs:", file=sys.stderr)
        for r in api.runs(project_path):
            print(f"  {r.name}  ({r.state})", file=sys.stderr)
        return 2

    # Preserve order the user gave on the CLI.
    runs = {n: found[n] for n in args.runs}
    ref = args.runs[0]

    if args.json:
        json.dump(as_json(runs), sys.stdout, default=str, indent=2)
        print()
        return 0

    print(f"project: {project_path}")
    print(f"runs:")
    for n, r in runs.items():
        marker = "  (reference)" if n == ref else ""
        print(f"  {n:<24} id={r.id}  state={r.state}  {r.url}{marker}")
    print()

    print_summary_table(runs, ref)
    print_trajectory(runs, ref, "val/bpb")
    print_train_loss_stats(runs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
