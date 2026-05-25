"""Pull metrics from wandb (and local base_eval archives) and compare two or
more runs side by side.

Uses the credentials already in ~/.netrc from `wandb login` (no extra setup).
The first positional run is the reference; deltas are computed against it.

Usage:
    # Basic A/B (zloss vs baseline)
    uv run python -m tools.compare_runs d6_baseline d6_zloss

    # Three-way comparison (e.g., a coefficient sweep)
    uv run python -m tools.compare_runs d24_baseline d24_zloss_1e4 d24_zloss_1e3

    # Different project / entity (rare — we default to base_train.py's project="nanochat")
    uv run python -m tools.compare_runs --project nanochat --entity my_team d6_baseline d6_zloss

    # Different local archive directory (default: sandbox/results/)
    uv run python -m tools.compare_runs --results-dir /path/to/archives d24_baseline d24_zloss

    # JSON output for piping into other tools / spreadsheets
    uv run python -m tools.compare_runs --json d6_baseline d6_zloss

What it pulls (per run):
  - Wandb summary scalars: val/bpb, train/loss, train/tok_per_sec, train/mfu,
    total_training_time, total_training_flops, and an in-training CORE
    subsample (if logged — `--core-metric-every > 0`).
  - Local `base_eval` CORE from `<results-dir>/<run-name>/eval.csv` (authoritative,
    full per-task budget). Two CORE columns are shown side by side because the
    in-training subsample uses a smaller per-task budget and can disagree with
    the post-training base_eval by ~0.01; the base_eval number is the citable one.
  - LM tokenization boundary-crossing rate from `lm_boundary_stats.json` (written
    by wrappers/base_eval.py). Non-zero means LM-task CORE for that run isn't
    directly comparable to prefix-stable baselines — see wrappers/_patches.py.
  - Trajectory: val/bpb at every eval step (shared steps shown side by side)
  - train/loss summary: count, max, min, last-5 mean

Extend by adding to SUMMARY_FIELDS / EVAL_CSV_FIELDS / TRAJECTORY_FIELDS below.

What gets logged to wandb is defined by `scripts/base_train.py` (lines ~430,
~447, ~580 in upstream). The in-training CORE is logged only if
`--core-metric-every > 0` (disabled in runs/runcpu.sh; defaulted on at last
step in upstream base_train).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from statistics import mean
from typing import Any


# (wandb summary key, display name, format spec). Extend freely.
SUMMARY_FIELDS: list[tuple[str, str, str]] = [
    ("val/bpb",              "val/bpb (final)",            "{:.6f}"),
    ("train/loss",           "train/loss (final)",         "{:.6f}"),
    ("core_metric",          "CORE (wandb, mid-train sub)","{:.6f}"),
    ("train/tok_per_sec",    "train tok/sec",              "{:.1f}"),
    ("train/mfu",            "train MFU",                  "{:.4f}"),
    ("total_training_time",  "total time (s)",             "{:.1f}"),
    ("total_training_flops", "total FLOPs",                "{:.3e}"),
]

# (task name in eval.csv, display name, format spec). Read from the local
# `<results-dir>/<run-name>/eval.csv` archive that runs/speedrun.sh writes
# after `base_eval` completes. The "CORE" row at the bottom of that CSV is
# the authoritative post-training value (full per-task budget). Add more rows
# here to surface specific tasks (e.g. piqa, lambada_openai).
EVAL_CSV_FIELDS: list[tuple[str, str, str]] = [
    ("CORE",                 "CORE (base_eval, full)",     "{:.6f}"),
]

# Keys to fetch full step-by-step history for (for trajectory analysis).
TRAJECTORY_FIELDS: list[str] = ["val/bpb", "train/loss"]


def _default_results_dir() -> Path:
    """Sandbox's results/ directory — two levels up from this file
    (tools/compare_runs.py → sandbox/results/)."""
    return Path(__file__).resolve().parent.parent / "results"


def _read_run_name(archive_dir: Path) -> str | None:
    """Recover the wandb run name from an archive directory. `speedrun.sh`
    archives under `results/$MODEL_TAG/`, which is *not* necessarily the same
    as `$WANDB_RUN` — e.g. a coefficient sweep may set distinct values to
    keep wandb dashboards readable while sharing a checkpoint dir layout.
    The wandb name is recorded in two places by the archive step; we prefer
    `meta.json` (structured) and fall back to the `ENV` file (bash-style)."""
    meta = archive_dir / "meta.json"
    if meta.exists():
        try:
            with meta.open() as f:
                data = json.load(f)
            run = data.get("user_config", {}).get("run")
            if run and run != "dummy":
                return run
        except Exception:
            pass
    env = archive_dir / "ENV"
    if env.exists():
        try:
            for line in env.read_text().splitlines():
                if line.startswith("WANDB_RUN="):
                    run = line.split("=", 1)[1].strip()
                    if run and run != "dummy":
                        return run
        except Exception:
            pass
    return None


def _build_run_index(results_dir: Path) -> dict[str, Path]:
    """Scan `results_dir/*/` and build `wandb_run_name -> archive_dir`. If two
    archives claim the same wandb name (re-run with same `$WANDB_RUN` but
    different `$MODEL_TAG`), the most recently modified one wins."""
    index: dict[str, Path] = {}
    if not results_dir.is_dir():
        return index
    # Sort by mtime ascending so later entries overwrite earlier ones — final
    # state is "most recent archive wins per run name."
    subdirs = [p for p in results_dir.iterdir() if p.is_dir()]
    subdirs.sort(key=lambda p: p.stat().st_mtime)
    for subdir in subdirs:
        run_name = _read_run_name(subdir)
        if run_name:
            index[run_name] = subdir
    return index


def _load_eval_csv(run_name: str, run_index: dict[str, Path]) -> tuple[dict[str, float], Path | None]:
    """Look up archive for `run_name` via the index (built from `meta.json`'s
    `user_config.run`, NOT from directory name). Returns `({}, None)` if no
    archive is indexed for this run name. Otherwise parses `eval.csv` and
    returns `(scores, archive_dir)`."""
    archive_dir = run_index.get(run_name)
    if archive_dir is None:
        return {}, None
    path = archive_dir / "eval.csv"
    if not path.exists():
        return {}, archive_dir
    out: dict[str, float] = {}
    with path.open() as f:
        reader = csv.reader(f)
        next(reader, None)  # header: Task, Accuracy, Centered
        for row in reader:
            if len(row) < 3:
                continue
            task = row[0].strip()
            centered = row[2].strip()
            if not task or not centered:
                continue
            try:
                out[task] = float(centered)
            except ValueError:
                continue
    return out, archive_dir


def _load_boundary_stats(run_name: str, run_index: dict[str, Path]) -> dict | None:
    """Read `lm_boundary_stats.json` from the run's archive if present.

    Written by wrappers/base_eval.py at end-of-eval. Records how often
    LM-style CORE prompts had a tokenization-boundary-crossing merge under
    the prefix-safety patch (wrappers/_patches.py); non-zero rates mean
    LM-task CORE numbers from this run aren't directly comparable to runs
    with a different tokenization. Returns None if archive missing, json
    missing, or json malformed (graceful degrade — older archives predate
    this file, treated the same as runs that didn't trigger LM eval)."""
    archive_dir = run_index.get(run_name)
    if archive_dir is None:
        return None
    path = archive_dir / "lm_boundary_stats.json"
    if not path.exists():
        return None
    try:
        with path.open() as f:
            return json.load(f)
    except Exception:
        return None


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


def print_summary_table(
    runs: dict[str, Any],
    ref_name: str,
    eval_scores: dict[str, dict[str, float]],
    boundary_stats: dict[str, dict | None],
) -> None:
    ordered = [ref_name] + [n for n in runs if n != ref_name]
    col_w = max(20, max(len(n) for n in ordered) + 2)
    headers = ["metric"] + ordered + [f"Δ vs {ref_name}" for _ in ordered[1:]]
    label_w = max(30, max(len(label) for _, label, _ in SUMMARY_FIELDS + EVAL_CSV_FIELDS) + 2)
    widths = [label_w] + [col_w] * len(ordered) + [col_w] * (len(ordered) - 1)

    sep = "  "
    def row(cells: list[str]) -> str:
        return sep.join(c.rjust(w) if i > 0 else c.ljust(w) for i, (c, w) in enumerate(zip(cells, widths)))

    total_w = sum(widths) + len(sep) * (len(widths) - 1)
    print("=" * total_w)
    print(row(headers))
    print("-" * total_w)

    # Wandb-sourced rows (run.summary)
    for key, label, spec in SUMMARY_FIELDS:
        cells: list[str] = [label]
        ref_val = runs[ref_name].summary.get(key)
        for name in ordered:
            cells.append(_fmt(runs[name].summary.get(key), spec))
        for name in ordered[1:]:
            v = runs[name].summary.get(key)
            cells.append(_fmt_delta(v, ref_val, spec) if (v is not None and ref_val is not None) else "—")
        print(row(cells))

    # base_eval-sourced rows (local eval.csv archives)
    if EVAL_CSV_FIELDS:
        print("-" * total_w)
        for task, label, spec in EVAL_CSV_FIELDS:
            cells = [label]
            ref_val = eval_scores.get(ref_name, {}).get(task)
            for name in ordered:
                cells.append(_fmt(eval_scores.get(name, {}).get(task), spec))
            for name in ordered[1:]:
                v = eval_scores.get(name, {}).get(task)
                cells.append(_fmt_delta(v, ref_val, spec) if (v is not None and ref_val is not None) else "—")
            print(row(cells))

    # LM tokenization-boundary-crossing rate (sidecar JSON written by
    # wrappers/base_eval.py). Non-zero means LM-task CORE for that run isn't
    # directly comparable to prefix-stable baselines — see wrappers/_patches.py.
    # Empty value (—) means: archive predates the JSON, or no LM eval ran.
    if any(b for b in boundary_stats.values()):
        print("-" * total_w)
        cells = ["LM bndry-crossing %"]
        for name in ordered:
            b = boundary_stats.get(name)
            cells.append(
                f"{b['boundary_crossing_pct']:.2f}% ({b['n_boundary_crossing']}/{b['n_total']})"
                if b and b.get("n_total") else "—"
            )
        # No delta column — this is a per-run quality marker, not a metric to subtract.
        cells.extend([""] * (len(ordered) - 1))
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


def as_json(
    runs: dict[str, Any],
    eval_scores: dict[str, dict[str, float]],
    eval_dirs: dict[str, Path | None],
    boundary_stats: dict[str, dict | None],
) -> dict:
    out: dict[str, dict] = {}
    for name, run in runs.items():
        archive_dir = eval_dirs.get(name)
        rec: dict[str, Any] = {
            "id": run.id,
            "state": run.state,
            "created_at": str(run.created_at),
            "url": run.url,
            "summary": {k: run.summary.get(k) for k, _, _ in SUMMARY_FIELDS},
            "archive_dir": str(archive_dir) if archive_dir else None,
            "eval_csv": {task: eval_scores.get(name, {}).get(task) for task, _, _ in EVAL_CSV_FIELDS},
            "lm_boundary_stats": boundary_stats.get(name),
            "trajectory": {key: _hist(run, key) for key in TRAJECTORY_FIELDS},
        }
        out[name] = rec
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Compare metrics across wandb runs.")
    p.add_argument("runs", nargs="+", help="Run names to compare. First is the reference for deltas.")
    p.add_argument("--project", default="nanochat", help="wandb project (default: nanochat, matching base_train.py)")
    p.add_argument("--entity", default=None, help="wandb entity (default: auto via api.default_entity)")
    p.add_argument("--results-dir", default=None, help="Local archive root (default: sandbox/results/; indexed by wandb run name from each subdir's meta.json)")
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

    # Index local archives by wandb run name (NOT by directory name — the two
    # differ for any sweep that sets distinct $WANDB_RUN and $MODEL_TAG).
    results_dir = Path(args.results_dir) if args.results_dir else _default_results_dir()
    run_index = _build_run_index(results_dir)
    eval_scores: dict[str, dict[str, float]] = {}
    eval_dirs: dict[str, Path | None] = {}
    boundary_stats: dict[str, dict | None] = {}
    for name in args.runs:
        scores, archive_dir = _load_eval_csv(name, run_index)
        eval_scores[name] = scores
        eval_dirs[name] = archive_dir
        boundary_stats[name] = _load_boundary_stats(name, run_index)

    if args.json:
        json.dump(as_json(runs, eval_scores, eval_dirs, boundary_stats), sys.stdout, default=str, indent=2)
        print()
        return 0

    print(f"project:     {project_path}")
    print(f"results-dir: {results_dir}  ({len(run_index)} archives indexed by wandb run name)")
    print(f"runs:")
    for n, r in runs.items():
        marker = "  (reference)" if n == ref else ""
        archive = eval_dirs.get(n)
        if eval_scores.get(n):
            csv_marker = f"  [eval.csv ✓ at {archive.name}/]"
        elif archive is not None:
            csv_marker = f"  [archive at {archive.name}/ but no eval.csv]"
        else:
            csv_marker = "  [no indexed archive — pull with `bash runs/lambda.sh pull <id>` or check meta.json]"
        print(f"  {n:<24} id={r.id}  state={r.state}  {r.url}{marker}{csv_marker}")
    print()

    print_summary_table(runs, ref, eval_scores, boundary_stats)
    print_trajectory(runs, ref, "val/bpb")
    print_train_loss_stats(runs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
