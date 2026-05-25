"""
Cumulative ablation driver for the mined FORCED_PAIRS list.

Trains a sequence of tokenizers using the top-K mined pairs for several K
values, runs eval_tokenizer on each, and emits a delta table showing which
metric changes at each step. The intent mirrors the user's original
incremental hand-curation workflow: add candidates in batches, verify each
step.

Usage:
    uv run python -m tools.ablate_mined --k-values 25,50,75,87

Each K value triggers a fresh tokenizer training (~5-6 min each on CPU) and
writes to ~/.cache/nanochat-variants/force_merges_mined_k${K}/tokenizer/.
After all training completes, eval_tokenizer is invoked with all K dirs and
the resulting JSON is diffed.

This tool exists mainly to validate that no individual bucket of mined
candidates causes a Tier-1 regression. With the current Tier-1 metrics the
signal floor is coarse — most regression-worthy effects only surface at
Lambda-scale CORE/val/bpb eval. The output identifies suspicious buckets to
re-test there; a fully-clean run gives confidence to ship the larger mined
list.
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def train_one(k: int) -> Path:
    """Train the mined_subset variant at MINED_TOP_K=k. Return tokenizer dir."""
    out_dir = Path.home() / f".cache/nanochat-variants/force_merges_mined_k{k}/tokenizer"
    if out_dir.exists() and (out_dir / "tokenizer.pkl").exists():
        print(f"  [skip] K={k} already trained at {out_dir}", file=sys.stderr)
        return out_dir
    print(f"  [train] K={k} ...", file=sys.stderr)
    env = {**os.environ, "MINED_TOP_K": str(k)}
    res = subprocess.run(
        ["uv", "run", "python", "-m",
         "wrappers.tok_train_force_merges_mined_subset"],
        env=env, capture_output=True, text=True,
    )
    if res.returncode != 0:
        print(res.stdout[-2000:], file=sys.stderr)
        print(res.stderr[-2000:], file=sys.stderr)
        raise RuntimeError(f"training failed at K={k}")
    return out_dir


def eval_one(tokenizer_dir: Path, json_out: Path) -> dict:
    """Run eval_tokenizer on one tokenizer and parse the JSON output."""
    if json_out.exists():
        return json.loads(json_out.read_text())
    res = subprocess.run(
        ["uv", "run", "python", "-m", "tools.eval_tokenizer",
         "--tokenizers", str(tokenizer_dir),
         "--json", str(json_out)],
        capture_output=True, text=True,
    )
    if res.returncode != 0:
        print(res.stdout[-2000:], file=sys.stderr)
        print(res.stderr[-2000:], file=sys.stderr)
        raise RuntimeError(f"eval failed for {tokenizer_dir}")
    return json.loads(json_out.read_text())


def diff_runs(reports: list[tuple[int, dict]]) -> None:
    """Print a delta table: which compression / probe values change at each K step."""
    print("\n=== Cumulative ablation: compression bytes/token deltas ===\n")
    # Each report is a list of {name, compression: {corpus: {bytes_per_token}}, ...}
    corpora = sorted({
        c for _, r in reports for c in r[0]["compression"].keys()
    })
    header = "corpus".ljust(18) + " ".join(f"K={k:<6}".rjust(8) for k, _ in reports)
    print(header)
    print("-" * len(header))
    for corpus in corpora:
        row = [corpus.ljust(18)]
        prev = None
        for k, r in reports:
            bpt = r[0]["compression"][corpus]["bytes_per_token"]
            cell = f"{bpt:.3f}"
            if prev is not None:
                delta = bpt - prev
                if abs(delta) > 1e-6:
                    sign = "+" if delta > 0 else ""
                    cell = f"{bpt:.3f}({sign}{delta:.3f})"
            row.append(cell.rjust(8))
            prev = bpt
        print(" ".join(row))

    print("\n=== Cumulative ablation: structural battery ===\n")
    tests = sorted({
        t for _, r in reports for t in r[0]["structural"].keys()
    })
    header = "test".ljust(22) + " ".join(f"K={k:<5}".rjust(8) for k, _ in reports)
    print(header)
    print("-" * len(header))
    for test in tests:
        row = [test.ljust(22)]
        for k, r in reports:
            res = r[0]["structural"][test]
            cell = "PASS" if res.get("ok", False) else "FAIL"
            row.append(cell.rjust(8))
        print(" ".join(row))

    print("\n=== Cumulative ablation: task probes (token count) ===\n")
    probes = sorted({
        p for _, r in reports for p in r[0]["task_probes"].keys()
    })
    header = "probe".ljust(18) + " ".join(f"K={k:<6}".rjust(8) for k, _ in reports)
    print(header)
    print("-" * len(header))
    for probe in probes:
        row = [probe.ljust(18)]
        prev = None
        for k, r in reports:
            n = r[0]["task_probes"][probe]["tokens"]
            cell = f"{n:>4}"
            if prev is not None:
                delta = n - prev
                if delta != 0:
                    sign = "+" if delta > 0 else ""
                    cell = f"{n:>4}({sign}{delta})"
            row.append(cell.rjust(8))
            prev = n
        print(" ".join(row))


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--k-values", type=str, default="25,50,75,87",
                   help="Comma-separated K values to train (default: 25,50,75,87)")
    p.add_argument("--results-dir", type=Path,
                   default=Path("results/mined_ablation"),
                   help="Directory for per-K JSON outputs")
    args = p.parse_args()

    k_values = sorted(int(x) for x in args.k_values.split(","))
    args.results_dir.mkdir(parents=True, exist_ok=True)

    reports = []
    for k in k_values:
        td = train_one(k)
        json_out = args.results_dir / f"eval_k{k}.json"
        rep = eval_one(td, json_out)
        reports.append((k, rep))

    diff_runs(reports)


if __name__ == "__main__":
    main()
