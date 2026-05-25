"""
DERIVED-pair cumulative ablation driver.

Trains a sequence of tokenizers with DERIVED_TOP_K ∈ {0, 5, 10, 15, 20}
(numeric DERIVED + full FORCED stay fixed) and diffs the eval output. K=0
is symlinked to the already-trained force_merges_mined_v1 (numeric DERIVED
only); K=20 to force_merges_mined_v2 (full kitchen sink). The middle K
values are trained fresh.

Usage:
    uv run python -m tools.ablate_derived

Outputs per-K eval JSONs at results/derived_ablation/eval_d${K}.json plus a
diff table on stdout.
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

HOME = Path.home()


def ensure_baseline_links():
    """Symlink k=0 → force_merges_mined_v1, k=20 → force_merges_mined_v2."""
    aliases = {
        0: HOME / ".cache/nanochat-variants/force_merges_mined_v1",
        20: HOME / ".cache/nanochat-variants/force_merges_mined_v2",
    }
    for k, target in aliases.items():
        link = HOME / f".cache/nanochat-variants/force_merges_mined_v2_d{k}"
        if not link.exists() and target.exists():
            link.symlink_to(target)


def train_one(k: int) -> Path:
    out_dir = HOME / f".cache/nanochat-variants/force_merges_mined_v2_d{k}/tokenizer"
    if out_dir.exists() and (out_dir / "tokenizer.pkl").exists():
        print(f"  [skip] DERIVED_TOP_K={k} already trained at {out_dir}",
              file=sys.stderr)
        return out_dir
    print(f"  [train] DERIVED_TOP_K={k} ...", file=sys.stderr)
    env = {**os.environ, "DERIVED_TOP_K": str(k)}
    res = subprocess.run(
        ["uv", "run", "python", "-m",
         "wrappers.tok_train_force_merges_mined_v2_subset"],
        env=env, capture_output=True, text=True,
    )
    if res.returncode != 0:
        print(res.stdout[-2000:], file=sys.stderr)
        print(res.stderr[-2000:], file=sys.stderr)
        raise RuntimeError(f"training failed at DERIVED_TOP_K={k}")
    return out_dir


def eval_one(tokenizer_dir: Path, json_out: Path) -> list:
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


def diff_runs(reports: list[tuple[int, list]]) -> None:
    print("\n=== DERIVED ablation: compression bytes/token (deltas vs prev K) ===\n")
    corpora = sorted({c for _, r in reports for c in r[0]["compression"].keys()})
    header = "corpus".ljust(18) + " ".join(f"K={k:<6}".rjust(10) for k, _ in reports)
    print(header)
    print("-" * len(header))
    for corpus in corpora:
        row = [corpus.ljust(18)]
        prev = None
        for k, r in reports:
            bpt = r[0]["compression"][corpus]["bytes_per_token"]
            cell = f"{bpt:.3f}"
            if prev is not None:
                d = bpt - prev
                if abs(d) > 1e-6:
                    sign = "+" if d > 0 else ""
                    cell = f"{bpt:.3f}({sign}{d:.3f})"
            row.append(cell.rjust(10))
            prev = bpt
        print(" ".join(row))

    print("\n=== DERIVED ablation: structural battery ===\n")
    tests = sorted({t for _, r in reports for t in r[0]["structural"].keys()})
    header = "test".ljust(22) + " ".join(f"K={k:<5}".rjust(8) for k, _ in reports)
    print(header)
    print("-" * len(header))
    any_fail = False
    for test in tests:
        row = [test.ljust(22)]
        for k, r in reports:
            ok = r[0]["structural"][test].get("ok", False)
            cell = "PASS" if ok else "FAIL"
            if not ok:
                any_fail = True
            row.append(cell.rjust(8))
        print(" ".join(row))
    if not any_fail:
        print("\n(all structural tests PASS at every K)")

    print("\n=== DERIVED ablation: task probes (token count) ===\n")
    probes = sorted({p for _, r in reports for p in r[0]["task_probes"].keys()})
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
                d = n - prev
                if d != 0:
                    sign = "+" if d > 0 else ""
                    cell = f"{n:>4}({sign}{d})"
            row.append(cell.rjust(8))
            prev = n
        print(" ".join(row))


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--k-values", type=str, default="0,5,10,15,20",
                   help="Comma-separated DERIVED_TOP_K values (default: 0,5,10,15,20)")
    p.add_argument("--results-dir", type=Path,
                   default=Path("results/derived_ablation"),
                   help="Directory for per-K JSON outputs")
    args = p.parse_args()

    k_values = sorted(int(x) for x in args.k_values.split(","))
    args.results_dir.mkdir(parents=True, exist_ok=True)

    ensure_baseline_links()

    reports = []
    for k in k_values:
        td = train_one(k)
        json_out = args.results_dir / f"eval_d{k}.json"
        rep = eval_one(td, json_out)
        reports.append((k, rep))

    diff_runs(reports)


if __name__ == "__main__":
    main()
