"""
Cumulative ablation over the DERIVED list. For each K value, trains a
tokenizer with all 105 mined FORCED + top-K mined DERIVED + 2 numeric
DERIVED (via `wrappers.tok_train_force_merges_mined_derived_subset` + env
`DERIVED_TOP_K`), runs `tools.eval_tokenizer`, diffs the JSONs. Outputs a
delta table on stdout + per-K JSON cache at `results/derived_ablation/`.

Drill-down workflow after this surfaces a regressing bucket: re-run with
finer `--k-values` (e.g. `15,16,17,18,19,20`) to single-step; if multiple
pairs in the bucket are individually-sufficient causes, use
`wrappers.tok_train_force_merges_mined_ablate` with `INCLUDE_INDICES` to
attribute.

Cache invalidation: see `tools.ablate_mined` — same scheme, applied to the
DERIVED-subset's input set.

Usage:
    uv run python -m tools.ablate_derived --k-values 0,5,10,15,20
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

from tools._ablate_common import compute_src_hash, diff_runs, eval_one

_REPO = Path(__file__).parent.parent
_INPUT_FILES = [
    _REPO / "rustbpe_variants/force_merges/mined/forced_pairs_climbmix_20rg_twopass.py",
    _REPO / "rustbpe_variants/force_merges/pairs_mined_derived_subset.py",
    _REPO / "rustbpe_variants/force_merges/pairs.py",
    _REPO / "wrappers/tok_train_force_merges_mined_derived_subset.py",
    _REPO / "rustbpe_variants/force_merges/src/lib.rs",
]
_SRC_HASH = compute_src_hash(_INPUT_FILES, _REPO)


def variant_dir(k: int) -> Path:
    return (Path.home()
            / f".cache/nanochat-variants/force_merges_mined_derived_d{k}_{_SRC_HASH}")


def train_one(k: int, force: bool = False) -> Path:
    base = variant_dir(k)
    out_dir = base / "tokenizer"
    if out_dir.exists() and (out_dir / "tokenizer.pkl").exists() and not force:
        print(f"  [skip] DERIVED_TOP_K={k} already trained at {out_dir}",
              file=sys.stderr)
        return out_dir
    if force and base.exists():
        print(f"  [force] removing cached {out_dir}", file=sys.stderr)
        shutil.rmtree(base)
    print(f"  [train] DERIVED_TOP_K={k} (src_hash={_SRC_HASH}) ...", file=sys.stderr)
    env = {**os.environ, "DERIVED_TOP_K": str(k), "VARIANT_BASE": str(base)}
    res = subprocess.run(
        ["uv", "run", "python", "-m",
         "wrappers.tok_train_force_merges_mined_derived_subset"],
        env=env, capture_output=True, text=True,
    )
    if res.returncode != 0:
        print(res.stdout[-2000:], file=sys.stderr)
        print(res.stderr[-2000:], file=sys.stderr)
        raise RuntimeError(f"training failed at DERIVED_TOP_K={k}")
    return out_dir


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--k-values", type=str, default="0,5,10,15,20",
                   help="Comma-separated DERIVED_TOP_K values (default: 0,5,10,15,20)")
    p.add_argument("--results-dir", type=Path,
                   default=Path("results/derived_ablation"),
                   help="Directory for per-K JSON outputs")
    p.add_argument("--force", action="store_true",
                   help="Bypass cache: retrain tokenizers + re-run evals "
                        "even if hash-matched cache files exist")
    args = p.parse_args()

    k_values = sorted(int(x) for x in args.k_values.split(","))
    args.results_dir.mkdir(parents=True, exist_ok=True)

    reports = []
    for k in k_values:
        td = train_one(k, force=args.force)
        json_out = args.results_dir / f"eval_d{k}_{_SRC_HASH}.json"
        reports.append((k, eval_one(td, json_out, force=args.force)))

    diff_runs(reports, k_label="DERIVED_TOP_K")


if __name__ == "__main__":
    main()
