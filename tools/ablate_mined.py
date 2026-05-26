"""
Cumulative ablation over the FORCED list. For each K value, trains a
tokenizer with the top-K mined FORCED pairs (via
`wrappers.tok_train_force_merges_mined_forced_subset` + env `MINED_TOP_K`),
runs `tools.eval_tokenizer`, then diffs the JSONs to surface which Tier-1
metric changes at which K step. Output is a delta table on stdout +
per-K JSON cache at `results/mined_ablation/`.

Tier-1 signal floor is coarse — most regression-worthy effects only show up
at Lambda-scale CORE/val/bpb. This tool flags suspicious buckets to drill
into there; a clean run is permission to skip further mining-list scrutiny.

Cache invalidation: variant dir + eval JSON include an 8-char hash of all
tokenizer-defining inputs (the mining artifact, the subset module,
`pairs.py`, the wrapper, `rustbpe_variants/force_merges/src/lib.rs`, and a
git fingerprint of upstream `nanochat`). Any of these changing → fresh
tokenizer + fresh eval automatically. `--force` bypasses both layers for
the rare case where the installed `.so` was rebuilt outside this tree.

Usage:
    uv run python -m tools.ablate_mined --k-values 25,50,75,90,105
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
    # Canonical post-shadow-fix mining artifact (subset loads its FORCED_PAIRS).
    _REPO / "rustbpe_variants/force_merges/mined/forced_pairs_climbmix_20rg_twopass.py",
    _REPO / "rustbpe_variants/force_merges/pairs_mined_forced_subset.py",
    _REPO / "rustbpe_variants/force_merges/pairs.py",
    _REPO / "wrappers/tok_train_force_merges_mined_forced_subset.py",
    _REPO / "rustbpe_variants/force_merges/src/lib.rs",
]
_SRC_HASH = compute_src_hash(_INPUT_FILES, _REPO)


def variant_dir(k: int) -> Path:
    return Path.home() / f".cache/nanochat-variants/force_merges_mined_k{k}_{_SRC_HASH}"


def train_one(k: int, force: bool = False) -> Path:
    """Train the FORCED-subset variant at MINED_TOP_K=k. Return tokenizer dir."""
    base = variant_dir(k)
    out_dir = base / "tokenizer"
    if out_dir.exists() and (out_dir / "tokenizer.pkl").exists() and not force:
        print(f"  [skip] K={k} already trained at {out_dir}", file=sys.stderr)
        return out_dir
    if force and base.exists():
        print(f"  [force] removing cached {out_dir}", file=sys.stderr)
        shutil.rmtree(base)
    print(f"  [train] K={k} (src_hash={_SRC_HASH}) ...", file=sys.stderr)
    env = {**os.environ, "MINED_TOP_K": str(k), "VARIANT_BASE": str(base)}
    res = subprocess.run(
        ["uv", "run", "python", "-m",
         "wrappers.tok_train_force_merges_mined_forced_subset"],
        env=env, capture_output=True, text=True,
    )
    if res.returncode != 0:
        print(res.stdout[-2000:], file=sys.stderr)
        print(res.stderr[-2000:], file=sys.stderr)
        raise RuntimeError(f"training failed at K={k}")
    return out_dir


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--k-values", type=str, default="25,50,75,90,105",
                   help="Comma-separated K values to train (default: 25,50,75,90,105 — "
                        "covers the current 105-entry canonical FORCED list)")
    p.add_argument("--results-dir", type=Path,
                   default=Path("results/mined_ablation"),
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
        json_out = args.results_dir / f"eval_k{k}_{_SRC_HASH}.json"
        reports.append((k, eval_one(td, json_out, force=args.force)))

    diff_runs(reports, k_label="K")


if __name__ == "__main__":
    main()
