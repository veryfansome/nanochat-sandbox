"""
Cumulative ablation over the DERIVED list. For each K value, trains a
tokenizer with all 87 mined FORCED + top-K mined DERIVED + 2 numeric
DERIVED (via `wrappers.tok_train_force_merges_mined_derived_subset` + env
`DERIVED_TOP_K`), runs `tools.eval_tokenizer`, diffs the JSONs. Outputs a
delta table on stdout + per-K JSON cache at `results/derived_ablation/`.

Drill-down workflow after this surfaces a regressing bucket: re-run with
finer `--k-values` (e.g. `15,16,17,18,19,20`) to single-step the bucket;
if multiple pairs in the bucket are individually-sufficient causes, use
`wrappers.tok_train_force_merges_mined_ablate` with `INCLUDE_INDICES` to
attribute.

Usage:
    uv run python -m tools.ablate_derived --k-values 0,5,10,15,20
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

HOME = Path.home()

_REPO = Path(__file__).parent.parent


def _upstream_nanochat_fingerprint() -> bytes:
    """Best-effort fingerprint of upstream nanochat (next to the sandbox).
    See tools/ablate_mined.py for the full rationale."""
    upstream = _REPO.parent / "nanochat"
    try:
        sha = subprocess.run(
            ["git", "-C", str(upstream), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True, timeout=5,
        ).stdout.strip()
        return f"nanochat@{sha}".encode()
    except Exception:
        pass
    parts = []
    for rel in ("nanochat/tokenizer.py", "scripts/tok_train.py"):
        f = upstream / rel
        if f.exists():
            parts.append(f.read_bytes())
    return b"\n".join(parts) if parts else b"<upstream-unavailable>"


# Hash of ALL tokenizer-defining inputs for the DERIVED-subset wrapper.
# Edits to any of these change the hash → variant dirs change →
# trained-tokenizer and eval-JSON caches both invalidate automatically.
# (For truly bypassing the cache without changing source — e.g. you suspect
# stale state from a `.so` rebuilt outside this tree — pass `--force`.)
_INPUT_FILES = [
    _REPO / "rustbpe_variants/force_merges/mined/forced_pairs_climbmix_20rg_twopass.py",
    _REPO / "rustbpe_variants/force_merges/pairs_mined_derived_subset.py",
    _REPO / "rustbpe_variants/force_merges/pairs.py",
    _REPO / "wrappers/tok_train_force_merges_mined_derived_subset.py",
    _REPO / "rustbpe_variants/force_merges/src/lib.rs",
]
_SRC_HASH = hashlib.sha1(
    b"\n".join([p.read_bytes() for p in _INPUT_FILES]
               + [_upstream_nanochat_fingerprint()])
).hexdigest()[:8]


def variant_dir(k: int) -> Path:
    return HOME / f".cache/nanochat-variants/force_merges_mined_derived_d{k}_{_SRC_HASH}"


def train_one(k: int, force: bool = False) -> Path:
    base = variant_dir(k)
    out_dir = base / "tokenizer"
    if out_dir.exists() and (out_dir / "tokenizer.pkl").exists() and not force:
        print(f"  [skip] DERIVED_TOP_K={k} already trained at {out_dir}",
              file=sys.stderr)
        return out_dir
    if force and out_dir.exists():
        print(f"  [force] removing cached {out_dir}", file=sys.stderr)
        import shutil
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


def eval_one(tokenizer_dir: Path, json_out: Path, force: bool = False) -> list:
    """Run eval_tokenizer; invalidate cache if the cached spec doesn't
    match the expected tokenizer_dir (e.g. wrapper output path renamed).
    `force=True` deletes the cached JSON unconditionally."""
    if force and json_out.exists():
        print(f"  [force] removing cached {json_out.name}", file=sys.stderr)
        json_out.unlink()
    if json_out.exists():
        cached = json.loads(json_out.read_text())
        cached_spec = cached[0].get("spec") if cached else None
        if cached_spec == str(tokenizer_dir):
            return cached
        print(
            f"  [stale] {json_out.name}: spec {cached_spec!r} != expected "
            f"{str(tokenizer_dir)!r}; re-running eval",
            file=sys.stderr,
        )
        json_out.unlink()
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
        rep = eval_one(td, json_out, force=args.force)
        reports.append((k, rep))

    diff_runs(reports)


if __name__ == "__main__":
    main()
