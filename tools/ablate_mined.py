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

Usage:
    uv run python -m tools.ablate_mined --k-values 25,50,75,87
"""

import argparse
import json
import hashlib
import os
import subprocess
import sys
from pathlib import Path

_REPO = Path(__file__).parent.parent


def _upstream_nanochat_fingerprint() -> bytes:
    """Best-effort fingerprint of upstream nanochat (next to the sandbox).

    The wrapper imports `nanochat.tokenizer` and `runpy`s `scripts.tok_train`
    — both upstream, both pull-able. Without including upstream state in the
    cache key, a `git pull ../nanochat` can change tokenizer behavior with
    our hash unchanged. We prefer the git HEAD SHA (covers all of upstream
    in one shot); fall back to hashing the two files the wrapper directly
    touches; if neither works, return a stable sentinel so the rest of the
    hash still functions.
    """
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


# Hash of ALL tokenizer-defining inputs for the FORCED-subset wrapper.
# Edits to any of these change the hash → variant dirs change →
# trained-tokenizer and eval-JSON caches both invalidate automatically.
# (For truly bypassing the cache without changing source — e.g. you suspect
# stale state from a `.so` rebuilt outside this tree — pass `--force`.)
_INPUT_FILES = [
    # Pair source the subset module exec()s.
    _REPO / "rustbpe_variants/force_merges/mined/forced_pairs_climbmix_20rg.py",
    # Subset module: controls how pairs are loaded + filtered.
    _REPO / "rustbpe_variants/force_merges/pairs_mined_forced_subset.py",
    # pairs.py: BLOCKED_PAIRS source.
    _REPO / "rustbpe_variants/force_merges/pairs.py",
    # Wrapper: controls SPLIT_PATTERN patching, train_from_iterator kwargs,
    # post-train density check.
    _REPO / "wrappers/tok_train_force_merges_mined_forced_subset.py",
    # Rust crate source: what `rustbpe_force_merges` actually does. We hash
    # the source rather than the installed .so because (a) the .so path is
    # platform-specific and (b) any behavior change requires editing source.
    _REPO / "rustbpe_variants/force_merges/src/lib.rs",
]
_SRC_HASH = hashlib.sha1(
    b"\n".join([p.read_bytes() for p in _INPUT_FILES]
               + [_upstream_nanochat_fingerprint()])
).hexdigest()[:8]


def variant_dir(k: int) -> Path:
    return Path.home() / f".cache/nanochat-variants/force_merges_mined_k{k}_{_SRC_HASH}"


def train_one(k: int, force: bool = False) -> Path:
    """Train the FORCED-subset variant at MINED_TOP_K=k. Return tokenizer dir."""
    base = variant_dir(k)
    out_dir = base / "tokenizer"
    if out_dir.exists() and (out_dir / "tokenizer.pkl").exists() and not force:
        print(f"  [skip] K={k} already trained at {out_dir}", file=sys.stderr)
        return out_dir
    if force and out_dir.exists():
        print(f"  [force] removing cached {out_dir}", file=sys.stderr)
        import shutil
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


def eval_one(tokenizer_dir: Path, json_out: Path, force: bool = False) -> list:
    """Run eval_tokenizer on one tokenizer and parse the JSON output.

    Cache invalidation: if a cached JSON exists but its `spec` field
    doesn't match the expected tokenizer_dir, the cache is stale (e.g.
    because the wrapper output path was renamed). Re-run rather than trust.
    `force=True` deletes the cached JSON unconditionally.
    """
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
        rep = eval_one(td, json_out, force=args.force)
        reports.append((k, rep))

    diff_runs(reports)


if __name__ == "__main__":
    main()
