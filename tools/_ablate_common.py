"""
Shared helpers for the ablate_{mined,derived}.py drivers.

The two drivers have identical scaffolding (hash inputs, run eval, diff
reports) and differ only in which subset wrapper they shell out to and
which env var sets K. This module owns the shared parts.
"""

import hashlib
import json
import subprocess
import sys
from pathlib import Path


def upstream_nanochat_fingerprint(repo: Path) -> bytes:
    """Best-effort fingerprint of upstream nanochat (next to the sandbox).

    The wrappers import `nanochat.tokenizer` and `runpy` `scripts.tok_train`
    — both upstream, both pull-able. Without including upstream state in the
    cache key, a `git pull ../nanochat` can change tokenizer behavior with
    our hash unchanged. Prefer the git HEAD SHA (covers all of upstream in
    one shot); fall back to hashing the two files the wrapper directly
    touches; if neither works, return a stable sentinel.
    """
    upstream = repo.parent / "nanochat"
    try:
        sha = subprocess.run(
            ["git", "-C", str(upstream), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True, timeout=5,
        ).stdout.strip()
        return f"nanochat@{sha}".encode()
    except Exception:
        pass
    parts = [
        f.read_bytes()
        for rel in ("nanochat/tokenizer.py", "scripts/tok_train.py")
        if (f := upstream / rel).exists()
    ]
    return b"\n".join(parts) if parts else b"<upstream-unavailable>"


def compute_src_hash(input_files: list[Path], repo: Path) -> str:
    """SHA1[:8] over all tokenizer-defining inputs + upstream fingerprint."""
    blobs = [p.read_bytes() for p in input_files]
    blobs.append(upstream_nanochat_fingerprint(repo))
    return hashlib.sha1(b"\n".join(blobs)).hexdigest()[:8]


def eval_one(tokenizer_dir: Path, json_out: Path, force: bool = False) -> list:
    """Run eval_tokenizer on one tokenizer, caching to json_out.

    Invalidates when (a) `force=True`, or (b) cached `spec` doesn't match
    the expected tokenizer_dir (catches wrapper-output-path renames).
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


def diff_runs(reports: list[tuple[int, list]], k_label: str = "K") -> None:
    """Print compression-delta + structural + task-probe tables.

    `reports` is a list of (K_value, eval_json) tuples in K-ascending order.
    `k_label` controls the column headers (e.g. "K", "DERIVED_TOP_K").
    """
    print("\n=== Cumulative ablation: compression bytes/token deltas ===\n")
    corpora = sorted({c for _, r in reports for c in r[0]["compression"]})
    header = "corpus".ljust(18) + " ".join(
        f"{k_label}={k:<6}".rjust(10) for k, _ in reports
    )
    print(header)
    print("-" * len(header))
    for corpus in corpora:
        row = [corpus.ljust(18)]
        prev = None
        for k, r in reports:
            bpt = r[0]["compression"][corpus]["bytes_per_token"]
            cell = f"{bpt:.3f}"
            if prev is not None and abs(bpt - prev) > 1e-6:
                d = bpt - prev
                cell = f"{bpt:.3f}({'+' if d > 0 else ''}{d:.3f})"
            row.append(cell.rjust(10))
            prev = bpt
        print(" ".join(row))

    print("\n=== Cumulative ablation: structural battery ===\n")
    tests = sorted({t for _, r in reports for t in r[0]["structural"]})
    header = "test".ljust(22) + " ".join(
        f"{k_label}={k:<5}".rjust(8) for k, _ in reports
    )
    print(header)
    print("-" * len(header))
    any_fail = False
    for test in tests:
        row = [test.ljust(22)]
        for k, r in reports:
            ok = r[0]["structural"][test].get("ok", False)
            any_fail = any_fail or not ok
            row.append(("PASS" if ok else "FAIL").rjust(8))
        print(" ".join(row))
    if not any_fail:
        print("\n(all structural tests PASS at every K)")

    print("\n=== Cumulative ablation: task probes (token count) ===\n")
    probes = sorted({p for _, r in reports for p in r[0]["task_probes"]})
    header = "probe".ljust(18) + " ".join(
        f"{k_label}={k:<6}".rjust(8) for k, _ in reports
    )
    print(header)
    print("-" * len(header))
    for probe in probes:
        row = [probe.ljust(18)]
        prev = None
        for k, r in reports:
            n = r[0]["task_probes"][probe]["tokens"]
            cell = f"{n:>4}"
            if prev is not None and n != prev:
                d = n - prev
                cell = f"{n:>4}({'+' if d > 0 else ''}{d})"
            row.append(cell.rjust(8))
            prev = n
        print(" ".join(row))
