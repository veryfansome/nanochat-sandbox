"""Generate rustbpe_variants/auto_tune/pass2_candidates.py for the held-out-Pareto
build-up (pass 2).

Candidate set = every pair that EVER showed a whole-word coverage gain in the
single-pair passes: the coverage-driven KEEPs (deltas.p1 > 0) plus all FLAGs
(coverage up but compression worse). Deliberately EXCLUDES the compression-only
and mangled-only keeps (coverage flat) — they can't pass the new "coverage must
increase" gate; revisit later. Ordered by original coverage gain desc, then
compression asc (free-lunch first), which is the build-up order (redundancy
elimination is order-dependent). Regenerate with
`uv run python -m tools.gen_pass2_candidates`.
"""
import json
from pathlib import Path

CHECKPOINTS = [
    ("R1", "rustbpe_variants/auto_tune/audit/checkpoint.json"),
    ("R2", "rustbpe_variants/auto_tune/audit/manual_pass1/checkpoint.json"),
    ("R3", "rustbpe_variants/auto_tune/audit/mined_pass1/checkpoint.json"),
]
OUT = Path("rustbpe_variants/auto_tune/pass2_candidates.py")


def _unit(r):
    return tuple(r["pairs"][0]) if "pairs" in r else tuple(r["pair"])


def main():
    seen = set()
    rows = []  # (pair, p1, comp, source, verdict)
    for tag, cp in CHECKPOINTS:
        for r in json.loads(Path(cp).read_text())["rows"]:
            v, d = r["verdict"], r.get("deltas")
            if d is None:
                continue
            # coverage-gain pairs only: coverage-driven KEEPs (p1>0) or any FLAG.
            if not (d["p1"] > 0 and v in ("KEEP", "FLAG")):
                continue
            p = _unit(r)
            if p in seen:
                continue
            seen.add(p)
            rows.append((p, d["p1"], d["comp"], tag, v))
    rows.sort(key=lambda x: (-x[1], x[2]))  # p1 desc, comp asc

    lines = [
        '"""Auto-generated pass-2 candidate list — coverage-gain pairs (KEEP p1>0 + all FLAGs)',
        "for the held-out-Pareto build-up. DO NOT edit by hand; regenerate with",
        "`uv run python -m tools.gen_pass2_candidates`. Comment = original deltas + source/verdict.",
        '"""',
        "",
        "CANDIDATES_PASS2: list[tuple[str, str]] = [",
    ]
    for (L, R), p1, comp, tag, v in rows:
        lines.append(f"    ({L!r}, {R!r}),  # {tag} {v} p1{p1:+} comp{comp:+}")
    lines.append("]")
    OUT.write_text("\n".join(lines) + "\n")
    nk = sum(1 for *_, v in rows if v == "KEEP")
    nf = sum(1 for *_, v in rows if v == "FLAG")
    print(f"[gen] wrote {OUT} with {len(rows)} candidates ({nk} coverage-KEEPs + {nf} FLAGs)")


if __name__ == "__main__":
    main()
