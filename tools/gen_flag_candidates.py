"""Generate rustbpe_variants/auto_tune/flag_candidates.py — the FLAG queue from the
single-pair audit passes, for the flag-revisit pass.

FLAGs were P1-positive (whole-word coverage up) but cost compression, so the
lexicographic rule deferred them. The held-out generalization check showed
compression is eval-shard noise while coverage and mangled GENERALIZE — so these
flags' real gains are worth re-testing on top of the final promoted keep set
(order-dependence: a gain measured against an earlier, smaller reference may or
may not survive once the 96 keeps are in place).

Only single-pair passes are included here; the few group-mode (route-set) flags
are handled separately. Regenerate with `uv run python -m tools.gen_flag_candidates`.
"""
import json
from pathlib import Path

CHECKPOINTS = [
    ("R1", "rustbpe_variants/auto_tune/audit/checkpoint.json"),
    ("R2", "rustbpe_variants/auto_tune/audit/manual_pass1/checkpoint.json"),
    ("R3", "rustbpe_variants/auto_tune/audit/mined_pass1/checkpoint.json"),
]
OUT = Path("rustbpe_variants/auto_tune/flag_candidates.py")


def _unit(r):
    return tuple(r["pairs"][0]) if "pairs" in r else tuple(r["pair"])


def main():
    seen = set()
    rows = []
    for tag, cp in CHECKPOINTS:
        for r in json.loads(Path(cp).read_text())["rows"]:
            if r["verdict"] != "FLAG":
                continue
            p = _unit(r)
            if p in seen:
                continue
            seen.add(p)
            rows.append((p, r["deltas"], tag))

    lines = [
        '"""Auto-generated FLAG queue for the flag-revisit pass.',
        "DO NOT edit by hand — regenerate with `uv run python -m tools.gen_flag_candidates`.",
        "",
        "FLAGs from the single-pair audit passes (coverage up, compression worse → deferred",
        "by the lexicographic rule). Comment = original deltas vs the reference AT FLAG TIME",
        "+ source round. Re-tested on top of the promoted keeps via:",
        "  --mode single --candidates-attr CANDIDATES_FLAGS --state-dir audit/flag_revisit",
        '"""',
        "",
        "CANDIDATES_FLAGS: list[tuple[str, str]] = [",
    ]
    for (L, R), d, tag in rows:
        lines.append(f"    ({L!r}, {R!r}),  # {tag} p1{d['p1']:+} comp{d['comp']:+} mangled{d['mangled']:+}")
    lines.append("]")
    OUT.write_text("\n".join(lines) + "\n")
    byr = {t: sum(1 for _, _, x in rows if x == t) for t in ("R1", "R2", "R3")}
    print(f"[gen] wrote {OUT} with {len(rows)} flags  (R1={byr['R1']} R2={byr['R2']} R3={byr['R3']})")


if __name__ == "__main__":
    main()
