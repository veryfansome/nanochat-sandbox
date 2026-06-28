"""Readout for the surface-factoring λ-screen (called by runs/runpod_lambda_screen.sh
verifydl). Per λ arm reads the MC-CORE robust trio (PRIMARY) + CORE from eval.csv and the
joint/content/overhead bpb split (SECONDARY) from the run log's last `[surface bpb]` line,
and ENFORCES the documented promote conjunction vs the control.

PROMOTE GATE (a candidate λ is worth a fuller run only if ALL hold vs the control):
  - Δ(MC-trio mean, centered) ≥ +0.01      (PRIMARY: the actual capability signal)
  - each of arc_easy/piqa/hellaswag individually ABOVE the control (sign-stable, not a 1-task fluke)
  - Δ(joint bpb) ≤ +0.002                   (the lever didn't cost bpb)
  - Δ(content bpb) ≤ +0.002                 (content non-regressing)
  - no trio task at floor (|centered| < 0.05 → that task is ~random at this cooled K, uninformative)
A single-seed 3-task sign test has an ~10-25% per-arm false-positive rate, so a PROMOTE here is
NOT a green light to a full run — it must be REPLICATED at a 2nd seed (SURFACE_SEED) at the same
cooled K first. content-bpb is CORE-DECOUPLED and, on a decreasing-λ grid, drifts in the expected
direction by construction → it is a non-regression sanity check, never the promote driver.

SCOPE: cooled K is a partial proxy — a flat result bounds a STRONG/EARLY λ effect, NOT a modest
(~0.005, the wash's own size) or LATE-emerging full-d24 one. Read a null as "λ is not a strong/early
lever," not "λ is not the lever." Absolutes are near floor (undertrained); read the Δ, not the level.
"""
import csv
import os
import re

LAMBDAS = os.environ["LAMBDAS"].split()
PREFIX = os.environ["TAG_PREFIX"]
NUM_ITERS = os.environ["NUM_ITERS"]
CONTROL = os.environ["CONTROL_LAM"]
ROOT = os.environ["SANDBOX_DIR"]
SEED = os.environ.get("SEED", "")
TRIO = ["arc_easy", "piqa", "hellaswag"]

PROMOTE_TRIO = 0.01     # Δ(trio mean) threshold
JOINT_MAX = 0.002       # Δ(joint bpb) must be ≤ this (no bpb cost)
CONTENT_MAX = 0.002     # Δ(content bpb) must be ≤ this (content non-regressing)
FLOOR = 0.05            # |centered| below this = task at floor (uninformative at cooled K)


def tag_of(lam):
    t = f"{PREFIX}_lam{lam.replace('.', 'p')}_k{NUM_ITERS}"
    return t + (f"_s{SEED}" if SEED else "")


def read_eval(tag):
    p = os.path.join(ROOT, "results", tag, "eval.csv")
    d = {}
    if not os.path.exists(p):
        return d
    for row in csv.reader(open(p)):
        if not row:
            continue
        k = row[0].strip()
        if k == "CORE" and len(row) >= 3 and row[-1].strip():
            d["_CORE"] = float(row[-1])
        elif len(row) >= 3 and row[2].strip():
            try:
                d[k] = float(row[2])
            except ValueError:
                pass
    return d


def read_bpb(tag):
    p = os.path.join(ROOT, "results", tag, f"{tag}.log")
    joint = content = overhead = valbpb = None
    if not os.path.exists(p):
        return joint, content, overhead, valbpb
    txt = open(p, errors="ignore").read()
    m = re.findall(r"\[surface bpb\] joint ([0-9.]+) = content ([0-9.]+) \+ overhead ([0-9.]+)", txt)
    if m:
        joint, content, overhead = (float(x) for x in m[-1])
    v = re.findall(r"(?:val bpb|Minimum validation bpb):\s*([0-9.]+)", txt)
    if v:
        valbpb = float(v[-1])
    return joint, content, overhead, valbpb


def trio_mean(ev):
    vals = [ev[t] for t in TRIO if t in ev]
    return sum(vals) / len(vals) if len(vals) == len(TRIO) else None


def fmt(x, n=4):
    return f"{x:.{n}f}" if isinstance(x, float) else "  ?  "


def verdict(lam, r, ctl):
    """Returns (label, reasons[]) — PROMOTE only if the full conjunction holds."""
    if lam == CONTROL:
        return "control", []
    fail = []
    if r["trio"] is None or ctl.get("trio") is None:
        return "INCOMPLETE", ["missing trio scores"]
    dtrio = r["trio"] - ctl["trio"]
    if dtrio < PROMOTE_TRIO:
        fail.append(f"Δtrio {dtrio:+.4f} < +{PROMOTE_TRIO}")
    # each trio task must individually beat the control (sign-stable, not 1-task fluke)
    per = [(t, r["ev"].get(t), ctl["ev"].get(t)) for t in TRIO]
    if not all(a is not None and b is not None and a > b for _, a, b in per):
        fail.append("not all 3 trio tasks beat control (1-task fluke?)")
    # floor check: any trio task ~random at this cooled K is uninformative
    floored = [t for t in TRIO if r["ev"].get(t) is not None and abs(r["ev"][t]) < FLOOR]
    if floored:
        fail.append(f"trio task(s) at floor (|centered|<{FLOOR}): {','.join(floored)} — weight toward the others")
    if r["joint"] is not None and ctl.get("joint") is not None and (r["joint"] - ctl["joint"]) > JOINT_MAX:
        fail.append(f"Δjoint {r['joint']-ctl['joint']:+.4f} > +{JOINT_MAX} (bpb cost)")
    if r["content"] is not None and ctl.get("content") is not None and (r["content"] - ctl["content"]) > CONTENT_MAX:
        fail.append(f"Δcontent {r['content']-ctl['content']:+.4f} > +{CONTENT_MAX} (content regressed)")
    return ("PROMOTE", []) if not fail else ("HOLD", fail)


def main():
    rows = {}
    for lam in LAMBDAS:
        tag = tag_of(lam)
        ev = read_eval(tag)
        j, c, o, vb = read_bpb(tag)
        rows[lam] = dict(tag=tag, ev=ev, trio=trio_mean(ev), core=ev.get("_CORE"),
                         joint=j, content=c, overhead=o, valbpb=vb)
    ctl = rows.get(CONTROL, {})
    print(f"\n=========== λ-SCREEN RESULT  (cooled {NUM_ITERS} iters{', seed '+SEED if SEED else ''}, control λ={CONTROL}) ===========")
    print("Read the Δ (vs control), not the level (absolutes are near floor / undertrained).\n")
    hdr = f"{'λ':>5} {'MCtrio':>8} {'Δtrio':>8} {'CORE':>7} {'val_bpb':>8} {'joint':>8} {'Δjoint':>8} {'content':>8} {'Δcont':>8} {'overhd':>7}  verdict"
    print(hdr); print("-" * len(hdr))
    any_promote = False
    for lam in LAMBDAS:
        r = rows[lam]
        d = lambda k: (r[k] - ctl[k]) if (r.get(k) is not None and ctl.get(k) is not None and lam != CONTROL) else None
        lab, _ = verdict(lam, r, ctl)
        any_promote = any_promote or lab == "PROMOTE"
        print(f"{lam:>5} {fmt(r['trio']):>8} {('—' if lam==CONTROL else fmt(d('trio'))):>8} "
              f"{fmt(r['core']):>7} {fmt(r['valbpb']):>8} {fmt(r['joint']):>8} {('—' if lam==CONTROL else fmt(d('joint'))):>8} "
              f"{fmt(r['content']):>8} {('—' if lam==CONTROL else fmt(d('content'))):>8} {fmt(r['overhead']):>7}  {lab}")
    print("\nPer-arm MC-trio detail (centered; ★=at floor, uninformative):")
    for lam in LAMBDAS:
        ev = rows[lam]["ev"]
        print(f"  λ={lam}: " + ", ".join(
            f"{t}={fmt(ev.get(t))}{'★' if (ev.get(t) is not None and abs(ev[t])<FLOOR) else ''}" for t in TRIO))
    print("\n--- VERDICT ---")
    for lam in LAMBDAS:
        if lam == CONTROL:
            continue
        lab, reasons = verdict(lam, rows[lam], ctl)
        if lab == "PROMOTE":
            print(f"  λ={lam}: PROMOTE — clears the conjunction. NEXT (do NOT skip): REPLICATE at a 2nd seed")
            print(f"           (SEED=2 LAMBDAS=\"{CONTROL} {lam}\" bash runs/runpod_lambda_screen.sh) at the same K")
            print(f"           BEFORE any full run — a single-seed 3-task pass has ~10-25% false-positive odds.")
        else:
            print(f"  λ={lam}: {lab} — " + "; ".join(reasons))
    if not any_promote:
        print("\n  => No λ clears the conjunction. The most likely + intended outcome: BANK the surface-λ")
        print("     negative. Scope it to 'λ is not a STRONG/EARLY lever' (cooled K can't exclude a modest")
        print("     ~0.005 or LATE-emerging full-d24 effect). content-bpb is a CORE-decoupled sanity channel;")
        print("     its expected drift tells you ~nothing new on a decreasing-λ grid.")


if __name__ == "__main__":
    main()
