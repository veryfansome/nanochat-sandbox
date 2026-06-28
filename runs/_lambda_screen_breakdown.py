"""Deeper λ-screen breakdown — reproduces the per-task / boolq / noise figures cited in
ideas/surface_factoring/README.md (the "λ-screen result" subsection). The headline verdict
table comes from runs/_lambda_screen_readout.py; this script is the source for everything
below it (the boolq decomposition, the 1–2-example trio deltas, the σ figures).

Numbers derive from results/<tag>/eval.csv (col 1 = raw accuracy, col 2 = centered). The
per-task N (#datapoints) and random baseline rb come from the eval bundle's authoritative
eval_meta_data.csv (NOT inferred — inferring N from accuracy precision is unreliable because
divisors coincide; that bug once mislabeled boolq as N=529 / +43 when it is N=3270 / +266).

Usage (tags default to the screen's two arms; EVAL_BUNDLE overrides the bundle path):
  uv run python runs/_lambda_screen_breakdown.py [CONTROL_TAG] [CAND_TAG]
"""
import csv
import math
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRIO = ["arc_easy", "piqa", "hellaswag"]
NTASKS_CORE = 22
_EB_CANDIDATES = [
    os.environ.get("EVAL_BUNDLE", ""),
    os.path.expanduser("~/.cache/nanochat/eval_bundle"),
    os.path.expanduser("~/.cache/nanochat-variants/surface_only/eval_bundle"),
]


def find_bundle():
    for d in _EB_CANDIDATES:
        if d and os.path.exists(os.path.join(d, "eval_meta_data.csv")):
            return d
    raise SystemExit("no eval_meta_data.csv found; set EVAL_BUNDLE=<dir>")


def load_meta(eb):
    """task -> (N datapoints, rb fraction) from eval_meta_data.csv."""
    M = {}
    for r in csv.reader(open(os.path.join(eb, "eval_meta_data.csv"))):
        if len(r) >= 6 and r[4].strip().isdigit():
            M[r[0].strip()] = (int(r[4]), float(r[5]) / 100.0)
    return M


def read_arm(tag):
    """task -> (raw_acc, centered) + _CORE; from results/<tag>/eval.csv."""
    d = {}
    for row in csv.reader(open(os.path.join(ROOT, "results", tag, "eval.csv"))):
        if not row:
            continue
        k = row[0].strip()
        if k == "CORE" and len(row) >= 3 and row[-1].strip():
            d["_CORE"] = (None, float(row[-1]))
        elif len(row) >= 3 and row[1].strip() and row[2].strip():
            try:
                d[k] = (float(row[1]), float(row[2]))
            except ValueError:
                pass
    return d


def main():
    ctl_tag = sys.argv[1] if len(sys.argv) > 1 else "d24_surf_lam0p5_k1850"
    cnd_tag = sys.argv[2] if len(sys.argv) > 2 else "d24_surf_lam0p1_k1850"
    eb = find_bundle()
    M = load_meta(eb)
    a, b = read_arm(ctl_tag), read_arm(cnd_tag)
    print(f"control = {ctl_tag}   candidate = {cnd_tag}")
    print(f"N + rb from {eb}/eval_meta_data.csv\n")

    # ---- CORE decomposition: per-task contribution to ΔCORE (= Δcentered / 22) ----
    core_a, core_b = a["_CORE"][1], b["_CORE"][1]
    tasks = [t for t in a if t in b and t != "_CORE"]
    print(f"CORE: control {core_a:.4f}  candidate {core_b:.4f}  Δ {core_b-core_a:+.4f}")
    deltas = sorted(((b[t][1] - a[t][1], t) for t in tasks), key=lambda x: -abs(x[0]))
    print(f"{'task':<24}{'Δcentered':>10}{'contrib(Δ/22)':>14}")
    for dlt, t in deltas[:6]:
        print(f"{t:<24}{dlt:>+10.4f}{dlt/NTASKS_CORE:>+14.4f}")
    if "boolq" in b and "boolq" in a:
        bd = b["boolq"][1] - a["boolq"][1]
        print(f"  boolq contributes {bd/NTASKS_CORE:+.4f} of the {core_b-core_a:+.4f} ΔCORE "
              f"= {100*(bd/NTASKS_CORE)/(core_b-core_a):.0f}%")

    # ---- per-task example deltas + sampling SE (trio + boolq) ----
    print(f"\n{'task':<11}{'N':>7}{'rb':>5}{'accΔ':>9}{'Δ#ex':>7}{'SE_raw':>9}{'Δ/SE':>8}")
    secent2 = 0.0; dcent_sum = 0.0
    for t in TRIO + ["boolq"]:
        if t not in M or t not in a:
            continue
        N, rb = M[t]
        acc_a = a[t][0]; acc_b = b[t][0]
        dacc = acc_b - acc_a
        se = math.sqrt(acc_a * (1 - acc_a) / N)
        print(f"{t:<11}{N:>7}{rb:>5.2f}{dacc:>+9.4f}{round(acc_b*N)-round(acc_a*N):>+7}{se:>9.4f}{dacc/se:>+8.2f}")
        if t in TRIO:
            dcent_sum += (b[t][1] - a[t][1]) / 3
            secent2 += (se / (1 - rb) / 3) ** 2
    se_trio = math.sqrt(secent2)
    print(f"\ntrio-mean Δcentered {dcent_sum:+.4f}  sampling-SE {se_trio:.4f}  "
          f"=> {abs(dcent_sum)/se_trio:.2f}σ   (promote threshold +0.01 = {0.01/se_trio:.1f}σ)")
    if "boolq" in a:
        N, rb = M["boolq"]
        print(f"boolq: raw {a['boolq'][0]:.4f}->{b['boolq'][0]:.4f}  "
              f"correct {round(a['boolq'][0]*N)}->{round(b['boolq'][0]*N)} of {N}  "
              f"Δ {round(b['boolq'][0]*N)-round(a['boolq'][0]*N):+d} questions")


if __name__ == "__main__":
    main()
