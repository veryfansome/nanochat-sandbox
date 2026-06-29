"""boolq passage-swap counterfactual — does the model USE the passage, or answer from a yes/no prior?

For each boolq question we score yes/no twice: (a) the correct passage, (b) a MISMATCHED passage
(shuffled across questions; question + few-shot intact). Low %changed + unmoved dist ⇒ ignoring the
passage (prior); high %changed + acc→chance ⇒ reading. Reuses the EXACT eval scoring so behavior
matches the real CORE eval. `--baseline` scores a vanilla checkpoint (standard token-CE) so surf-vs-
baseline passage-use is comparable (the (2) follow-up).

  uv run python -m wrappers.boolq_swap --ckpt .../d24_surf_fw_lam0p1_s8 --tok .../surface_only_fw/tokenizer --n 800
  uv run python -m wrappers.boolq_swap --baseline --ckpt .../d24_baseline_fw_s8 --tok .../baseline_fw/tokenizer --n 800
"""
import argparse
import json
import os
import random
from collections import Counter

import yaml
import torch

from nanochat.common import get_base_dir
from wrappers._mc_scoring import make_scorer

DELIM = "\nQuestion: "


def boolq_task_meta(bundle):
    cfg = yaml.safe_load(open(os.path.join(bundle, "core.yaml")))
    for t in cfg["icl_tasks"]:
        if "boolq" in t["dataset_uri"]:
            return {"task_type": t["icl_task_type"], "dataset_uri": t["dataset_uri"],
                    "num_fewshot": t["num_fewshot"][0],
                    "continuation_delimiter": t.get("continuation_delimiter", " ")}
    raise SystemExit("boolq not found in core.yaml")


def shuffled_passages(data, seed=0):
    """Each question keeps its text but gets a DIFFERENT question's passage."""
    parts = []
    for r in data:
        q = r["query"]
        p, ques = (q.split(DELIM, 1) + [""])[:2] if DELIM in q else (q, "")
        parts.append((p, DELIM + ques if DELIM in q else ""))
    n = len(parts)
    perm = list(range(n)); random.Random(seed).shuffle(perm)
    for i in range(n):                                  # force perm[i] != i (real mismatch)
        if perm[i] == i:
            j = (i + 1) % n; perm[i], perm[j] = perm[j], perm[i]
    return [parts[perm[i]][0] + parts[i][1] for i in range(n)]


def mc_pred(idx, data, tm, score, query_override=None):
    item = data[idx] if query_override is None else {**data[idx], "query": query_override}
    rng = random.Random(1234 + idx)
    avail = [i for i in range(len(data)) if i != idx]
    fewshot = [data[i] for i in rng.sample(avail, tm["num_fewshot"])]   # ORIGINAL passages intact
    return score(item, fewshot, tm)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--tok", required=True)
    ap.add_argument("--baseline", action="store_true", help="score a vanilla checkpoint (standard token-CE)")
    ap.add_argument("--bundle", default=os.path.join(get_base_dir(), "eval_bundle"))
    ap.add_argument("--n", type=int, default=800, help="subsample size (deterministic)")
    ap.add_argument("--device", default="cuda"); ap.add_argument("--kmax", type=int, default=1)
    args = ap.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() or args.device != "cuda" else "cpu")

    tm = boolq_task_meta(args.bundle)
    data = [json.loads(l) for l in open(os.path.join(args.bundle, "eval_data", tm["dataset_uri"]))]
    swapped = shuffled_passages(data)
    mode = "baseline" if args.baseline else "surface"
    _, score = make_scorer(mode, args.ckpt, args.tok, device, args.kmax)

    idxs = list(range(len(data)))
    random.Random(7).shuffle(idxs); idxs = sorted(idxs[:args.n])
    po, ps, go = [], [], []
    for i in idxs:
        go.append(data[i]["gold"])
        po.append(mc_pred(i, data, tm, score))
        ps.append(mc_pred(i, data, tm, score, query_override=swapped[i]))
    ch = data[0]["choices"]
    acc_o = sum(p == g for p, g in zip(po, go)) / len(go)
    acc_s = sum(p == g for p, g in zip(ps, go)) / len(go)
    changed = sum(a != b for a, b in zip(po, ps)) / len(po)
    print(f"\nboolq passage-swap probe [{mode}]  (n={len(idxs)}, choices={ch})")
    print(f"  pred dist ORIG: {dict(Counter(ch[p] for p in po))}")
    print(f"  pred dist SWAP: {dict(Counter(ch[p] for p in ps))}")
    print(f"  accuracy  ORIG: {acc_o:.3f}   SWAP: {acc_s:.3f}")
    print(f"  preds CHANGED by swap: {100*changed:.1f}%")
    print("  → low %changed + dist unmoved ⇒ ignoring passage (prior); high %changed + acc→chance ⇒ reading.")


if __name__ == "__main__":
    main()
