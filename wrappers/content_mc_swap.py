"""content-MC premise/stem-swap counterfactual — does the model USE the question stem, or pick by
choice-intrinsic fluency/register?

The reading test content-MC never got: choice-order shuffle is a NO-OP for content tasks (scoring is
position-invariant), so it only rules out a position artifact. Here we swap each item's STEM (its
`query` — piqa goal / arc question / hellaswag context) with a DIFFERENT item's stem, keep its own
choices + gold, and re-score (few-shot intact). The gold is now paired with the WRONG stem, so:
  acc COLLAPSES toward chance ⇒ the model reads the stem (capability);
  acc HOLDS ≈ orig        ⇒ the model picks by choice fluency/register, NOT the stem.
Run on both arms × both corpora: if the ClimbMix-over-FineWeb advantage SURVIVES the swap it's
register/domain-match (the boolq pattern); if it collapses it's reading. `--baseline` for vanilla
checkpoints. Lettered tasks are excluded (their options live in the query — premise-swap isn't clean).

  uv run python -m wrappers.content_mc_swap --ckpt .../d24_baseline_s7 --tok .../baseline/tokenizer --baseline --n 400
"""
import argparse
import json
import os
import random

import yaml
import torch

from nanochat.common import get_base_dir
from wrappers._mc_scoring import make_scorer


def swapped_queries(data, seed=0):
    """Each item keeps its choices+gold but gets a DIFFERENT item's stem."""
    n = len(data)
    perm = list(range(n)); random.Random(seed).shuffle(perm)
    for i in range(n):
        if perm[i] == i:
            j = (i + 1) % n; perm[i], perm[j] = perm[j], perm[i]
    return [data[perm[i]]["query"] for i in range(n)]


def mc_pred(idx, data, tm, score, query_override=None):
    item = data[idx] if query_override is None else {**data[idx], "query": query_override}
    rng = random.Random(1234 + idx)
    avail = [j for j in range(len(data)) if j != idx]
    fewshot = [data[j] for j in rng.sample(avail, tm["num_fewshot"])]   # few-shot intact (own stems)
    return score(item, fewshot, tm)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--tok", required=True)
    ap.add_argument("--baseline", action="store_true", help="vanilla checkpoint (standard token-CE)")
    ap.add_argument("--bundle", default=os.path.join(get_base_dir(), "eval_bundle"))
    ap.add_argument("--tasks", default="piqa,arc_easy,arc_challenge,hellaswag")
    ap.add_argument("--n", type=int, default=400); ap.add_argument("--device", default="cuda")
    ap.add_argument("--kmax", type=int, default=1)
    args = ap.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() or args.device != "cuda" else "cpu")
    cfg = yaml.safe_load(open(os.path.join(args.bundle, "core.yaml")))
    meta = {os.path.basename(t["dataset_uri"]).replace(".jsonl", ""): t for t in cfg["icl_tasks"]}
    mode = "baseline" if args.baseline else "surface"
    _, score = make_scorer(mode, args.ckpt, args.tok, device, args.kmax)

    print(f"\n[{mode}]  {'task':>16} {'#ch':>4} {'acc_orig':>9} {'acc_swap':>9} {'pred_chg':>9} {'chance':>7}")
    for t in args.tasks.split(","):
        t = t.strip()
        if t not in meta:
            continue
        td = meta[t]
        if td["icl_task_type"] not in ("multiple_choice", "schema"):
            continue
        tm = {"task_type": td["icl_task_type"], "num_fewshot": td["num_fewshot"][0],
              "continuation_delimiter": td.get("continuation_delimiter", " ")}
        data = [json.loads(l) for l in open(os.path.join(args.bundle, "eval_data", td["dataset_uri"]))]
        if isinstance(data[0]["choices"][0], str) and all(len(c.strip()) == 1 for c in data[0]["choices"]):
            print(f"{'':>2}{t:>16}  (lettered — skipped; premise-swap not clean)")
            continue
        sw = swapped_queries(data)
        idxs = list(range(len(data))); random.Random(7).shuffle(idxs); idxs = sorted(idxs[:args.n])
        co = cs = chg = 0
        for i in idxs:
            g = data[i]["gold"]
            po = mc_pred(i, data, tm, score)
            ps = mc_pred(i, data, tm, score, query_override=sw[i])
            co += (po == g); cs += (ps == g); chg += (po != ps)
        n = len(idxs); nch = len(data[0]["choices"])
        print(f"{'':>2}{t:>16} {nch:>4} {co/n:>9.3f} {cs/n:>9.3f} {100*chg/n:>8.1f}% {1/nch:>7.2f}")
    print("\nacc_swap → chance ⇒ model reads the stem (capability); acc_swap ≈ acc_orig ⇒ picks by "
          "choice fluency/register (not reading). Compare the cm-vs-fw gap orig vs swap.")


if __name__ == "__main__":
    main()
