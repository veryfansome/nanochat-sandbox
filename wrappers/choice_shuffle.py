"""choice-order-shuffle counterfactual — does the model track the correct ANSWER when the choices
are reordered (reading), or stick to a letter/position (prior)?

Per multiple-choice task we score twice: choices as-presented vs reordered (gold remapped). Auto-
detected per item: LETTERED (choices 'A'..'D', option text in the query — commonsense_qa,
language_id) reorders the text↔letter mapping → tests a LETTER prior; CONTENT (choices are the
answer text — hellaswag/arc/piqa) reorders the choices list → SHOULD be accuracy-inert (each choice
scored independently), which the probe confirms rather than assumes. acc collapses / pred stays
skewed under shuffle ⇒ prior; acc holds ⇒ reading. `--baseline` scores a vanilla checkpoint
(standard token-CE) so surf-vs-baseline is comparable (the (2) follow-up).

  uv run python -m wrappers.choice_shuffle --ckpt .../d24_surf_fw_lam0p1_s8 --tok .../surface_only_fw/tokenizer --n 500
  uv run python -m wrappers.choice_shuffle --baseline --ckpt .../d24_baseline_fw_s8 --tok .../baseline_fw/tokenizer --n 500
"""
import argparse
import json
import os
import random
import re

import yaml
import torch

from nanochat.common import get_base_dir
from wrappers._mc_scoring import make_scorer

_OPT = re.compile(r"^([A-Z])\.\s(.*)$")


def is_lettered(item):
    ch = item["choices"]
    return all(isinstance(c, str) and len(c.strip()) == 1 and c.strip().isupper() for c in ch)


def shuffle_item(item, rng):
    """Return (shuffled_item, new_gold_idx)."""
    if is_lettered(item):
        lines = item["query"].split("\n")
        oi = [i for i, ln in enumerate(lines) if _OPT.match(ln)]
        letters = [_OPT.match(lines[i]).group(1) for i in oi]
        texts = [_OPT.match(lines[i]).group(2) for i in oi]
        gold_text = texts[item["gold"]]
        new_texts = texts[:]; rng.shuffle(new_texts)
        for j, i in enumerate(oi):
            lines[i] = f"{letters[j]}. {new_texts[j]}"
        return {**item, "query": "\n".join(lines)}, new_texts.index(gold_text)
    order = list(range(len(item["choices"]))); rng.shuffle(order)
    return {**item, "choices": [item["choices"][i] for i in order]}, order.index(item["gold"])


def fewshot_for(idx, data, nf):
    rng = random.Random(1234 + idx)
    avail = [j for j in range(len(data)) if j != idx]
    return [data[j] for j in rng.sample(avail, nf)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--tok", required=True)
    ap.add_argument("--baseline", action="store_true", help="score a vanilla checkpoint (standard token-CE)")
    ap.add_argument("--bundle", default=os.path.join(get_base_dir(), "eval_bundle"))
    ap.add_argument("--tasks", default="commonsense_qa,bigbench_language_identification,hellaswag,arc_easy,piqa")
    ap.add_argument("--n", type=int, default=400); ap.add_argument("--device", default="cuda")
    ap.add_argument("--kmax", type=int, default=1)
    args = ap.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() or args.device != "cuda" else "cpu")
    cfg = yaml.safe_load(open(os.path.join(args.bundle, "core.yaml")))
    meta = {os.path.basename(t["dataset_uri"]).replace(".jsonl", ""): t for t in cfg["icl_tasks"]}
    mode = "baseline" if args.baseline else "surface"
    _, score = make_scorer(mode, args.ckpt, args.tok, device, args.kmax)

    print(f"\n[{mode}]  {'task':>30} {'kind':>8} {'acc_orig':>9} {'acc_shuf':>9} {'pred_chg':>9}")
    for t in args.tasks.split(","):
        t = t.strip()
        if t not in meta:
            continue
        td = meta[t]
        tm = {"task_type": td["icl_task_type"], "num_fewshot": td["num_fewshot"][0],
              "continuation_delimiter": td.get("continuation_delimiter", " ")}
        data = [json.loads(l) for l in open(os.path.join(args.bundle, "eval_data", td["dataset_uri"]))]
        idxs = list(range(len(data))); random.Random(7).shuffle(idxs); idxs = sorted(idxs[:args.n])
        kind = "lettered" if data and is_lettered(data[0]) else "content"
        co = cs = chg = 0
        for i in idxs:
            fewshot = fewshot_for(i, data, tm["num_fewshot"])
            po = score(data[i], fewshot, tm)
            sh_item, new_gold = shuffle_item(data[i], random.Random(99 + i))
            ps = score(sh_item, fewshot, tm)
            co += (po == data[i]["gold"]); cs += (ps == new_gold); chg += (po != ps)
        n = len(idxs)
        print(f"{'':>6}{t:>30} {kind:>8} {co/n:>9.3f} {cs/n:>9.3f} {100*chg/n:>8.1f}%")
    print("\ncontent: acc ~unchanged (inert) — moves ⇒ position effect. lettered: acc holds ⇒ reading; acc drops ⇒ letter prior.")


if __name__ == "__main__":
    main()
