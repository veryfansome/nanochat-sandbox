"""
Visual inspection of the surface model on CORE tasks. For a few LM and MC examples,
print the context/question, the gold answer, and the model's prediction under BOTH
the OLD base-only scoring (the OOD bug) and the NEW surface-fed scoring (the fix),
decoded to readable text. Lets us SEE the fix working rather than trust an aggregate.

Run on the eval pod after the checkpoint + tokenizer are in place:
  NANOCHAT_BASE_DIR=~/.cache/nanochat-variants/surface_only \
    uv run python -m wrappers.inspect_surface_eval --model-tag d24_surface_only_ab --n 4
"""
import argparse
import glob
import json
import os
import random

import torch
import yaml

from nanochat.common import get_base_dir
from nanochat.tokenizer import get_tokenizer
from nanochat.core_eval import (render_prompts_lm, render_prompts_mc,
                                find_common_length, forward_model)

from tools.stack_fm_spaceless import SPACELESS_BODY
from overlay.surface_tokenizer import SurfaceTokenizer
from overlay.surface_checkpoint import surface_build_model
from overlay.surface_core_eval import (set_surface_tokenizer, _encode_prompt, _stack,
                                       content_preds, joint_per_token, pick_min_continuation)


def dec(enc, ids):
    """Decode content-stream token ids to (folded, spaceless) text — readable enough
    to eyeball; the surface side channels render spacing/case, not shown here."""
    try:
        return enc.decode(ids)
    except Exception:
        return "".join(enc.decode_single_token_bytes(i).decode("utf-8", "replace") for i in ids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-tag", required=True)
    ap.add_argument("--n", type=int, default=4)
    args = ap.parse_args()
    base = get_base_dir()
    device = torch.device("cuda")

    KMAX = int(os.environ.get("SURFACE_KMAX", "1"))   # surface-only: single-word tokens
    os.environ["SURFACE_KMAX"] = str(KMAX)
    enc = get_tokenizer().enc
    TOK = SurfaceTokenizer(enc, SPACELESS_BODY, kmax=KMAX)
    set_surface_tokenizer(TOK)

    ckpt_dir = os.path.join(base, "base_checkpoints", args.model_tag)
    steps = sorted(int(os.path.basename(p).split("_")[1].split(".")[0])
                   for p in glob.glob(os.path.join(ckpt_dir, "model_*.pt")))
    assert steps, f"no model_*.pt in {ckpt_dir}"
    model, tokenizer, meta = surface_build_model(ckpt_dir, steps[-1], device, "eval")
    model.eval()
    print(f"loaded {args.model_tag} step {steps[-1]} (K_max={TOK.kmax})")

    bundle = os.path.join(base, "eval_bundle")
    config = yaml.safe_load(open(os.path.join(bundle, "core.yaml")))
    tasks = {t["label"]: t for t in config["icl_tasks"]}
    data_base = os.path.join(bundle, "eval_data")

    def load(label):
        t = tasks[label]
        m = {"task_type": t["icl_task_type"], "num_fewshot": t["num_fewshot"][0],
             "continuation_delimiter": t.get("continuation_delimiter", " "), "dataset_uri": t["dataset_uri"]}
        data = [json.loads(l) for l in open(os.path.join(data_base, m["dataset_uri"]))]
        random.Random(1337).shuffle(data)
        return m, data

    # --- LM tasks: OLD (base-only, OOD bug) vs NEW (surface-fed, fix) ---
    for label in [l for l in ["lambada_openai", "squad", "coqa", "jeopardy"] if l in tasks]:
        m, data = load(label)
        print(f"\n{'=' * 74}\nLM TASK: {label}  ({m['num_fewshot']}-shot)\n{'=' * 74}")
        old_hits = new_hits = shown = 0
        for item in data:
            if shown >= args.n:
                break
            prompts = render_prompts_lm(item, m["continuation_delimiter"], [])
            sw, sh = _encode_prompt(prompts[0]), _encode_prompt(prompts[1])
            without, wth = sw[0], sh[0]
            start = find_common_length([without, wth], "left")
            end = len(wth)
            if start >= end:
                continue
            shown += 1
            gold_ids = wth[start:end]
            ids = torch.tensor([wth], dtype=torch.long, device=device)
            _, old_p = forward_model(model, ids)                      # OLD: base-only -> OOD trunk
            old_ids = old_p[0, start - 1:end - 1].tolist()
            b, sp, cb, cm = _stack([sh], TOK.kmax, TOK.bos, device)
            new_ids = content_preds(model, b, sp, cb, cm)[0, start - 1:end - 1].tolist()   # NEW: surface-fed
            old_hits += int(old_ids == gold_ids)
            new_hits += int(new_ids == gold_ids)
            print(f"\n[{shown}] CONTEXT : ...{prompts[0][-150:]!r}")
            print(f"    GOLD    : {dec(enc, gold_ids)!r}")
            print(f"    OLD pred: {dec(enc, old_ids)!r}   {'== MATCH' if old_ids == gold_ids else 'x miss'}")
            print(f"    NEW pred: {dec(enc, new_ids)!r}   {'== MATCH' if new_ids == gold_ids else 'x miss'}")
        print(f"\n  [{label}] shown={shown}  OLD exact-match={old_hits}  NEW exact-match={new_hits}")

    # --- one MC task (scoring unchanged — sanity that it picks sensibly) ---
    for label in [l for l in ["arc_easy", "piqa"] if l in tasks][:1]:
        m, data = load(label)
        print(f"\n{'=' * 74}\nMC TASK: {label}  (surface-fed joint-NLL, unchanged)\n{'=' * 74}")
        shown = 0
        for item in data:
            if shown >= args.n:
                break
            shown += 1
            prompts = render_prompts_mc(item, m["continuation_delimiter"], [])
            streams = [_encode_prompt(p) for p in prompts]
            base_streams = [s[0] for s in streams]
            start = find_common_length(base_streams, "left")
            start_idxs = [start] * len(streams)
            end_idxs = [len(bs) for bs in base_streams]
            bb, sp, cb, cm = _stack(streams, TOK.kmax, TOK.bos, device)
            joint = joint_per_token(model, bb, sp, cb, cm)
            pick = pick_min_continuation(joint, start_idxs, end_idxs)
            print(f"\n[{shown}] Q: {item.get('query', '')!r}")
            for j, ch in enumerate(item["choices"]):
                tag = " <<PICK" if j == pick else ""
                gold = " (gold)" if j == item["gold"] else ""
                print(f"      [{j}] {ch!r}{gold}{tag}")
            print(f"    -> picked {pick}, gold {item['gold']}: {'CORRECT' if pick == item['gold'] else 'wrong'}")


if __name__ == "__main__":
    main()
