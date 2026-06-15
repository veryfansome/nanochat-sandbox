"""
Per-example probe for a CORE multiple-choice task (default: boolq).

Reproduces nanochat's CORE MC scoring for ONE task, but logs per example:
gold, predicted choice, the mean per-token loss of each choice, and correctness.
Two questions it answers:

  1. BIAS  -- the model's answer-class prior (boolq "pred-1 / yes-rate") vs the
     true gold rate. A no-biased base model scores ~the majority-class rate.
  2. PASSAGE-USE -- via --ablate, an A/B on the test example's passage:
       none          : query unchanged (reproduces the real CORE number; self-check)
       swap          : replace the test passage with another example's passage,
                       keeping the question + few-shot intact (format/length held
                       constant, only content-relevance destroyed)
       question_only : drop the passage entirely, keep "Question: <q>"
     Non-'none' modes ALSO run the 'none' baseline, so the report includes
     delta = acc(none) - acc(ablation). delta ~ 0 => the model isn't using the
     passage content (answers by prior); delta large => it reads the passage.

Faithful to scripts/base_eval.py + nanochat/core_eval.py (same Random(1337) data
shuffle, same Random(1234+idx) few-shot, same lowest-mean-per-token-loss decision
and max_seq_len truncation), so --ablate none reproduces the CORE-reported number.
Does NOT edit nanochat -- imports its helpers.

Run SINGLE-process (plain python, not torchrun) against a saved base checkpoint:

    cd ~/sandbox && uv run python tools/boolq_probe.py \
        --model-tag d24_force_merges_r194 --task boolq --ablate swap

NANOCHAT_BASE_DIR must point at the run's base dir (the variant base dir for a
tokenizer-variant run) so the checkpoint + eval_bundle resolve.
"""
import os
import csv
import json
import random
import argparse

import yaml

from nanochat.common import (
    compute_init, compute_cleanup, print0, get_base_dir, autodetect_device_type,
)
from nanochat.checkpoint_manager import load_model
from nanochat.core_eval import (
    render_prompts_mc, batch_sequences_mc, forward_model, stack_sequences,
)


def split_query(query):
    """boolq query == 'Passage: <p>\\nQuestion: <q>'. Return (passage_part, question_part)
    where passage_part == 'Passage: ...' and question_part == '\\nQuestion: ...'.
    Returns (None, query) if the marker is absent (not ablatable)."""
    j = query.find("\nQuestion:")
    if j == -1:
        return None, query
    return query[:j], query[j:]


def ablate_item(item, idx, data, mode):
    """Return a (possibly modified) COPY of item per the ablation mode. Never mutates
    the shared data objects -- those are reused verbatim as few-shot context, so only
    the TEST item's passage is touched."""
    if mode == "none":
        return item
    passage_part, question_part = split_query(item["query"])
    if passage_part is None:
        return item  # can't isolate a passage; leave unchanged
    if mode == "question_only":
        new_query = question_part.lstrip("\n")             # 'Question: <q>' (passage dropped)
    elif mode == "swap":
        donor_passage, _ = split_query(data[(idx + 1) % len(data)]["query"])
        new_query = (donor_passage or "") + question_part  # wrong passage + right question
    else:
        raise ValueError(f"unknown ablate mode: {mode}")
    out = dict(item)
    out["query"] = new_query
    return out


def score_example_mc(model, tokenizer, item, task_meta, device, pool, idx, max_seq_len):
    """Mirror of core_eval.evaluate_example (MC path), returning details. `item` is the
    (possibly ablated) test example; few-shot is drawn from the unmodified `pool`."""
    num_fewshot = task_meta["num_fewshot"]
    delim = task_meta["continuation_delimiter"]

    fewshot_examples = []
    if num_fewshot > 0:
        rng = random.Random(1234 + idx)  # same seed scheme as core_eval
        available = [i for i in range(len(pool)) if i != idx]
        fewshot_examples = [pool[i] for i in rng.sample(available, num_fewshot)]

    prompts = render_prompts_mc(item, delim, fewshot_examples)
    tokens, start_idxs, end_idxs = batch_sequences_mc(tokenizer, prompts)

    if max_seq_len is not None:
        nt, ns, ne = [], [], []
        for t, s, e in zip(tokens, start_idxs, end_idxs):
            if len(t) > max_seq_len:
                crop = len(t) - max_seq_len
                nt.append(t[-max_seq_len:]); ns.append(s - crop); ne.append(e - crop)
            else:
                nt.append(t); ns.append(s); ne.append(e)
        tokens, start_idxs, end_idxs = nt, ns, ne

    pad = tokenizer.get_bos_token_id()
    input_ids = stack_sequences(tokens, pad).to(device)
    losses, _ = forward_model(model, input_ids)
    mean_losses = [losses[i, s - 1:e - 1].mean().item()
                   for i, (s, e) in enumerate(zip(start_idxs, end_idxs))]
    pred = mean_losses.index(min(mean_losses))
    return pred, item["gold"], mean_losses


def run_pass(model, tokenizer, data, task_meta, device, max_seq_len, mode, tag, task, base_dir):
    """Score the whole task under one ablation mode; write a per-example CSV; return (acc, pred1_rate)."""
    n = len(data)
    rows = []
    n_correct = n_pred1 = 0
    for idx, item in enumerate(data):
        test_item = ablate_item(item, idx, data, mode)
        pred, gold, ml = score_example_mc(model, tokenizer, test_item, task_meta, device, data, idx, max_seq_len)
        n_correct += int(pred == gold)
        n_pred1 += int(pred == 1)
        rows.append([idx, gold, pred, int(pred == gold)] + ml)
        if idx % 500 == 0:
            print0(f"  [{mode}] {idx}/{n}")
    acc, pred1 = n_correct / n, n_pred1 / n
    out = os.path.join(base_dir, "base_eval", f"{task}_probe_{tag}_{mode}.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["idx", "gold", "pred", "correct"] + [f"mean_loss_{i}" for i in range(len(rows[0]) - 4)])
        w.writerows(rows)
    print0(f"[{mode}] acc {acc:.4f}  pred-1 {pred1:.4f}  -> {out}")
    return acc, pred1


def main():
    ap = argparse.ArgumentParser(description="Per-example CORE MC probe (default boolq) with passage ablation")
    ap.add_argument("--model-tag", type=str, default=None, help="nanochat base checkpoint tag")
    ap.add_argument("--step", type=int, default=None, help="checkpoint step (default = last)")
    ap.add_argument("--task", type=str, default="boolq", help="core.yaml task label (must be multiple_choice)")
    ap.add_argument("--max-per-task", type=int, default=-1, help="cap examples (-1 = all, matches default eval)")
    ap.add_argument("--device-type", type=str, default="", help="cuda|cpu|mps (empty = autodetect)")
    ap.add_argument("--ablate", type=str, default="none", choices=["none", "swap", "question_only"],
                    help="passage-use test; non-'none' also runs the 'none' baseline to report the delta")
    args = ap.parse_args()

    device_type = autodetect_device_type() if args.device_type == "" else args.device_type
    _ddp, rank, _local, world, device = compute_init(device_type)
    if world > 1:
        raise SystemExit("Run single-process (plain python, not torchrun); the probe does not stride ranks.")

    model, tokenizer, meta = load_model("base", device, phase="eval",
                                        model_tag=args.model_tag, step=args.step)
    max_seq_len = meta["model_config"]["sequence_len"]

    base_dir = get_base_dir()
    eb = os.path.join(base_dir, "eval_bundle")
    cfg = yaml.safe_load(open(os.path.join(eb, "core.yaml"), encoding="utf-8"))
    task = next((t for t in cfg["icl_tasks"] if t["label"] == args.task), None)
    if task is None:
        raise SystemExit(f"task {args.task!r} not found in core.yaml")
    task_meta = {
        "task_type": task["icl_task_type"],
        "num_fewshot": task["num_fewshot"][0],
        "continuation_delimiter": task.get("continuation_delimiter", " "),
    }
    if task_meta["task_type"] != "multiple_choice":
        raise SystemExit(f"{args.task} is {task_meta['task_type']}; this probe handles multiple_choice only")

    data = [json.loads(l) for l in open(os.path.join(eb, "eval_data", task["dataset_uri"]), encoding="utf-8")]
    random.Random(1337).shuffle(data)  # mirror base_eval's subsample shuffle
    if args.max_per_task > 0:
        data = data[:args.max_per_task]
    n = len(data)
    gold1_rate = sum(d["gold"] == 1 for d in data) / n

    modes = ["none"] if args.ablate == "none" else ["none", args.ablate]
    results = {}
    for mode in modes:
        results[mode] = run_pass(model, tokenizer, data, task_meta, device,
                                 max_seq_len, mode, args.model_tag, args.task, base_dir)

    # ---- summary ----
    print0("=" * 64)
    print0(f"task={args.task}  model_tag={args.model_tag}  n={n}")
    print0(f"choices={data[0]['choices']}  (pred-1 / gold-1 refer to choice idx 1)")
    print0(f"gold-1 rate          : {gold1_rate:.4f}   (majority baselines: always-0 {1 - gold1_rate:.4f} | always-1 {gold1_rate:.4f})")
    for mode in modes:
        acc, p1 = results[mode]
        print0(f"  {mode:13s}: acc {acc:.4f}   pred-1 {p1:.4f}")

    acc_none, p1_none = results["none"]
    biased = abs(p1_none - 0.5) > 0.10
    print0(f"bias                 : pred-1 {p1_none:.4f} -> {'SKEWED (' + ('toward 1/yes' if p1_none > 0.5 else 'toward 0/no') + ')' if biased else 'roughly balanced ~50/50'}")
    if len(modes) > 1:
        abl = modes[1]
        delta = acc_none - results[abl][0]
        uses = delta > 0.03
        print0(f"passage-use          : delta(none - {abl}) = {delta:+.4f} -> "
               f"{'USES the passage' if uses else 'does NOT use the passage content (answers by prior)'}")
        if not uses and not biased and abs(acc_none - 0.5) < 0.05:
            print0("read                 : unbiased AND ~chance AND swap-insensitive -> debiased but NOT reading")
        elif uses:
            print0("read                 : swap hurts accuracy -> the model is genuinely reading the passage")

    compute_cleanup()


if __name__ == "__main__":
    main()
