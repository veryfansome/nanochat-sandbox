"""
Surface-aware CORE scoring — a drop-in for `nanochat.core_eval.evaluate_example`
that scores multiple-choice / schema options by the JOINT NLL (content + space +
cap) of the surface-encoded continuation instead of plain token CE.

`evaluate_task` (core_eval.py:254) resolves `evaluate_example` from the core_eval
module globals at call time, so the eval launcher installs
`nanochat.core_eval.evaluate_example = surface_evaluate_example`. Everything
downstream stays upstream: `evaluate_task`, `evaluate_core` centering/aggregation
(base_eval.py), the CSV/report writer.

Scoring mirrors core_eval.py:232-237 EXACTLY — per option, the **mean over the
continuation slice `[si-1:ei-1]` of the per-token loss, argmin wins** — with the
λ-free joint NLL (from `SurfaceFactoringGPT.forward(..., loss_reduction='none')`)
replacing the content-only CE. CORE normalizes by token COUNT, not bytes; the
surface tokenization's differing per-option token count is a known caveat (the
mean partially compensates — do NOT byte-normalize CORE).

LM tasks (e.g. lambada) keep baseline-comparable semantics: content-only argmax
on the base stream (surface bits are a side channel, not part of "did it predict
the word"), prefix-safe via `find_common_length` (subsumes the _patches.py fix).
"""
from __future__ import annotations

import json
import os
import random

import torch

from nanochat.core_eval import (
    render_prompts_mc, render_prompts_schema, render_prompts_lm,
    find_common_length, forward_model,
)
from nanochat.common import get_dist_info

_SURFACE_TOK = None
_LOG_PATH = os.environ.get("SURFACE_EVAL_INCORRECT_LOG")   # None = off; set by the launcher
_LOG_FH = None


def set_surface_tokenizer(tok):
    """Inject the dual-stream SurfaceTokenizer (the launcher calls this — the
    `tokenizer` evaluate_example receives is nanochat's plain RustBPETokenizer)."""
    global _SURFACE_TOK
    _SURFACE_TOK = tok


def set_incorrect_log(path):
    """Enable per-WRONG-answer JSONL logging (the launcher calls this; or set env
    SURFACE_EVAL_INCORRECT_LOG). One row per INCORRECT example: {task, idx, gold, pred[,
    scores]} for MC/schema, {task, idx, gold, pred} (decoded text) for LM. Correct answers
    are omitted — they're recoverable from the gold. Written per-rank under torchrun."""
    global _LOG_PATH, _LOG_FH
    _LOG_PATH = path
    _LOG_FH = None


def _task_name(task_meta):
    uri = task_meta.get("dataset_uri", "")
    return os.path.splitext(os.path.basename(uri))[0] or task_meta.get("task_type", "task")


def _rank_path(path, world, rank):
    if world <= 1:
        return path
    root, ext = os.path.splitext(path)                 # insert before the real suffix —
    return f"{root}.rank{rank}{ext or '.jsonl'}"        # an override path may not end in .jsonl


def _log_incorrect(task_meta, idx, **fields):
    global _LOG_FH, _LOG_PATH
    if not _LOG_PATH:
        return
    try:
        if _LOG_FH is None:
            _, rank, _, world = get_dist_info()
            path = _rank_path(_LOG_PATH, world, rank)
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            _LOG_FH = open(path, "w")                   # fresh per eval run
        _LOG_FH.write(json.dumps({"task": _task_name(task_meta), "idx": idx, **fields}) + "\n")
        _LOG_FH.flush()
    except Exception as e:                              # a diagnostic must NEVER abort a paid eval
        import sys
        print(f"[surface-eval] incorrect-answer logging disabled (write failed: {e})", file=sys.stderr)
        _LOG_PATH = None                               # disable so we don't retry every example


def _encode_prompt(prompt):
    """str -> four BOS-prepended streams (base, space, cap_bits[K], cap_mask[K]),
    mirroring the dataloader's _encode_doc; BOS carries a null surface label."""
    K = _SURFACE_TOK.kmax
    st = _SURFACE_TOK.encode(prompt)
    base = [_SURFACE_TOK.bos] + [s[0] for s in st]
    space = [0] + [s[1] for s in st]
    capb = [[0] * K] + [list(s[2]) for s in st]
    capm = [[0] * K] + [list(s[3]) for s in st]
    return base, space, capb, capm


def _stack(streams, K, bos, device):
    """Right-pad each option's 4 streams to the longest -> (B,T) / (B,T,K)."""
    B = len(streams)
    T = max(len(s[0]) for s in streams)
    base = torch.full((B, T), bos, dtype=torch.long)
    space = torch.zeros((B, T), dtype=torch.long)
    capb = torch.zeros((B, T, K), dtype=torch.long)
    capm = torch.zeros((B, T, K), dtype=torch.long)
    for i, (b, sp, cb, cm) in enumerate(streams):
        n = len(b)
        base[i, :n] = torch.tensor(b, dtype=torch.long)
        space[i, :n] = torch.tensor(sp, dtype=torch.long)
        capb[i, :n] = torch.tensor(cb, dtype=torch.long)
        capm[i, :n] = torch.tensor(cm, dtype=torch.long)
    return (base.to(device), space.to(device), capb.to(device), capm.to(device))


def _surface_keys(stream):
    """Per-position identity over ALL FOUR streams (base + space + cap) — so the
    common-prefix/suffix boundary treats options differing ONLY in case or factored
    spacing as DISTINCT there, keeping their distinguishing surface NLL in the scored
    slice (a base-IDs-only boundary folds them into the shared region → empty/NaN)."""
    base, space, capb, capm = stream
    return [(base[t], space[t], tuple(capb[t]), tuple(capm[t])) for t in range(len(base))]


@torch.no_grad()
def joint_per_token(model, base, space, capb, capm):
    """(B,T) per-token λ-free joint NLL via the model's loss_reduction='none' path.
    Targets are the left-roll of each stream, matching forward_model's alignment
    (position t predicts token t+1)."""
    tgt = torch.roll(base, -1, dims=1)
    sp_y = torch.roll(space, -1, dims=1)
    cb_y = torch.roll(capb, -1, dims=1)
    cm_y = torch.roll(capm, -1, dims=1)
    return model(base, tgt, loss_reduction='none',
                 space_x=space, cap_bits_x=capb, cap_mask_x=capm,
                 space_y=sp_y, cap_bits_y=cb_y, cap_mask_y=cm_y)


def pick_min_with_scores(joint, start_idxs, end_idxs):
    """CORE normalization (verbatim core_eval.py:234): mean over [si-1:ei-1] per option.
    Returns (argmin_index, per_option_means)."""
    means = [joint[i, si - 1:ei - 1].mean().item()
             for i, (si, ei) in enumerate(zip(start_idxs, end_idxs))]
    return means.index(min(means)), means


def pick_min_continuation(joint, start_idxs, end_idxs):
    """Argmin option index (back-compat wrapper around pick_min_with_scores)."""
    return pick_min_with_scores(joint, start_idxs, end_idxs)[0]


@torch.no_grad()
def content_preds(model, base, space, capb, capm):
    """(B,T) content-only next-token argmax with the trunk fed the surface INPUT
    streams so it runs IN-DISTRIBUTION (targets=None -> the forward returns the
    vocab-cropped content logits). Position t predicts token t+1 (forward_model
    alignment)."""
    logits = model(base, space_x=space, cap_bits_x=capb, cap_mask_x=capm)
    return logits.argmax(dim=-1)


@torch.no_grad()
def lossless_lm_match(model, base, space, capb, capm, start, end):
    """Teacher-forced LOSSLESS exact-match for an LM continuation: argmax content +
    threshold the space/cap heads, require ALL THREE streams (content + space + cap) to
    match the gold at every continuation position [start:end]. Because the tokenizer is
    lossless, matching all three == matching the exact rendered text — tokenizer-
    independent and apples-to-apples with a standard tokenizer (whose tokens ARE the
    text). Position t predicts token t+1, so [start-1:end-1] covers the continuation.
    Returns (matched_all, content_ok, space_ok, cap_ok) — the per-stream flags let the
    incorrect-answer log explain a miss whose content argmax matched but space/cap did not."""
    tgt = torch.roll(base, -1, dims=1)   # next-content targets condition the surface heads
    clogits, slogit, kloglt = model(base, tgt, space_x=space, cap_bits_x=capb,
                                    cap_mask_x=capm, return_surface_logits=True)
    sl = slice(start - 1, end - 1)
    c_ok = (clogits[0, sl].argmax(dim=-1) == base[0, start:end]).all()
    s_ok = ((slogit[0, sl] > 0).long() == space[0, start:end]).all()
    m = capm[0, start:end].bool()        # only active cap slots are part of the rendering
    k_ok = (((kloglt[0, sl] > 0).long() == capb[0, start:end]) | ~m).all()
    return (bool((c_ok & s_ok & k_ok).item()),
            bool(c_ok.item()), bool(s_ok.item()), bool(k_ok.item()))


@torch.no_grad()
def surface_evaluate_example(idx, model, tokenizer, data, device, task_meta):
    """Surface-aware drop-in for nanochat.core_eval.evaluate_example."""
    assert _SURFACE_TOK is not None, "set_surface_tokenizer() must be called first"
    item = data[idx]
    task_type = task_meta['task_type']
    num_fewshot = task_meta['num_fewshot']
    cd = task_meta['continuation_delimiter']

    fewshot = []
    if num_fewshot > 0:
        rng = random.Random(1234 + idx)
        avail = [i for i in range(len(data)) if i != idx]
        fewshot = [data[i] for i in rng.sample(avail, num_fewshot)]

    if task_type == 'language_modeling':
        # LOSSLESS exact-match (content + space + cap = exact rendered text). Feed the
        # prompt's known surface INPUT streams so the trunk runs IN-DISTRIBUTION (the old
        # base-only forward_model omitted them → OOD → tanked squad/lambada), then require
        # all three OUTPUT streams to match — NOT content-only, which was lenient on
        # space/case and not apples-to-apples with the standard tokenizer's token==text.
        prompts = render_prompts_lm(item, cd, fewshot)
        without = _encode_prompt(prompts[0])[0]
        sh = _encode_prompt(prompts[1])
        wth = sh[0]
        start = find_common_length([without, wth], 'left')
        end = len(wth)
        assert start < end, "continuation empty after surface encode"
        base, space, capb, capm = _stack([sh], _SURFACE_TOK.kmax, _SURFACE_TOK.bos, device)
        matched, c_ok, s_ok, k_ok = lossless_lm_match(model, base, space, capb, capm, start, end)
        if not matched and _LOG_PATH:                  # gold vs content-argmax pred (content-only,
            gold_ids = base[0, start:end].tolist()      # spaceless/folded); flags show which stream missed
            pred_ids = content_preds(model, base, space, capb, capm)[0, start - 1:end - 1].tolist()
            _log_incorrect(task_meta, idx, gold=_SURFACE_TOK.enc.decode(gold_ids),
                           pred=_SURFACE_TOK.enc.decode(pred_ids),
                           content_ok=c_ok, space_ok=s_ok, cap_ok=k_ok)
        return matched

    if task_type == 'multiple_choice':
        prompts = render_prompts_mc(item, cd, fewshot)
        streams = [_encode_prompt(p) for p in prompts]
        end_idxs = [len(s[0]) for s in streams]
        # common prefix over (base,space,cap), NOT base IDs alone — so case/space-only
        # differences stay in the scored slice (F3). Clamp so no option is empty-sliced.
        start = find_common_length([_surface_keys(s) for s in streams], 'left')
        start = max(1, min(start, min(end_idxs) - 1))
        start_idxs = [start] * len(streams)
    elif task_type == 'schema':
        prompts = render_prompts_schema(item, cd, fewshot)
        streams = [_encode_prompt(p) for p in prompts]
        end_idxs = [len(s[0]) for s in streams]
        # common suffix over (base,space,cap), NOT base IDs alone (F3). Clamp so no
        # option is empty-sliced.
        suf = find_common_length([_surface_keys(s) for s in streams], 'right')
        suf = min(suf, min(end_idxs) - 1)
        start_idxs = [e - suf for e in end_idxs]
    else:
        raise ValueError(f"Unsupported task type: {task_type}")

    # (No max_seq_len truncation: nanochat GPT sets no max_seq_len and over-computes
    # the rotary cache 10×, so CORE-length prompts never need cropping.)
    base, space, capb, capm = _stack(streams, _SURFACE_TOK.kmax, _SURFACE_TOK.bos, device)
    joint = joint_per_token(model, base, space, capb, capm)   # (B,T)
    pred, means = pick_min_with_scores(joint, start_idxs, end_idxs)
    gold = item['gold']
    if pred != gold and _LOG_PATH:
        _log_incorrect(task_meta, idx, gold=gold, pred=pred, scores=[round(m, 5) for m in means])
    return pred == gold
