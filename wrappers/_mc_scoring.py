"""Shared multiple-choice scoring for the counterfactual probes (boolq_swap, choice_shuffle).

Two backends, same prompt rendering + few-shot, so a probe runs identically on either arm:
  - SURFACE  : joint content+space+cap per-token NLL (surface checkpoints).
  - BASELINE : standard token-CE via nanochat's own MC path (vanilla checkpoints) — this is the
               (2) follow-up that makes surf-vs-base reading / letter-prior comparable.

make_scorer(mode, ...) -> (model, score_fn) where score_fn(item, fewshot, task_meta) -> pred_idx
(argmin mean continuation NLL over the choices). The probe owns item/few-shot/swap construction.
"""
import glob
import os

import torch

from nanochat.core_eval import (
    render_prompts_mc, batch_sequences_mc, stack_sequences, forward_model, find_common_length,
)


def latest_step(ckpt):
    return sorted(int(os.path.basename(p).split("_")[1].split(".")[0])
                  for p in glob.glob(ckpt + "/model_*.pt"))[-1]


def make_scorer(mode, ckpt, tok, device, kmax=1):
    """mode in {'surface','baseline'}. Returns (model, score_fn)."""
    if mode == "baseline":
        from nanochat.tokenizer import RustBPETokenizer
        import nanochat.checkpoint_manager as cm
        tk = RustBPETokenizer.from_directory(tok)
        model = cm.build_model(ckpt, latest_step(ckpt), device, "eval")[0]
        model.eval()

        @torch.no_grad()
        def score(item, fewshot, tm):
            prompts = render_prompts_mc(item, tm["continuation_delimiter"], fewshot)
            tokens, si, ei = batch_sequences_mc(tk, prompts)        # standard tokenizer encode
            ids = stack_sequences(tokens, tk.get_bos_token_id()).to(device)
            losses, _ = forward_model(model, ids)                  # per-token CE
            means = [losses[i, s - 1:e - 1].mean().item() for i, (s, e) in enumerate(zip(si, ei))]
            return means.index(min(means))                         # argmin mean continuation CE
        return model, score

    # surface backend (joint scoring, reused from the eval path)
    from overlay.surface_tokenizer import SurfaceTokenizer
    from overlay.surface_checkpoint import surface_build_model
    from overlay.surface_core_eval import (
        set_surface_tokenizer, _encode_prompt, _stack, joint_per_token, pick_min_with_scores, _surface_keys,
    )
    from tools.stack_fm_spaceless import SPACELESS_BODY
    os.environ["SURFACE_KMAX"] = str(kmax)
    surf = SurfaceTokenizer.from_directory(tok, SPACELESS_BODY, kmax=kmax)
    set_surface_tokenizer(surf)
    model = surface_build_model(ckpt, latest_step(ckpt), device, "eval")[0]
    model.eval()

    @torch.no_grad()
    def score(item, fewshot, tm):
        prompts = render_prompts_mc(item, tm["continuation_delimiter"], fewshot)
        streams = [_encode_prompt(p) for p in prompts]
        end_idxs = [len(s[0]) for s in streams]
        start = max(1, min(find_common_length([_surface_keys(s) for s in streams], "left"), min(end_idxs) - 1))
        b, sp, cb, cmk = _stack(streams, surf.kmax, surf.bos, device)
        joint = joint_per_token(model, b, sp, cb, cmk)
        pred, _ = pick_min_with_scores(joint, [start] * len(streams), end_idxs)
        return pred
    return model, score
