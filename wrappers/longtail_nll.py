"""Long-tail capability eval — does the surface model achieve lower bits/byte on the RARE
tail of REAL held-out text than baseline? (Test 2 of the eval suite.) This is the direct,
non-tautological version of "surface helps rare words": it measures modeling of the genuine
μ<1 tail on natural text, not robustness to synthetic corruption.

Cross-tokenizer comparison on the SHARED frequency axis: each model's per-token NLL is
bucketed by that token's TRUE full-corpus frequency (counted over a train sample with the
model's OWN tokenizer), and reported as bits per RENDERED byte per bucket. The token sets
differ (different tokenizers — inherent), but the rarity axis is shared, so "surface's rare
tokens vs baseline's rare tokens, per byte" is a fair capability comparison. Surface's rare
tokens are the variant-POOLED content atoms (more effective exposure than baseline's rarer
per-variant tokens) — so a surface advantage here IS the pooling mechanism paying off.

Run on the persisted d24 checkpoints:
  uv run python -m wrappers.longtail_nll \
    --surf-ckpt .../surface_only/base_checkpoints/d24_surf_lam0p1_s7 \
    --base-ckpt .../baseline/base_checkpoints/d24_baseline_s7 \
    --surf-tok  .../surface_only/tokenizer --base-tok .../baseline/tokenizer \
    --val-docs 400 --freq-chars 40000000
"""
import argparse
import glob
import math
import os
from collections import Counter

import torch

from nanochat.dataset import parquets_iter_batched
from overlay.dataset_fineweb import maybe_repoint
from nanochat.tokenizer import RustBPETokenizer
from overlay.surface_tokenizer import SurfaceTokenizer
from overlay.surface_core_eval import set_surface_tokenizer, _encode_prompt
from overlay.surface_checkpoint import surface_build_model
import nanochat.checkpoint_manager as cm
from tools.stack_fm_spaceless import SPACELESS_BODY

LN2 = math.log(2)
FREQ_EDGES = [1, 10, 100, 1_000, 10_000]
FREQ_LABELS = ["0(unseen)", "1-9", "10-99", "100-999", "1k-9999", "10k+"]


def _bucket(c):
    return sum(1 for e in FREQ_EDGES if c >= e)


def _step(ckpt):
    return sorted(int(os.path.basename(p).split("_")[1].split(".")[0])
                  for p in glob.glob(os.path.join(ckpt, "model_*.pt")))[-1]


def surf_token_freq(surf, chars, cap=10_000):
    f = Counter(); seen = 0
    for batch in parquets_iter_batched(split="train"):
        for doc in batch:
            for tid, *_ in surf.encode(doc[:cap]):
                f[tid] += 1
            seen += min(len(doc), cap)
        if seen >= chars:
            break
    return f


def base_token_freq(enc, chars, cap=10_000):
    f = Counter(); seen = 0
    for batch in parquets_iter_batched(split="train"):
        for doc in batch:
            f.update(enc.enc.encode_ordinary(doc[:cap]))
            seen += min(len(doc), cap)
        if seen >= chars:
            break
    return f


@torch.no_grad()
def surf_per_token(model, doc, device):
    """(target_base_id, joint_NLL_nats, rendered_bytes) per target token for one doc."""
    b, s, cb, cm_ = (torch.tensor([x], device=device) for x in _encode_prompt(doc))
    joint = model(b[:, :-1], b[:, 1:], loss_reduction='none', space_x=s[:, :-1], cap_bits_x=cb[:, :-1],
                  cap_mask_x=cm_[:, :-1], space_y=s[:, 1:], cap_bits_y=cb[:, 1:], cap_mask_y=cm_[:, 1:]).reshape(-1)
    return b[0, 1:], joint, s[0, 1:]


@torch.no_grad()
def base_per_token(model, enc, doc, device):
    ids = torch.tensor([[enc.get_bos_token_id()] + enc.enc.encode_ordinary(doc)], device=device)
    ce = model(ids[:, :-1], ids[:, 1:], loss_reduction='none').reshape(-1)
    return ids[0, 1:], ce


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--surf-ckpt", required=True); ap.add_argument("--base-ckpt", required=True)
    ap.add_argument("--surf-tok", required=True); ap.add_argument("--base-tok", required=True)
    ap.add_argument("--val-docs", type=int, default=400); ap.add_argument("--doc-cap", type=int, default=4000)
    ap.add_argument("--freq-chars", type=int, default=40_000_000)
    ap.add_argument("--device", default="cuda"); ap.add_argument("--kmax", type=int, default=1)
    args = ap.parse_args()
    maybe_repoint()                              # honor NANOCHAT_DATASET=fineweb → base_data_fineweb
    device = torch.device(args.device if torch.cuda.is_available() or args.device != "cuda" else "cpu")
    os.environ["SURFACE_KMAX"] = str(args.kmax)

    surf = SurfaceTokenizer.from_directory(args.surf_tok, SPACELESS_BODY, kmax=args.kmax)
    set_surface_tokenizer(surf)
    base = RustBPETokenizer.from_directory(args.base_tok)
    surf_tb = torch.load(os.path.join(args.surf_tok, "token_bytes.pt")).long()
    base_tb = torch.load(os.path.join(args.base_tok, "token_bytes.pt")).long()

    print(f"counting token frequencies over ~{args.freq_chars/1e6:.0f}M chars (each tokenizer) ...")
    sfreq = surf_token_freq(surf, args.freq_chars)
    bfreq = base_token_freq(base, args.freq_chars)
    sfb = torch.zeros(surf.n_vocab, dtype=torch.long); [sfb.__setitem__(k, v) for k, v in sfreq.items()]
    bfb = torch.zeros(base.enc.n_vocab, dtype=torch.long); [bfb.__setitem__(k, v) for k, v in bfreq.items()]

    docs = []
    for batch in parquets_iter_batched(split="val"):
        for d in batch:
            docs.append(d[:args.doc_cap])
            if len(docs) >= args.val_docs:
                break
        if len(docs) >= args.val_docs:
            break

    nb = len(FREQ_EDGES) + 1
    s_bits = torch.zeros(nb); s_byte = torch.zeros(nb)
    b_bits = torch.zeros(nb); b_byte = torch.zeros(nb)

    smodel, _, _ = surface_build_model(args.surf_ckpt, _step(args.surf_ckpt), device, "eval"); smodel.eval()
    for doc in docs:
        yb, joint, sy = surf_per_token(smodel, doc, device)
        for i in range(yb.numel()):
            bb = surf_tb[yb[i]].item() + sy[i].item()
            if bb <= 0:
                continue
            k = _bucket(sfb[yb[i]].item())
            s_bits[k] += joint[i].item() / LN2; s_byte[k] += bb
    del smodel
    if device.type == "cuda":
        torch.cuda.empty_cache()

    bmodel, _, _ = cm.build_model(args.base_ckpt, _step(args.base_ckpt), device, "eval"); bmodel.eval()
    for doc in docs:
        y, ce = base_per_token(bmodel, base, doc, device)
        for i in range(y.numel()):
            bb = base_tb[y[i]].item()
            if bb <= 0:
                continue
            k = _bucket(bfb[y[i]].item())
            b_bits[k] += ce[i].item() / LN2; b_byte[k] += bb

    print(f"\n{'freq bucket':>12} {'surf bpb':>10} {'base bpb':>10} {'Δ(s-b)':>9} {'surf %byte':>10} {'base %byte':>10}")
    st, bt = s_byte.sum().item(), b_byte.sum().item()
    for k in range(nb):
        sp = s_bits[k] / s_byte[k] if s_byte[k] > 0 else float('nan')
        bp = b_bits[k] / b_byte[k] if b_byte[k] > 0 else float('nan')
        print(f"{FREQ_LABELS[k]:>12} {sp:>10.4f} {bp:>10.4f} {sp-bp:>+9.4f} "
              f"{100*s_byte[k].item()/st:>9.1f}% {100*b_byte[k].item()/bt:>9.1f}%")
    print("\nΔ<0 in the rare buckets (left) = surface models the long tail better per byte.")
    print("NB token sets differ (different tokenizers); the rarity axis is shared. Surface's rare")
    print("tokens are the variant-POOLED atoms, so an edge here is the pooling mechanism paying off.")


if __name__ == "__main__":
    main()
