"""FineWeb-Edu holdout bpb (eval-suite item 12a) — does the ClimbMix-trained surface model
generalize OUT of its training distribution (to FineWeb-Edu) better than baseline?

Computes clean per-RENDERED-byte bpb (surface joint vs baseline token-CE) on a held-out shard of
EACH corpus. The signal is the CROSS-DISTRIBUTION SHIFT in Δ(surf−base): if surface's gap to
baseline improves on the OOD corpus, its representation transfers better than in-distribution bpb
(which is a wash) shows. Not a substitute for a FineWeb training A/B (item 12b) — a cheap probe of
the same question on checkpoints we already have.

OOD corpus = karpathy/fineweb-edu-100b-shuffle LAST shard (= a future FineWeb run's val set).

  uv run python -m wrappers.fineweb_holdout \
    --surf-ckpt .../surface_only/base_checkpoints/d24_surf_lam0p1_s7 \
    --base-ckpt .../baseline/base_checkpoints/d24_baseline_s7 \
    --surf-tok  .../surface_only/tokenizer --base-tok .../baseline/tokenizer --val-docs 400
"""
import argparse
import os

import torch

from nanochat.tokenizer import RustBPETokenizer
from overlay.surface_tokenizer import SurfaceTokenizer
from overlay.surface_core_eval import set_surface_tokenizer
from overlay.surface_checkpoint import surface_build_model
import nanochat.checkpoint_manager as cm
from tools.stack_fm_spaceless import SPACELESS_BODY
from wrappers.surface_robustness import surface_bpb, baseline_bpb, _step
from wrappers.rare_entity_cloze import _load_val, _load_fineweb


def corpus_bpb(scorer, model, tok_or_enc, tb, docs, device):
    bits = byt = 0.0
    for doc in docs:
        b, y = scorer(model, tok_or_enc, tb, doc, device)
        bits += b; byt += y
    return bits / byt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--surf-ckpt", required=True); ap.add_argument("--base-ckpt", required=True)
    ap.add_argument("--surf-tok", required=True); ap.add_argument("--base-tok", required=True)
    ap.add_argument("--val-docs", type=int, default=400); ap.add_argument("--doc-cap", type=int, default=4000)
    ap.add_argument("--device", default="cuda"); ap.add_argument("--kmax", type=int, default=1)
    args = ap.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() or args.device != "cuda" else "cpu")
    os.environ["SURFACE_KMAX"] = str(args.kmax)

    surf = SurfaceTokenizer.from_directory(args.surf_tok, SPACELESS_BODY, kmax=args.kmax)
    set_surface_tokenizer(surf)
    base = RustBPETokenizer.from_directory(args.base_tok)
    surf_tb = torch.load(os.path.join(args.surf_tok, "token_bytes.pt")).to(device).long()
    base_tb = torch.load(os.path.join(args.base_tok, "token_bytes.pt")).to(device).long()

    corpora = [
        ("climbmix-val (in-dist)", _load_val(args.val_docs, args.doc_cap)),
        ("fineweb-edu (OOD)", _load_fineweb(args.val_docs, args.doc_cap)),
    ]
    smodel = surface_build_model(args.surf_ckpt, _step(args.surf_ckpt), device, "eval")[0]; smodel.eval()
    bmodel = cm.build_model(args.base_ckpt, _step(args.base_ckpt), device, "eval")[0]; bmodel.eval()

    print(f"\n{'corpus':>24} {'surf bpb':>10} {'base bpb':>10} {'Δ(s-b)':>9}")
    deltas = []
    for name, docs in corpora:
        s = corpus_bpb(surface_bpb, smodel, surf, surf_tb, docs, device)
        b = corpus_bpb(baseline_bpb, bmodel, base, base_tb, docs, device)
        deltas.append(s - b)
        print(f"{name:>24} {s:>10.4f} {b:>10.4f} {s - b:>+9.4f}")
    print(f"\ncross-distribution shift in Δ(surf−base): {deltas[1] - deltas[0]:+.4f}")
    print("shift < 0 = surface's gap to baseline IMPROVES out-of-distribution (transfers better).")


if __name__ == "__main__":
    main()
