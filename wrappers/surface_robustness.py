"""Surface-form ROBUSTNESS eval — does the surface model degrade LESS than baseline under
casing/spacing perturbations the educational-value-filtered ClimbMix corpus lacks?

Motivation: ClimbMix is curated with an educational-value classifier (nvidia/ClimbMix card),
so it is clean / surface-regular and never stresses surface factoring's designed advantage.
We inject that missing messiness on held-out text and measure the cost to each model.

Cross-tokenizer-FAIR: bits per RENDERED BYTE on the SAME text (never per-token). Surface NLL
= joint content+space+cap; baseline NLL = standard token CE; both ÷ bytes. We report
Δbpb = bpb(perturbed) − bpb(clean) for each model:
  - titlecase flips (The↔the) + leading-space  → surface's DESIGNED invariances → predict it
    degrades LESS (content token unchanged, only the cheap bit moves).
  - ALLCAPS (THE)  → HONESTY CONTROL: surface folds only titlecase, so no edge here (maybe worse).

Run on the persisted d24 checkpoints once training finishes:
  NANOCHAT_BASE_DIR unused; pass explicit dirs.
  uv run python -m wrappers.surface_robustness \
    --surf-ckpt /workspace/nanochat-variants/surface_only/base_checkpoints/d24_surf_lam0p1_s7 \
    --base-ckpt /workspace/nanochat-variants/baseline/base_checkpoints/d24_baseline_s7 \
    --surf-tok  /workspace/nanochat-variants/surface_only/tokenizer \
    --base-tok  /workspace/nanochat-variants/baseline/tokenizer \
    --val-docs 400 --p 0.5
"""
import argparse
import glob
import math
import os

import regex
import torch

from nanochat.dataset import parquets_iter_batched
from nanochat.tokenizer import RustBPETokenizer
from overlay.surface_tokenizer import SurfaceTokenizer
from overlay.surface_core_eval import set_surface_tokenizer, _encode_prompt
from overlay.surface_checkpoint import surface_build_model
import nanochat.checkpoint_manager as cm
from tools.stack_fm_spaceless import SPACELESS_BODY

LN2 = math.log(2)
_WORD = regex.compile(r"\p{L}+")


# --------------------------- perturbations (on raw text) --------------------------- #
def flip_titlecase(text, p, rng):
    """Flip each word between Title and lower (The↔the) with prob p — surface's designed
    invariance (the content token is the same; only the cap bit changes)."""
    def repl(m):
        w = m.group()
        if rng.random() >= p:
            return w
        if len(w) > 1 and w[0].isupper() and w[1:].islower():
            return w.lower()
        if w.islower():
            return w[0].upper() + w[1:]
        return w
    return _WORD.sub(repl, text)


def allcaps(text, p, rng):
    """Uppercase whole words with prob p (THE) — surface does NOT fold all-caps → no edge."""
    return _WORD.sub(lambda m: m.group().upper() if rng.random() < p else m.group(), text)


def double_space(text, p, rng):
    """Double a single space with prob p — probes leading-space handling."""
    return regex.sub(r"(?<=\S) (?=\S)", lambda m: "  " if rng.random() < p else " ", text)


PERTURB = {"titlecase": flip_titlecase, "allcaps": allcaps, "spacing": double_space}


# --------------------------- per-byte NLL (each model) --------------------------- #
@torch.no_grad()
def surface_bpb(model, tok, token_bytes, doc, device):
    """Joint content+space+cap bits / rendered byte for one doc (the λ-free eval path)."""
    base, space, capb, capm = (torch.tensor([s], device=device) for s in _encode_prompt(doc))
    xb, yb = base[:, :-1], base[:, 1:]
    xs, ys = space[:, :-1], space[:, 1:]
    xcb, ycb = capb[:, :-1], capb[:, 1:]
    xcm, ycm = capm[:, :-1], capm[:, 1:]
    joint = model(xb, yb, loss_reduction='none', space_x=xs, cap_bits_x=xcb, cap_mask_x=xcm,
                  space_y=ys, cap_bits_y=ycb, cap_mask_y=ycm).reshape(-1)
    yf = yb.reshape(-1)
    bb = token_bytes[yf].to(torch.float32) + ys.reshape(-1).to(torch.float32)
    m = bb > 0
    return joint[m].sum().item() / LN2, bb[m].sum().item()


@torch.no_grad()
def baseline_bpb(model, enc, token_bytes, doc, device):
    """Standard token CE bits / byte for one doc."""
    ids = [enc.get_bos_token_id()] + enc.enc.encode_ordinary(doc)
    t = torch.tensor([ids], device=device)
    x, y = t[:, :-1], t[:, 1:]
    loss = model(x, y, loss_reduction='none').reshape(-1)
    yf = y.reshape(-1)
    bb = token_bytes[yf].to(torch.float32)
    m = bb > 0
    return loss[m].sum().item() / LN2, bb[m].sum().item()


def _step(ckpt):
    return sorted(int(os.path.basename(p).split("_")[1].split(".")[0])
                  for p in glob.glob(os.path.join(ckpt, "model_*.pt")))[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--surf-ckpt", required=True)
    ap.add_argument("--base-ckpt", required=True)
    ap.add_argument("--surf-tok", required=True)
    ap.add_argument("--base-tok", required=True)
    ap.add_argument("--val-docs", type=int, default=400)
    ap.add_argument("--doc-cap", type=int, default=4000)
    ap.add_argument("--p", type=float, default=0.5, help="perturbation probability per word/space")
    ap.add_argument("--perturbs", default="titlecase,allcaps,spacing")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--kmax", type=int, default=int(os.environ.get("SURFACE_KMAX", "1")))
    args = ap.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() or args.device != "cuda" else "cpu")
    os.environ["SURFACE_KMAX"] = str(args.kmax)
    import random
    rng = random.Random(1234)

    # tokenizers
    surf = SurfaceTokenizer.from_directory(args.surf_tok, SPACELESS_BODY, kmax=args.kmax)
    set_surface_tokenizer(surf)
    base = RustBPETokenizer.from_directory(args.base_tok)
    surf_tb = torch.load(os.path.join(args.surf_tok, "token_bytes.pt")).to(device).long()
    base_tb = torch.load(os.path.join(args.base_tok, "token_bytes.pt")).to(device).long()

    # held-out val docs
    docs = []
    for batch in parquets_iter_batched(split="val"):
        for d in batch:
            docs.append(d[:args.doc_cap])
            if len(docs) >= args.val_docs:
                break
        if len(docs) >= args.val_docs:
            break
    print(f"{len(docs)} val docs; perturbations: {args.perturbs} @ p={args.p}")

    # build the clean + each-perturbed text once (same rng draw per doc across models = fair)
    variants = {"clean": [d for d in docs]}
    for name in args.perturbs.split(","):
        fn = PERTURB[name.strip()]
        variants[name.strip()] = [fn(d, args.p, random.Random(hash(d) & 0xffff)) for d in docs]

    smodel, _, _ = surface_build_model(args.surf_ckpt, _step(args.surf_ckpt), device, "eval"); smodel.eval()
    bmodel, _, _ = cm.build_model(args.base_ckpt, _step(args.base_ckpt), device, "eval"); bmodel.eval()

    def corpus_bpb(scorer, *a):
        bits = byt = 0.0
        for doc in a[-1]:
            b, y = scorer(*a[:-1], doc, device)
            bits += b; byt += y
        return bits / byt

    print(f"\n{'variant':>12} {'surface bpb':>12} {'baseline bpb':>13} {'Δsurf':>9} {'Δbase':>9}")
    s_clean = corpus_bpb(surface_bpb, smodel, surf, surf_tb, variants["clean"])
    b_clean = corpus_bpb(baseline_bpb, bmodel, base, base_tb, variants["clean"])
    print(f"{'clean':>12} {s_clean:>12.4f} {b_clean:>13.4f} {'—':>9} {'—':>9}")
    for name in args.perturbs.split(","):
        name = name.strip()
        s = corpus_bpb(surface_bpb, smodel, surf, surf_tb, variants[name])
        b = corpus_bpb(baseline_bpb, bmodel, base, base_tb, variants[name])
        print(f"{name:>12} {s:>12.4f} {b:>13.4f} {s-s_clean:>+9.4f} {b-b_clean:>+9.4f}")
    print("\nΔ = bpb(perturbed) − bpb(clean). Surface should show a SMALLER Δ on titlecase/spacing")
    print("(its factored axes) and ~no advantage on allcaps (the honesty control).")


if __name__ == "__main__":
    main()
