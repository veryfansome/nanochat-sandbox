"""
Content-only bpb reporter for a surface checkpoint. Splits the surface joint NLL into
its CONTENT stream vs the space+cap side channel, per RAW byte (same byte basis as
surface_evaluate_bpb: token_bytes[base_y] + space_y), and reports content bpb, joint
bpb, and the overhead.

VALID USE — PAIRED within-surface ablations: run on two surface arms differing in one
knob (e.g. λ_test vs λ_control) at the same commit/seed/data-order and diff the content
bpb. The offloading cancels, so Δ(content-bpb) cleanly measures whether the knob freed
content capacity (the cheap-ablation KILL signal; see the idea doc).

NOT a cross-tokenizer comparison: a surface content stream encodes LESS than a
phrase/standard tokenizer's tokens (spacing/casing is offloaded), so a lower content
bpb is accounting, not better modeling — comparing it to a force_merges/baseline bpb is
confounded. The clean cross-tokenizer signal is the JOINT bpb. Runs locally (MPS).

  NANOCHAT_BASE_DIR=~/.cache/nanochat-variants/surface_only \
    uv run python -m wrappers.content_bpb --ckpt-dir results/<run>/checkpoint --steps 60
"""
import argparse
import glob
import math
import os

import torch
import torch.nn.functional as F

from nanochat.common import get_base_dir
from nanochat.tokenizer import get_tokenizer, get_token_bytes
from tools.stack_fm_spaceless import SPACELESS_BODY
from overlay.surface_tokenizer import SurfaceTokenizer
from overlay.surface_checkpoint import surface_build_model
from overlay.surface_core_eval import set_surface_tokenizer
from overlay.surface_dataloader import surface_data_loader
from overlay.surface_context import get_surface


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--B", type=int, default=8)
    ap.add_argument("--T", type=int, default=512)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--ckpt-dir", default="results/d24_surface_only_ab/checkpoint",
                    help="dir with model_*.pt + meta_*.json (a surface checkpoint)")
    args = ap.parse_args()
    device = torch.device(args.device if (args.device != "mps" or torch.backends.mps.is_available()) else "cpu")
    base = get_base_dir()

    KMAX = int(os.environ.get("SURFACE_KMAX", "1"))   # surface-only: single-word tokens
    os.environ["SURFACE_KMAX"] = str(KMAX)
    enc = get_tokenizer().enc
    TOK = SurfaceTokenizer(enc, SPACELESS_BODY, kmax=KMAX)
    set_surface_tokenizer(TOK)
    token_bytes = get_token_bytes().to(device)

    ckpt_dir = args.ckpt_dir
    avail = sorted(int(os.path.basename(p).split("_")[1].split(".")[0])
                   for p in glob.glob(ckpt_dir + "/model_*.pt"))
    model, _, _ = surface_build_model(ckpt_dir, avail[-1], device, "eval")
    model.eval()
    print(f"loaded surface checkpoint {os.path.basename(ckpt_dir.rstrip('/'))} "
          f"step {avail[-1]} on {device}, K_max={KMAX}")

    loader = surface_data_loader(TOK, args.B, args.T, "val", device)
    it = iter(loader)
    content_nats = torch.zeros((), device=device)
    joint_nats = torch.zeros((), device=device)
    total_bytes = torch.zeros((), dtype=torch.int64, device=device)
    with torch.no_grad():
        for s in range(args.steps):
            x, y = next(it)
            sc = get_surface()
            space_y = sc["space_y"] if sc is not None else torch.zeros_like(y)
            # joint NLL (content + space + cap), as the A/B's surface_evaluate_bpb does
            joint2d = model(x, y, loss_reduction="none").reshape(-1)
            # content-only NLL: raw content logits → CE against the content targets
            clogits, _, _ = model(x, y, return_surface_logits=True)
            content2d = F.cross_entropy(clogits.reshape(-1, clogits.size(-1)), y.reshape(-1),
                                        ignore_index=-1, reduction="none")
            yf, spf = y.reshape(-1), space_y.reshape(-1)
            valid = yf >= 0
            ys = torch.where(valid, yf, torch.zeros_like(yf))
            bb = torch.where(valid, token_bytes[ys], torch.zeros_like(yf, dtype=token_bytes.dtype))
            nb = bb + (spf * (bb > 0)).to(token_bytes.dtype)          # raw bytes = base + factored space
            content_nats += (content2d * (nb > 0)).sum()
            joint_nats += (joint2d * (nb > 0)).sum()
            total_bytes += nb.sum()
            if (s + 1) % 10 == 0:
                print(f"  step {s+1}/{args.steps} ...")

    ln2 = math.log(2)
    tb = total_bytes.item()
    cbpb = content_nats.item() / (ln2 * tb)
    jbpb = joint_nats.item() / (ln2 * tb)
    print("\n=== RESULT ===")
    print(f"bytes scored        : {tb:,}")
    print(f"joint bpb           : {jbpb:.4f}")
    print(f"content-only bpb    : {cbpb:.4f}")
    print(f"space+cap overhead  : {jbpb - cbpb:.4f} bits/byte")
    print("\nFor a verdict, diff content-only bpb against a PAIRED surface control (same")
    print("commit/seed/data-order, one knob changed). Absolute content bpb is NOT comparable")
    print("across tokenizers (offloading confound) — see the module docstring.")


if __name__ == "__main__":
    main()
