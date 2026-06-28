"""Rarity-bucketed content-NLL for surface checkpoints, GROUNDED in the tokenizer pooling gain
(the step-3 companion to tools/pooling_gain.py). Generic + reusable: point it at any surface
tokenizer + one or two checkpoints.

For each held-out val target token it accumulates the model's CONTENT-only bits/byte into a
bucket given by that token's POOLING GAIN (total firings of its baseline space/case variants /
busiest variant, from tools.pooling_gain) — plus a NEW-atom bucket (content absent from
baseline). With ONE checkpoint you get the content-bits/byte profile across the manipulation;
with TWO (e.g. a λ-control vs candidate, same tokenizer + val order = paired) you get the
per-bucket Δ — i.e. does a content-bpb difference concentrate where pooling actually acted?

Reads everything from NANOCHAT_BASE_DIR (tokenizer + data + base_checkpoints/<tag>). Forward
only — a single modest GPU is plenty; no torchrun.

Usage (on a pod with the surface volume mounted, NANOCHAT_BASE_DIR=/workspace/nanochat-variants/<v>):
  SURFACE_KMAX=1 uv run python -m wrappers.rarity_delta \
      --tags d24_surf_lam0p5_k1850 d24_surf_lam0p1_k1850 --val-steps 120 --pool-chars 20000000
"""
import argparse
import glob
import math
import os

import torch
import torch.nn.functional as F

from nanochat.tokenizer import RustBPETokenizer
from overlay.surface_tokenizer import SurfaceTokenizer
from overlay.surface_checkpoint import surface_build_model
from overlay.surface_core_eval import set_surface_tokenizer
from overlay.surface_dataloader import surface_data_loader
from overlay.surface_context import get_surface
from tools.pooling_gain import baseline_content_stats, content_pooling, pooling_bucket, EDGES, LAB
from tools.stack_fm_spaceless import SPACELESS_BODY

LN2 = math.log(2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-dir", default=os.environ.get("NANOCHAT_BASE_DIR", os.path.expanduser("~/.cache/nanochat")))
    ap.add_argument("--tok-dir", default=None, help="default {base-dir}/tokenizer")
    ap.add_argument("--baseline-tok", default=os.path.expanduser("~/.cache/nanochat/tokenizer"))
    ap.add_argument("--pattern", default=SPACELESS_BODY, help="surface spaceless split pattern")
    ap.add_argument("--kmax", type=int, default=int(os.environ.get("SURFACE_KMAX", "1")))
    ap.add_argument("--tags", nargs="+", required=True, help="1 or 2 checkpoint tags under base-dir/base_checkpoints/")
    ap.add_argument("--pool-chars", type=int, default=20_000_000)
    ap.add_argument("--val-steps", type=int, default=120)
    ap.add_argument("--B", type=int, default=8)
    ap.add_argument("--T", type=int, default=512)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    tok_dir = args.tok_dir or os.path.join(args.base_dir, "tokenizer")
    device = torch.device(args.device if torch.cuda.is_available() or args.device != "cuda" else "cpu")
    os.environ["SURFACE_KMAX"] = str(args.kmax)

    # --- surface tokenizer + the pooling-gain bucket per content id ---
    TOK = SurfaceTokenizer.from_directory(tok_dir, args.pattern, kmax=args.kmax)
    set_surface_tokenizer(TOK)
    surf_enc = TOK.enc
    V = surf_enc.n_vocab
    token_bytes = torch.load(os.path.join(tok_dir, "token_bytes.pt")).to(device).long()

    print(f"grounding pooling gain from {args.baseline_tok} over ~{args.pool_chars/1e6:.0f}M chars ...")
    base = RustBPETokenizer.from_directory(args.baseline_tok).enc
    bfir, id2content, _ = baseline_content_stats(base, args.pool_chars)
    stats = content_pooling(bfir, id2content)              # fired contents -> [total, max, nvar]
    base_content_vocab = set(id2content)                  # ALL baseline vocab content forms (sample-INDEPENDENT)
    NEWB = len(EDGES) + 1                                   # new-atom bucket index
    labels = LAB + ["NEW-atom"]
    bucket_of = torch.zeros(V, dtype=torch.long)           # default bucket 0 (=1 / low / undecodable)
    for i in range(V):
        try:
            s = surf_enc.decode_single_token_bytes(i).decode("utf-8")
        except UnicodeDecodeError:
            continue                                       # partial-byte token -> bucket 0
        if not s:
            continue
        if s not in base_content_vocab:
            bucket_of[i] = NEWB                            # content baseline lacks in any variant = genuinely new
        else:
            g = stats.get(s)                               # in baseline vocab: bucket by pooling gain if it fired
            if g:
                bucket_of[i] = pooling_bucket(g[0] / g[1])
    bucket_of = bucket_of.to(device)
    nb = len(labels)
    print("bucket sizes (vocab):", {labels[k]: int((bucket_of == k).sum()) for k in range(nb)})

    # --- per checkpoint: accumulate content bits + bytes per bucket over the SAME val tokens ---
    avail = lambda tag: sorted(int(os.path.basename(p).split("_")[1].split(".")[0])
                               for p in glob.glob(os.path.join(args.base_dir, "base_checkpoints", tag, "model_*.pt")))
    results = {}
    for tag in args.tags:
        ck = os.path.join(args.base_dir, "base_checkpoints", tag)
        step = avail(tag)[-1]
        model, _, _ = surface_build_model(ck, step, device, "eval")
        model.eval()
        bits = torch.zeros(nb, device=device); byte = torch.zeros(nb, device=device)
        cnt = torch.zeros(nb, device=device)
        it = iter(surface_data_loader(TOK, args.B, args.T, "val", device))
        with torch.no_grad():
            for s in range(args.val_steps):
                x, y = next(it)
                sc = get_surface()
                space_y = sc["space_y"] if sc is not None else torch.zeros_like(y)
                clogits, _, _ = model(x, y, return_surface_logits=True)
                cb = F.cross_entropy(clogits.reshape(-1, clogits.size(-1)), y.reshape(-1),
                                     ignore_index=-1, reduction="none") / LN2
                yf, spf = y.reshape(-1), space_y.reshape(-1)
                ys = torch.where(yf >= 0, yf, torch.zeros_like(yf))
                bb = token_bytes[ys].to(torch.float32)
                m = (yf >= 0) & (bb > 0)
                rk = bucket_of[ys][m]
                bits.index_add_(0, rk, cb[m])
                byte.index_add_(0, rk, (bb + spf.to(torch.float32))[m])
                cnt.index_add_(0, rk, torch.ones_like(cb[m]))
        results[tag] = (bits.cpu(), byte.cpu(), cnt.cpu())
        tb = byte.sum().item()
        print(f"  {tag}: overall content bpb {bits.sum().item()/tb:.4f}  ({int(cnt.sum()):,} positions, {int(tb):,} bytes)")
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # --- report ---
    tags = args.tags
    tot_byte = results[tags[0]][1].sum().item()
    print(f"\n############ content bits/byte by POOLING-GAIN bucket ({'paired Δ' if len(tags)==2 else 'single arm'}) ############")
    hdr = f"{'bucket':>16} {'byteshare':>10} {'positions':>10}" + "".join(f"{t[-12:]:>16}" for t in tags)
    if len(tags) == 2:
        hdr += f"{'Δ(b-a)':>10}"
    print(hdr)
    for k in range(nb):
        share = 100 * results[tags[0]][1][k].item() / tot_byte
        pos = int(results[tags[0]][2][k].item())
        if pos == 0:
            continue
        bpb = [results[t][0][k].item() / results[t][1][k].item() for t in tags]
        row = f"{labels[k]:>16} {share:>9.2f}% {pos:>10,}" + "".join(f"{v:>16.4f}" for v in bpb)
        if len(tags) == 2:
            row += f"{bpb[1]-bpb[0]:>+10.4f}"
        print(row)
    if len(tags) == 2:
        d = sum(results[tags[1]][0]).item() / sum(results[tags[1]][1]).item() - \
            sum(results[tags[0]][0]).item() / sum(results[tags[0]][1]).item()
        print(f"\noverall content-bpb Δ ({tags[1]} − {tags[0]}) = {d:+.4f}")


if __name__ == "__main__":
    main()
