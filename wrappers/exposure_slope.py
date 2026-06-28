"""
Exposure-slope diagnostic for surface factoring — FREE (existing checkpoint, 0 new training).

QUESTION: is the d24 surface wash an UNDERTRAINING floor rather than a ceiling? The
whole-word analysis found surface's +4.45% density gain is ~77.5% RARE LONG single-word
atoms. Rare atoms get few gradient updates, so their embeddings may be undertrained at
the d24 step budget — meaning more exposure (a longer/larger run) could convert the
compression into capability. This probes that on the checkpoint we have, with 0 new steps.

CHECKPOINT: the clean surface-only d24 checkpoint was LOST (download failed before the pod
was terminated); the only local surface checkpoint is the deprecated TRIPLE
(results/d24_surface_factoring_ab, K_max=3, fm phrases + space + case). The triple mixes
fm cross-word PHRASE tokens (frequent glue, well-trained) with surface's single-word atoms,
so we SPLIT single-word vs phrase and read the SINGLE-WORD cut as the surface-densification
proxy. Same spaceless+folded BPE mechanism as surface-only => directional, not exact.

CROSS-SECTIONAL, not temporal: only the final checkpoint survived (no intermediates), so
this answers "are rare/long atoms underfit AT 5569 steps?", NOT directly "would more steps
help" (exposure is not short-run triageable — only a LONGER run confirms). Two cuts on
held-out val CONTENT-NLL (the surface model's content stream, bits per RAW byte =
token_bytes[base] + space_bit, same basis as surface_evaluate_bpb / content_bpb):

  (a) content bits/byte & bits/token vs TARGET-TOKEN BYTE-LENGTH  [sample-independent, headline]
      A well-fit compressor AMORTIZES: a long token costs ~one prediction but covers many
      bytes, so bits/byte should FALL as tokens get longer. If long single-word atoms instead
      stay flat / rise in bits/byte, the compression is not realized in modeling => underfit.

  (b) content bits/token vs TARGET-TOKEN TRAIN-FREQUENCY (exposure proxy) + a UNIGRAM
      baseline: context_gain = (-log2 p_unigram) - model_bits. If context_gain collapses
      toward 0 / negative for rare tokens, the model can't beat the base rate even with
      context => the rare-token embeddings are undertrained.

Each bucket also reports its share of total bytes — if the rare/long regime is a tiny slice
of text, even a big per-token penalty barely moves aggregate bpb (which itself partly
explains why the +4.90% compression didn't move val_bpb).

Usage (DATA comes from the default base dir; the TRIPLE tokenizer is loaded explicitly):
  uv run python -m wrappers.exposure_slope --val-steps 120 --freq-chars 40000000
"""
import argparse
import glob
import math
import os
from collections import Counter

import torch
import torch.nn.functional as F

from nanochat.dataset import parquets_iter_batched
from tools.stack_triple import CFG_TRIPLE, FOLDED_PAIRS
from overlay.surface_tokenizer import SurfaceTokenizer
from overlay.surface_checkpoint import surface_build_model
from overlay.surface_core_eval import set_surface_tokenizer
from overlay.surface_dataloader import surface_data_loader
from overlay.surface_context import get_surface

LN2 = math.log(2)
TRIPLE_TOK = os.path.expanduser("~/.cache/nanochat-variants/surface_triple/tokenizer")

# byte-length buckets (edges are lower-inclusive starts of buckets 1..n; bucket 0 = below first edge)
LEN_EDGES = [2, 4, 7, 11]                    # 1 | 2-3 | 4-6 | 7-10 | 11+
LEN_LABELS = ["1", "2-3", "4-6", "7-10", "11+"]
FREQ_EDGES = [1, 10, 100, 1_000, 10_000]     # 0 | 1-9 | 10-99 | 100-999 | 1k-9999 | 10k+
FREQ_LABELS = ["0(unseen)", "1-9", "10-99", "100-999", "1k-9999", "10k+"]
# merge rank (== token id for rustbpe): later merge = rarer pattern = less exposure.
# Sample-INDEPENDENT rarity proxy (greedy BPE merges most-frequent pairs first), so it
# avoids the local-frequency noise in cut (b).
RANK_EDGES = [512, 2_048, 8_192, 16_384]     # <512 | 512-2k | 2k-8k | 8k-16k | 16k+
RANK_LABELS = ["<512", "512-2k", "2k-8k", "8k-16k", "16k+"]


def bucketize(vals, edges):
    idx = torch.zeros_like(vals, dtype=torch.long)
    for e in edges:
        idx += (vals >= e).long()
    return idx


@torch.no_grad()
def rank_pass(model, TOK, token_bytes, is_phrase, split, steps, B, T, device):
    """Single-word-atom content bits/byte per MERGE-RANK bucket over `split`. Used to
    compare HELD-IN train vs HELD-OUT val per rank — the ceiling/convergence probe the
    cross-sectional snapshot lacks: a train-val gap that WIDENS toward the rare tail =
    rare-specific generalization gap (exposure/regularization live); a flat gap = the
    tail penalty is the same on seen+unseen data (inherent / uniform, not rare-specific)."""
    nR = len(RANK_EDGES) + 1
    bits = torch.zeros(nR, device=device); byte = torch.zeros(nR, device=device)
    cnt = torch.zeros(nR, device=device)
    it = iter(surface_data_loader(TOK, B, T, split, device))
    for _ in range(steps):
        x, y = next(it)
        sc = get_surface()
        space_y = sc["space_y"] if sc is not None else torch.zeros_like(y)
        clogits, _, _ = model(x, y, return_surface_logits=True)
        cbits = F.cross_entropy(clogits.reshape(-1, clogits.size(-1)), y.reshape(-1),
                                ignore_index=-1, reduction="none") / LN2
        yf, spf = y.reshape(-1), space_y.reshape(-1)
        ys = torch.where(yf >= 0, yf, torch.zeros_like(yf))
        bb = token_bytes[ys].to(torch.float32)
        m = (yf >= 0) & (bb > 0) & (is_phrase[ys] == 0)            # single-word, real-byte
        if not m.any():
            continue
        rk = bucketize(ys, RANK_EDGES)[m]
        bits.index_add_(0, rk, cbits[m]); byte.index_add_(0, rk, (bb + spf.to(torch.float32))[m])
        cnt.index_add_(0, rk, torch.ones_like(cbits[m]))
    return bits, byte, cnt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-steps", type=int, default=120)
    ap.add_argument("--train-steps", type=int, default=0,
                    help="if >0, also run a HELD-IN train pass and report the train-val gap per rank")
    ap.add_argument("--freq-chars", type=int, default=40_000_000)
    ap.add_argument("--B", type=int, default=8)
    ap.add_argument("--T", type=int, default=512)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--ckpt-dir", default="results/d24_surface_factoring_ab/checkpoint")
    args = ap.parse_args()
    device = torch.device(args.device if (args.device != "mps" or torch.backends.mps.is_available()) else "cpu")

    KMAX = max(2, max(len((a + b).split()) for a, b in FOLDED_PAIRS))    # 3 for the triple
    os.environ["SURFACE_KMAX"] = str(KMAX)
    # TRIPLE tokenizer (32890 vocab) loaded explicitly; DATA stays on the default base dir.
    TOK = SurfaceTokenizer.from_directory(TRIPLE_TOK, CFG_TRIPLE["spaceless_pattern"], kmax=KMAX)
    set_surface_tokenizer(TOK)
    enc = TOK.enc
    V = enc.n_vocab
    token_bytes = torch.load(os.path.join(TRIPLE_TOK, "token_bytes.pt")).to(device).long()
    assert token_bytes.numel() == V, f"token_bytes {token_bytes.numel()} != vocab {V}"

    id2bytes = [enc.decode_single_token_bytes(i) for i in range(V)]
    byte_len = token_bytes                                               # counted text bytes (specials=0)
    # phrase = an INTERNAL space byte (the triple's cross-word fm phrase tokens; spaceless
    # single-word atoms never contain a space — the leading space is factored to a bit)
    is_phrase = torch.tensor([(b" " in b[1:]) for b in id2bytes], device=device)

    avail = sorted(int(os.path.basename(p).split("_")[1].split(".")[0])
                   for p in glob.glob(args.ckpt_dir + "/model_*.pt"))
    assert avail, f"no model_*.pt in {args.ckpt_dir}"
    model, _, _ = surface_build_model(args.ckpt_dir, avail[-1], device, "eval")
    model.eval()
    print(f"loaded TRIPLE checkpoint step {avail[-1]} on {device}, K_max={KMAX}, vocab={V}")

    # ---- Pass 1: train-corpus frequencies (exposure proxy) -------------------
    print(f"counting train base-token frequencies over ~{args.freq_chars/1e6:.0f}M chars ...")
    freq = Counter()
    seen = 0
    for batch in parquets_iter_batched(split="train"):
        for doc in batch:
            d = doc[:10_000]
            for tid, *_ in TOK.encode(d):
                freq[tid] += 1
            seen += len(d)
        if seen >= args.freq_chars:
            break
    total_tok = sum(freq.values())
    freq_cpu = torch.zeros(V, dtype=torch.long)                          # build on CPU (MPS has no f64)
    for tid, c in freq.items():
        freq_cpu[tid] = c
    p_uni = (freq_cpu.double() + 1.0) / (total_tok + V)                  # add-1 smoothed
    uni_bits = (-torch.log2(p_uni)).float().to(device)
    freq_t = freq_cpu.to(device)
    print(f"  counted {total_tok:,} base tokens; {int((freq_cpu == 0).sum())}/{V} vocab ids unseen locally")

    # ---- Pass 2: held-out val content-NLL, accumulate into buckets -----------
    nL, nF, nR = len(LEN_EDGES) + 1, len(FREQ_EDGES) + 1, len(RANK_EDGES) + 1
    len_bits = torch.zeros(2, nL, device=device); len_byte = torch.zeros(2, nL, device=device)
    len_cnt = torch.zeros(2, nL, device=device); len_lensum = torch.zeros(2, nL, device=device)
    frq_bits = torch.zeros(2, nF, device=device); frq_byte = torch.zeros(2, nF, device=device)
    frq_cnt = torch.zeros(2, nF, device=device); frq_uni = torch.zeros(2, nF, device=device)
    rnk_bits = torch.zeros(2, nR, device=device); rnk_byte = torch.zeros(2, nR, device=device)
    rnk_cnt = torch.zeros(2, nR, device=device); rnk_lensum = torch.zeros(2, nR, device=device)
    rnk_uni = torch.zeros(2, nR, device=device)
    tot_bits = torch.zeros((), device=device); tot_byte = torch.zeros((), device=device)

    it = iter(surface_data_loader(TOK, args.B, args.T, "val", device))
    with torch.no_grad():
        for s in range(args.val_steps):
            x, y = next(it)
            sc = get_surface()
            space_y = sc["space_y"] if sc is not None else torch.zeros_like(y)
            clogits, _, _ = model(x, y, return_surface_logits=True)
            cbits = F.cross_entropy(clogits.reshape(-1, clogits.size(-1)), y.reshape(-1),
                                    ignore_index=-1, reduction="none") / LN2
            yf, spf = y.reshape(-1), space_y.reshape(-1)
            ys = torch.where(yf >= 0, yf, torch.zeros_like(yf))
            bb = token_bytes[ys].to(torch.float32)
            counted = (yf >= 0) & (bb > 0)                               # real text tokens only (specials=0 bytes)
            rawb = (bb + spf.to(torch.float32))
            ph = is_phrase[ys].long()
            lb = bucketize(byte_len[ys], LEN_EDGES)
            fb = bucketize(freq_t[ys], FREQ_EDGES)
            rk = bucketize(ys, RANK_EDGES)
            ub = uni_bits[ys]
            for pcat in (0, 1):
                m = counted & (ph == pcat)
                if not m.any():
                    continue
                lbm, fbm, rkm = lb[m], fb[m], rk[m]
                cbm, rwm, ubm, bbm = cbits[m], rawb[m], ub[m], bb[m]
                ones = torch.ones_like(cbm)
                len_bits[pcat].index_add_(0, lbm, cbm); len_byte[pcat].index_add_(0, lbm, rwm)
                len_cnt[pcat].index_add_(0, lbm, ones); len_lensum[pcat].index_add_(0, lbm, bbm)
                frq_bits[pcat].index_add_(0, fbm, cbm); frq_byte[pcat].index_add_(0, fbm, rwm)
                frq_cnt[pcat].index_add_(0, fbm, ones); frq_uni[pcat].index_add_(0, fbm, ubm)
                rnk_bits[pcat].index_add_(0, rkm, cbm); rnk_byte[pcat].index_add_(0, rkm, rwm)
                rnk_cnt[pcat].index_add_(0, rkm, ones); rnk_lensum[pcat].index_add_(0, rkm, bbm)
                rnk_uni[pcat].index_add_(0, rkm, ubm)
            mc = counted.to(torch.float32)
            tot_bits += (cbits * mc).sum(); tot_byte += (rawb * mc).sum()
            if (s + 1) % 20 == 0:
                print(f"  val step {s+1}/{args.val_steps} ...")

    tb = tot_byte.item()
    print(f"\n=== overall single+phrase content-only bpb (sanity): {tot_bits.item()/tot_byte.item():.4f}  "
          f"({int(len_cnt.sum().item()):,} positions, {int(tb):,} bytes) ===")

    print("\n############ CUT (a): content cost vs TOKEN BYTE-LENGTH ############")
    print("(headline: for a well-fit compressor bits/BYTE should FALL as length rises — amortization)")
    for pcat, name in ((0, "SINGLE-WORD atoms (surface densification)"), (1, "PHRASE tokens (fm glue)")):
        print(f"\n-- {name} --")
        print(f"{'len':>6} {'positions':>10} {'bits/tok':>9} {'bits/byte':>9} {'byteshare':>9} {'meanlen':>7}")
        for i, lab in enumerate(LEN_LABELS):
            n = len_cnt[pcat, i].item()
            if n == 0:
                print(f"{lab:>6} {'—':>10}"); continue
            bpt = len_bits[pcat, i].item() / n
            bpb = len_bits[pcat, i].item() / len_byte[pcat, i].item()
            share = 100 * len_byte[pcat, i].item() / tb
            ml = len_lensum[pcat, i].item() / n
            print(f"{lab:>6} {int(n):>10,} {bpt:>9.3f} {bpb:>9.4f} {share:>8.2f}% {ml:>7.1f}")

    print("\n############ CUT (b): content cost vs TRAIN-FREQUENCY (exposure proxy) ############")
    print("(context_gain = unigram_bits - model_bits; collapse toward 0/neg for rare = underfit)")
    for pcat, name in ((0, "SINGLE-WORD atoms"), (1, "PHRASE tokens")):
        print(f"\n-- {name} --")
        print(f"{'freq':>10} {'positions':>10} {'bits/tok':>9} {'uni_bits':>9} {'ctx_gain':>9} {'byteshare':>9}")
        for i, lab in enumerate(FREQ_LABELS):
            n = frq_cnt[pcat, i].item()
            if n == 0:
                print(f"{lab:>10} {'—':>10}"); continue
            bpt = frq_bits[pcat, i].item() / n
            ubt = frq_uni[pcat, i].item() / n
            share = 100 * frq_byte[pcat, i].item() / tb
            print(f"{lab:>10} {int(n):>10,} {bpt:>9.3f} {ubt:>9.3f} {ubt - bpt:>9.3f} {share:>8.2f}%")

    print("\n############ CUT (c): content cost vs MERGE-RANK (sample-independent rarity) ############")
    print("(later merge = rarer. UNDERFIT vs INHERENT-ENTROPY control: frac = ctx_gain/uni_bits =")
    print(" fraction of base-rate uncertainty the model removes. FLAT frac across rarity => the")
    print(" bits/byte penalty is inherent rare-content entropy, NOT undertrained embeddings.)")
    for pcat, name in ((0, "SINGLE-WORD atoms"), (1, "PHRASE tokens")):
        print(f"\n-- {name} --")
        print(f"{'rank':>9} {'positions':>10} {'bits/tok':>9} {'bits/byte':>9} {'meanlen':>7} "
              f"{'uni_bits':>9} {'frac_rm':>8} {'byteshare':>9}")
        for i, lab in enumerate(RANK_LABELS):
            n = rnk_cnt[pcat, i].item()
            if n == 0:
                print(f"{lab:>9} {'—':>10}"); continue
            bpt = rnk_bits[pcat, i].item() / n
            bpb = rnk_bits[pcat, i].item() / rnk_byte[pcat, i].item()
            share = 100 * rnk_byte[pcat, i].item() / tb
            ml = rnk_lensum[pcat, i].item() / n
            ubt = rnk_uni[pcat, i].item() / n
            frac = (ubt - bpt) / ubt if ubt > 0 else float('nan')
            print(f"{lab:>9} {int(n):>10,} {bpt:>9.3f} {bpb:>9.4f} {ml:>7.1f} "
                  f"{ubt:>9.3f} {frac:>8.3f} {share:>8.2f}%")

    if args.train_steps > 0:
        print("\n############ CUT (d): HELD-IN train vs HELD-OUT val bits/byte by rank (single-word) ############")
        print("(ceiling/convergence probe: gap WIDENING toward the rare tail = rare-specific generalization")
        print(" gap => exposure/regularization live; FLAT gap => tail penalty is not rare-specific overfit)")
        tr_b, tr_y, tr_c = rank_pass(model, TOK, token_bytes, is_phrase, "train",
                                     args.train_steps, args.B, args.T, device)
        print(f"{'rank':>9} {'val b/byte':>10} {'train b/byte':>12} {'val-train gap':>13} {'train pos':>10}")
        for i, lab in enumerate(RANK_LABELS):
            vn = rnk_cnt[0, i].item(); tn = tr_c[i].item()
            if vn == 0 or tn == 0:
                print(f"{lab:>9} {'—':>10}"); continue
            vbpb = rnk_bits[0, i].item() / rnk_byte[0, i].item()
            tbpb = tr_b[i].item() / tr_y[i].item()
            print(f"{lab:>9} {vbpb:>10.4f} {tbpb:>12.4f} {vbpb - tbpb:>13.4f} {int(tn):>10,}")

    print("\nNOTE: train freq is from the LOCAL shards (a subset of the 170 the model trained on),")
    print("so the frequency cut is a noisy exposure proxy; the byte-length cut is sample-independent.")
    print("Cross-sectional read on the TRIPLE (single-word ≈ surface atoms). Local shards ARE held-in")
    print("(model trained on shards 0-7); val = the pinned held-out last shard.")


if __name__ == "__main__":
    main()
