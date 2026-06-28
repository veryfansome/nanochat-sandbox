"""Tokenizer-grounded pooling-gain analysis for surface factoring (offline, no model).

Grounds "rare/affected" in the ACTUAL baseline->surface manipulation, not generic rarity
(the gap exposure_slope.py had: it bucketed the surface model's own vocab by merge-rank /
frequency, never by the tokenizer difference or the firing-mass redistribution).

(1) DIFF the vocabs over a corpus:
    POOLED families - baseline tokens that FOLD to the same content form (strip ONE leading
                      space + titlecase-fold the first letter, via case_factor_compression).
                      Surface collapses each family to one base content token; a family with
                      >=2 *fired* variants is where pooling redistributes firing mass.
    NEW atoms       - surface content tokens whose content is absent from baseline's folded
                      vocab (a denser merge baseline lacks in any space/case variant).

(2) Per pooled family:  POOLING GAIN = total_variant_firings / busiest_variant_firing
    (= the extra exposure pooling actually buys the base token) + BYTE-SHARE (fraction of
    corpus text bytes the family covers). The rare-word-benefit hypothesis lives in the
    HIGH-gain tail; if that tail carries little byte-share, there is little text mass for the
    benefit to act on (regardless of any model).

The core (baseline_content_stats / pooling_bucket / content_form / EDGES) is imported by
wrappers/rarity_delta.py, which buckets model NLL by the SAME pooling-gain grounding.

Usage:  uv run python -m tools.pooling_gain --chars 20000000 [--base-tok DIR] [--surf-tok DIR]
"""
import argparse
import os
import statistics
from collections import Counter, defaultdict

from nanochat.dataset import parquets_iter_batched
from nanochat.tokenizer import RustBPETokenizer
from overlay.surface_tokenizer import SurfaceTokenizer
from tools.case_factor_compression import fold_case
from tools.stack_fm_spaceless import SPACELESS_BODY

BASE_DIR = os.path.expanduser("~/.cache/nanochat/tokenizer")
SURF_DIR = os.path.expanduser("~/.cache/nanochat-variants/surface_only/tokenizer")

# pooling-gain buckets (gain = total variant firings / busiest variant). NEW atoms are a
# separate category the caller adds; these edges cover multi/single-variant families only.
EDGES = [1.0000001, 1.05, 1.2, 1.5, 2.0, 3.0]
LAB = ["=1 (1 variant)", "1.00-1.05", "1.05-1.20", "1.20-1.50", "1.50-2.00", "2.00-3.00", "3.00+"]


def content_form(b):
    """Fold a token's raw bytes to its surface content form (strip one leading space +
    titlecase-fold), or None if the bytes are not valid UTF-8 (a partial multi-byte token)."""
    try:
        s = b.decode("utf-8")
    except UnicodeDecodeError:
        return None
    core = s[1:] if s.startswith(" ") else s      # factor a single leading space (the bit)
    return fold_case(core)                          # titlecase-fold only (leaves caps/camelCase)


def pooling_bucket(gain):
    return sum(1 for e in EDGES if gain >= e)


def baseline_content_stats(base_enc, chars, doc_cap=10_000):
    """Encode ~`chars` of train text with the baseline tokenizer; return the raw materials for
    grounding pooling gain (used by main() AND wrappers/rarity_delta.py):
      bfir       : Counter, baseline token id -> firing count over the sample
      id2content : list, baseline token id -> folded content form (or "\\x00RAW{i}")
      id2blen    : list, baseline token id -> raw byte length
    """
    V = base_enc.n_vocab
    id2content, id2blen = [], []
    for i in range(V):
        b = base_enc.decode_single_token_bytes(i)
        id2blen.append(len(b))
        c = content_form(b)
        id2content.append(c if c is not None else f"\x00RAW{i}")
    bfir = Counter()
    seen = 0
    for batch in parquets_iter_batched(split="train"):
        for doc in batch:
            d = doc[:doc_cap]
            bfir.update(base_enc.encode_ordinary(d))
            seen += len(d)
        if seen >= chars:
            break
    return bfir, id2content, id2blen


def content_pooling(bfir, id2content):
    """Group baseline firings by content form -> content -> [total, max, n_variants]."""
    stats = defaultdict(lambda: [0, 0, 0])
    for tid, c in bfir.items():
        g = stats[id2content[tid]]
        g[0] += c; g[1] = max(g[1], c); g[2] += 1
    return dict(stats)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chars", type=int, default=20_000_000)
    ap.add_argument("--base-tok", default=BASE_DIR)
    ap.add_argument("--surf-tok", default=SURF_DIR)
    ap.add_argument("--pattern", default=SPACELESS_BODY)
    ap.add_argument("--kmax", type=int, default=1)
    args = ap.parse_args()

    base = RustBPETokenizer.from_directory(args.base_tok).enc
    surf = SurfaceTokenizer.from_directory(args.surf_tok, args.pattern, kmax=args.kmax)
    surf_enc = surf.enc
    print(f"baseline vocab {base.n_vocab:,}  surface vocab {surf_enc.n_vocab:,}")

    print(f"counting baseline firings over ~{args.chars/1e6:.0f}M chars ...")
    bfir, id2content, id2blen = baseline_content_stats(base, args.chars)
    stats = content_pooling(bfir, id2content)
    base_content_set = set(id2content)   # ALL baseline vocab content forms (sample-INDEPENDENT)
    # byte mass per content = Σ firing·(baseline raw byte len) over the family's variants
    cbytes = defaultdict(float)
    for tid, c in bfir.items():
        cbytes[id2content[tid]] += c * id2blen[tid]
    total_bytes = sum(cbytes.values())

    # surface new-atom byte-share needs a surface encode pass (content absent from baseline)
    print("surface-encoding the same sample for new-atom byte-share ...")
    surf_new = {}
    for i in range(surf_enc.n_vocab):
        try:
            s = surf_enc.decode_single_token_bytes(i).decode("utf-8")
            surf_new[i] = (bool(s) and s not in base_content_set, len(surf_enc.decode_single_token_bytes(i)))
        except UnicodeDecodeError:
            surf_new[i] = (False, len(surf_enc.decode_single_token_bytes(i)))
    snew = stot = seen = 0
    for batch in parquets_iter_batched(split="train"):
        for doc in batch:
            d = doc[:10_000]
            try:
                for tid, *_ in surf.encode(d):
                    isnew, bl = surf_new[tid]; stot += bl
                    if isnew:
                        snew += bl
            except Exception:
                pass
            seen += len(d)
        if seen >= args.chars:
            break

    nb = len(EDGES) + 1
    fam = [0] * nb; occ = [0] * nb; bm = [0.0] * nb
    multi = []
    for c, (tot, mx, nv) in stats.items():
        bi = pooling_bucket(tot / mx)
        fam[bi] += 1; occ[bi] += tot; bm[bi] += cbytes[c]
        if nv >= 2:
            multi.append((tot / mx, cbytes[c], mx, tot))

    print("\n############ (2) POOLING-GAIN distribution, weighted by corpus byte-share ############")
    print(f"{'bucket':>16} {'families':>9} {'occurrences':>13} {'byte-share':>11}")
    for i in range(nb):
        print(f"{LAB[i]:>16} {fam[i]:>9,} {occ[i]:>13,} {100*bm[i]/total_bytes:>10.2f}%")
    sh = lambda th: 100 * sum(m for g, m, *_ in multi if g >= th) / total_bytes
    print(f"\nbyte-share with pooling gain  >=1.20: {sh(1.2):.2f}%   >=1.50: {sh(1.5):.2f}%   >=2.00: {sh(2.0):.2f}%")
    if multi:
        g = [x[0] for x in multi]
        occ_conc = sum(x[2] for x in multi) / sum(x[3] for x in multi)
        med_conc = statistics.median([x[2] / x[3] for x in multi])
        print(f"multi-variant families: {len(multi):,}  (byte-share {100*sum(x[1] for x in multi)/total_bytes:.2f}%)")
        print(f"  median pooling gain {statistics.median(g):.3f}  |  busiest-variant concentration: "
              f"median {med_conc:.3f}, occurrence-weighted {occ_conc:.3f}")

    print("\n############ (1b) NEW atoms (surface content absent from baseline's folded vocab) ############")
    n_new = sum(1 for i in range(surf_enc.n_vocab) if surf_new[i][0])
    print(f"surface content tokens with no baseline single-token variant: {n_new:,} / {surf_enc.n_vocab:,}")
    print(f"corpus byte-share covered by surface NEW atoms: {100*snew/max(stot,1):.2f}%  (of {stot:,} surface-content bytes)")
    ex = sorted((surf_enc.decode_single_token_bytes(i) for i in range(surf_enc.n_vocab) if surf_new[i][0]),
                key=len, reverse=True)[:12]
    print("  longest:", ", ".join(repr(b.decode("utf-8", "replace")) for b in ex))


if __name__ == "__main__":
    main()
