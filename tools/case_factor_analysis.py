"""
Offline potential analysis for factoring CAPITALIZATION out of token identity
(the generalization of the space-as-a-bit idea to a second "token feature").

Mirrors the space analysis: how much vocab does capitalization duplicate, and
how predictable is the case bit? "Titlecase" = first letter upper, rest lower
(`\\p{Lu}\\p{Ll}*`) — this cleanly INCLUDES sentence-start + simple proper nouns
and EXCLUDES acronyms (NASA), camelCase, iPhone, McDonald (none are titlecase),
so the fold is unambiguous and lossless.

Reports:
  A. Vocab duplication
     - titlecase tokens whose lowercase twin is also a token (case duplicates)
     - space x case 4-families {x, ' '+x, Cap(x), ' '+Cap(x)} — joint potential
  B. Corpus predictability of the case bit (held-out val)
     - marginal P(capitalized), conditional H(cap | prev token)
     - sentence-start vs mid-sentence (proper-noun) split + per-regime entropy

Usage: uv run python -m tools.case_factor_analysis [--val-chars N]
"""
import argparse
from collections import Counter
from math import log2

import regex

from tools.eval_tokenizer import load_tokenizer, _safe_token_bytes
from tools._corpus_iter import iter_capped_batches

_TITLE = regex.compile(r"\p{Lu}\p{Ll}*")          # whole-core titlecase
_SENT_END = regex.compile(r"[.!?:\n]\s*$")         # prev token ends a sentence


def core_of(s):
    return s[1:] if s.startswith(" ") else s

def is_titlecase(s):
    c = core_of(s)
    return len(c) >= 1 and _TITLE.fullmatch(c) is not None

def lower_twin_bytes(b):
    """Bytes of the lowercased-first-letter twin of a (space?+)titlecase token."""
    s = b.decode("utf-8", "replace")
    lead = " " if s.startswith(" ") else ""
    c = core_of(s)
    return (lead + c[0].lower() + c[1:]).encode("utf-8")

def binent(ps):
    return -sum(p * log2(p) for p in ps if p > 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-chars", type=int, default=20_000_000)
    args = ap.parse_args()

    tok = load_tokenizer("ours")
    V = tok.get_vocab_size()
    specials = set(tok.get_special_tokens())

    # id -> bytes / decoded / flags
    id_bytes, id_str, cap_flag = {}, {}, {}
    byteset = set()
    for tid in range(V):
        b = _safe_token_bytes(tok, tid)
        if b is None:
            continue
        s = b.decode("utf-8", "replace")
        if s in specials:
            continue
        id_bytes[tid] = b
        id_str[tid] = s
        cap_flag[tid] = is_titlecase(s)
        byteset.add(b)

    # ---- A. vocab duplication ----
    title_ids = [t for t in id_bytes if cap_flag[t]]
    case_dups = sum(1 for t in title_ids if lower_twin_bytes(id_bytes[t]) in byteset)

    # space x case families keyed by bare-lowercase core
    fam = {}  # core_lower -> set of variants present
    for t, b in id_bytes.items():
        s = id_str[t]
        lead = s.startswith(" ")
        c = core_of(s)
        if not c or not c[0:1].isalpha():
            continue
        base = c[0].lower() + c[1:]
        cap = c[0].isupper() and _TITLE.fullmatch(c) is not None
        if c[0].isupper() and not cap:
            continue  # not clean titlecase (camelCase/acronym) — skip
        key = base.lower()
        fam.setdefault(key, set()).add((lead, cap))
    multi_fam = {k: v for k, v in fam.items() if len(v) >= 2}
    # reclaimable slots = sum(variants-1) over families (collapse to 1 base + bits)
    reclaim_case = sum(1 for t in title_ids if lower_twin_bytes(id_bytes[t]) in byteset)
    reclaim_joint = sum(len(v) - 1 for v in fam.values())

    print(f"=== A. Vocab duplication (vocab={V}, non-special={len(id_bytes)}) ===")
    print(f"titlecase tokens (Cap word, incl. ' ' prefix): {len(title_ids)} "
          f"({100*len(title_ids)/len(id_bytes):.1f}%)")
    print(f"  with a lowercase twin in vocab (case-duplicate): {case_dups} "
          f"({100*case_dups/len(id_bytes):.1f}% of vocab reclaimable by case alone)")
    print(f"space x case families (>=2 variants of one base word): {len(multi_fam)}")
    print(f"  reclaimable slots if collapse ALL space+case variants -> base+2 bits: "
          f"{reclaim_joint} ({100*reclaim_joint/len(id_bytes):.1f}% of vocab)")
    # show a few 4-variant families
    four = [k for k, v in fam.items() if len(v) == 4][:8]
    print(f"  sample full 4-way families {{x, ' '+x, Cap, ' '+Cap}}: {four}")

    # ---- B. corpus predictability ----
    corpus = list(iter_capped_batches("val", args.val_chars))
    pair = Counter()            # (prev_id, cap) over content tokens
    sent_pair = Counter()       # same, sentence-start regime
    mid_pair = Counter()
    total_bytes = 0
    n_tokens = 0
    n_cap = 0
    for docs in corpus:
        for d in docs:
            total_bytes += len(d.encode("utf-8"))
        ids_batch = tok.enc.encode_ordinary_batch(docs, num_threads=8)
        for ids in ids_batch:
            prev = -1
            for tid in ids:
                n_tokens += 1
                cap = 1 if cap_flag.get(tid, False) else 0
                n_cap += cap
                pair[(prev, cap)] += 1
                prev_s = id_str.get(prev, "")
                if _SENT_END.search(prev_s) or prev == -1:
                    sent_pair[(prev, cap)] += 1
                else:
                    mid_pair[(prev, cap)] += 1
                prev = tid

    def cond_bits(counter):
        by_prev = {}
        for (p, c), n in counter.items():
            d = by_prev.setdefault(p, [0, 0]); d[c] += n
        bits = 0.0; tot = 0
        for p, (n0, n1) in by_prev.items():
            n = n0 + n1; tot += n
            bits += n * binent([n0/n, n1/n])
        return bits, tot

    marg = n_cap / n_tokens
    cond_bits_all, _ = cond_bits(pair)
    sb, sn = cond_bits(sent_pair)
    mb, mn = cond_bits(mid_pair)

    print(f"\n=== B. Case-bit predictability (held-out val, {total_bytes:,} bytes, {n_tokens:,} tokens) ===")
    print(f"marginal P(token capitalized): {marg:.4f}  (marginal H={binent([1-marg,marg]):.3f} bit/tok)")
    print(f"conditional on prev token: {cond_bits_all/n_tokens:.4f} bit/tok  "
          f"({cond_bits_all/total_bytes:.4f} bits/byte)")
    print(f"  capitalized tokens: {n_cap:,} ({100*n_cap/n_tokens:.1f}% of tokens)")
    print(f"  of capitalized, sentence-start regime: {100*sum(n for (p,c),n in sent_pair.items() if c)/n_cap:.1f}%  "
          f"mid-sentence (proper-noun-ish): {100*sum(n for (p,c),n in mid_pair.items() if c)/n_cap:.1f}%")
    print(f"  H(cap|prev) sentence-start regime: {sb/sn:.4f} bit/tok   "
          f"mid-sentence regime: {mb/mn:.4f} bit/tok")


if __name__ == "__main__":
    main()
