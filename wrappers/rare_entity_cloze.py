"""Rare-entity cloze eval (Test 3) — does the model assign higher likelihood to RARE entities
given their natural context? The "generating rare entities" question, teacher-forced (true
generation needs the unbuilt surface generate path).

Pass 1 = EXTRACTION (offline, no model): pull rare proper-noun-like titlecase spans (1-2 words;
mid-sentence only, >= min-ctx chars of preceding context) whose rarest word fires <= max-freq in
ClimbMix train (= the s7 models' training corpus, so "rare" means rarely-seen-in-training).
Tagged kind = entity (2-word) | term (1-word). `--source` picks where spans come from:
  climbmix  — local ClimbMix val shard
  fineweb   — karpathy/fineweb-edu-100b-shuffle LAST shard (the OOD corpus; = a future FineWeb
              run's val set). rarity stays vs ClimbMix either way.
`--review` prints the set for inspection before we trust it. KNOWN noise to clean in the curated
pass: glossary/word-list docs spawn fake-entity clusters; a few truncations (e.g. 'Schwarzen').

Pass 2 = SCORING (needs checkpoints): per-RENDERED-byte NLL of the entity span given its
context, surface vs baseline (cross-tokenizer-fair). Δ<0 = surface models rare entities better.

  uv run python -m wrappers.rare_entity_cloze --review --source climbmix --max-freq 20 --n 80
  uv run python -m wrappers.rare_entity_cloze --review --source fineweb  --max-freq 20 --n 80
  uv run python -m wrappers.rare_entity_cloze --surf-ckpt ... --base-ckpt ... \  # on the pod
      --surf-tok ... --base-tok ... --source climbmix --max-freq 20 --n 300
"""
import argparse
import glob
import math
import os
import random
from collections import Counter

import regex

LN2 = math.log(2)
_WORD = regex.compile(r"\p{L}+")
# a titlecase word, optionally ONE more (1-2 words: real entities are short; 3-4-word
# titlecase runs are usually headlines/table-headers, not entities)
_SPAN = regex.compile(r"(?<![\p{L}'])\p{Lu}\p{Ll}+(?: \p{Lu}\p{Ll}+){0,1}")


def word_freq(chars, cap=10_000):
    from nanochat.dataset import parquets_iter_batched
    f = Counter(); seen = 0
    for batch in parquets_iter_batched(split="train"):
        for doc in batch:
            f.update(w.lower() for w in _WORD.findall(doc[:cap]))
            seen += min(len(doc), cap)
        if seen >= chars:
            break
    return f


def extract_rare_entities(docs, wf, max_freq, n, ctx_chars=240, min_ctx=80):
    """-> list of dicts {entity, context, sentence_start, min_word_freq, n_words, kind}.
    Filters: rarest word freq <= max_freq, NON-sentence-start (true proper nouns, not trivial
    sentence-start casing), and >= min_ctx non-whitespace chars of preceding context (so the
    cloze has something to condition on). kind = 'entity' (multi-word) | 'term' (single word)."""
    out = []
    for doc in docs:
        for m in _SPAN.finditer(doc):
            span = m.group()
            words = [w.lower() for w in _WORD.findall(span)]
            if not words:
                continue
            mn = min(wf.get(w, 0) for w in words)
            if mn > max_freq:                          # not rare enough
                continue
            start = m.start()
            pre = doc[max(0, start - 2):start]
            ss = pre.endswith((". ", ".\n", "\n", "? ", "! ")) or start == 0
            if ss:                                     # drop sentence-start (trivial casing)
                continue
            ctx = doc[max(0, start - ctx_chars):start]
            if len(regex.sub(r"\s", "", ctx)) < min_ctx:   # FILTER 1: enough context
                continue
            out.append({"entity": span, "context": ctx, "sentence_start": ss, "min_word_freq": mn,
                        "n_words": len(words), "kind": "entity" if len(words) >= 2 else "term"})
            if len(out) >= n * 6:
                break
        if len(out) >= n * 6:
            break
    out.sort(key=lambda d: (d["min_word_freq"], -d["n_words"]))   # rarest first (2-word as tiebreak)
    return out[:n]


# lowercase 1-2-word span (≥4 chars/word to skip stopwords) — the CASE control: surface's cap-bit
# discount does NOT apply to lowercase, so a rarity/pooling effect persists here but a cap-bit one vanishes
_SPAN_LOWER = regex.compile(r"(?<![\p{L}'])\p{Ll}{4,}(?: \p{Ll}{4,})?")


def extract_band(docs, wf, span_re, lo, hi, n, ctx_chars=240, min_ctx=80, drop_ss=True):
    """Spans whose rarest word freq ∈ [lo, hi]. drop_ss drops sentence-start (titlecase only).
    Returns n: rarest-first when lo==0 (rare band), most-common-first otherwise (common band)."""
    out = []
    for doc in docs:
        for m in span_re.finditer(doc):
            words = [w.lower() for w in _WORD.findall(m.group())]
            if not words:
                continue
            mn = min(wf.get(w, 0) for w in words)
            if not (lo <= mn <= hi):
                continue
            start = m.start()
            pre = doc[max(0, start - 2):start]
            ss = pre.endswith((". ", ".\n", "\n", "? ", "! ")) or start == 0
            if drop_ss and ss:
                continue
            ctx = doc[max(0, start - ctx_chars):start]
            if len(regex.sub(r"\s", "", ctx)) < min_ctx:
                continue
            out.append({"entity": m.group(), "context": ctx, "min_word_freq": mn,
                        "n_words": len(words), "kind": "entity" if len(words) >= 2 else "term"})
            if len(out) >= n * 10:
                break
        if len(out) >= n * 10:
            break
    out.sort(key=lambda d: d["min_word_freq"])         # ascending freq
    return out[:n] if lo == 0 else out[::-1][:n]       # rare band: rarest; common band: most common


def boot_ci(surf_bits, base_bits, byts, B=2000, seed=0):
    """Bootstrap 95% CI on Δbpb = Σsurf/Σbytes − Σbase/Σbytes, resampling spans with replacement."""
    rng = random.Random(seed); n = len(byts); out = []
    for _ in range(B):
        s = [rng.randrange(n) for _ in range(n)]
        sb = sum(surf_bits[j] for j in s); bb = sum(base_bits[j] for j in s); by = sum(byts[j] for j in s)
        out.append(sb / by - bb / by)
    out.sort()
    return out[int(0.025 * B)], out[int(0.975 * B)]


def _load_val(val_docs, doc_cap):
    from nanochat.dataset import parquets_iter_batched
    docs = []
    for batch in parquets_iter_batched(split="val"):
        for d in batch:
            docs.append(d[:doc_cap])
            if len(docs) >= val_docs:
                break
        if len(docs) >= val_docs:
            break
    return docs


def _load_fineweb(val_docs, doc_cap, repo="karpathy/fineweb-edu-100b-shuffle"):
    """karpathy's PREPROCESSED FineWeb-Edu (nanochat shard format, what a real FineWeb run uses).
    Reads the LAST shard = the future val set; streams row-groups so we don't pull the whole file."""
    import fsspec
    import pyarrow.parquet as pq
    fs = fsspec.filesystem("hf")
    files = sorted(p for p in fs.ls(f"datasets/{repo}", detail=False) if p.endswith(".parquet"))
    pf = pq.ParquetFile(fs.open(files[-1]))
    col = "text" if "text" in pf.schema_arrow.names else pf.schema_arrow.names[0]
    docs = []
    for rg in range(pf.num_row_groups):
        for d in pf.read_row_group(rg).column(col).to_pylist():
            docs.append(d[:doc_cap])
            if len(docs) >= val_docs:
                return docs
    return docs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--review", action="store_true", help="extraction only (offline, no model)")
    ap.add_argument("--surf-ckpt"); ap.add_argument("--base-ckpt")
    ap.add_argument("--surf-tok"); ap.add_argument("--base-tok")
    ap.add_argument("--val-docs", type=int, default=600); ap.add_argument("--doc-cap", type=int, default=6000)
    ap.add_argument("--freq-chars", type=int, default=40_000_000)
    ap.add_argument("--max-freq", type=int, default=30, help="keep entities whose rarest word fires <= this")
    ap.add_argument("--min-ctx", type=int, default=80, help="min non-whitespace chars of preceding context")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--source", default="climbmix",
                    help="entity source: climbmix (local val) | fineweb (karpathy last shard). "
                         "rarity is always measured vs ClimbMix train = the s7 models' training corpus")
    ap.add_argument("--device", default="cuda"); ap.add_argument("--kmax", type=int, default=1)
    ap.add_argument("--control", action="store_true",
                    help="2x2 rarity×case control (rare/common × titlecase/lowercase) + bootstrap CI — "
                         "separates surface's cap-bit discount from a rarity/pooling effect")
    ap.add_argument("--common-min-freq", type=int, default=1000, help="freq floor for the COMMON band")
    args = ap.parse_args()
    from overlay.dataset_fineweb import maybe_repoint
    maybe_repoint()                              # NANOCHAT_DATASET=fineweb → rarity counted over FineWeb train

    _corpus = os.environ.get("NANOCHAT_DATASET", "climbmix")
    print(f"counting word frequencies over ~{args.freq_chars/1e6:.0f}M chars ({_corpus} train) ...")
    wf = word_freq(args.freq_chars)
    docs = (_load_fineweb if args.source == "fineweb" else _load_val)(args.val_docs, args.doc_cap)
    ents = extract_rare_entities(docs, wf, args.max_freq, args.n, min_ctx=args.min_ctx)
    n_ent = sum(1 for e in ents if e["kind"] == "entity"); n_term = len(ents) - n_ent
    print(f"[{args.source}] extracted {len(ents)} rare items (rarest word freq <= {args.max_freq}, "
          f"min-ctx {args.min_ctx}, mid-sentence only): {n_ent} multi-word entities + {n_term} single-word terms")

    if args.review or not args.surf_ckpt:
        for kind, desc in (("entity", "multi-word entities"), ("term", "single-word rare titlecase")):
            grp = [e for e in ents if e["kind"] == kind][:20]
            print(f"\n=== {kind.upper()} — {desc} ({len(grp)} shown) ===")
            for e in grp:
                print(f"[f={e['min_word_freq']:>3}] {e['entity']!r:<30} ⟸ …{e['context'][-56:]!r}")
        if args.review:
            return

    # ---- Pass 2: cloze NLL (needs checkpoints) ----
    import torch
    from nanochat.tokenizer import RustBPETokenizer
    from overlay.surface_tokenizer import SurfaceTokenizer
    from overlay.surface_core_eval import set_surface_tokenizer, _encode_prompt
    from overlay.surface_checkpoint import surface_build_model
    import nanochat.checkpoint_manager as cm
    from tools.stack_fm_spaceless import SPACELESS_BODY
    device = torch.device(args.device if torch.cuda.is_available() or args.device != "cuda" else "cpu")
    os.environ["SURFACE_KMAX"] = str(args.kmax)
    surf = SurfaceTokenizer.from_directory(args.surf_tok, SPACELESS_BODY, kmax=args.kmax)
    set_surface_tokenizer(surf)
    base = RustBPETokenizer.from_directory(args.base_tok)
    surf_tb = torch.load(os.path.join(args.surf_tok, "token_bytes.pt")).long()
    base_tb = torch.load(os.path.join(args.base_tok, "token_bytes.pt")).long()
    _step = lambda c: sorted(int(os.path.basename(p).split("_")[1].split(".")[0])
                             for p in glob.glob(os.path.join(c, "model_*.pt")))[-1]

    @torch.no_grad()
    def surf_cloze(model, ctx, ent):
        # bytes of the entity span (rendered): chars of `ent`
        n_ctx = len(_encode_prompt(ctx)[0])
        b, s, cb, cm_ = (torch.tensor([x], device=device) for x in _encode_prompt(ctx + ent))
        joint = model(b[:, :-1], b[:, 1:], loss_reduction='none', space_x=s[:, :-1], cap_bits_x=cb[:, :-1],
                      cap_mask_x=cm_[:, :-1], space_y=s[:, 1:], cap_bits_y=cb[:, 1:], cap_mask_y=cm_[:, 1:]).reshape(-1)
        sl = slice(n_ctx - 1, None)                    # the entity's predicted tokens
        return joint[sl].sum().item() / LN2, len(ent.encode())

    @torch.no_grad()
    def base_cloze(model, ctx, ent):
        cids = [base.get_bos_token_id()] + base.enc.encode_ordinary(ctx)
        full = cids + base.enc.encode_ordinary(ent) if not ctx.endswith(" ") else \
            [base.get_bos_token_id()] + base.enc.encode_ordinary(ctx + ent)
        t = torch.tensor([full], device=device)
        ce = model(t[:, :-1], t[:, 1:], loss_reduction='none').reshape(-1)
        return ce[len(cids) - 1:].sum().item() / LN2, len(ent.encode())

    if args.control:
        cells = [("rare-titlecase", _SPAN, 0, args.max_freq, True),
                 ("common-titlecase", _SPAN, args.common_min_freq, 10 ** 9, True),
                 ("rare-lowercase", _SPAN_LOWER, 0, args.max_freq, False),
                 ("common-lowercase", _SPAN_LOWER, args.common_min_freq, 10 ** 9, False)]
        cell_ents = []
        for name, sre, lo, hi, dss in cells:
            ce = extract_band(docs, wf, sre, lo, hi, args.n, min_ctx=args.min_ctx, drop_ss=dss)
            cell_ents.append((name, ce))
            fr = sorted(e["min_word_freq"] for e in ce)
            print(f"[{name:>16}] {len(ce):>4} spans  freq {fr[0] if fr else 0}..{fr[-1] if fr else 0} "
                  f"median {fr[len(fr) // 2] if fr else 0}  ({sum(1 for e in ce if e['kind'] == 'entity')} two-word)")
        s = surface_build_model(args.surf_ckpt, _step(args.surf_ckpt), device, "eval")[0]; s.eval()
        surf_sc = {nm: [surf_cloze(s, e["context"], e["entity"]) for e in ce] for nm, ce in cell_ents}
        del s
        if device.type == "cuda":
            torch.cuda.empty_cache()
        bm = cm.build_model(args.base_ckpt, _step(args.base_ckpt), device, "eval")[0]; bm.eval()
        base_sc = {nm: [base_cloze(bm, e["context"], e["entity"]) for e in ce] for nm, ce in cell_ents}
        print(f"\n{'cell':>16} {'surf bpb':>9} {'base bpb':>9} {'Δ(s-b)':>9} {'95% CI (Δ)':>22}")
        for name, _ in cell_ents:
            sb = [x[0] for x in surf_sc[name]]; by = [x[1] for x in surf_sc[name]]; bb = [x[0] for x in base_sc[name]]
            d = sum(sb) / sum(by) - sum(bb) / sum(by)
            lo_ci, hi_ci = boot_ci(sb, bb, by)
            print(f"{name:>16} {sum(sb) / sum(by):>9.4f} {sum(bb) / sum(by):>9.4f} {d:>+9.4f}   [{lo_ci:+.4f}, {hi_ci:+.4f}]")
        print("\ncap-bit hypothesis ⇒ Δ<0 on BOTH titlecase cells, ~0 (CI spans 0) on lowercase.")
        print("rarity/pooling hypothesis ⇒ Δ<0 on BOTH rare cells, ~0 on common.")
        return

    s = surface_build_model(args.surf_ckpt, _step(args.surf_ckpt), device, "eval")[0]; s.eval()
    sb = sbb = 0.0
    for e in ents:
        bits, byt = surf_cloze(s, e["context"], e["entity"]); sb += bits; sbb += byt
    del s
    if device.type == "cuda":
        torch.cuda.empty_cache()
    bm = cm.build_model(args.base_ckpt, _step(args.base_ckpt), device, "eval")[0]; bm.eval()
    bb = bbb = 0.0
    for e in ents:
        bits, byt = base_cloze(bm, e["context"], e["entity"]); bb += bits; bbb += byt
    print(f"\nrare-entity cloze bpb — surface {sb/sbb:.4f}  baseline {bb/bbb:.4f}  Δ(s-b) {sb/sbb-bb/bbb:+.4f}")
    print("Δ<0 = surface assigns the rare entity lower bits/byte given context (models it better).")


if __name__ == "__main__":
    main()
