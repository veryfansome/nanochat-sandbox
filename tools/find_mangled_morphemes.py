"""
Find mangled-morpheme tokens in a reference tokenizer and emit BLOCKED_PAIRS
to prevent their formation in a retrained tokenizer.

A *mangled* morpheme is a token of the form `<bound-fragment> + <clean-suffix>`,
e.g. `'sembled'` = `'sembl'` + `'ed'`. The left operand `'sembl'` is bound —
it never appears at word-initial position — so the token splits the underlying
word at a non-morphemic boundary. Encoding `'assembled'` ends up as
`['as', 'sembled']` instead of the morpheme-aligned `['assemble', 'd']`.

The hypothesis (see `rustbpe_variants/blocked_morphemes/README.md`) is that
breaking such mangled tokens into `<stem>+<suffix>` 2-token encodings gives
the model better-trained semantic atoms even at small compression cost.

## Algorithm

For each token T in the reference tokenizer's vocab:

1. **Suffix-strip**: Find the longest match S from `SUFFIXES` that T ends with.
   Skip if no match or T == S (T is itself a pure suffix).
2. **L lookup**: Let `L_str = T_str[:-len(S)]`. Skip if L isn't a token.
3. **Bound check**: T is mangled iff
   - `word_start_prob(L_id) < bound_threshold` (default 0.05 — strict), AND
   - `(' ' + L_str)` is NOT a token. The space-prefixed sibling test
     distinguishes true bound fragments (`'sembl'`, no space-sibling) from
     mid-word allomorphs of free stems (`'organ'`, space-sibling `' organ'`
     exists). The latter combination — free-stem-allomorph + suffix — is a
     clean morphological composition and we leave it alone.
4. **Formation pair**: For each mangled T, scan all `(L', R')` byte-splits
   where both operands are in the reference vocab; pick the one with minimal
   `max(L'_id, R'_id)` (BPE's most-likely formation path at training time).
   Emit `(L'_str, R'_str)` as a block.

## Word-start probability

For each token id, the fraction of its firings that occur at the **start of
a pre-tok chunk**. Computed by iterating the corpus through the tokenizer's
pre-tok regex and encoding each chunk individually:
  - Tokens with a leading space have word_start_prob ≈ 1.0 (chunks split on
    spaces, so space-prefixed tokens are usually chunk-first).
  - Pure suffixes (`'ing'`, `'ed'`) have word_start_prob ≈ 0.0 (always
    composition-suffixes, never chunk-first).
  - Single-letter tokens (`'c'`, `'a'`) typically have low word_start_prob
    because their leading-space siblings (`' c'`, `' a'`) win at word starts.

## Suffix inventory

A curated list of English derivational + inflectional suffixes (see
`SUFFIXES`). Sorted longest-first so greedy matching strips the most
informative suffix. The list is the lexical knowledge injected into the
miner — review and edit it when adapting to a non-English corpus.

## Output

`BLOCKED_PAIRS = [(L_str, R_str), ...]` directly compatible with
`rustbpe_force_merges.Tokenizer.train_from_iterator(... blocked_pairs=...)`.

Usage (regenerates the canonical):
    uv run python -m tools.find_mangled_morphemes \\
        --reference ~/.cache/nanochat/tokenizer \\
        --max-chars 20_000_000 --bound-threshold 0.05 \\
        --out-py rustbpe_variants/blocked_morphemes/mined/mangled_climbmix_baseline.py
"""

import argparse
import sys
import time
from collections import Counter
from pathlib import Path


# English derivational + inflectional suffix inventory. Sorted longest-first
# at use-time so greedy match strips the most informative suffix (e.g.
# 'organization' → strip 'ization', not 'ation').
#
# Selection principle: include suffixes that the model should learn as
# *compositional atoms* — i.e. ones that combine productively with many
# stems to derive a predictable meaning shift. Excludes:
#   - Pure phonological alternants we can't distinguish from random endings
#     (`'a'`, `'i'`, etc.).
#   - Single-letter inflections that would generate noise (`'s'`, `'d'` as
#     standalone — caught by their longer-form derivatives `'ed'`, `'ing'`,
#     `'tion'`, etc.).
#
# Add/remove suffixes here to adapt to a non-English corpus or to test the
# miner's sensitivity to the inventory.
SUFFIXES = [
    # Long agglutinated forms first
    "izations", "isations",
    "ization", "isation",
    "ifications", "ification",
    "abilities", "ability",
    "ibilities", "ibility",
    "ologically", "ological",
    "fullness", "lessness", "ousness",
    "fully", "lessly", "ously",
    "ically", "ical",
    "ationally", "ationality",
    # Common derivational
    "ations", "ation",
    "tions", "tion",
    "sions", "sion",
    "ments", "ment",
    "nesses", "ness",
    "ities", "ity",
    "izing", "izes", "ized", "ize",
    "ising", "ises", "ised", "ise",
    "ifying", "ifies", "ified", "ify",
    "ables", "able",
    "ibles", "ible",
    "fulnesses", "ful",
    "less", "ous",
    "ative", "atives",
    "itive", "itives",
    "ives", "ive",
    "ally", "al",
    "ier", "iest",
    "ists", "ist",
    "isms", "ism",
    "ings", "ing",
    "ors", "or",
    "ers", "er",
    "ies",
    "ily", "ly",
    "ed",
    "es",
    # Added after a vocab-QA pass found mangled tokens slipping through on
    # suffixes the inventory lacked (e.g. ' abulary', 'ependent', 'ibrary',
    # 'riage'). The T-is-word / stem-bound / no-space-sibling filters guard
    # against false positives on real words ending in these (went, fish,
    # message, library-the-word, etc.).
    "ences", "ence", "ances", "ance",
    "ents", "ent", "ants", "ant",
    "ages", "age",
    "aries", "ary", "ories", "ory", "eries", "ery",
    "ians", "ian", "ish",
    # Added after a second vocab-QA pass (post-P4) for completeness. `-ial` is
    # the clear genuine gap (material, industrial, bacterial, terrestrial);
    # `-ure`/`-ice`/`-ety`/`-acy`/`-hood` are marginal-but-genuine. Deliberately
    # EXCLUDES false positives the QA scan over-flagged: -th (mouth/south/
    # tooth/algorithm), -ling (filling=fill+ing), -ster (Manchester), -ee
    # (coffee), -let (violet) — adding those would mangle real words.
    "icials", "icial", "ially", "ials", "ial",
    "itures", "iture", "atures", "ature", "ures", "ure",
    "ices", "ice",
    "eties", "ety",
    "acies", "acy",
    "hoods", "hood",
]


def compute_word_start_probs(tok, max_chars: int) -> tuple[Counter, Counter, int]:
    """Walk the corpus chunk-by-chunk through the tokenizer's pre-tok regex,
    encode each chunk individually, and tally (a) first-token-of-chunk
    firings vs (b) all firings per token id.

    Returns (word_start_count, total_count, total_chars).
    """
    import regex

    from tools._corpus_iter import iter_capped_batches

    pat = regex.compile(
        tok.enc._pat_str if hasattr(tok.enc, "_pat_str") else tok.enc.pat_str
    )

    word_start_count: Counter = Counter()
    total_count: Counter = Counter()
    total_chars = 0
    n_docs = 0
    t0 = time.time()

    for docs in iter_capped_batches("val", max_chars):
        all_chunks: list[str] = []
        for doc in docs:
            all_chunks.extend(m.group(0) for m in pat.finditer(doc))
        # Single big batch call keeps tiktoken's per-chunk parallelism intact.
        chunk_ids_list = tok.enc.encode_ordinary_batch(all_chunks, num_threads=8)
        for chunk_ids in chunk_ids_list:
            if not chunk_ids:
                continue
            word_start_count[chunk_ids[0]] += 1
            for tid in chunk_ids:
                total_count[tid] += 1
        total_chars += sum(len(d) for d in docs)
        n_docs += len(docs)
        if n_docs % 5000 == 0:
            print(
                f"  {n_docs:,} docs / {total_chars:,} chars "
                f"({time.time() - t0:.1f}s)",
                file=sys.stderr,
            )
    return word_start_count, total_count, total_chars


def build_vocab_maps(tok) -> tuple[dict[bytes, int], dict[int, bytes]]:
    """Bytes ↔ id maps over the tokenizer's mergeable vocab (specials excluded)."""
    enc = tok.enc
    special_set = set(tok.get_special_tokens())
    b2i: dict[bytes, int] = {}
    i2b: dict[int, bytes] = {}
    for tid in range(tok.get_vocab_size()):
        try:
            b = enc.decode_single_token_bytes(tid)
        except Exception:
            continue
        try:
            if b.decode("utf-8") in special_set:
                continue
        except UnicodeDecodeError:
            pass
        b2i[b] = tid
        i2b[tid] = b
    return b2i, i2b


def longest_suffix_match(token_str: str, suffixes_sorted: list[str]) -> str | None:
    """Return the longest entry from `suffixes_sorted` that `token_str` ends
    with (and is strictly shorter than `token_str`). `suffixes_sorted` must
    already be in descending-length order. Returns None if no match."""
    for suf in suffixes_sorted:
        if len(token_str) > len(suf) and token_str.endswith(suf):
            return suf
    return None


def find_formation_split(
    token_bytes: bytes,
    b2i: dict[bytes, int],
    token_id: int,
    exclude: set[tuple[bytes, bytes]] | None = None,
) -> tuple[bytes, bytes, int, int] | None:
    """For a target token T (bytes, id), return the (L', R') byte-split where
    both operands are in vocab and `max(L'_id, R'_id)` is minimal — BPE's
    most-likely formation pair at training time. Returns None if no in-vocab
    split exists.

    `exclude` is a set of (L_bytes, R_bytes) pairs to skip — used in
    iterative single-path blocking: pass the previous pass's blocks so this
    pass emits the NEXT-best path a surviving mangled token reformed through,
    rather than re-emitting an already-blocked pair.
    """
    best = None
    for i in range(1, len(token_bytes)):
        L = token_bytes[:i]
        R = token_bytes[i:]
        if exclude and (L, R) in exclude:
            continue
        L_id = b2i.get(L)
        R_id = b2i.get(R)
        if L_id is None or R_id is None:
            continue
        if L_id >= token_id or R_id >= token_id:
            # Operands must form before T at training time.
            continue
        key = max(L_id, R_id)
        if best is None or key < best[4]:
            best = (L, R, L_id, R_id, key)
    if best is None:
        return None
    L, R, L_id, R_id, _ = best
    return L, R, L_id, R_id


def load_exclude_blocks(paths: list[Path]) -> set[tuple[bytes, bytes]]:
    """Load (L_bytes, R_bytes) pairs from one or more block-list .py files
    (each exposing a `BLOCKED_PAIRS = [(L_str, R_str), ...]`). Used to feed
    `--exclude-blocks-py` into `find_formation_split`."""
    import importlib.util

    exclude: set[tuple[bytes, bytes]] = set()
    for path in paths:
        spec = importlib.util.spec_from_file_location("_excl", str(path))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        for L, R in mod.BLOCKED_PAIRS:
            exclude.add((L.encode("utf-8"), R.encode("utf-8")))
    return exclude


def find_mangled_tokens(
    tok,
    word_start_prob: dict[int, float],
    b2i: dict[bytes, int],
    i2b: dict[int, bytes],
    bound_threshold: float,
    suffixes_sorted: list[str],
    firings: Counter,
    min_firings: int = 1,
    sibling_min_stem_len: int = 1,
) -> list[dict]:
    """For every vocab token T, decide if T = L + S is mangled per the
    algorithm in this module's docstring. Returns the metadata for each
    mangled token; the caller derives blocks via `find_formation_split`.
    """
    suffix_set = set(suffixes_sorted)

    mangled = []
    skipped_T_is_suffix = 0
    skipped_T_is_word = 0
    skipped_no_suffix = 0
    skipped_L_not_token = 0
    skipped_L_is_suffix = 0
    skipped_L_leading_space = 0
    skipped_L_free = 0
    skipped_L_has_space_sibling = 0

    for tid, b in i2b.items():
        if firings.get(tid, 0) < min_firings:
            continue
        try:
            s = b.decode("utf-8")
        except UnicodeDecodeError:
            continue
        # T is itself a clean suffix — `'ized'`, `'izing'`, `'tions'`, ...
        # Don't try to strip a smaller suffix off a token that IS a suffix.
        if s in suffix_set:
            skipped_T_is_suffix += 1
            continue
        # T is also a standalone word — its space-prefixed whole-word form
        # `' other'`, `' water'`, `' after'` exists in the vocab. The
        # suffix-looking ending (`'other'`→`-er`) is coincidental, part of
        # the word, not a productive morpheme boundary. Leave it alone.
        if (" " + s).encode("utf-8") in b2i:
            skipped_T_is_word += 1
            continue
        suf = longest_suffix_match(s, suffixes_sorted)
        if suf is None:
            skipped_no_suffix += 1
            continue
        L_str = s[: -len(suf)]
        L_bytes = L_str.encode("utf-8")
        if L_bytes not in b2i:
            skipped_L_not_token += 1
            continue
        # L is itself a clean suffix — `'ational'` = `'ation'`+`'al'`,
        # `'ationally'` = `'ationally'`(skipped above) or `'ationall'`+`'y'`,
        # etc. Per the user's framing, suffix+suffix is a clean morpheme
        # combination, not a mangled bound-fragment.
        if L_str in suffix_set:
            skipped_L_is_suffix += 1
            continue
        # L starts with a space → word-start-capable by construction. The
        # leading-space form IS the word-initial allomorph, even if longer
        # space-prefixed tokens (`' nurses'`, `' nursing'`) absorb most of
        # its firings in this corpus and depress its measured ws_prob.
        if L_str.startswith(" "):
            skipped_L_leading_space += 1
            continue
        L_id = b2i[L_bytes]
        ws_prob = word_start_prob.get(L_id, 0.0)
        if ws_prob >= bound_threshold:
            skipped_L_free += 1
            continue
        # L is non-space-prefixed and bound. Check for space-prefixed sibling
        # — if it exists, L is just the mid-word allomorph of a free stem
        # (`'organ'`/`' organ'`), and T is a clean stem+suffix combination.
        #
        # `sibling_min_stem_len` guards a failure mode: for very short stems
        # (1-2 chars) the space-sibling is a common function word/letter
        # (` at`, ` it`, ` c`, ` t`), so this exemption wrongly clears
        # high-frequency mangles like `ated`(at+ed), `ating`(at+ing),
        # `ction`(c+tion). Default 1 = exempt at any length (the
        # `organ`-protecting behavior). Set to 3 to require a real (≥3-char)
        # stem before trusting the sibling, so the short-stem head family
        # gets flagged.
        space_L_bytes = (" " + L_str).encode("utf-8")
        if space_L_bytes in b2i and len(L_str) >= sibling_min_stem_len:
            skipped_L_has_space_sibling += 1
            continue
        mangled.append({
            "T_str": s,
            "T_id": tid,
            "T_firings": firings.get(tid, 0),
            "L_str": L_str,
            "L_id": L_id,
            "L_firings": firings.get(L_id, 0),
            "L_word_start_prob": ws_prob,
            "suffix": suf,
        })

    print(
        f"\nMangled-token decision tally:\n"
        f"  skipped (T is itself a suffix):         {skipped_T_is_suffix:>6}\n"
        f"  skipped (T is a word, ' '+T in vocab):  {skipped_T_is_word:>6}\n"
        f"  skipped (no suffix match):              {skipped_no_suffix:>6}\n"
        f"  skipped (L not in vocab):               {skipped_L_not_token:>6}\n"
        f"  skipped (L is itself a suffix):         {skipped_L_is_suffix:>6}\n"
        f"  skipped (L is leading-space):           {skipped_L_leading_space:>6}\n"
        f"  skipped (L is free, ws_prob >= thr):    {skipped_L_free:>6}\n"
        f"  skipped (L has ' '+L space-sibling):    {skipped_L_has_space_sibling:>6}\n"
        f"  MANGLED:                                {len(mangled):>6}",
        file=sys.stderr,
    )
    return mangled


def emit_blocks_py(
    blocks: list[dict],
    file=sys.stdout,
    reference: Path = None,
    mining_args: dict = None,
) -> None:
    """Emit BLOCKED_PAIRS list compatible with blocks.py."""
    print(
        "# Auto-generated by tools/find_mangled_morphemes.py — do not edit by hand.",
        file=file,
    )
    print(
        "# Each entry blocks a `(L, R)` pair whose concat is a mangled "
        "morpheme token in the reference tokenizer:",
        file=file,
    )
    print(
        "# concat T = <bound-fragment> + <clean-suffix>, where the bound "
        "fragment never appears at word-start AND has no space-prefixed "
        "sibling in the vocab.",
        file=file,
    )
    if reference:
        print(f"# Reference tokenizer: {reference}", file=file)
    if mining_args:
        print(f"# Mining args: {mining_args}", file=file)
    print(f"# Generated {len(blocks)} blocks.\n", file=file)
    print("BLOCKED_PAIRS = [", file=file)
    for b in blocks:
        comment = (
            f"  # → {b['T_str']!r} (id={b['T_id']}, fires={b['T_firings']}x); "
            f"L={b['L_str']!r} ws_prob={b['L_word_start_prob']:.3f}; suffix={b['suffix']!r}"
        )
        print(f"    ({b['Lp_str']!r}, {b['Rp_str']!r}),{comment}", file=file)
    print("]", file=file)


def main(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--reference", type=Path,
        default=Path("~/.cache/nanochat/tokenizer").expanduser(),
        help="Reference tokenizer directory to analyze (default: baseline)",
    )
    p.add_argument(
        "--max-chars", type=int, default=20_000_000,
        help="Cap corpus sample size in characters (default: 20M, matches "
        "eval_tokenizer Tier-2)",
    )
    p.add_argument(
        "--bound-threshold", type=float, default=0.05,
        help="Word-start probability threshold below which L is considered "
        "bound. Default 0.05 — strict; raise (e.g. 0.15) for moderate "
        "aggressiveness.",
    )
    p.add_argument(
        "--min-firings", type=int, default=1,
        help="Skip tokens firing fewer than this many times in the corpus "
        "(default 1 — keep everything; raise for tighter filtering)",
    )
    p.add_argument(
        "--sibling-min-stem-len", type=int, default=1,
        help="Minimum stem length for the space-sibling exemption to apply "
        "(default 1 = always exempt, protects free stems like 'organ'). Set "
        "to 3 to stop exempting 1-2 char stems, flagging the high-frequency "
        "short-stem mangle family (ated=at+ed, ction=c+tion, ...).",
    )
    p.add_argument(
        "--exclude-blocks-py", type=Path, action="append", default=None,
        help="Path to a prior pass's block-list .py (with BLOCKED_PAIRS). "
        "When choosing each mangled token's formation split, skip pairs "
        "already in these files so this pass emits the NEXT path a surviving "
        "token reformed through. Repeatable. Used for iterative single-path "
        "blocking (PASS_2 = re-mine the PASS_1 tokenizer, exclude PASS_1).",
    )
    p.add_argument(
        "--out-py", type=Path, default=None,
        help="Write BLOCKED_PAIRS .py to this path (else print to stdout)",
    )
    args = p.parse_args(argv)

    from nanochat.tokenizer import RustBPETokenizer

    print(f"Loading reference tokenizer from {args.reference}...", file=sys.stderr)
    tok = RustBPETokenizer.from_directory(str(args.reference))
    b2i, i2b = build_vocab_maps(tok)
    print(f"  {len(b2i):,} mergeable tokens.", file=sys.stderr)

    suffixes_sorted = sorted(set(SUFFIXES), key=len, reverse=True)
    print(
        f"\nSuffix inventory: {len(suffixes_sorted)} entries, "
        f"longest={suffixes_sorted[0]!r} ({len(suffixes_sorted[0])} chars), "
        f"shortest={suffixes_sorted[-1]!r} ({len(suffixes_sorted[-1])} chars)",
        file=sys.stderr,
    )

    print(f"\nComputing word-start probabilities over {args.max_chars:,} chars...",
          file=sys.stderr)
    word_start_count, total_count, total_chars = compute_word_start_probs(
        tok, args.max_chars,
    )
    word_start_prob: dict[int, float] = {}
    for tid, total in total_count.items():
        word_start_prob[tid] = word_start_count.get(tid, 0) / total if total else 0.0
    print(f"  {total_chars:,} chars, {sum(total_count.values()):,} firings",
          file=sys.stderr)

    print(f"\nIdentifying mangled tokens (bound_threshold={args.bound_threshold})...",
          file=sys.stderr)
    mangled = find_mangled_tokens(
        tok, word_start_prob, b2i, i2b,
        bound_threshold=args.bound_threshold,
        suffixes_sorted=suffixes_sorted,
        firings=total_count,
        min_firings=args.min_firings,
        sibling_min_stem_len=args.sibling_min_stem_len,
    )

    exclude_set: set[tuple[bytes, bytes]] = set()
    if args.exclude_blocks_py:
        exclude_set = load_exclude_blocks(args.exclude_blocks_py)
        print(f"\nExcluding {len(exclude_set)} already-blocked pairs from "
              f"{len(args.exclude_blocks_py)} prior-pass file(s) when picking "
              f"formation splits.", file=sys.stderr)

    print(f"\nDeriving formation-pair blocks for {len(mangled)} mangled tokens...",
          file=sys.stderr)
    blocks = []
    no_split = 0
    for m in mangled:
        T_bytes = m["T_str"].encode("utf-8")
        split = find_formation_split(T_bytes, b2i, m["T_id"], exclude=exclude_set)
        if split is None:
            no_split += 1
            continue
        Lp, Rp, Lp_id, Rp_id = split
        try:
            Lp_str = Lp.decode("utf-8")
            Rp_str = Rp.decode("utf-8")
        except UnicodeDecodeError:
            continue
        blocks.append({
            **m,
            "Lp_str": Lp_str, "Rp_str": Rp_str,
            "Lp_id": Lp_id, "Rp_id": Rp_id,
        })
    print(f"  {len(blocks)} blocks derived; {no_split} skipped (no valid split).",
          file=sys.stderr)

    # Dedupe — multiple targets may share the same (Lp, Rp) pair.
    seen = set()
    deduped = []
    for b in blocks:
        key = (b["Lp_str"], b["Rp_str"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(b)
    if len(deduped) < len(blocks):
        print(f"  deduped to {len(deduped)} unique pairs "
              f"(removed {len(blocks) - len(deduped)} duplicates).", file=sys.stderr)

    # Emit in target-id ascending order — matches force_merges/seed_tokens'
    # natural-firing-order convention.
    deduped.sort(key=lambda b: b["T_id"])

    mining_args = {
        "reference": str(args.reference),
        "max_chars": args.max_chars,
        "bound_threshold": args.bound_threshold,
        "min_firings": args.min_firings,
        "sibling_min_stem_len": args.sibling_min_stem_len,
        "exclude_blocks_py": [str(p) for p in (args.exclude_blocks_py or [])],
    }
    if args.out_py:
        args.out_py.parent.mkdir(parents=True, exist_ok=True)
        with args.out_py.open("w") as f:
            emit_blocks_py(
                deduped, file=f, reference=args.reference, mining_args=mining_args,
            )
        print(f"\nWrote {args.out_py}", file=sys.stderr)
    else:
        emit_blocks_py(deduped, mining_args=mining_args)


if __name__ == "__main__":
    main()
