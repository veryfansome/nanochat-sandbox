"""Classify a candidate block's added/lost whole-words as REAL coverage vs ARTIFACT /
AMBIENT — the per-round candidate-comparison lens for the auto_tune block-list audit.

The trap this exists to catch: a space-prefixed token's *firing count* is token-usage, NOT
standalone-word frequency. A token like ` airs` can fire 71× while the word "airs" barely
occurs — because blocking a merge reroutes longer words through it (` airship`→` airs`+`hip`
when `s+hip` is blocked), and the `_is_word` stem+suffix heuristic then mislabels the
fragment ("air"+s) as a whole word. So "+3 firing whole words" can be artifacts of the very
shattering the block causes, not new coverage. Reading firing counts without this lens is how
count-based picking commits junk (see the `autotune-firing33-orchestrated-judge` memory).

For each surface word (e.g. " airs") this reports, on a given corpus:
  - dict   : DICT = a real /usr/share/dict/words entry · INFL = only via the _is_word
             stem+suffix heuristic (e.g. "hards"=hard+s, often not a real standalone word)
  - standN : standalone-word occurrences (how often it appears AS a word, word-boundaried)
  - fireN  : how often the TOKEN fires (passed in from the tokenizer; CLI estimates it from
             standalone + sub-word usage is NOT computed — pass the inspect/propose count)
  - flags  : descriptive lenses, NOT a verdict —
             ART?     fires but ~0 standalone usage → firing is prefix/reroute, not coverage
             AMBIENT  the same word is born under many candidates → a slot-filler any block
                      gets (computed across a round's candidates, not from one word)

These are DATA to read the firing counts correctly. The keep decision stays a per-round hand
judgment weighing realness + frequency — there is no fixed weight (that's the whole point of
the orchestrated loop). This tool never picks; it annotates.
"""
import re
import argparse
from collections import Counter

from tools.blocked_pairs_report import load_wordlist, _is_word

# a born word shared by >= this many of a round's candidates is treated as an ambient
# slot-filler (the next merge waiting at that frequency margin, claimed by any freed slot)
AMBIENT_MIN_CANDIDATES = 3


def bare(surface):
    """The dictionary form of a surface token: drop the leading space if present."""
    return surface[1:] if surface.startswith(" ") else surface


def dict_status(surface, words):
    """DICT = bare word is a true dict entry · INFL = only via the _is_word stem+suffix
    heuristic (plural/inflection of a real word, often not a real standalone word) · — = neither."""
    w = bare(surface).lower()
    if not w:
        return "—"
    if w in words:
        return "DICT"
    if _is_word(w, words):
        return "INFL"
    return "—"


def standalone_counts(surfaces, docs):
    """One regex pass: standalone-word (word-boundaried, case-sensitive) occurrences of each
    surface's bare form in `docs` (an iterable of strings). Returns (Counter{bare: n}, chars)."""
    bares = {bare(s) for s in surfaces if bare(s)}
    if not bares:
        return Counter(), 0
    alt = "|".join(re.escape(b) for b in sorted(bares, key=len, reverse=True))
    pat = re.compile(r"(?<![A-Za-z])(" + alt + r")(?![A-Za-z])")
    cnt, chars = Counter(), 0
    for d in docs:
        chars += len(d)
        for m in pat.finditer(d):
            cnt[m.group(1)] += 1
    return cnt, chars


def flags(surface, words, stand_n, fire_n, ambient=False, ambient_tag="AMBIENT"):
    """Descriptive flags for reading the firing count — NOT a keep verdict. `ambient_tag` lets
    the caller pick the cross-set tag: AMBIENT (propose: born under many *different* candidates
    → slot-filler) vs SHARED (loo: born under many *leave-one-outs* of the *same* kept set →
    a word jointly held by several keeps — the OPPOSITE value signal)."""
    f = []
    if fire_n and stand_n == 0:        # fires yet never appears as a standalone word
        f.append("ART?")
    if ambient:
        f.append(ambient_tag)
    return f


def annotate(surface, fire_n, words, stand_map, ambient_set, ambient_tag="AMBIENT"):
    """Compact inline annotation for the propose/inspect/loo view, e.g.
    `insure:84[s14·DICT]` · `hards:56[s0·INFL·ART?]` · `cladding:31[s7·DICT·AMBIENT]`.
    Pass ambient_tag='SHARED' for the loo view (cross-set sharing means co-held, not slot-filler)."""
    stand_n = stand_map.get(bare(surface), 0)
    fl = flags(surface, words, stand_n, fire_n, ambient=surface in ambient_set, ambient_tag=ambient_tag)
    tail = "·".join([f"s{stand_n}", dict_status(surface, words)] + fl)
    return f"{repr(surface)[1:-1]}:{fire_n}[{tail}]"


def ambient_surfaces(candidate_born_lists, min_n=AMBIENT_MIN_CANDIDATES):
    """Surfaces born under >= min_n of the given born-lists. Two meanings by context:
    PROPOSE (lists = different candidates, min_n=3) → ambient slot-fillers any block gets;
    LOO (lists = leave-one-outs of one kept set, min_n=2) → words co-held by several keeps
    (removing any of them drops the word — a jointly-supported / slot-marginal tail word)."""
    seen = Counter()
    for born in candidate_born_lists:
        for s in set(born):
            seen[s] += 1
    return {s for s, n in seen.items() if n >= min_n}


# ---------------------------------------------------------------- CLI (deep dive w/ contexts)
def _iter_corpus(corpus, max_chars):
    if corpus == "val":
        from tools._corpus_iter import iter_capped_batches
        for batch in iter_capped_batches("val", max_chars):
            yield from batch
    elif corpus == "holdout":
        import glob
        from pathlib import Path
        from tools.heldout_buildup import iter_capped_from_path
        home = str(Path.home())
        shards = sorted(glob.glob(f"{home}/.cache/nanochat/base_data_climbmix_holdout/*.parquet"))
        per = max(max_chars // max(len(shards), 1), 1)
        for s in shards:
            for batch in iter_capped_from_path(s, per):  # yields batches (lists of docs)
                yield from batch
    else:
        raise SystemExit(f"unknown --corpus {corpus!r} (use 'val' or 'holdout')")


def main():
    p = argparse.ArgumentParser(description="Whole-word realness lens for auto_tune candidates.")
    p.add_argument("words", nargs="+", help="surface tokens to check, e.g. ' airs' ' hards' ' insure' "
                                            "(quote the leading space; a bare word is treated as space-prefixed)")
    p.add_argument("--corpus", default="val", choices=["val", "holdout"],
                   help="corpus for standalone-word counting + contexts (default val = the discovery shard)")
    p.add_argument("--max-chars", type=int, default=30_000_000)
    p.add_argument("--contexts", type=int, default=5, help="sample surrounding snippets per word (0 = none)")
    args = p.parse_args()

    words = load_wordlist()
    if words is None:
        raise SystemExit("no wordlist at /usr/share/dict/words")
    surfaces = [w if w.startswith(" ") else " " + w for w in args.words]

    bares = {bare(s) for s in surfaces}
    alt = "|".join(re.escape(b) for b in sorted(bares, key=len, reverse=True))
    pat = re.compile(r"(?<![A-Za-z])(" + alt + r")(?![A-Za-z])")
    cnt, examples, chars = Counter(), {b: [] for b in bares}, 0
    for d in _iter_corpus(args.corpus, args.max_chars):
        chars += len(d)
        for m in pat.finditer(d):
            b = m.group(1)
            cnt[b] += 1
            if len(examples[b]) < args.contexts:
                s, e = max(0, m.start() - 28), min(len(d), m.end() + 28)
                examples[b].append(d[s:e].replace("\n", " "))

    print(f"# word realness on {args.corpus} ({chars:,} chars) — DATA to judge by hand, not a verdict")
    print(f"# {'surface':<16} {'dict':>5} {'standalone':>18}   flags")
    for s in surfaces:
        b = bare(s)
        n = cnt[b]
        fl = flags(s, words, n, fire_n=1)  # fire_n unknown at CLI; ART? only fires if standalone==0
        print(f"  {repr(s)[1:-1]:<16} {dict_status(s, words):>5} {n:>8} ({1e6*n/max(chars,1):>5.2f}/M)   {' '.join(fl)}")
        for ex in examples[b]:
            print(f"      …{ex}…")


if __name__ == "__main__":
    main()
