"""
Data-driven candidate generation for the force_merges tokenizer.

Counts cross-boundary bigrams from a corpus sample: apply the BASELINE
pre-tokenization regex, walk adjacent chunk pairs (chunk_i, chunk_{i+1})
within each document, rank by frequency. Adjacent chunks are precisely the
pairs that BPE *can't* learn today because pre-tok splits them — i.e. the
candidates for forced cross-boundary merges.

Two filters constrain candidates before ranking:
  - Grammatical role: only pairs whose LHS is closed-class (punctuation,
    preposition, conjunction, auxiliary, determiner) and RHS is closed-class
    (determiner, auxiliary, pronoun, conjunction, capitalized sentence-
    starter, punctuation). See LHS_FUNCTION_WORDS / RHS_FUNCTION_WORDS.
    Frequency alone over-selects on content-word bigrams that don't
    generalize across domains.
  - Shadow detection: a pair `(c, d)` is shadowed by emitted `(a, b)` when
    `b == c` (operand consumed) or `a == d` (regex precedence pre-emption).
    Shadowed pairs would never fire under the leftmost-first alternation.

`--two-pass` adds a second pass that re-pre-tokenizes against an augmented
SPLIT_PATTERN with the pass-1 carve-outs, then mines bigrams whose LHS is a
pass-1 concatenation. Recovers depth-2 phrases (e.g. `(" of the", " most")`)
that single-pass can't see.

Output: a `pairs.py`-compatible Python module with `FORCED_PAIRS` (and, with
`--two-pass`, `DERIVED_PAIRS`) — see `rustbpe_variants/force_merges/mined/`
for the artifacts currently used by `pairs.py`.

Usage:
    # Two-pass mining, write artifact for the canonical pairs.py to import.
    uv run python -m tools.mine_forced_pairs --row-groups 20 --two-pass \\
        --top-k 150 --top-k-derived 20 \\
        --out-py rustbpe_variants/force_merges/mined/forced_pairs_climbmix_20rg_twopass.py

    # Diff a new mining run against the in-tree list.
    uv run python -m tools.mine_forced_pairs --row-groups 20 --top-k 150 \\
        --compare-to rustbpe_variants/force_merges/pairs.py

No GPU. ~10s per row group on a laptop; a 20-row-group two-pass run takes ~5 min.
"""

import argparse
import sys
import time
from collections import Counter
from pathlib import Path

import regex

# Baseline pre-tokenization regex — lifted verbatim from
# nanochat/tokenizer.py:30 (pristine, no force_merges carve-out). Kept inline
# so the miner doesn't depend on an installed nanochat.
BASELINE_SPLIT_PATTERN = (
    r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,2}"""
    r"""| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
)

# ---------------------------------------------------------------------------
# Grammatical-role inventory
#
# Closed-class words only (small, finite sets that don't grow with vocabulary
# expansion). The hand-picked FORCED_PAIRS list almost exclusively draws from
# these classes — see rustbpe_variants/force_merges/pairs.py:48-145. We
# enumerate them explicitly so the filter is auditable.
#
# All entries include the leading space because the baseline pre-tokenizer
# produces space-prefixed word chunks (" the", not "the"). Bare-word entries
# (no leading space) are for the start-of-document or post-punctuation case.

PUNCT_LHS = {",", ".", ";", ":", "?", "!"}

# Prepositions, conjunctions, articles, auxiliaries — function words that
# legitimately glue to a following noun phrase or clause.
LHS_FUNCTION_WORDS = {
    # prepositions
    " of", " in", " on", " at", " to", " for", " from", " with", " by",
    " as", " about", " into", " through", " during", " between", " over",
    " under", " against", " without", " within", " across", " around",
    " before", " after",
    # conjunctions
    " and", " or", " but", " nor", " yet", " so",
    # auxiliaries / common verbs
    " is", " are", " was", " were", " be", " been", " being",
    " has", " have", " had", " can", " could", " will", " would",
    " should", " may", " might", " must", " do", " does", " did",
    # determiners / quantifiers that frequently precede other determiners
    " all", " some", " any", " no", " every", " each", " both", " more",
    " most", " such", " many", " few",
    # demonstratives / quantifiers that can lead a phrase
    " that", " this", " these", " those", " one", " two",
    # discourse markers
    " However", " Also", " Therefore", " Moreover", " Furthermore",
}

# Right-hand side: what a glue phrase legitimately *points at*.
RHS_FUNCTION_WORDS = {
    # determiners / articles
    " the", " a", " an", " this", " that", " these", " those",
    " their", " its", " his", " her", " our", " your", " my",
    # auxiliaries / common verbs (continuations like " has been", " can be")
    " be", " been", " is", " are", " was", " were", " not",
    " do", " did", " have", " has", " make", " get", " go", " come",
    " know", " see", " find", " give", " take", " use", " want", " need",
    # pronouns (incl. expletive "there", "here")
    " it", " they", " we", " you", " he", " she", " I", " us", " them",
    " there", " here",
    # conjunctions / connectives
    " and", " or", " but", " which", " who", " that", " as", " in", " of",
    " to", " with", " from", " on", " for", " than", " then", " yet",
    # quantifiers / common modifiers that follow function words
    " same", " first", " other", " more", " most", " many", " all", " only",
    " also", " just", " still", " well", " ever", " even", " never", " any",
    " every", " each", " some",
    # common adverbial / participial right-anchors
    " including", " regarding", " concerning", " following",
    # capitalized sentence-starters (post-punctuation case)
    " The", " A", " An", " It", " This", " That", " These", " They",
    " We", " You", " He", " She", " I", " In", " For", " But",
    " As", " If", " There", " However", " Also", " When", " While",
    " So", " And", " Or", " On", " To", " With", " By", " At", " Of",
}

# Punctuation also legitimately appears on the right when the LHS is a
# discourse marker (" However" + ",", " Therefore" + ",", etc.).
RHS_PUNCT = {",", "."}


def classify_pair(a: str, b: str, mode: str) -> bool:
    """Return True if (a, b) passes the grammatical-role filter."""
    if mode == "none":
        return True
    lhs_ok = a in PUNCT_LHS or a in LHS_FUNCTION_WORDS
    if mode == "loose":
        # LHS-only filter — RHS just needs to be short and word-like.
        return lhs_ok and len(b) <= 6 and b.startswith((" ", "0"))
    # strict (default)
    rhs_ok = b in RHS_FUNCTION_WORDS or b in RHS_PUNCT
    return lhs_ok and rhs_ok


# ---------------------------------------------------------------------------
# Mining

def iter_corpus(row_groups: int, max_chars: int):
    """Yield documents from the local nanochat parquet shards.

    parquets_iter_batched yields one batch per parquet row group, not per
    shard (a shard has many row groups — typically ~84 for the climbmix
    shards). We expose row-group count as the cap so the user can control
    sample size at a fine granularity; ~3M chars per row group, ~84 per shard.
    """
    # Local import: keeps the module importable without nanochat for the
    # filter/conflict-detection logic alone (handy in tests).
    from nanochat.dataset import parquets_iter_batched

    total_chars = 0
    docs_yielded = 0
    row_groups_seen = 0
    for batch in parquets_iter_batched(split="train"):
        if row_groups > 0 and row_groups_seen >= row_groups:
            break
        for doc in batch:
            yield doc
            docs_yielded += 1
            total_chars += len(doc)
            if max_chars and total_chars >= max_chars:
                print(
                    f"  reached --max-chars cap ({total_chars:,} chars, "
                    f"{docs_yielded:,} docs)",
                    file=sys.stderr,
                )
                return
        row_groups_seen += 1
    print(
        f"  scanned {docs_yielded:,} docs / {total_chars:,} chars "
        f"across {row_groups_seen} row group(s)",
        file=sys.stderr,
    )


def count_bigrams(docs, split_pattern: str) -> Counter:
    """Count adjacent pre-tok chunk pairs across the document stream."""
    rgx = regex.compile(split_pattern)
    counts: Counter = Counter()
    n_docs = 0
    t0 = time.time()
    for doc in docs:
        # findall with a pure-alternation grouping pattern returns a list of
        # tuples (one element per group); we want raw chunk strings, so use
        # finditer + match.group(0).
        chunks = [m.group(0) for m in rgx.finditer(doc)]
        # bigrams within a document only — cross-document bigrams are
        # meaningless because BOS separates documents at training time.
        for i in range(len(chunks) - 1):
            counts[(chunks[i], chunks[i + 1])] += 1
        n_docs += 1
        if n_docs % 5000 == 0:
            elapsed = time.time() - t0
            print(
                f"  {n_docs:,} docs / {len(counts):,} unique bigrams "
                f"({elapsed:.1f}s)",
                file=sys.stderr,
            )
    return counts


# ---------------------------------------------------------------------------
# Two-pass mining: derived pairs

def build_augmented_pattern(pass1_pairs: list[tuple[str, str]]) -> str:
    """Prepend a forced-pair carve-out to BASELINE_SPLIT_PATTERN.

    Mirrors the runtime SPLIT_PATTERN derivation in
    rustbpe_variants/force_merges/pairs.py: pairs become the leftmost
    alternation, followed by `(?=([^a-z]|$))` to prevent matching inside a
    larger word (e.g. " of theory" -/-> " of the" + "ory"), then the rest of
    the baseline pattern.
    """
    if not pass1_pairs:
        return BASELINE_SPLIT_PATTERN
    forced_alt = "|".join(regex.escape(a + b) for a, b in pass1_pairs)
    return (
        f"({forced_alt})(?=([^a-z]|$))|"
        + BASELINE_SPLIT_PATTERN
    )


def classify_derived_pair(c: str, d: str, pass1_concats: set[str],
                          filter_mode: str) -> bool:
    """Pass-2 filter: candidate is valid iff c is a pass-1 concatenation.

    The pass-2 LHS comes from the carve-out alternation, so by construction
    it's a string like ", in" or ".  However,". Filtering on "c was produced
    by pass 1" guarantees we're only producing genuine derived pairs (an
    arbitrary baseline chunk on the LHS would already have been visible in
    pass 1).

    The RHS still goes through the standard role filter — derived pairs
    should still point at function-class continuations.
    """
    if c not in pass1_concats:
        return False
    if filter_mode == "none":
        return True
    rhs_ok = d in RHS_FUNCTION_WORDS or d in RHS_PUNCT
    if filter_mode == "loose":
        return rhs_ok or (len(d) <= 6 and d.startswith((" ", "0")))
    return rhs_ok


# ---------------------------------------------------------------------------
# Conflict detection

def detect_shadows(ordered_pairs, alternation_wins=False):
    """For each pair, list the *higher-ranked* pairs that would shadow it.

    Shadow case 1 (always applies): b == c — the emitted pair's right operand
    is the candidate's left operand. The emitted carve-out consumes that byte
    sequence wholesale, so the candidate's LHS token never appears in
    isolation and the forced merge can't fire.

    Shadow case 2 (suppressed when alternation_wins=True): pa == b — the
    emitted pair's left operand equals the candidate's right operand. This
    can pre-empt the candidate via regex leftmost-first IF the emitted pair
    is listed before the candidate. For DERIVED pairs (pass-2 output) this
    doesn't apply — derived alternations are emitted FIRST in the regex
    (DERIVED_PAIRS_EXPR precedes _FORCED_PAIRS_EXPR_BASE in pairs.py:165),
    so the longer derived match wins regardless of frequency rank. Pass
    alternation_wins=True for that case.

    The hand-picked list at rustbpe_variants/force_merges/pairs.py:81-118
    documents both cases in its inline comments — the `(" it", " is")`
    family is case 1 (a==b in the prior == c in the candidate? no, that's
    pa==b — same letters, different framing); `(" one", " of")` shadowed by
    pairs starting with `" of"` is case 2.

    Args:
        ordered_pairs: list of ((a, b), count) in priority order (highest first).
        alternation_wins: if True, suppress case 2 (use for derived pairs).

    Returns:
        dict mapping pair -> list of higher-ranked pairs that shadow it.
    """
    shadows: dict[tuple[str, str], list[tuple[str, str]]] = {}
    seen: list[tuple[str, str]] = []
    for pair, _ in ordered_pairs:
        a, b = pair
        conflicts = []
        for prior in seen:
            pa, pb = prior
            # Case 1: right-of-prior == left-of-candidate. Always shadows.
            if pb == a:
                conflicts.append(prior)
            # Case 2: left-of-prior == right-of-candidate. Regex precedence;
            # skip when the candidate wins precedence via alternation order.
            elif pa == b and not alternation_wins:
                conflicts.append(prior)
        if conflicts:
            shadows[pair] = conflicts
        seen.append(pair)
    return shadows


# ---------------------------------------------------------------------------
# Output

def fmt_pair(a: str, b: str) -> str:
    """Format a (a, b) tuple as a valid Python source repr."""
    return f"({a!r}, {b!r})"


def emit_text(ranked, shadows, file=sys.stdout):
    """Human-readable ranked dump."""
    width = max((len(fmt_pair(a, b)) for (a, b), _ in ranked), default=24)
    print(f"# rank  count        {'pair':<{width}}  shadowed-by", file=file)
    for i, ((a, b), count) in enumerate(ranked, 1):
        sh = shadows.get((a, b))
        sh_str = ""
        if sh:
            sh_str = "  ⚠ " + ", ".join(fmt_pair(*p) for p in sh[:3])
            if len(sh) > 3:
                sh_str += f", +{len(sh) - 3} more"
        print(
            f"  {i:>3}  {count:>9,}  {fmt_pair(a, b):<{width}}{sh_str}",
            file=file,
        )


def emit_pairs_py(ranked, shadows, file=sys.stdout, include_shadowed=False,
                  derived_ranked=None, derived_shadows=None):
    """Emit a pairs.py-compatible FORCED_PAIRS (+ optional DERIVED_PAIRS) list."""
    print("# Auto-generated by tools/mine_forced_pairs.py", file=file)
    print("# Each entry: (left, right)  # count=N", file=file)
    print(
        "# Shadowed entries (would be pre-empted by an earlier pair under "
        "leftmost-first alternation) are commented out.",
        file=file,
    )
    print("FORCED_PAIRS = [", file=file)
    for (a, b), count in ranked:
        prefix = "    "
        suffix = f"  # count={count:,}"
        if (a, b) in shadows and not include_shadowed:
            prefix = "    # "
            sh = shadows[(a, b)]
            suffix += f" SHADOWED by {fmt_pair(*sh[0])}"
        print(f"{prefix}{fmt_pair(a, b)},{suffix}", file=file)
    print("]", file=file)

    if derived_ranked is not None:
        print(
            "\n# DERIVED_PAIRS: pairs whose LHS only exists as a token after "
            "a pass-1 forced merge has fired.",
            file=file,
        )
        print(
            "# Mined by re-pre-tokenizing the corpus against an augmented "
            "SPLIT_PATTERN that includes the pass-1 carve-outs.",
            file=file,
        )
        print("DERIVED_PAIRS = [", file=file)
        derived_shadows = derived_shadows or {}
        for (a, b), count in derived_ranked:
            prefix = "    "
            suffix = f"  # count={count:,}"
            if (a, b) in derived_shadows and not include_shadowed:
                prefix = "    # "
                sh = derived_shadows[(a, b)]
                suffix += f" SHADOWED by {fmt_pair(*sh[0])}"
            print(f"{prefix}{fmt_pair(a, b)},{suffix}", file=file)
        print("]", file=file)


def emit_compare(ranked, hand_picked: list[tuple[str, str]],
                 raw_counts: Counter, args, file=sys.stdout):
    """Diff the mined list against an existing hand-picked list."""
    mined_set = {pair for pair, _ in ranked}
    hand_set = set(hand_picked)
    mined_count = {pair: c for pair, c in ranked}

    overlap = mined_set & hand_set
    mined_only = [(p, mined_count[p]) for p in mined_set - hand_set]
    mined_only.sort(key=lambda x: -x[1])
    hand_only = [p for p in hand_picked if p not in mined_set]

    print(f"Hand-picked size:      {len(hand_picked)}", file=file)
    print(f"Mined size:            {len(mined_set)}", file=file)
    print(f"Overlap:               {len(overlap)} pairs", file=file)
    print(
        f"  → coverage of hand:  {len(overlap) / max(1, len(hand_picked)):.1%}",
        file=file,
    )
    print(
        f"\nMined ∖ hand-picked    ({len(mined_only)} pairs — "
        "candidates to add):",
        file=file,
    )
    for (a, b), c in mined_only[:50]:
        print(f"  {c:>9,}  {fmt_pair(a, b)}", file=file)
    if len(mined_only) > 50:
        print(f"  ... +{len(mined_only) - 50} more", file=file)

    print(
        f"\nHand-picked ∖ mined    ({len(hand_only)} pairs — "
        "in hand list but did not make the top-K under current filters):",
        file=file,
    )
    for a, b in hand_only:
        c = raw_counts.get((a, b), 0)
        if c == 0:
            note = "(zero count in sample)"
        elif c < args.min_count:
            note = f"(below --min-count {args.min_count})"
        elif not classify_pair(a, b, args.filter):
            note = f"(rejected by --filter {args.filter})"
        else:
            note = "(below top-K)"
        print(f"  {c:>9,}  {fmt_pair(a, b):<28}  {note}", file=file)


def load_hand_picked(path: Path) -> list[tuple[str, str]]:
    """Extract FORCED_PAIRS from a pairs.py-style file."""
    namespace: dict = {}
    exec(compile(path.read_text(), str(path), "exec"), namespace)
    pairs = namespace.get("FORCED_PAIRS")
    if pairs is None:
        raise ValueError(f"{path} does not define FORCED_PAIRS")
    return [tuple(p) for p in pairs]


# ---------------------------------------------------------------------------
# CLI

def main(argv=None):
    p = argparse.ArgumentParser(
        description="Mine forced-merge candidates from the pretraining corpus.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--row-groups", type=int, default=10,
        help=(
            "Number of parquet row groups to scan (default: 10; ~3M chars "
            "each; ~84 row groups per shard). Use 0 to scan everything "
            "available."
        ),
    )
    p.add_argument(
        "--max-chars", type=int, default=0,
        help="Cap total characters scanned (0 = no cap)",
    )
    p.add_argument(
        "--top-k", type=int, default=100,
        help="Emit top-K pairs after filtering (default: 100)",
    )
    p.add_argument(
        "--min-count", type=int, default=100,
        help="Drop pairs below this count (default: 100)",
    )
    p.add_argument(
        "--filter", choices=["function_word", "loose", "none"],
        default="function_word",
        help="Grammatical-role filter mode (default: function_word)",
    )
    p.add_argument(
        "--out-py", type=Path, default=None,
        help="Write a pairs.py-style FORCED_PAIRS file to this path",
    )
    p.add_argument(
        "--compare-to", type=Path, default=None,
        help="Path to existing pairs.py to compare against",
    )
    p.add_argument(
        "--include-shadowed", action="store_true",
        help="When emitting --out-py, keep shadowed entries un-commented",
    )
    p.add_argument(
        "--two-pass", action="store_true",
        help=(
            "After pass 1, re-pre-tokenize against an augmented "
            "SPLIT_PATTERN with pass-1 carve-outs and mine DERIVED_PAIRS "
            "(pairs whose LHS only exists post-merge). Doubles run time."
        ),
    )
    p.add_argument(
        "--top-k-derived", type=int, default=15,
        help="--two-pass only: top-K derived pairs to emit (default: 15)",
    )
    p.add_argument(
        "--min-count-derived", type=int, default=50,
        help="--two-pass only: min count for derived pairs (default: 50)",
    )
    args = p.parse_args(argv)

    print(f"Scanning {args.row_groups} row group(s), "
          f"filter={args.filter} ...", file=sys.stderr)
    counts = count_bigrams(
        iter_corpus(args.row_groups, args.max_chars),
        BASELINE_SPLIT_PATTERN,
    )
    print(f"Counted {sum(counts.values()):,} bigrams "
          f"({len(counts):,} unique).", file=sys.stderr)

    filtered = [
        (pair, c) for pair, c in counts.most_common()
        if c >= args.min_count and classify_pair(pair[0], pair[1], args.filter)
    ]
    ranked = filtered[: args.top_k]
    print(f"After filter + min-count + top-K: {len(ranked)} pairs.",
          file=sys.stderr)

    shadows = detect_shadows(ranked)
    print(f"Shadowed entries: {len(shadows)} / {len(ranked)}.", file=sys.stderr)

    derived_ranked = None
    derived_shadows = None
    if args.two_pass:
        # Pass-1 emitted set: non-shadowed pairs only (those that would
        # actually fire in the augmented regex; shadowed ones are dead code).
        emitted_p1 = [pair for pair, _ in ranked if pair not in shadows]
        pass1_concats = {a + b for a, b in emitted_p1}
        augmented = build_augmented_pattern(emitted_p1)
        print(
            f"Pass 2: re-scanning with {len(emitted_p1)}-pair "
            "augmented SPLIT_PATTERN...",
            file=sys.stderr,
        )
        counts2 = count_bigrams(
            iter_corpus(args.row_groups, args.max_chars),
            augmented,
        )
        print(f"  Pass-2 bigrams: {sum(counts2.values()):,} total, "
              f"{len(counts2):,} unique.", file=sys.stderr)
        # Filter: LHS must be a pass-1 concatenation (= new chunk type
        # introduced by the carve-out, not visible in pass 1).
        filtered2 = [
            (pair, c) for pair, c in counts2.most_common()
            if c >= args.min_count_derived
            and classify_derived_pair(
                pair[0], pair[1], pass1_concats, args.filter
            )
        ]
        derived_ranked = filtered2[: args.top_k_derived]
        # Shadow detection: derived pairs win regex-alternation precedence
        # (DERIVED_PAIRS_EXPR is listed before _FORCED_PAIRS_EXPR_BASE), so
        # only case-1 shadowing (b==c — operand actually consumed) applies.
        # Case-2 shadowing (pa==b — precedence) is suppressed.
        derived_shadows = detect_shadows(
            ranked + derived_ranked, alternation_wins=True
        )
        derived_pair_set = {pair for pair, _ in derived_ranked}
        derived_shadows = {
            k: v for k, v in derived_shadows.items() if k in derived_pair_set
        }
        print(
            f"Pass-2 derived: {len(derived_ranked)} pairs after filter+top-K "
            f"(shadowed: {len(derived_shadows)}).",
            file=sys.stderr,
        )

    if args.compare_to:
        hand = load_hand_picked(args.compare_to)
        emit_compare(ranked, hand, counts, args)
    elif args.out_py:
        with args.out_py.open("w") as f:
            emit_pairs_py(
                ranked, shadows, file=f,
                include_shadowed=args.include_shadowed,
                derived_ranked=derived_ranked,
                derived_shadows=derived_shadows,
            )
        print(f"Wrote {args.out_py}", file=sys.stderr)
    else:
        emit_text(ranked, shadows)
        if derived_ranked:
            print("\n-- DERIVED (pass 2) --")
            emit_text(derived_ranked, derived_shadows)


if __name__ == "__main__":
    main()
