"""
FORCED_PAIRS / BLOCKED_PAIRS / DERIVED_PAIRS + derived SPLIT_PATTERN for
the force_merges tokenizer variant.

## History (replaced 2026-05-25)

Originally a ~70-pair hand-curated list lifted from
veryfansome/nanochat@force_merges_wip — built incrementally with per-pair
verification. As of 2026-05-25, replaced wholesale with a data-driven list
produced via:

  1. **Two-pass corpus mining** (`tools/mine_forced_pairs.py --two-pass`)
     on 20 climbmix row groups (~60M chars). Pass 1 counts cross-boundary
     bigrams in the baseline pre-tokenization; pass 2 re-pre-tokenizes
     against an augmented SPLIT_PATTERN with pass-1 carve-outs and mines
     bigrams whose LHS is a pass-1 concatenation (recovers DERIVED-class
     merges that single-pass mining can't see).
  2. **Closed-class grammatical-role filter** restricting candidates to
     LHS = {punctuation, prepositions, conjunctions, auxiliaries,
     determiners} and RHS = {determiners, auxiliaries, pronouns,
     conjunctions, capitalized sentence-starters, punctuation}.
  3. **Shadow detection** flagging pairs pre-empted under the
     leftmost-first regex alternation (case 1: `b==c` operand consumed;
     case 2: `pa==b` precedence, suppressed for DERIVED since those win
     alternation order).
  4. **Cumulative DERIVED ablation** (`tools/ablate_derived.py`) to find
     the K where Tier-1 metrics start to regress (K=15 of 20 mined
     DERIVED). Pairs beyond K=15 each individually displace a borderline
     BPE merge (`izards`, ` acclim`, `ampton`, `uran`) because the
     32K-vocab budget can't accommodate more carve-outs without
     sacrificing low-rank content-word merges.
  5. **Numeric DERIVED hand-additions** (`("00","0")`, `(",","000")`).
     These fail the function-word filter by design but are needed to
     recover the `currency` task probe vs the original hand-picked list.

Real-corpus measurement (15M-char sample): this list compresses to
**−6.5%** total tokens vs the baseline nanochat tokenizer (3.27M → 3.06M
tokens), trading ~237 rare content-word BPE merges (` Staphylococcus`,
` philosophers`, ` consultations`, inflected forms) for ~237 high-frequency
phrasal merges (` of the`, `, and the`, `. The`, ` can be`, etc.). On
Tier-1 probes the trade is net-zero or positive on every corpus.

The full exploration sequence (v0 through v5 candidates, +5/+10/+20 vocab
bracketing experiment) is documented in [`STATUS.md`](../../STATUS.md);
mining + ablation tooling lives at `tools/{mine_forced_pairs,ablate_*}.py`
and is preserved for re-running this curation against future corpus
distributions.

## Per-pair DERIVED rationale (Phase 3 ablation, top-20 mined)

  rank   pair                           verdict  evidence
  ----   ----                           -------  --------
   1   (', and', ' the')                KEEP     Tier-1 silent
   2   ('. If', ' you')                 KEEP     Tier-1 silent
   3   ('. It', ' is')                  KEEP     Tier-1 silent
   4   (', it', ' is')                  KEEP     Tier-1 silent
   5   ('. This', ' is')                KEEP     Tier-1 silent
   6   (', but', ' it')                 KEEP     +0.070 bpt modern_prose;
                                                 token-diff: ` this`+`, but`
                                                 +` it` → ` this`+`, but it`
   7   (', which', ' is')               KEEP     Tier-1 silent
   8   ('. There', ' are')              KEEP     Tier-1 silent
   9   (' of the', ' most')             KEEP     Tier-1 silent
  10   ('. In', ' the')                 KEEP     Tier-1 silent
  11   ('. They', ' are')               KEEP     Tier-1 silent
  12   (', and', ' it')                 KEEP     Tier-1 silent
  13   (', and', ' a')                  KEEP     Tier-1 silent
  14   ('. In', ' this')                KEEP     Tier-1 silent
  15   (' of the', ' following')        KEEP     Tier-1 silent
  --- vocab-budget threshold (K=15 + numerics is the empirical max) ---
  16   (', but', ' the')                DROP     each of pairs #16-20,
  17   ('. If', ' the')                 DROP     when added individually
  18   (', and', ' other')              DROP     to the K=15 baseline,
  19   (' can be', ' a')                DROP     displaces the `izards`
  20   (' to be', ' a')                 DROP     BPE merge → english -0.088
                                                 bpt (47→48 tokens). The
                                                 identity of the regressor
                                                 is "whichever pair you
                                                 add 16th"; the threshold
                                                 is the vocab budget, not
                                                 any specific pair.

Mechanism: each forced merge has an effective ~2× vocab cost. The
`apply_forced_merges_at_end` reservation consumes 1 slot for the carve-out
token; the regex carve-out also enables ~1 BPE-priority shift to the
punctuation-less variant (` but the`, ` If the`, ` and other`, ` be a`
become top-budget BPE merges once their punctuated forms are carved out).
Bracketing experiment confirmed: adding +5 vocab + 5 derived pairs does
NOT compensate (still regresses); +10 vocab does. See STATUS.md for the
full table.

Numeric DERIVED pairs (`("00","0")` and `(",","000")`) are retained as
hand-additions; they fail the function-word filter by design but recover
the `currency` task probe regression.
"""

import re
from pathlib import Path

# ---------------------------------------------------------------------------
# BLOCKED_PAIRS
#
# Prevent BPE from learning undesirable cross-boundary merges that the regex
# carve-out (FORCED_PAIRS_EXPR) makes structurally possible. These guard
# against accidental "letter + space" fragments showing up as tokens because
# certain pre-tok chunks now span what would normally be a boundary.
#
# Kept verbatim from the original hand-picked list — the carve-out shape
# hasn't fundamentally changed (still LHS=punctuation/function-word,
# RHS=word-like), so the same blocking rules apply.

BLOCKED_PAIRS = [
    (",", " "),
    (".", " "),
    ("d", " "),
    ("e", " "),
    ("f", " "),
    ("g", " "),
    ("h", " "),
    ("l", " "),
    ("m", " "),
    ("n", " "),
    ("o", " "),
    ("r", " "),
    ("s", " "),
    ("t", " "),
    ("y", " "),
]

# ---------------------------------------------------------------------------
# FORCED / DERIVED PAIRS
#
# Loaded from the two-pass mining artifact and curated per the Phase 3
# ablation result table above.

_ARTIFACT = Path(__file__).parent / "mined" / "forced_pairs_climbmix_20rg_twopass.py"
_ns: dict = {}
exec(compile(_ARTIFACT.read_text(), str(_ARTIFACT), "exec"), _ns)  # noqa: S102
_MINED_FORCED = [tuple(p) for p in _ns["FORCED_PAIRS"]]
_MINED_DERIVED_ALL = [tuple(p) for p in _ns["DERIVED_PAIRS"]]

# Curation: keep top-15 of the 20 mined DERIVED. Pairs #16-20 each
# individually displace a borderline BPE merge under the 32K vocab budget;
# see the rationale table in the module docstring.
_KEEP_DERIVED_K = 15
_MINED_DERIVED = _MINED_DERIVED_ALL[:_KEEP_DERIVED_K]

# Numeric DERIVED — hand-additions, recover the currency probe regression.
# Two-pass mining cannot surface these because they fail the function-word
# grammatical-role filter by design (the RHS is a digit, not a word).
_NUMERIC_DERIVED: list[tuple[str, str]] = [
    ("00", "0"),
    (",", "000"),
]

DERIVED_PAIRS = _MINED_DERIVED + _NUMERIC_DERIVED
FORCED_PAIRS = _MINED_FORCED + DERIVED_PAIRS

# ---------------------------------------------------------------------------
# Regex carve-out derivation
#
# The regex matches forced phrases as single pre-tok chunks. Derived pairs
# must come FIRST in the alternation (leftmost-first ⇒ longer derived match
# beats its base components). Trailing lookahead `(?=([^a-z]|$))` prevents
# matching e.g. ` of theory` as ` of the` + `ory`.

_FORCED_PAIRS_EXPR_BASE = "|".join(re.escape(a + b) for a, b in _MINED_FORCED)
_DERIVED_PAIRS_EXPR = "|".join(re.escape(a + b) for a, b in DERIVED_PAIRS)

FORCED_PAIRS_EXPR = (
    "(" + _DERIVED_PAIRS_EXPR + "|" + _FORCED_PAIRS_EXPR_BASE
    + ")(?=([^a-z]|$))"
)

# Full SPLIT_PATTERN. Differs from current Karpathy master in two ways:
#   1. FORCED_PAIRS_EXPR prepended — carves out cross-boundary phrases.
#   2. ` ?\p{N}{1,2}` (with optional leading space) — the space-prefix-digit
#      micro-optimization (+1.09–1.23% compression for ~220 token cost).
SPLIT_PATTERN = (
    FORCED_PAIRS_EXPR
    + r"""|'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+| ?\p{N}{1,2}"""
    r"""| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
)
