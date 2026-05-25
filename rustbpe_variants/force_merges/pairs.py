"""
FORCED_PAIRS / BLOCKED_PAIRS / DERIVED_PAIRS + derived SPLIT_PATTERN.

87 mined FORCED + 15 mined DERIVED + 2 numeric-hand-add DERIVED = 104
forced merges, vocab=32768. Compresses **−6.5%** vs baseline on a 15M-char
sample of climbmix prose, trading ~237 rare-content BPE merges for ~237
high-frequency phrasal merges.

Replaced the original ~70-pair hand-picked list on 2026-05-25 via
`tools/mine_forced_pairs.py --two-pass` + `tools/ablate_derived.py`. Full
construction narrative, per-pair attribution table (which 5 of the 20
mined DERIVED were dropped and why), and the +5/+10/+20 vocab-cost
bracketing experiment live in [STATUS.md](../../STATUS.md) §force_merges.

To re-curate against a different corpus: regenerate the mining artifacts
at `mined/`, re-run `ablate_derived`, update `_KEEP_DERIVED_K` below.
"""

import re
from pathlib import Path

# BLOCKED_PAIRS guard against "letter + space" fragments that the carve-out
# makes structurally possible (kept verbatim from the hand-picked list — the
# carve-out shape is unchanged: LHS=punctuation/function-word, RHS=word-like).
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

# Mined artifact (two-pass): FORCED_PAIRS = 87 pass-1 phrases,
# DERIVED_PAIRS = 20 pass-2 phrases whose LHS is a pass-1 concatenation.
_ARTIFACT = Path(__file__).parent / "mined" / "forced_pairs_climbmix_20rg_twopass.py"
_ns: dict = {}
exec(compile(_ARTIFACT.read_text(), str(_ARTIFACT), "exec"), _ns)  # noqa: S102
_MINED_FORCED = [tuple(p) for p in _ns["FORCED_PAIRS"]]
_MINED_DERIVED_ALL = [tuple(p) for p in _ns["DERIVED_PAIRS"]]

# Curation cutoff. Pairs #16-20 each individually displace a borderline
# BPE merge (`izards`, ` acclim`, `ampton`, `uran`) under the 32K vocab
# budget — adding any 16th derived pair regresses `english` by 0.088 bpt.
# See STATUS.md §force_merges for the per-pair attribution table.
_KEEP_DERIVED_K = 15
_MINED_DERIVED = _MINED_DERIVED_ALL[:_KEEP_DERIVED_K]

# Numeric DERIVED — hand-additions; fail the function-word filter (RHS is
# numeric) but recover the `currency` task probe (count=20 vs 21 tokens).
_NUMERIC_DERIVED: list[tuple[str, str]] = [
    ("00", "0"),
    (",", "000"),
]

DERIVED_PAIRS = _MINED_DERIVED + _NUMERIC_DERIVED
FORCED_PAIRS = _MINED_FORCED + DERIVED_PAIRS

# Carve-out regex: derived pairs first in the alternation (leftmost-first ⇒
# longer match wins). Trailing lookahead prevents ` of theory` matching as
# ` of the` + `ory`.
_FORCED_PAIRS_EXPR_BASE = "|".join(re.escape(a + b) for a, b in _MINED_FORCED)
_DERIVED_PAIRS_EXPR = "|".join(re.escape(a + b) for a, b in DERIVED_PAIRS)
FORCED_PAIRS_EXPR = (
    "(" + _DERIVED_PAIRS_EXPR + "|" + _FORCED_PAIRS_EXPR_BASE
    + ")(?=([^a-z]|$))"
)

# SPLIT_PATTERN = carve-out prepended to nanochat/tokenizer.py:30, plus the
# ` ?\p{N}{1,2}` space-prefix-digit tweak (+1.09–1.23% compression, ~220
# token cost).
SPLIT_PATTERN = (
    FORCED_PAIRS_EXPR
    + r"""|'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+| ?\p{N}{1,2}"""
    r"""| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
)
