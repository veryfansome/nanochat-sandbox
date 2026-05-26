"""
FORCED_PAIRS / BLOCKED_PAIRS / DERIVED_PAIRS + derived SPLIT_PATTERN.

87 mined FORCED + 15 mined DERIVED + 2 numeric-hand-add DERIVED = 104
forced merges, vocab=32768. Compresses **−6.5%** vs baseline on a 15M-char
sample of climbmix prose, trading ~237 rare-content BPE merges for ~237
high-frequency phrasal merges.

Replaced the original ~70-pair hand-picked list on 2026-05-25 via
`tools/mine_forced_pairs.py --two-pass` + `tools/ablate_derived.py`. The
pair tuples below are frozen verbatim from that curation; the mining
artifacts at `mined/forced_pairs_climbmix_20rg{,_twopass}.py` are kept as
provenance, not imported. Full construction narrative, per-pair attribution
table, and the +5/+10/+20 vocab-cost bracketing experiment live in
[STATUS.md](../../STATUS.md) §force_merges.

To re-curate against a different corpus:
  1. regenerate the mining artifacts (`tools/mine_forced_pairs.py --two-pass`),
  2. re-run `tools/ablate_derived.py` to find the new K cutoff,
  3. paste the new mined tuples into the `_MINED_FORCED` / `_MINED_DERIVED`
     literals below (do not re-introduce runtime artifact loading —
     `pairs.py` is intentionally self-contained so it's grep-able and
     review-able as a flat list),
  4. re-validate the `_NUMERIC_DERIVED` hand-adds for the new corpus + probe
     battery. They were added here because the `currency` Tier-1 probe
     regressed by 1 token without them; a different corpus may need different
     numeric concats (or none).
"""

import re

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

# Pass-1 mined FORCED pairs (87 entries, ordered by corpus frequency).
# Regenerate via:
#   uv run python -m tools.mine_forced_pairs --two-pass --row-groups 20 \
#       --top-k 150 --min-count 200 --top-k-derived 20 --min-count-derived 100 \
#       --out-py rustbpe_variants/force_merges/mined/forced_pairs_climbmix_20rg_twopass.py
# (canonical defaults; ~20 row groups ≈ 60M chars from the climbmix train
# split; shadow-filtered drops removed by the miner before output).
_MINED_FORCED = [
    (' of', ' the'),
    (',', ' and'),
    (' in', ' the'),
    ('.', ' The'),
    (' to', ' the'),
    (',', ' the'),
    (' on', ' the'),
    (' is', ' a'),
    (' is', ' the'),
    (',', ' but'),
    ('.', ' It'),
    ('.', ' This'),
    (' can', ' be'),
    (' to', ' be'),
    (',', ' it'),
    (' for', ' the'),
    (' in', ' a'),
    (' of', ' a'),
    (' from', ' the'),
    (',', ' which'),
    (',', ' you'),
    (' with', ' the'),
    (' such', ' as'),
    (',', ' or'),
    ('.', ' In'),
    (' with', ' a'),
    (' at', ' the'),
    (',', ' a'),
    (' that', ' the'),
    ('.', ' If'),
    ('.', ' They'),
    ('.', ' I'),
    (' for', ' a'),
    (' to', ' a'),
    (' by', ' the'),
    (' have', ' a'),
    (',', ' we'),
    (' will', ' be'),
    ('.', ' You'),
    (',', ' they'),
    (' is', ' not'),
    ('.', ' A'),
    (' into', ' the'),
    (',', ' I'),
    (' on', ' a'),
    (' to', ' make'),
    ('.', ' These'),
    (':', ' The'),
    (' has', ' been'),
    ('.', ' We'),
    (' have', ' been'),
    (' may', ' be'),
    (' are', ' the'),
    ('.', ' For'),
    ('.', ' But'),
    (' that', ' you'),
    (' should', ' be'),
    (' to', ' get'),
    (' about', ' the'),
    ('.', ' As'),
    ('.', ' When'),
    (' more', ' than'),
    ('.', ' There'),
    (' all', ' the'),
    (' has', ' a'),
    ('.', ' So'),
    ('.', ' And'),
    (' through', ' the'),
    (' from', ' a'),
    (' can', ' also'),
    (' during', ' the'),
    (' for', ' your'),
    (' at', ' a'),
    (' on', ' your'),
    (' does', ' not'),
    (' over', ' the'),
    ('.', ' By'),
    (' by', ' a'),
    (' was', ' the'),
    ('.', ' To'),
    (' between', ' the'),
    ('.', ' He'),
    ('.', ' That'),
    (' into', ' a'),
    (' around', ' the'),
    ('.', ' With'),
    (' was', ' a'),
]

# Pass-2 mined DERIVED pairs (top-15 of 20). Pairs #16-20 (mined ranks
# 16-20: `(', but', ' the')`, `('. If', ' the')`, `(', and', ' other')`,
# `(' can be', ' a')`, `(' to be', ' a')`) each individually displace a
# borderline BPE merge (`izards`, ` acclim`, `ampton`, `uran`) under the
# 32K vocab budget — see STATUS.md §force_merges for the per-pair table.
_MINED_DERIVED = [
    (', and', ' the'),
    ('. If', ' you'),
    ('. It', ' is'),
    (', it', ' is'),
    ('. This', ' is'),
    (', but', ' it'),
    (', which', ' is'),
    ('. There', ' are'),
    (' of the', ' most'),
    ('. In', ' the'),
    ('. They', ' are'),
    (', and', ' it'),
    (', and', ' a'),
    ('. In', ' this'),
    (' of the', ' following'),
]

# Numeric DERIVED — hand-additions; fail the continuation-whitelist filter
# by design (RHS is numeric) but recover the `currency` task probe
# (count=20 vs 21 tokens).
_NUMERIC_DERIVED = [
    ('00', '0'),
    (',', '000'),
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
