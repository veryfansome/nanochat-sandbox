"""
FORCED_PAIRS / BLOCKED_PAIRS / DERIVED_PAIRS + derived SPLIT_PATTERN.

105 mined FORCED + 15 mined DERIVED + 2 numeric DERIVED = 122 forced
merges, vocab=32890 (32768 nanochat default + 122, fully ADDITIVE — the forced
phrases are bolted on rather than displacing borderline base merges). The old +20
bracket at 32788 is retired: block_leading_space frees the natural-bigram slots,
so no displacement bracketing is needed. See RECOMMENDED_VOCAB_SIZE below + the
BLOCKED_PAIRS note, and STATUS.md §force_merges curation details.

## History

- **2026-05-25**: Replaced the ~70-pair hand-picked list (lifted from
  `veryfansome/nanochat@force_merges_wip`) with a data-driven curation via
  `tools/mine_forced_pairs.py --two-pass` + `tools/ablate_derived.py`.
  104 forced merges at vocab=32768. Speedrun result: **CORE +5.81% vs
  `d24_baseline`, val/bpb tied (slight edge), ChatCORE +0.62%** ($115.76,
  wandb `7aesostl` vs `qmsi105c`).
- **2026-05-26**: Re-curated after Codex reviews surfaced two `detect_shadows`
  bugs (case-2 "regex precedence" overflag + chained false-shadowing).
  Fixed rule produces +18 pass-1 entries (` one of`, ` that is`, ` have to`,
  ` However,`, etc.) + 4 swapped pass-2 entries. Vocab-bracketed to +20.
  Locally Tier-1 parity with the 2026-05-25 version + 0.42% better
  real-corpus compression. Lambda-pending.

The 122 tuples below are frozen verbatim from the 2026-05-26 curation.
Provenance: `mined/forced_pairs_climbmix_20rg_twopass.py`. Historical
lineage preserved at `mined/forced_pairs_climbmix_20rg.py` (single-pass,
pre-shadow-fix) and `mined/forced_pairs_climbmix_20rg_twopass_pre_shadow_fix.py`
(2026-05-25's two-pass output, source for the speedrun tokenizer).

To re-curate against a different corpus (the mining/ablation tooling —
`tools/mine_forced_pairs.py`, `tools/ablate_derived.py` — was removed 2026-06-05;
restore it from git history first, see STATUS.md §force_merges):
  1. Re-mine (`tools/mine_forced_pairs.py --two-pass`).
  2. Re-ablate K (`tools/ablate_derived.py`).
  3. Paste new tuples into `_MINED_FORCED` / `_MINED_DERIVED` below
     (intentionally a flat literal list, not a runtime loader).
  4. Re-validate `_NUMERIC_DERIVED` — added for `currency` probe; a
     different corpus may need different (or no) numeric concats.
  5. Set `RECOMMENDED_VOCAB_SIZE = 32768 + len(FORCED_PAIRS)` (additive; the old
     +20 displacement bracket no longer applies — see the summary above).
"""

import re

# BLOCKED_PAIRS: empty. The forced-phrase cannibalizers — inner bigrams (' It'+' is')
# AND shared prefixes ('.'+' T') — are now killed wholesale by the crate's
# block_leading_space=True predicate (bans any base merge whose right operand STARTS
# with a space and whose left operand isn't all whitespace), the mirror of
# block_trailing_space. Both predicates are passed by wrappers/tok_train_force_merges.py
# (see src/lib.rs), so no enumerated pair list is needed. (Multi-word phrases come only
# from the injected forced list, so no legitimate base merge has a space-leading R —
# except pure-whitespace runs, which the all-whitespace-L guard preserves.)
BLOCKED_PAIRS = []

# Pass-1 mined FORCED pairs (105 entries, ordered by corpus frequency).
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
    (',', ' as'),
    (' will', ' be'),
    ('.', ' You'),
    (',', ' they'),
    (' one', ' of'),
    (' is', ' not'),
    ('.', ' A'),
    (' into', ' the'),
    (',', ' I'),
    (' on', ' a'),
    (' to', ' make'),
    ('.', ' These'),
    (':', ' The'),
    (' that', ' is'),
    (' has', ' been'),
    ('.', ' We'),
    (' have', ' been'),
    (' may', ' be'),
    (' are', ' the'),
    (' However', ','),
    ('.', ' However'),
    ('.', ' For'),
    ('.', ' But'),
    (' that', ' you'),
    (' should', ' be'),
    (' that', ' are'),
    (' to', ' get'),
    (' in', ' your'),
    (' about', ' the'),
    (' to', ' do'),
    (' have', ' to'),
    ('.', ' As'),
    ('.', ' When'),
    (' more', ' than'),
    ('.', ' There'),
    (' all', ' the'),
    (' has', ' a'),
    ('.', ' So'),
    ('.', ' And'),
    (' through', ' the'),
    (' would', ' be'),
    (' in', ' this'),
    (' from', ' a'),
    (' that', ' it'),
    (' can', ' also'),
    (' during', ' the'),
    (' for', ' your'),
    (' that', ' they'),
    (' at', ' a'),
    (' on', ' your'),
    (' does', ' not'),
    (' some', ' of'),
    (' over', ' the'),
    ('.', ' By'),
    (' by', ' a'),
    (' was', ' the'),
    (' could', ' be'),
    (' in', ' their'),
    ('.', ' To'),
    (' between', ' the'),
    (' have', ' the'),
    ('.', ' He'),
    ('.', ' That'),
    (' into', ' a'),
    (' around', ' the'),
    ('.', ' With'),
    (' in', ' an'),
    (' was', ' a'),
]

# Pass-2 mined DERIVED pairs (top-15 of 20 produced by the two-pass miner).
# Pairs #16-20 from the same mining run (`(', and', ' a')`, `('. In', ' this')`,
# `(', but', ' the')`, `(' of the', ' following')`, `('. If', ' the')`) each
# individually push the BPE budget past the `izards`-class borderline — see
# STATUS.md §force_merges curation details for the per-pair table.
_MINED_DERIVED = [
    (' one of', ' the'),
    (', and', ' the'),
    ('. If', ' you'),
    ('. It', ' is'),
    (', it', ' is'),
    ('. However', ','),
    ('. This', ' is'),
    (', but', ' it'),
    (', which', ' is'),
    (' some of', ' the'),
    ('. There', ' are'),
    (', as', ' well'),
    ('. In', ' the'),
    ('. They', ' are'),
    (', and', ' it'),
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

# Recommended vocab_size when training this tokenizer. 32890 = 32768 (nanochat
# default) + 122 forced phrases, fully ADDITIVE — the forced merges are bolted onto
# the normal vocab rather than displacing borderline base merges (the old +20 bracket
# at 32788). This keeps force_merges' base-merge budget identical to baseline AND to
# the auto_tune audit (which also runs at 32890), so cross-tokenizer comparisons are
# apples-to-apples. With block_leading_space freeing the natural-bigram slots, the old
# displacement bracketing no longer applies. `tok_train_force_merges.py` reads this.
RECOMMENDED_VOCAB_SIZE = 32890
