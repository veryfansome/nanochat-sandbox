"""
DERIVED-pair ablation variant of pairs_mined_v2.

Same as pairs_mined_v2 except the two-pass-mined DERIVED list is truncated
to the top-K entries via the DERIVED_TOP_K environment variable (default:
all). Used by tools/ablate_derived.py to attribute the v0→v2 `english`
regression to specific DERIVED additions.

  FORCED_PAIRS = mined-v0 (87 pass-1 pairs)
                 + (top-K of 20 two-pass DERIVED)
                 + 2 numeric DERIVED (always included; needed for currency)

K=0  ≡ pairs_mined_v1 (mined-v0 FORCED + numeric DERIVED only)
K=20 ≡ pairs_mined_v2 (full kitchen sink)

The 20 two-pass DERIVED entries are ordered by frequency rank in the
artifact at mined/forced_pairs_climbmix_20rg_twopass.py — see that file for
counts. Cumulative ablation walks K = 0, 5, 10, 15, 20.
"""

import os
import re
from pathlib import Path

from rustbpe_variants.force_merges.pairs import BLOCKED_PAIRS

_ARTIFACT = Path(__file__).parent / "mined" / "forced_pairs_climbmix_20rg_twopass.py"
_ns: dict = {}
exec(compile(_ARTIFACT.read_text(), str(_ARTIFACT), "exec"), _ns)  # noqa: S102
_MINED_FORCED = [tuple(p) for p in _ns["FORCED_PAIRS"]]
_MINED_DERIVED_ALL = [tuple(p) for p in _ns["DERIVED_PAIRS"]]

_NUMERIC_DERIVED: list[tuple[str, str]] = [
    ("00", "0"),
    (",", "000"),
]

_k = int(os.environ.get("DERIVED_TOP_K", len(_MINED_DERIVED_ALL)))
_k = max(0, min(_k, len(_MINED_DERIVED_ALL)))
_MINED_DERIVED = _MINED_DERIVED_ALL[:_k]

DERIVED_PAIRS = _MINED_DERIVED + _NUMERIC_DERIVED
FORCED_PAIRS = _MINED_FORCED + DERIVED_PAIRS

_FORCED_PAIRS_EXPR_BASE = "|".join(re.escape(a + b) for a, b in _MINED_FORCED)
_DERIVED_PAIRS_EXPR = "|".join(re.escape(a + b) for a, b in DERIVED_PAIRS)

if DERIVED_PAIRS:
    FORCED_PAIRS_EXPR = (
        "(" + _DERIVED_PAIRS_EXPR + "|" + _FORCED_PAIRS_EXPR_BASE
        + ")(?=([^a-z]|$))"
    )
else:
    FORCED_PAIRS_EXPR = (
        "(" + _FORCED_PAIRS_EXPR_BASE + ")(?=([^a-z]|$))"
    )

SPLIT_PATTERN = (
    FORCED_PAIRS_EXPR
    + r"""|'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+| ?\p{N}{1,2}"""
    r"""| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
)
