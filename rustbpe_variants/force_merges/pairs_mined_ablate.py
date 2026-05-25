"""
Arbitrary-subset DERIVED variant for single-pair attribution. Env
`INCLUDE_INDICES` = comma-separated 0-based indices into the 20 mined
DERIVED list; only those are included (all 87 mined FORCED + 2 numeric
DERIVED always present). Used when cumulative ablation
(pairs_mined_derived_subset.py) narrows the regressor to a bucket and you
need to isolate which specific pair(s) within it are responsible.

Examples:
  INCLUDE_INDICES="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,16"  → K=15 + pair #16
  INCLUDE_INDICES=""                                       → no mined DERIVED
  INCLUDE_INDICES unset                                    → all 20
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

_raw = os.environ.get("INCLUDE_INDICES")
if _raw is None:
    _indices = list(range(len(_MINED_DERIVED_ALL)))
elif _raw == "":
    _indices = []
else:
    _indices = sorted({int(x) for x in _raw.split(",") if x.strip()})

_MINED_DERIVED = [_MINED_DERIVED_ALL[i] for i in _indices]

_NUMERIC_DERIVED: list[tuple[str, str]] = [
    ("00", "0"),
    (",", "000"),
]

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
    FORCED_PAIRS_EXPR = "(" + _FORCED_PAIRS_EXPR_BASE + ")(?=([^a-z]|$))"

SPLIT_PATTERN = (
    FORCED_PAIRS_EXPR
    + r"""|'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+| ?\p{N}{1,2}"""
    r"""| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
)
