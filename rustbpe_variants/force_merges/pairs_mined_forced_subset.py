"""
FORCED-list subset for ablation. Loads the pass-1 mining artifact and
truncates to the top-K entries by frequency rank (env `MINED_TOP_K`,
default: all). Used by `tools/ablate_mined.py`. No DERIVED.
"""

import os
import re
from pathlib import Path

from rustbpe_variants.force_merges.pairs import BLOCKED_PAIRS

_MINED_ARTIFACT = (
    Path(__file__).parent / "mined" / "forced_pairs_climbmix_20rg.py"
)
_ns: dict = {}
exec(compile(_MINED_ARTIFACT.read_text(), str(_MINED_ARTIFACT), "exec"), _ns)  # noqa: S102
_ALL = [tuple(p) for p in _ns["FORCED_PAIRS"]]

_top_k = int(os.environ.get("MINED_TOP_K", len(_ALL)))
_top_k = max(0, min(_top_k, len(_ALL)))
FORCED_PAIRS = _ALL[:_top_k]

DERIVED_PAIRS: list[tuple[str, str]] = []

_FORCED_PAIRS_EXPR_BASE = "|".join(re.escape(a + b) for a, b in FORCED_PAIRS)

if FORCED_PAIRS:
    FORCED_PAIRS_EXPR = (
        "(" + _FORCED_PAIRS_EXPR_BASE + ")(?=([^a-z]|$))"
    )
    SPLIT_PATTERN = (
        FORCED_PAIRS_EXPR
        + r"""|'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+| ?\p{N}{1,2}"""
        r"""| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
    )
else:
    # K=0 corner case: no carve-out at all, baseline regex.
    FORCED_PAIRS_EXPR = ""
    SPLIT_PATTERN = (
        r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+| ?\p{N}{1,2}"""
        r"""| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
    )
