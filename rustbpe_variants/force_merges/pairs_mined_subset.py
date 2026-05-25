"""
Cumulative-subset variant of pairs_mined.py for incremental ablation.

Loads the same mined FORCED_PAIRS artifact as pairs_mined.py but truncates
to the top-K entries by frequency rank. K is read from the MINED_TOP_K
environment variable at import time (default: all entries).

Used by tools/ablate_mined.py to train a sequence of tokenizers at K ∈
{25, 50, 75, 87} and probe where (if anywhere) adding more mined-only
candidates causes a Tier-1 regression. The intent mirrors the user's
original incremental hand-curation workflow: add candidates in batches,
verify each step.

DERIVED_PAIRS is empty (single-pass) — ablation is on the FORCED list only.
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
