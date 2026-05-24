"""
Force-merges Python data: blocked pairs, forced pairs, derived pairs, and the
derived SPLIT_PATTERN with regex carve-out.

Lifted verbatim from veryfansome/nanochat@force_merges_wip:nanochat/tokenizer.py.
The lists are kept here (next to the Rust crate they pair with) rather than in
the wrapper, so the wrapper stays minimal and these data assets are
version-controlled with the variant they support.

The crucial insight from the prior work: forced cross-boundary merges only work
if pre-tokenization also treats the phrase as a single chunk. The
FORCED_PAIRS_EXPR alternation, prepended to SPLIT_PATTERN, guarantees that
tiktoken at inference sees " of the" as one pre-tok chunk and applies the
forced merge consistently. The trailing lookahead `(?=([^a-z]|$))` prevents
matching ` of theory` as ` of the` + `ory`.

Updating this file is the primary lever for tuning the variant — add/remove
pairs, reorder for alternation priority (leftmost-first in the regex), or
adjust BLOCKED_PAIRS to suppress accidental merges the carve-out exposes.
"""

import re

BLOCKED_PAIRS = [
    # May not be needed. But, in theory, since we create holes in the pre-tokenization regex for certain
    # cross-boundary tokens, we should prevent BPE from accidentally learning certain undesirable merges.
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

# Forced merges: when BOTH tokens exist in the merge DAG, add a merge (left,right)->(left||right).
# NOTE: Considerations for sorting:
# - Frequency of occurrence in text, since the order here reflects in the pre-tokenization regex
# - Special cases, as noted by comments
FORCED_PAIRS = [
    (" of", " the"),
    (",", " and"),
    (",", " in"),  # Moved from after (".", " They") to before (" in", " the") and (" in", " a")
    (" in", " the"),
    (".", " The"),
    (",", " the"),
    (" to", " the"),
    (" on", " the"),
    (" and", " the"),
    (" to", " be"),
    (",", " but"),
    (" is", " an"),  # Moved from after (" during", " the") to before (" is", " a")
    (" is", " a"),
    (" for", " the"),
    (" from", " the"),
    (".", " In"),
    (".", " It"),
    (".", " This"),
    (" of", " a"),
    (",", " with"),  # Moved from after (" and", " a") to before (" with", " the") and (" with", " a")
    (" with", " the"),
    (",", " which"),
    (" in", " a"),
    (" by", " the"),
    (" can", " be"),
    (" at", " the"),
    (",", " a"),
    (" is", " the"),
    (" that", " the"),
    (" as", " a"),
    (",", " it"),
    (",", " or"),
    #(" it", " is"),  # Conflicts with pairs that begins with " is"
    (" such", " as"),
    (",", " as"),
    (" as", " the"),
    (" with", " a"),
    #(" the", " same"),  # Conflicts with pairs that ends with " the"
    (".", " A"),
    (".", " They"),
    (" to", " a"),
    (" have", " been"),
    #(" one", " of"),  # Conflicts with pairs that begins with " of"
    (" will", " be"),
    (" has", " been"),
    #(" the", " first"),  # Conflicts with pairs that ends with " the"
    (".", " If"),
    (" for", " a"),
    (",", " you"),
    (" is", " not"),
    #(" the", " most"),  # Conflicts with pairs that ends with " the"
    (",", " we"),
    #(" the", " world"),  # Conflicts with pairs that ends with " the"
    (" may", " be"),
    (",", " they"),
    (" as", " well"),
    (".", " For"),
    (" into", " the"),
    (".", " But"),
    #(" It", " is"),  # Conflicts with (".", " It") and pairs that begins with " is"
    (".", " He"),
    (" and", " a"),
    #(" part", " of"),  # Conflicts with pairs that begins with " of"
    ("00", "0"),
    (" of", " this"),
    (" on", " a"),
    #(" number", " of"),  # Conflicts with pairs that begin with " of"
    #(" they", " are"),  # Conflicts with (",", " they")
    (" have", " a"),
    #(" you", " can"),  # Conflicts with (",", " you") and (" can", " be")
    (" more", " than"),
    #(" need", " to"),  # Conflicts with pairs that ends with " to"
    (".", " We"),
    (" about", " the"),
    (".", " These"),
    (" However", ","),
    (" to", " make"),
    (".", " As"),
    (" of", " these"),
    (" should", " be"),
    (",", " including"),
    (",", " he"),
    (" of", " their"),
    #(" that", " is"),
    (",", " there"),
    (" during", " the"),
    (",", " who"),
    (".", " There"),
    (".", " You"),
    (" through", " the"),
    (" was", " a"),
    (" by", " a"),
    (" would", " be"),
    (" all", " the"),
    (" are", " the"),
    (" are", " not"),
]
_FORCED_PAIRS_EXPR_BASE = "|".join([re.escape(a + b) for a, b in FORCED_PAIRS])

# Derived pairs, i.e. one or both members are the results of another forced merge.
DERIVED_PAIRS = [
    (", in", " the"),
    (", and", " the"),
    (", with", " the"),
    (" one", " of the"),
    (",", "000"),
    (".", " However,"),
    (",", " such as"),
    (", in", " a"),
    (", and", " a"),
    (", with", " a"),
]
FORCED_PAIRS = FORCED_PAIRS + DERIVED_PAIRS  # Derived from previous merges so BPE merge must come after
_DERIVED_PAIRS_EXPR = "|".join([re.escape(a + b) for a, b in DERIVED_PAIRS])

# Regex for derived pairs must match first (alternation is leftmost-first).
FORCED_PAIRS_EXPR = "(" + _DERIVED_PAIRS_EXPR + "|" + _FORCED_PAIRS_EXPR_BASE + ")(?=([^a-z]|$))"

# Full SPLIT_PATTERN used by this variant. Differs from current Karpathy master in two ways:
#   1. FORCED_PAIRS_EXPR prepended — carves out cross-boundary phrases.
#   2. ` ?\p{N}{1,2}` (with optional leading space) — the space-prefix-digit micro-optimization.
# Note: this variant subsumes the space_digits variant; running both wrappers in
# sequence is redundant (and would conflict, since both patch SPLIT_PATTERN).
SPLIT_PATTERN = FORCED_PAIRS_EXPR + r"""|'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+| ?\p{N}{1,2}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
