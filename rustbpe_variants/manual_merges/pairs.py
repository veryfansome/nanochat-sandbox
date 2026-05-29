"""
MANUAL_PAIRS — hand-curated merges, applied FIRST, for the manual_merges tokenizer.

Hand-written counterpart to seed_tokens (same rank-insertion engine). For each
`(L, R)`, the concat `S = L + R` is inserted into `mergeable_ranks` at its
natural-firing rank `max(id_L, id_R) + 1` (clamped to 256). For byte-bigram
pairs (single-char operands), that clamp lands them at **rank 256 — the first
merges**, so they win the merge race wherever their operands are adjacent.
That's the intent here: the merges are done *first* (highest priority), unlike
`force_merges`, whose `apply_forced_merges_at_end` appends at the lowest
priority (right only when you must force a pair that otherwise can't be
mergeable, paired with a regex carve-out).

Properties:
  - **Pairs are hand-written** here, not mined.
  - **Standard pre-tokenization regex** (no carve-out), via pristine `rustbpe`.
    So every pair must be one the standard pre-tokenizer keeps *within a single
    chunk* — a within-word / sub-word composition. For a pair with multi-char
    operands (e.g. `("educ","ation")`), "first" means right after its operands
    (`max(id_L,id_R)+1`), not literally rank 256, since the operands must form
    before the merge can fire. Cross-boundary phrase pairs (` of`, ` the`)
    cannot fire here — those belong in `force_merges`.
  - If `S` already exists naturally, it is PROMOTED to its insertion rank (moved
    to the front), not skipped — the point is priority.

Fill `MANUAL_PAIRS` below.
"""

MANUAL_PAIRS: list[tuple[str, str]] = [
    # (L, R) pairs to force-merge — within-chunk / sub-word only.
    # e.g. ("educ", "ation"), ("un", "happy"), ...
    ("e", "i"),
    ("i", "e"),
    ("n", "g"),
    ("q", "u"),
]

# Slot reservation is disabled (see tok_train_manual_merges.py): natural BPE
# trains to this full vocab size, and seeds are added on top. So the FINAL
# mergeable vocab = RECOMMENDED_VOCAB_SIZE + (count of genuinely-new seeds);
# promoted-existing seeds don't change the count. `tok_train_manual_merges.py`
# reads this.
RECOMMENDED_VOCAB_SIZE = 32768
