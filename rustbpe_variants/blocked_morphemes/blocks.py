"""
Canonical BLOCKED_PAIRS list for the blocked_morphemes tokenizer.

Built by successive single-path mining passes, ALL using the full suffix
inventory in tools/find_mangled_morphemes.py:

  - PASS_1 mines the pristine baseline tokenizer once, comprehensively.
  - Each later pass re-mines the PRIOR pass's retrained tokenizer (excluding
    all earlier passes), catching (a) tokens that reformed via an alternate
    byte-split after their primary path was blocked, and (b) newly-emerged
    mangled tokens. The loop converges when re-mining returns ~0.

Why multiple passes: single-path blocking opens reformation paths that don't
exist in the baseline tokenizer, so a single comprehensive pass leaves a tail
of survivors (empirically ~74) that only become visible after retraining.
The passes chase that tail down to zero.

History: this sequence was restarted from a single comprehensive full-inventory
PASS_1 after an earlier build had accreted passes while the suffix inventory was
still being expanded (messy lineage). An experimental head-family pass —
blocking the high-frequency short-stem mangles (`ures`=ur+es, `ication`=ic+ation)
via `--sibling-min-stem-len 3` — was tried and rolled back as net-negative
(compression cost ~doubled, dead tokens up, cascade spawned more mangled, no
concentration gain). See git history.

The wrapper wrappers/tok_train_blocked_morphemes.py reads BLOCKED_PAIRS and
passes it to rustbpe_force_merges.train_from_iterator.
"""

# PASS_1: comprehensive mine of the pristine baseline (full suffix inventory).
from rustbpe_variants.blocked_morphemes.mined.mangled_climbmix_pass1 import (
    BLOCKED_PAIRS as PASS_1,
)
# PASS_2: re-mined from the PASS_1-trained tokenizer, excluding PASS_1 —
# survivors (reformed via alternate splits) + newly-emerged mangled.
from rustbpe_variants.blocked_morphemes.mined.mangled_climbmix_pass2 import (
    BLOCKED_PAIRS as PASS_2,
)
# PASS_3: re-mined from the PASS_1+2 tokenizer, excluding both.
from rustbpe_variants.blocked_morphemes.mined.mangled_climbmix_pass3 import (
    BLOCKED_PAIRS as PASS_3,
)

BLOCKED_PAIRS = PASS_1 + PASS_2 + PASS_3

# vocab size stays at nanochat default — blocked merges get reallocated by BPE
# to alternative merges, no slot growth needed.
RECOMMENDED_VOCAB_SIZE = 32768
