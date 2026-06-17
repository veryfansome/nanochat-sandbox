"""
auto_tune — block list for the auto_tune tokenizer overlay.

Pairs (L, R) that must NOT merge during BPE training (rustbpe_force_merges'
`blocked_pairs=`): the trainer drops any such merge candidate, so words route
through the cleaner [stem][suffix] split. Blocking only — no forced merges, no
vocab bump.
"""

# BLOCKED_PAIRS — the committed block list that wrappers/tok_train_auto_tune.py bakes
# (override per-run with the AUTO_TUNE_BLOCKED_PAIRS_JSON env var). INTENTIONALLY EMPTY:
# the list is between generations, so we don't ship a stale one.
#   - The original 96 pairs (single-shard build-up, rounds 1-3) were SUPERSEDED by the
#     held-out-Pareto fixpoint (tools/heldout_fixpoint_cached.py).
#   - The current downstream-tested list is that fixpoint's r194 set (129 pairs) — it
#     got the d24 A/B (CORE +5.59%, val/bpb tied; single seed) baked via the env
#     override, NOT from here. Recover it from the canonical comptol6 run
#     (audit/checkpoint.json + audit/ledger.md) by journal-replaying to round 194 (the
#     `--upto 194` rule).
#   - A fresh re-tune INSIDE force_merges' pretokenization (carve-out + 122 forced
#     phrases + block_leading_space, vocab 32890) is in progress at
#     audit/heldout_fixpoint_fm/ (`heldout_fixpoint_cached --fm-context`).
# Promote the chosen held-out-vetted prefix here once a 2nd seed confirms (STATUS.md
# §next steps). Until then this stays empty (a no-block bake == the pristine baseline).
BLOCKED_PAIRS: list[tuple[str, str]] = []

# Blocking only removes merge candidates; freed slots get reallocated by BPE.
RECOMMENDED_VOCAB_SIZE = 32768
