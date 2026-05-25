"""
Smoke test for `wrappers/_patches.py` (used by `wrappers/base_eval.py` and
`wrappers/base_train.py`). Pure function test — no model / data, <1s.

Pins five behaviors:
  (a) Prefix-stable tokenization → `common_len == len(tokens_without)`.
      No semantic shift.
  (b) Non-prefix-stable tokenization → `common_len < len(tokens_without)`.
      The loss window EXPANDS LEFT past the original string boundary. This
      is the **known and accepted** semantic shift documented in
      `wrappers/_patches.py`; assertion is pinned so future "fixes" don't
      silently make this an equality check.
  (c) Boundary-crossing counter increments correctly across both cases.
  (d) String-prefix invariant still rejects mismatched prompt pairs.
  (e) `aggregate_boundary_stats` returns local counters + rank=0 when
      dist isn't initialized (single-process / smoke path). The dist-
      initialized path can't be smoked here; it's exercised end-to-end by
      torchrun-invoked wrappers.base_eval.
"""

from wrappers._patches import (
    _patched_batch_sequences_lm,
    aggregate_boundary_stats,
    get_boundary_stats,
    reset_boundary_stats,
)


class _FakeTokenizer:
    """Injects per-string tokenizations so the test can pose the
    non-prefix-stable case without a real tokenizer."""
    def __init__(self, table):
        self.table = table
    def __call__(self, prompts, prepend=None):
        return [([prepend] if prepend is not None else []) + list(self.table[p])
                for p in prompts]
    def get_bos_token_id(self):
        return 999


reset_boundary_stats()

# (a) Prefix-stable: tokens_without is a token-level prefix of tokens_with.
tok_stable = _FakeTokenizer({
    "The cat sat":            [1, 2, 3],
    "The cat sat on the mat": [1, 2, 3, 4, 5, 6],
})
tokens, starts, ends = _patched_batch_sequences_lm(
    tok_stable, ["The cat sat", "The cat sat on the mat"],
)
without_len = 4  # 1 bos + 3 ids
assert starts == [without_len] and ends == [7] and tokens == [[999, 1, 2, 3, 4, 5, 6]]
print(f"[a] prefix-stable: start={starts}, end={ends} — common_len == len(tokens_without)")

# (b) Non-prefix-stable: force_merges-style divergence. ' on' (id 4) collapses
# into ' on the' (id 40) when the carve-out lookahead fires in the longer prompt.
tok_unstable = _FakeTokenizer({
    "The cat sat on":         [1, 2, 3, 4],
    "The cat sat on the mat": [1, 2, 3, 40, 5],
})
tokens, starts, ends = _patched_batch_sequences_lm(
    tok_unstable, ["The cat sat on", "The cat sat on the mat"],
)
without_len = 5  # 1 bos + 4 ids
# Upstream would assert-crash here. Patch produces common_len=4, end=6 —
# scores tokens [4, 6) = [' on the', ' mat'], expanding the window 1 token
# LEFT of the prompt/continuation byte boundary. Accepted semantic shift.
assert starts == [4] and ends == [6]
assert starts[0] < without_len, (
    f"loss window should expand left of len(tokens_without)={without_len}; "
    f"this is the accepted semantic shift pinned by this assertion"
)
print(f"[b] non-prefix-stable: start={starts}, end={ends} — common_len ({starts[0]}) "
      f"< len(tokens_without) ({without_len}); window expanded {without_len - starts[0]} "
      f"token(s) left")

# (c) Counter: 2 total calls, 1 crossed.
stats = get_boundary_stats()
assert stats == {"n_total": 2, "n_boundary_crossing": 1}, f"unexpected stats: {stats}"
print(f"[c] boundary stats: {stats}")

# (d) String-prefix invariant catches unrelated-prompt pairs.
try:
    _patched_batch_sequences_lm(tok_stable, ["The cat sat", "A dog ran far"])
    raise AssertionError("expected AssertionError on string-prefix violation")
except AssertionError as e:
    assert "string prefix" in str(e), f"unexpected message: {e}"
print(f"[d] string-prefix invariant rejects unrelated prompts")

# (e) Single-process aggregate path returns local counters with rank=0.
# (Multi-rank aggregation requires a live torch.distributed group; not
# something we can synthesize in a single-process smoke. End-to-end coverage
# happens via runs/speedrun.sh under torchrun.)
agg_stats, agg_rank = aggregate_boundary_stats()
assert agg_rank == 0, f"single-process aggregate: expected rank=0, got {agg_rank}"
assert agg_stats == get_boundary_stats(), (
    f"single-process aggregate should equal local stats, got "
    f"{agg_stats} vs {get_boundary_stats()}"
)
print(f"[e] aggregate_boundary_stats (single-process): rank={agg_rank}, stats={agg_stats}")

print("\nall smoke checks passed")
