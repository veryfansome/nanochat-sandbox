"""
Smoke test for wrappers/tok_train_blocked_morphemes.py.

Verifies:
  (a) The canonical block list loads and is structurally well-formed
      (list of 2-tuples of non-empty str).
  (b) rustbpe_force_merges imports and the sys.modules['rustbpe'] alias works
      before nanochat.tokenizer loads.
  (c) `train_from_iterator` accepts the full canonical blocked_pairs without
      error, AND the block mechanism works: a control token with ALL its
      byte-split paths blocked does NOT appear in the resulting vocab.
      (Single-path blocking alone does NOT guarantee absence — a token with
      multiple formation paths reforms via an unblocked split. That routing
      is the v3→v7 cascade and the reason all-path blocking exists; the
      control here blocks all paths to isolate the mechanism.)
  (d) Encode/decode roundtrip is intact.

No real data; runs in <2s.
"""
import os
import sys
import tempfile

with tempfile.TemporaryDirectory() as tmp:
    os.environ["VARIANT_BASE"] = tmp
    os.environ["NANOCHAT_BASE_DIR"] = tmp

    # (a) canonical block list is well-formed
    from rustbpe_variants.blocked_morphemes.blocks import (
        BLOCKED_PAIRS,
        RECOMMENDED_VOCAB_SIZE,
    )
    assert isinstance(BLOCKED_PAIRS, list) and BLOCKED_PAIRS, "BLOCKED_PAIRS empty/not list"
    for i, pair in enumerate(BLOCKED_PAIRS):
        assert isinstance(pair, tuple) and len(pair) == 2, f"entry {i} not a 2-tuple: {pair!r}"
        L, R = pair
        assert isinstance(L, str) and isinstance(R, str), f"entry {i} not (str, str): {pair!r}"
        assert L and R, f"entry {i} has empty operand: {pair!r}"
    assert RECOMMENDED_VOCAB_SIZE == 32768, f"expected no vocab bump, got {RECOMMENDED_VOCAB_SIZE}"
    print(f"[a] BLOCKED_PAIRS: {len(BLOCKED_PAIRS)} well-formed (str,str) pairs; vocab={RECOMMENDED_VOCAB_SIZE}")

    # (b) variant module + alias
    import rustbpe_force_merges
    assert hasattr(rustbpe_force_merges, "Tokenizer")
    from wrappers._tokenizer_variant_common import alias_rustbpe
    alias_rustbpe("rustbpe_force_merges")
    import rustbpe
    assert rustbpe is rustbpe_force_merges, "alias_rustbpe didn't swap sys.modules"
    print(f"[b] sys.modules['rustbpe'] -> rustbpe_force_merges")

    # (c) Train with the full canonical list (must be accepted by rustbpe)
    # PLUS an all-path block for a control token, to isolate the mechanism.
    # Single-path blocking can't guarantee absence (multi-path reformation),
    # so the control blocks every byte-split of "ished" to prove that when
    # all holes are plugged, the token genuinely dies.
    control = "ished"
    control_all_paths = [(control[:i], control[i:]) for i in range(1, len(control))]
    concat = control.encode("utf-8")

    # Corpus heavy in the control byte sequence so greedy BPE would WANT to
    # merge it absent the block.
    synthetic = [
        f"finished published established vanished punished {control} {control} {control}. " * 40,
        f"the {control} text was {control} and {control} again and again. " * 40,
    ]
    tok = rustbpe.Tokenizer()
    tok.train_from_iterator(
        iter(synthetic),
        vocab_size=512,
        pattern=r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+""",
        forced_pairs=[],
        blocked_pairs=BLOCKED_PAIRS + control_all_paths,
    )
    mergeable = dict((bytes(k), v) for k, v in tok.get_mergeable_ranks())
    assert concat not in mergeable, (
        f"BLOCK FAILED: control concat {concat!r} appeared in vocab despite "
        f"blocking all {len(control_all_paths)} byte-split paths"
    )
    print(
        f"[c] trained {len(mergeable)} merges with {len(BLOCKED_PAIRS)} canonical "
        f"+ {len(control_all_paths)} control blocks; all-path-blocked {concat!r} correctly absent"
    )

    # (d) roundtrip
    import tiktoken
    enc = tiktoken.Encoding(
        name="smoke", pat_str=tok.get_pattern(),
        mergeable_ranks=mergeable, special_tokens={},
    )
    test = f"the {control} document was {control}."
    ids = enc.encode_ordinary(test)
    assert enc.decode(ids) == test, f"roundtrip failed: {enc.decode(ids)!r} != {test!r}"
    print(f"[d] roundtrip OK ({len(ids)} tokens for {test!r})")

print("\nall smoke checks passed")
