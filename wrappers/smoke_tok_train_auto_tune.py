"""
Smoke test for wrappers/tok_train_auto_tune.py.

Verifies:
  (a) BLOCKED_PAIRS loads and is well-formed (list of (str,str)); auto_tune
      forces nothing (no MANUAL_PAIRS).
  (b) The shim installs (rustbpe_auto_tune crate, for blocked_pairs support)
      and trains under the PRISTINE pattern (no delimiter-glue tweak).
  (c) BLOCKING: a synthetic blocked pair ('x','y') does NOT merge — 'xy' is absent
      from the vocab even though it's frequent in the corpus.
  (d) SPLIT_PATTERN is pristine: opening delimiters STILL glue to a following
      word (no delimiter-glue carve-out), and that flows through to the saved
      pattern.
  (e) config-portability: encode/decode roundtrip (no overlay-only fields leak).

No real data; runs in <2s.
"""
import os
import tempfile

with tempfile.TemporaryDirectory() as tmp:
    os.environ["VARIANT_BASE"] = tmp
    os.environ["NANOCHAT_BASE_DIR"] = tmp

    # (a) block list well-formed; auto_tune forces nothing
    from rustbpe_variants.auto_tune import pairs as P
    assert isinstance(P.BLOCKED_PAIRS, list), "BLOCKED_PAIRS must be a list"
    for i, pr in enumerate(P.BLOCKED_PAIRS):
        assert isinstance(pr, tuple) and len(pr) == 2, f"BLOCKED_PAIRS[{i}] not a 2-tuple: {pr!r}"
        L, R = pr
        assert isinstance(L, str) and isinstance(R, str) and L and R, f"BLOCKED_PAIRS[{i}] bad: {pr!r}"
    assert not hasattr(P, "MANUAL_PAIRS"), "auto_tune forces nothing — MANUAL_PAIRS must be absent"
    print(f"[a] BLOCKED_PAIRS={len(P.BLOCKED_PAIRS)} well-formed")

    # (b) inject a synthetic blocked pair, install shim, train
    P.BLOCKED_PAIRS = [("x", "y")]
    from wrappers.tok_train_auto_tune import apply_patches
    apply_patches()
    import nanochat.tokenizer as tk
    corpus = [
        "the quick brown xy fox xy jumps over xy. xy lazy xy dog xy now. " * 40,
        "data of xy the xy analysis xy showed xy results xy here xy ok. " * 40,
    ]
    tok = tk.RustBPETokenizer.train_from_iterator(iter(corpus), 512)
    b2i = {}
    for tid in range(tok.get_vocab_size()):
        try:
            b2i[tok.enc.decode_single_token_bytes(tid)] = tid
        except Exception:
            pass
    print(f"[b] trained ({len(b2i)} tokens) on rustbpe_auto_tune via shim")

    # (c) blocked pair never merges
    assert b"xy" not in b2i, "blocked pair 'xy' merged despite BLOCKED_PAIRS"
    print("[c] blocked 'xy' correctly absent from vocab")

    # (d) SPLIT_PATTERN is pristine — delimiters STILL glue (tweak NOT applied)
    import regex as _re
    assert r'[^\r\n\p{L}\p{N}"(,“]' not in tk.SPLIT_PATTERN, \
        "delimiter-exclusion tweak leaked in — auto_tune must use the pristine pattern"
    if hasattr(tok.enc, "_pat_str"):
        assert tok.enc._pat_str == tk.SPLIT_PATTERN, "trained pattern != SPLIT_PATTERN"
    assert [m.group() for m in _re.compile(tk.SPLIT_PATTERN).finditer('"The')] == ['"The'], \
        'pristine pattern should keep \'"The\' glued (delimiter+word)'
    print('[d] SPLIT_PATTERN pristine (" ( , “ still glue to words)')

    # (e) roundtrip
    test = "the quick xy fox."
    assert tok.enc.decode(tok.enc.encode_ordinary(test)) == test, "roundtrip failed"
    print("[e] roundtrip OK")

print("\nall smoke checks passed")
