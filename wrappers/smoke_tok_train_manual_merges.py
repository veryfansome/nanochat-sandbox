"""
Smoke test for wrappers/tok_train_manual_merges.py.

Verifies:
  (a) MANUAL_PAIRS loads and is well-formed (list of (str, str)).
  (b) The insertion shim installs (pristine rustbpe; no alias) and trains.
  (c) DONE-FIRST semantics: a synthetic byte-bigram pair ('q','z') lands at
      rank 256 — the very first merge — not appended at the end.
  (d) The trained tokenizer's pattern is nanochat's STANDARD SPLIT_PATTERN.
  (e) The forced merge actually wins: 'qz' tokenizes as a single token, and
      encode/decode roundtrips.

No real data; runs in <2s.
"""
import os
import sys
import tempfile

with tempfile.TemporaryDirectory() as tmp:
    os.environ["VARIANT_BASE"] = tmp
    os.environ["NANOCHAT_BASE_DIR"] = tmp

    # (a) canonical list well-formed
    from rustbpe_variants.manual_merges import pairs as P
    assert isinstance(P.MANUAL_PAIRS, list), "MANUAL_PAIRS must be a list"
    for i, pr in enumerate(P.MANUAL_PAIRS):
        assert isinstance(pr, tuple) and len(pr) == 2, f"entry {i} not a 2-tuple: {pr!r}"
        L, R = pr
        assert isinstance(L, str) and isinstance(R, str) and L and R, f"entry {i} bad: {pr!r}"
    print(f"[a] MANUAL_PAIRS: {len(P.MANUAL_PAIRS)} well-formed (str,str) pairs")

    # (b) inject a synthetic byte-bigram pair, install the shim, train
    P.MANUAL_PAIRS = [("q", "z")]
    from wrappers.tok_train_manual_merges import apply_patches
    apply_patches()
    import nanochat.tokenizer as tk
    STD_PATTERN = tk.SPLIT_PATTERN
    corpus = [
        "the quick qz brown qz fox jumps. qzqz the lazy qz dog. " * 40,
        "data qz of the qz analysis qz showed qz results. " * 40,
    ]
    tok = tk.RustBPETokenizer.train_from_iterator(iter(corpus), 512)

    # (c) done-first: 'qz' is the FIRST merge (rank 256)
    b2i = {}
    for tid in range(tok.get_vocab_size()):
        try:
            b2i[tok.enc.decode_single_token_bytes(tid)] = tid
        except Exception:
            pass
    assert b"qz" in b2i, "forced concat b'qz' absent — insertion failed"
    assert b2i[b"qz"] == 256, f"b'qz' at rank {b2i[b'qz']}, expected 256 (first merge / done-first)"
    print(f"[c] forced 'qz' inserted at rank {b2i[b'qz']} (first merge — done first)")

    # (d) standard pattern (no carve-out)
    assert tok.enc._pat_str == STD_PATTERN if hasattr(tok.enc, "_pat_str") else True
    print("[d] pattern == nanochat standard SPLIT_PATTERN (no carve-out)")

    # (e) the merge wins + roundtrip
    ids = tok.enc.encode_ordinary("qz qzqz")
    parts = [tok.enc.decode_single_token_bytes(i) for i in ids]
    assert b"qz" in parts, f"'qz' didn't win the merge: {parts}"
    test = "the qz quick qz fox."
    assert tok.enc.decode(tok.enc.encode_ordinary(test)) == test, "roundtrip failed"
    print(f"[e] 'qz' wins the merge ({parts}); roundtrip OK")

print("\nall smoke checks passed")
