"""
Smoke test for wrappers/tok_train_manual_merges.py.

Verifies:
  (a) MANUAL_PAIRS and BLOCKED_PAIRS load and are well-formed (lists of (str,str)).
  (b) The shim installs (rustbpe_force_merges crate, for blocked_pairs support)
      and trains under the STANDARD pattern.
  (c) DONE-FIRST seed: a synthetic byte-bigram ('q','z') lands at rank 256 — the
      very first merge.
  (d) BLOCKING: a synthetic blocked pair ('x','y') does NOT merge — 'xy' is absent
      from the vocab even though it's frequent in the corpus.
  (e) The trained pattern is nanochat's standard SPLIT_PATTERN (no carve-out).
  (f) Seed wins + encode/decode roundtrip.

No real data; runs in <2s.
"""
import os
import tempfile

with tempfile.TemporaryDirectory() as tmp:
    os.environ["VARIANT_BASE"] = tmp
    os.environ["NANOCHAT_BASE_DIR"] = tmp

    # (a) both lists well-formed
    from rustbpe_variants.manual_merges import pairs as P
    for name, lst in [("MANUAL_PAIRS", P.MANUAL_PAIRS), ("BLOCKED_PAIRS", P.BLOCKED_PAIRS)]:
        assert isinstance(lst, list), f"{name} must be a list"
        for i, pr in enumerate(lst):
            assert isinstance(pr, tuple) and len(pr) == 2, f"{name}[{i}] not a 2-tuple: {pr!r}"
            L, R = pr
            assert isinstance(L, str) and isinstance(R, str) and L and R, f"{name}[{i}] bad: {pr!r}"
    print(f"[a] MANUAL_PAIRS={len(P.MANUAL_PAIRS)}, BLOCKED_PAIRS={len(P.BLOCKED_PAIRS)} well-formed")

    # (b) inject synthetic seed + blocked pair, install shim, train
    P.MANUAL_PAIRS = [("q", "z")]
    P.BLOCKED_PAIRS = [("x", "y")]
    from wrappers.tok_train_manual_merges import apply_patches
    apply_patches()
    import nanochat.tokenizer as tk
    STD_PATTERN = tk.SPLIT_PATTERN
    corpus = [
        "the quick qz brown xy fox qz jumps xy. qzqz xy lazy qz dog. " * 40,
        "data qz of xy the qz analysis xy showed qz results xy now. " * 40,
    ]
    tok = tk.RustBPETokenizer.train_from_iterator(iter(corpus), 512)
    b2i = {}
    for tid in range(tok.get_vocab_size()):
        try:
            b2i[tok.enc.decode_single_token_bytes(tid)] = tid
        except Exception:
            pass
    print(f"[b] trained ({len(b2i)} tokens) on rustbpe_force_merges via shim")

    # (c) done-first seed
    assert b2i.get(b"qz") == 256, f"'qz' at rank {b2i.get(b'qz')}, expected 256 (done first)"
    print("[c] seed 'qz' at rank 256 (first merge — done first)")

    # (d) blocked pair never merges
    assert b"xy" not in b2i, "blocked pair 'xy' merged despite BLOCKED_PAIRS"
    print("[d] blocked 'xy' correctly absent from vocab")

    # (e) standard pattern
    if hasattr(tok.enc, "_pat_str"):
        assert tok.enc._pat_str == STD_PATTERN, "pattern != standard (carve-out leaked?)"
    print("[e] standard SPLIT_PATTERN (no carve-out)")

    # (f) roundtrip
    test = "the qz quick xy fox."
    assert tok.enc.decode(tok.enc.encode_ordinary(test)) == test, "roundtrip failed"
    print("[f] roundtrip OK")

print("\nall smoke checks passed")
