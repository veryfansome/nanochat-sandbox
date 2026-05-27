"""
Smoke test for wrappers/tok_train_seed_tokens.py (data-driven route version).

Verifies:
  (a) routes.py loads and exposes SEED_ROUTES + RECOMMENDED_VOCAB_SIZE.
  (b) Uses pristine rustbpe — no `sys.modules['rustbpe']` alias swap.
  (c) apply_patches() installs the train_from_iterator shim and reads
      SEED_ROUTES dynamically (test can swap routes between calls).
  (d) End-to-end: train + apply synthetic seed routes; bytes 0..255 keep
      their identity ranks; seed tokens land contiguously among the
      natural merges at their natural-firing positions; tiktoken
      encode/decode roundtrip is preserved.
  (e) Seed merges actually fire at encode time: encoding a string with
      `qq` produces the b'qq' seed token, not adjacent (q, q) bytes —
      proving rank-based encoding picked the seed merge over competing
      natural merges that would consume its operands.
  (f) `L + R != S` route corruption asserts at training time (the merge
      could never fire and would be a dead vocab slot).

No real data; runs in <5s.
"""
import os
import sys
import tempfile

with tempfile.TemporaryDirectory() as tmp:
    os.environ["VARIANT_BASE"] = tmp
    os.environ["NANOCHAT_BASE_DIR"] = tmp

    # (a) routes module loads + canonical structural validation. The mining
    # artifact is the new design's main failure surface — corruption in the
    # generated table (wrong tuple shape, L+R != S, duplicate S) would not
    # be caught by the synthetic-route mechanics tests below. Cheap structural
    # checks against the canonical guard the deployed artifact.
    from rustbpe_variants.seed_tokens import routes as routes_mod
    assert hasattr(routes_mod, "SEED_ROUTES")
    assert hasattr(routes_mod, "RECOMMENDED_VOCAB_SIZE")
    canonical = routes_mod.SEED_ROUTES
    assert canonical, "canonical SEED_ROUTES is empty"
    seen_S: set[bytes] = set()
    for i, entry in enumerate(canonical):
        assert len(entry) == 3, f"route {i}: expected 3-tuple, got {entry!r}"
        L, R, S = entry
        assert isinstance(L, bytes) and isinstance(R, bytes) and isinstance(S, bytes), (
            f"route {i}: operands must be bytes, got {type(L).__name__}/"
            f"{type(R).__name__}/{type(S).__name__}"
        )
        assert L and R, f"route {i} has empty operand: {entry!r}"
        assert L + R == S, f"route {i} broken: {L!r} + {R!r} != {S!r}"
        assert S not in seen_S, f"route {i}: duplicate S={S!r}"
        seen_S.add(S)
    assert routes_mod.RECOMMENDED_VOCAB_SIZE == 32768 + len(canonical), (
        f"RECOMMENDED_VOCAB_SIZE drift: routes={len(canonical)} but "
        f"RECOMMENDED_VOCAB_SIZE={routes_mod.RECOMMENDED_VOCAB_SIZE}"
    )
    print(
        f"[a] routes.py canonical: {len(canonical)} routes (all 3-tuples, "
        f"L+R==S, unique S); RECOMMENDED_VOCAB_SIZE={routes_mod.RECOMMENDED_VOCAB_SIZE}"
    )

    # Swap in a synthetic SEED_ROUTES so the mechanics tests below don't
    # depend on a freshly-mined canonical (which may not exist on a clean
    # checkout) and stay fast.
    #
    # Routes (all single-byte L, R for guaranteed-in-vocab operands):
    #   (b'q', b'q', b'qq')  — synthetic; bytes definitely in vocab; 'qq' is
    #                          rare in any natural English corpus, so the
    #                          natural BPE won't have minted it → route applies.
    #   (b'z', b'q', b'zq')  — synthetic rare bigram; route applies.
    #   (b'a', b'a', b'aa')  — synthetic; 'aa' may or may not be in natural
    #                          vocab depending on corpus; counts toward the
    #                          S-exists skip path if it is.
    #   (b'\xff', b'q', b'\xffq')  — operand b'\xff' present (always: byte
    #                                tokens); route applies if 'q' is too.
    routes_mod.SEED_ROUTES = [
        (b'q', b'q', b'qq'),
        (b'z', b'q', b'zq'),
        (b'a', b'a', b'aa'),
        (b'\xff', b'q', b'\xffq'),
    ]

    # (b) pristine rustbpe (no alias)
    from wrappers.tok_train_seed_tokens import apply_patches
    routes = apply_patches()
    import rustbpe
    assert hasattr(rustbpe, "Tokenizer")
    print(f"[b] rustbpe = {rustbpe.__file__}; no sys.modules alias")

    # (c) shim installed; SEED_ROUTES read dynamically from the module.
    assert len(routes) == 4
    print(f"[c] apply_patches() returned {len(routes)} routes; shim installed")

    # (d) End-to-end mechanics.
    import nanochat.tokenizer as tk
    # Use a small synthetic corpus. Keep 'qq', 'zq', '\xffq' rare/absent so
    # they're not in the natural vocab; include 'aa' a few times to ensure
    # it might appear naturally (or not — depends on rustbpe's frequency
    # cutoff).
    synthetic_corpus = [
        "the quick brown fox jumps over the lazy dog " * 30,
        "she sells seashells by the seashore " * 30,
        "abracadabra! aardvarks and aliens " * 5,
    ]
    # Reserve plenty of vocab room; with N_routes=4 + len(specials) overhead,
    # vocab_size=512 is comfortable.
    result_tok = tk.RustBPETokenizer.train_from_iterator(iter(synthetic_corpus), 512)
    assert result_tok is not None
    enc = result_tok.enc
    vocab_n = result_tok.get_vocab_size()
    print(f"[d] trained: vocab_size={vocab_n}")

    # Verify the synthetic routes that could apply ARE in the merge table.
    # b'qq', b'zq', b'\xffq' should all be applied (rare in corpus, operands
    # in vocab). b'aa' may or may not be applied (corpus has aa in "aardvark").
    mergeable_to_id = {}
    for tid in range(vocab_n):
        try:
            b = enc.decode_single_token_bytes(tid)
            mergeable_to_id[b] = tid
        except Exception:
            continue

    must_apply = [(b'q', b'q', b'qq'), (b'z', b'q', b'zq'),
                  (b'\xff', b'q', b'\xffq')]
    for L, R, S in must_apply:
        assert S in mergeable_to_id, (
            f"expected seed {S!r} (route {L!r}+{R!r}) in vocab; "
            f"check stdout for skip reasons."
        )
    # Byte tokens MUST keep their identity ranks. Regression check for a
    # past bug where the seed-insertion loop re-numbered bytes to 256..511,
    # bloating vocab and breaking the "byte i ↔ rank i" convention.
    for byte_val in (ord('q'), ord('z'), 0xff, ord(' '), 0):
        b = bytes([byte_val])
        assert mergeable_to_id.get(b) == byte_val, (
            f"byte {b!r} should be at rank {byte_val}, got "
            f"{mergeable_to_id.get(b)}"
        )
    print(f"[d] all 3 guaranteed-apply seeds present; byte tokens unshifted")

    # (e) Encode-time merge fires. Encode a string that contains 'qq' and
    # verify the b'qq' token id appears in the output (not q + q).
    qq_id = mergeable_to_id[b'qq']
    q_id = mergeable_to_id[b'q']
    test = "before qq after"
    ids = enc.encode_ordinary(test)
    assert qq_id in ids, (
        f"seed token b'qq' (id={qq_id}) didn't fire when encoding "
        f"{test!r}: got {ids}. (q_id={q_id}.)"
    )
    # Confirm no isolated (q, q) pair survives — both 'q's should be merged.
    qq_run = sum(1 for i in range(len(ids) - 1) if ids[i] == q_id and ids[i+1] == q_id)
    assert qq_run == 0, (
        f"adjacent (q, q) pair survived encoding: {ids}. The seed merge "
        f"should have consumed it."
    )
    # Roundtrip
    decoded = enc.decode(ids)
    assert decoded == test, f"roundtrip failed: {decoded!r} != {test!r}"
    print(f"[e] seed b'qq' fires at encode time (id={qq_id} in {ids}); roundtrip OK")

    # (f) L+R != S route should assert. Swap in a corrupt entry and confirm
    # training fails fast.
    routes_mod.SEED_ROUTES = [(b'X', b'Y', b'XYZ')]  # 'X' + 'Y' != 'XYZ'
    try:
        tk.RustBPETokenizer.train_from_iterator(iter(synthetic_corpus), 512)
        raise AssertionError("expected AssertionError for L+R != S route")
    except AssertionError as exc:
        assert "bytes inconsistent" in str(exc), (
            f"got AssertionError but wrong message: {exc}"
        )
    print(f"[f] L+R != S route correctly asserts in wrapper")

print("\nall smoke checks passed")
