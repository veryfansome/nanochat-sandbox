"""
Smoke test for wrappers/tok_train_seed_tokens.py.

Verifies:
  (a) The rustbpe_seed_tokens variant module imports and exposes Tokenizer.
  (b) sys.modules['rustbpe'] alias works pre-nanochat-import.
  (c) Seed-token YAML loads + produces the expected position-aware variants.
  (d) `train_from_iterator` accepts seed_tokens kwarg.
  (e) End-to-end: train on a tiny synthetic corpus, encode/decode roundtrip.

No real data; runs in <2s.
"""
import os
import sys
import tempfile

with tempfile.TemporaryDirectory() as tmp:
    os.environ["VARIANT_BASE"] = tmp
    os.environ["NANOCHAT_BASE_DIR"] = tmp

    # (a) variant imports
    import rustbpe_seed_tokens
    assert hasattr(rustbpe_seed_tokens, "Tokenizer")
    print(f"[a] rustbpe_seed_tokens.Tokenizer = {rustbpe_seed_tokens.Tokenizer}")

    # (b) alias swap
    from wrappers._tokenizer_variant_common import alias_rustbpe
    alias_rustbpe("rustbpe_seed_tokens")
    import rustbpe
    assert rustbpe is rustbpe_seed_tokens, "alias didn't swap sys.modules"
    print(f"[b] sys.modules['rustbpe'] points to rustbpe_seed_tokens")

    # (c) seed-token loading produces expected variants
    from wrappers.tok_train_seed_tokens import _load_seeds
    seeds = _load_seeds()
    assert len(seeds) > 100, f"too few seeds loaded: {len(seeds)}"
    # Check a known versatile_morpheme produces all three variants (per the loader).
    # 'able' should produce ' able', ' Able', 'able'.
    for expected in [" able", " Able", "able"]:
        assert expected in seeds, f"expected seed {expected!r} not in {len(seeds)} seeds"
    print(f"[c] loaded {len(seeds)} seeds; ' able', ' Able', 'able' all present")

    # (d) train kwargs accepted
    tok = rustbpe.Tokenizer()
    synthetic = [
        "the brand new able-bodied actor demonstrated cred and credibility. " * 20,
        "informed observers noted the chronic mismatch in finite forms. " * 20,
        "macro and micro patterns formed identifiable cycles over time. " * 20,
    ]
    tok.train_from_iterator(
        iter(synthetic),
        vocab_size=512,
        pattern=r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,2}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+""",
        seed_tokens=seeds[:50],  # subset to keep smoke fast
    )
    mergeable = dict((bytes(k), v) for k, v in tok.get_mergeable_ranks())
    print(f"[d] trained with seed_tokens kwarg: {len(mergeable)} merges produced")

    # (e) roundtrip
    import tiktoken
    enc = tiktoken.Encoding(
        name="smoke", pat_str=tok.get_pattern(),
        mergeable_ranks=mergeable, special_tokens={},
    )
    test = "the chronic ability to be informed."
    ids = enc.encode_ordinary(test)
    decoded = enc.decode(ids)
    assert decoded == test, f"roundtrip failed: {decoded!r} != {test!r}"
    print(f"[e] roundtrip OK ({len(ids)} tokens for {len(test)} bytes)")

print("\nall smoke checks passed")
