"""
Smoke test for wrappers/tok_train_space_digits.py.

Verifies:
  (a) The SPLIT_PATTERN substitution actually changes the regex (catches
      upstream-regex drift that would silently no-op the patch).
  (b) The variant base dir / symlink setup runs without error.
  (c) The patched regex is applied as a pre-tokenizer to a known string and
      produces the expected number of chunks (sanity check on the regex).

No data needed; runs in <1s.
"""
import os
import sys
import tempfile

# Use a throwaway VARIANT_BASE so we don't touch the real ~/.cache/nanochat-variants
with tempfile.TemporaryDirectory() as tmp:
    os.environ["VARIANT_BASE"] = tmp

    from wrappers._tokenizer_variant_common import setup_variant_base
    base = setup_variant_base("space_digits_smoke")
    assert base == tmp, f"setup_variant_base returned {base}, expected {tmp}"
    assert os.environ["NANOCHAT_BASE_DIR"] == tmp, "NANOCHAT_BASE_DIR not set"
    print(f"[a] variant base set up at {base}")

    import nanochat.tokenizer as tk
    ORIG = tk.SPLIT_PATTERN
    assert r"\p{N}{1,2}" in ORIG, "Current SPLIT_PATTERN doesn't contain the digit clause"

    # Re-import the wrapper module — but importing it runs the patch + runpy.
    # Instead, replay the patch logic inline to test it in isolation.
    SPACE_DIGITS_PATTERN = ORIG.replace(r"|\p{N}{1,2}|", r"| ?\p{N}{1,2}|")
    assert SPACE_DIGITS_PATTERN != ORIG, "Patch was a no-op"
    assert r"| ?\p{N}{1,2}|" in SPACE_DIGITS_PATTERN, "Expected substitution missing"
    print(f"[b] regex patch applied; diff = ' ?' inserted before digit clause")

    # (c) Apply both regexes to a test string and check chunking differs as expected.
    # We use fancy-regex-compatible matching via the `regex` package (a tiktoken dep).
    try:
        import regex as re_mod  # PyPI 'regex', supports \p{N}
    except ImportError:
        import re as re_mod
    test_str = "the number 42 appears"
    orig_chunks = re_mod.findall(ORIG, test_str)
    new_chunks = re_mod.findall(SPACE_DIGITS_PATTERN, test_str)
    # With original: " 42" -> [" "] + ["42"] (space and digits split)
    # With new:      " 42" -> [" 42"] (merged)
    assert "42" in orig_chunks, f"orig regex didn't find '42': {orig_chunks}"
    assert " 42" in new_chunks, (
        f"new regex didn't merge ' 42' into one chunk: {new_chunks}"
    )
    print(f"[c] chunking diff verified:")
    print(f"      original:  {orig_chunks}")
    print(f"      patched:   {new_chunks}")

print("\nall smoke checks passed")
