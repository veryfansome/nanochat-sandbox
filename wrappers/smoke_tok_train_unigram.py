"""
Smoke test for wrappers/tok_train_unigram.py.

Verifies (8 checks total):
  (a)  HF Unigram + UnigramTrainer + ByteLevel pre-tok all wire together
       (build_tokenizer / train end-to-end on a synthetic corpus).
  (b)  Vocab size is in a sensible range relative to the request
       (UnigramTrainer may stop short of the target with a small corpus).
  (c)  All SPECIAL_TOKENS (+ `<|unk|>`) are present in the trained vocab.
  (d)  Save → HuggingFaceTokenizer.from_directory roundtrip works.
  (e)  Encode/decode is lossless on prose, contractions, code, Unicode
       (CJK + accents + emoji) — exercises the ByteLevel decoder.
  (e2) tools/eval_tokenizer.py:_safe_token_bytes returns ByteLevel-decoded
       bytes (not internal glyph bytes) — pins ' the' token to b' the'.
  (f)  write_token_bytes produces a tensor with sensible structure
       (zero entries for specials, positive for normal tokens).
  (f2) Single-byte tokens for bytes 0x80–0xFF have token_bytes == 1 (not 3)
       — pins the partial-byte overcount fix in write_token_bytes.

No real data; runs in <10s.

This smoke can run on this M1 because it doesn't require `nanochat.dataset`
(no corpus iteration). It DOES import `nanochat.tokenizer` for
`SPECIAL_TOKENS` / `SPLIT_PATTERN` / `HuggingFaceTokenizer`, which is
why the sibling `../nanochat/` checkout is needed.
"""

import os
import tempfile

import torch
from tokenizers import Tokenizer

with tempfile.TemporaryDirectory() as tmp:
    os.environ["VARIANT_BASE"] = tmp
    os.environ["NANOCHAT_BASE_DIR"] = tmp

    # (a) build + train on synthetic corpus
    from wrappers.tok_train_unigram import build_tokenizer, train, write_token_bytes

    tok_empty, specials = build_tokenizer()
    assert isinstance(tok_empty, Tokenizer)
    assert "<|unk|>" in specials
    from nanochat.tokenizer import SPECIAL_TOKENS as NANOCHAT_SPECIALS
    for s in NANOCHAT_SPECIALS:
        assert s in specials, f"missing nanochat special token: {s}"
    print(f"[a] build_tokenizer OK; {len(specials)} specials "
          f"(`<|unk|>` + {len(NANOCHAT_SPECIALS)} from nanochat)")

    # Synthetic corpus — diverse enough that Unigram has signal to converge.
    # Repetition matters for Unigram (it estimates token probabilities from
    # observed frequencies), so we repeat each line ~50x for a stable EM.
    base_lines = [
        "The quick brown fox jumps over the lazy dog. ",
        "She sells seashells by the seashore. ",
        "I'm not sure if you're going to believe this, but it's already happened. ",
        "import os\nfrom typing import Optional\ndef foo(x): return x + 1\n",
        "const fetchUser = async (id) => { return fetch(`/api/${id}`).json(); }\n",
        "Visit https://example.com for details. Email alice@example.com. ",
        "Numbers: 123 4567 89012 0.001 3.14159 2.71828 ",
        "Café naïve résumé. 你好世界. こんにちは. 안녕하세요. 🚀🎉💯 ",
    ]
    synthetic = [line * 50 for line in base_lines]

    VOCAB_SIZE = 512  # small so smoke runs fast; initial_alphabet contributes 256
    hf_tok = train(iter(synthetic), VOCAB_SIZE)
    print(f"[a] trained Unigram on {len(synthetic)} docs "
          f"({sum(len(s) for s in synthetic):,} chars)")

    # (b) vocab size is in a sensible range
    # Note: UnigramTrainer can stop short of the target if its EM can't
    # differentiate further pieces (no equivalent of BPE's `min_frequency=0`).
    # With a tiny smoke corpus this is expected — the production run on a
    # 2B-char corpus will saturate the target. For smoke we just require:
    #   - vocab is meaningfully bigger than just the 256-byte alphabet + specials
    #   - vocab is no larger than requested (UnigramTrainer never exceeds the cap)
    actual_vocab = hf_tok.get_vocab_size()
    floor = 256 + len(specials)  # byte alphabet + specials
    assert floor < actual_vocab <= VOCAB_SIZE, (
        f"vocab size {actual_vocab} outside ({floor}, {VOCAB_SIZE}] — "
        f"trainer should produce more than just bytes+specials but not exceed target"
    )
    print(f"[b] vocab_size = {actual_vocab} (target {VOCAB_SIZE}; "
          f"UnigramTrainer stops at convergence, doesn't pad to target)")

    # (c) all required special tokens present
    for s in specials:
        tid = hf_tok.token_to_id(s)
        assert tid is not None, f"special token {s!r} missing from trained vocab"
    print(f"[c] all {len(specials)} special tokens present in trained vocab")

    # (d) save → reload roundtrip via HuggingFaceTokenizer
    tokenizer_dir = os.path.join(tmp, "tokenizer")
    os.makedirs(tokenizer_dir, exist_ok=True)
    hf_tok.save(os.path.join(tokenizer_dir, "tokenizer.json"))

    from nanochat.tokenizer import HuggingFaceTokenizer
    reloaded = HuggingFaceTokenizer.from_directory(tokenizer_dir)
    assert reloaded.get_vocab_size() == actual_vocab
    print(f"[d] save → HuggingFaceTokenizer.from_directory roundtrip OK")

    # (e) encode/decode lossless on a battery — prose, code, unicode, emoji
    probes = [
        "The quick brown fox jumps over the lazy dog.",
        "I'm not sure if you're going to believe this.",
        "def foo(x):\n    return x + 1",
        "Café naïve résumé",
        "你好世界",
        "こんにちは",
        "🚀🎉💯",
        "https://example.com/path?a=1&b=2",
    ]
    for text in probes:
        ids = reloaded.encode(text)
        decoded = reloaded.decode(ids)
        assert decoded == text, (
            f"roundtrip failed for {text!r}:\n"
            f"  encoded ids: {ids}\n"
            f"  decoded:     {decoded!r}"
        )
    print(f"[e] encode/decode roundtrip OK on {len(probes)} probes "
          f"(prose, code, CJK, accents, emoji, URL)")

    # (e2) eval_tokenizer.py's _safe_token_bytes returns ByteLevel-decoded
    # bytes (not the internal glyph form). Guards against the regression where
    # id_to_token().encode('utf-8') gets used directly for HF ByteLevel
    # tokens, which overcounts byte lengths and breaks longest-token /
    # digit-only inspection (see tools/eval_tokenizer.py:_safe_token_bytes
    # comment for the failure mode).
    from tools.eval_tokenizer import _safe_token_bytes, _is_bytelevel_hf
    assert _is_bytelevel_hf(reloaded), (
        "ByteLevel detection failed on a tokenizer that uses ByteLevel decoder"
    )
    # Find tokens that decode to a clean UTF-8 string (no replacement char) —
    # for these, _safe_token_bytes's recovered length should match the
    # decoder's encoded length. Partial-byte tokens (single bytes of a
    # multi-byte UTF-8 sequence) are excluded because decode([tid]) returns
    # U+FFFD (replacement, 3 bytes), which is NOT the actual represented
    # byte length — _safe_token_bytes correctly returns 1 byte for those,
    # so a strict equality check would false-fail.
    REPLACEMENT = "�"
    mismatches = []
    checked = 0
    for tid in range(reloaded.get_vocab_size()):
        s = reloaded.id_to_token(tid)
        if s is None or s in specials:
            continue
        decoded = reloaded.decode([tid])
        if REPLACEMENT in decoded:
            continue  # partial-byte token; can't use decoder as ground truth
        expected = len(decoded.encode("utf-8"))
        recovered = _safe_token_bytes(reloaded, tid)
        if recovered is None:
            continue
        checked += 1
        if len(recovered) != expected:
            mismatches.append((tid, s, decoded, len(recovered), expected))
            if len(mismatches) >= 5:
                break
    assert not mismatches, (
        f"_safe_token_bytes byte-length mismatches (first {len(mismatches)}):\n"
        + "\n".join(
            f"  id={tid} id_to_token={s!r} decode={decoded!r} "
            f"recovered={got}B expected={exp}B"
            for tid, s, decoded, got, exp in mismatches
        )
    )
    # Also sanity-check a known-shape token: the ByteLevel-prefixed " the"
    # token (Ġthe) — _safe_token_bytes must return 4 bytes (b' the'), not 5
    # (b'\xc4\xa0the' from naive utf-8 encode of 'Ġthe').
    sample_id = reloaded.tokenizer.token_to_id("Ġthe")
    if sample_id is not None:
        b = _safe_token_bytes(reloaded, sample_id)
        assert b == b" the", (
            f"' the' token (id {sample_id}) recovered as {b!r}; "
            f"expected b' the'. Likely cause: ByteLevel reverse-map regression."
        )
    print(f"[e2] _safe_token_bytes byte lengths match decoder output on "
          f"{checked} cleanly-decoding tokens (ByteLevel reverse-map correct)")

    # (f) write_token_bytes produces a sensible tensor
    write_token_bytes(tokenizer_dir, hf_tok)
    tb_path = os.path.join(tokenizer_dir, "token_bytes.pt")
    token_bytes = torch.load(tb_path)
    assert token_bytes.shape == (actual_vocab,), (
        f"token_bytes shape {token_bytes.shape} != ({actual_vocab},)"
    )
    assert token_bytes.dtype == torch.int32
    # Specials should be zero.
    n_zero = int((token_bytes == 0).sum().item())
    assert n_zero >= len(specials), (
        f"only {n_zero} zero entries; expected ≥ {len(specials)} (specials)"
    )
    # Normal tokens should be positive.
    n_positive = int((token_bytes > 0).sum().item())
    assert n_positive >= actual_vocab - len(specials) - 10, (
        f"only {n_positive} positive entries; expected ≈ {actual_vocab - len(specials)}"
    )
    print(f"[f] token_bytes.pt OK ({n_zero} zero entries, "
          f"{n_positive} positive, dtype={token_bytes.dtype})")

    # (f2) Single-byte initial-alphabet tokens representing bytes 0x80–0xFF
    # must have token_bytes == 1, NOT 3. The previous naive
    # `len(decode(...).encode("utf-8"))` implementation returned 3 for these
    # because `decode([id])` produces U+FFFD (3 UTF-8 bytes) when the
    # underlying byte doesn't decode cleanly as UTF-8 in isolation — true
    # for all 128 bytes in 0x80–0xFF (continuation bytes, invalid starts,
    # overlong encodings). The fixed implementation uses the ByteLevel
    # reverse-map and reports the correct 1 byte.
    #
    # Note: these are regular Unigram pieces from `initial_alphabet`, NOT
    # HF byte_fallback tokens (which our build doesn't enable — see
    # `tok_train_unigram.py:build_tokenizer`).
    #
    # Pin a specific token: the ByteLevel char for byte 0xC0 is 'À' (codepoint
    # U+00C0 — bytes in 0xA1–0xFF except 0xAD map to themselves in the GPT-2
    # bytes_to_unicode algorithm). If 'À' is in the vocab (it's in the
    # `initial_alphabet`), its token_bytes entry must be 1.
    high_byte_char = "À"  # ByteLevel char for byte 0xC0
    sample_id = reloaded.tokenizer.token_to_id(high_byte_char)
    if sample_id is not None:
        actual = int(token_bytes[sample_id].item())
        assert actual == 1, (
            f"token_bytes[{sample_id}] (the single-byte token for "
            f"byte 0xC0, char {high_byte_char!r}) = {actual}; expected 1. "
            f"Indicates write_token_bytes regressed to using "
            f"len(decode([id]).encode('utf-8')), which returns 3 for "
            f"U+FFFD-decoding partial-byte tokens and would inflate "
            f"val/bpb downstream."
        )
        print(f"[f2] high-byte single-byte token (byte 0xC0, id={sample_id}) "
              f"has token_bytes == 1 (not 3) — partial-byte overcount fix OK")
    else:
        # Fallback: find ANY single-char token whose byte is in 0x80–0xFF
        # (i.e., its `decode([tid])` returns U+FFFD). Plain "non-ASCII single
        # char" is NOT a sufficient discriminator — chars like 'Ġ' (U+0120,
        # the ByteLevel form of byte 0x20 = space) are non-ASCII but represent
        # ASCII bytes that decode cleanly, so checking them wouldn't exercise
        # the bug. Use U+FFFD in the decoded form as the precise filter.
        REPLACEMENT_CHAR = "�"
        for tid in range(reloaded.get_vocab_size()):
            s = reloaded.id_to_token(tid)
            if s is None or s in specials or len(s) != 1:
                continue
            decoded = reloaded.decode([tid])
            if REPLACEMENT_CHAR not in decoded:
                continue  # byte decodes cleanly — bug wouldn't trigger here
            actual = int(token_bytes[tid].item())
            assert actual == 1, (
                f"token_bytes[{tid}] (single-byte token, ByteLevel char "
                f"{s!r}, byte ≥ 0x80) = {actual}; expected 1. "
                f"Partial-byte overcount regression."
            )
            print(f"[f2] high-byte token id={tid} ({s!r}) "
                  f"has token_bytes == 1 (decode={decoded!r})")
            break
        else:
            print(f"[f2] skipped — no single-byte tokens for 0x80–0xFF in "
                  f"smoke vocab (corpus too narrow to allocate them)")

print("\nall smoke checks passed")