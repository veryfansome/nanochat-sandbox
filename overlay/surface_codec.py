"""
Surface-factoring dual-stream CODEC — the production encode/decode that turns
text into `(base_id, space_bit, cap_bits[K], cap_mask[K])` per token and back,
losslessly. Tokenizer-agnostic: every function takes `(enc, sre, id2len)` so the
same logic drives the tiny smoke tokenizer and the real triple tokenizer.

Contract (see ../ideas/surface_factoring/README.md "Architecture"):
  - leading SPACE before a word is factored out as a per-token `space_bit`;
  - first-letter CASE of each word-start a token covers is factored out as a
    per-word-start cap label; cap SLOTS are OCCURRENCE-active (a token owns the
    word-starts whose first byte falls in its span — a mid-word continuation owns
    none), cap VALUES are per-occurrence (was that word-start titlecase-folded).
Unicode-safe: word-starts are detected at the CHARACTER level and decode operates
on the FULL reconstructed string (proper `.upper()`), so multi-byte case folds
(Łódź, Greek) and byte-split tokens round-trip.
"""
from __future__ import annotations

from tools.case_factor_compression import fold_case


def char_word_starts(s):
    """[(byte_offset, char_index)] of each word-start char in s — a LETTER char
    preceded by a non-letter (or at start). CHAR-level → unicode-safe."""
    out, byte, prev_letter = [], 0, False
    for ci, ch in enumerate(s):
        if ch.isalpha() and not prev_letter:
            out.append((byte, ci))
        prev_letter = ch.isalpha()
        byte += len(ch.encode("utf-8"))
    return out


def compute_kmax(enc, sre, id2len, sample_texts):
    """Cap-slot tensor width = max word-starts a single token owns across a
    sample. Tokenizer/corpus-derived, ≥1."""
    K = 1
    for t in sample_texts:
        folded = fold_case(t)
        ws = [b for b, _ in char_word_starts(folded)]
        goff = 0
        for pt in (m.group(0) for m in sre.finditer(folded)):
            acc = 0
            for tid in enc.encode_ordinary(pt):
                lo, hi = goff + acc, goff + acc + id2len[tid]
                K = max(K, sum(1 for b in ws if lo <= b < hi))
                acc += id2len[tid]
            goff += len(pt.encode("utf-8"))
    return K


def surface_encode(text, enc, sre, id2len, K_max):
    """Dual-stream encode -> per token (base_id, space_bit, cap_bits[K], cap_mask[K]).
    Cap slots = the word-starts a token OWNS (a word-start byte that falls in the
    token's span — OCCURRENCE-active, so a mid-word continuation owns none);
    value = whether that word-start char was titlecase-folded. Unicode-safe."""
    folded = fold_case(text)
    assert len(folded) == len(text), "case fold must be char-length-preserving"
    ws = char_word_starts(folded)
    capped = {byte: (text[ci] != folded[ci]) for byte, ci in ws}
    ws_off = [b for b, _ in ws]
    pretoks = [m.group(0) for m in sre.finditer(folded)]
    out, pending, goff = [], 0, 0
    # Two-pointer over the SORTED word-start offsets: tokens are emitted in
    # increasing global byte order, so each word-start is visited once total
    # (O(tokens + word_starts), not the O(tokens·word_starts) of a per-token scan).
    wi, W = 0, len(ws_off)
    for i, pt in enumerate(pretoks):
        ptb = pt.encode("utf-8")
        if pt == " " and i + 1 < len(pretoks) and not pretoks[i + 1][:1].isspace():
            pending = 1
            goff += len(ptb)
            continue
        acc = 0
        for k, tid in enumerate(enc.encode_ordinary(pt)):
            lo, hi = goff + acc, goff + acc + id2len[tid]
            while wi < W and ws_off[wi] < lo:        # skip any word-start before this token
                wi += 1
            start = wi
            while wi < W and ws_off[wi] < hi:        # the word-starts this token owns
                wi += 1
            cap_bits, cap_mask = [0] * K_max, [0] * K_max
            for s in range(start, min(wi, start + K_max)):
                cap_mask[s - start] = 1
                cap_bits[s - start] = 1 if capped[ws_off[s]] else 0
            out.append((tid, pending if k == 0 else 0, cap_bits, cap_mask))
            acc += id2len[tid]
        pending = 0
        goff += len(ptb)
    return out


def surface_decode(stream, enc):
    """Reconstruct text from the (base_id, space_bit, cap_bits, cap_mask) stream.
    Operates on the FULL reconstructed string (proper unicode .upper()), so it
    handles multi-byte folds AND byte-split tokens — not per-token byte ops."""
    out, spans = bytearray(), []
    for tid, sbit, cap_bits, cap_mask in stream:
        if sbit:
            out += b" "
        lo = len(out)
        out += enc.decode_single_token_bytes(tid)
        spans.append((lo, len(out), cap_bits))
    folded = out.decode("utf-8", "replace")
    ws = char_word_starts(folded)
    byte_to_ci = {b: ci for b, ci in ws}
    ws_off = [b for b, _ in ws]
    chars = list(folded)
    wi, W = 0, len(ws_off)                            # two-pointer (spans are byte-ordered)
    for lo, hi, cap_bits in spans:
        while wi < W and ws_off[wi] < lo:
            wi += 1
        start = wi
        while wi < W and ws_off[wi] < hi:
            wi += 1
        for s in range(start, wi):
            o = ws_off[s]
            if s - start < len(cap_bits) and cap_bits[s - start] and o in byte_to_ci:
                chars[byte_to_ci[o]] = chars[byte_to_ci[o]].upper()
    return "".join(chars)
