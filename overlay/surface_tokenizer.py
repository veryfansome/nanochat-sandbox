"""
SurfaceTokenizer — the production dual-stream tokenizer for surface factoring.

Wraps a trained triple (force_merges + spaceless + case-fold) tokenizer and the
surface codec into one object the dataloader/eval/generation can use:

    tok = SurfaceTokenizer.from_directory(tok_dir, SPACELESS_PATTERN, kmax=K)
    stream = tok.encode(text)          # [(base_id, space_bit, cap_bits[K], cap_mask[K]), ...]
    text   = tok.decode(stream)        # lossless inverse (unicode/byte-split safe)
    nbytes = tok.byte_count(stream)    # Σ base-token bytes + Σ space_bit  (the bpb denominator)

The underlying tokenizer must have been trained with the SPACELESS pattern (so a
leading space is its own pretoken and can be factored to a bit) — see
tools/stack_fm_spaceless.py / tools/stack_triple.py for the build. `kmax` is the
cap-slot width; if omitted it is derived from a sample via the codec's
compute_kmax (pass a representative `sample_texts`).
"""
from __future__ import annotations

import regex

from overlay.surface_codec import surface_encode, surface_decode, compute_kmax


class SurfaceTokenizer:
    def __init__(self, enc, spaceless_pattern, kmax=None, sample_texts=None):
        self.enc = enc
        self.sre = regex.compile(spaceless_pattern)
        self.id2len = [len(enc.decode_single_token_bytes(i)) for i in range(enc.n_vocab)]
        self.bos = enc.encode_single_token("<|bos|>")
        if kmax is None:
            assert sample_texts, "pass kmax= or sample_texts= to derive it"
            kmax = compute_kmax(enc, self.sre, self.id2len, sample_texts)
        self.kmax = kmax

    @classmethod
    def from_directory(cls, tokenizer_dir, spaceless_pattern, kmax=None, sample_texts=None):
        """Load a tiktoken encoding saved by RustBPETokenizer.from_directory."""
        from nanochat.tokenizer import RustBPETokenizer
        enc = RustBPETokenizer.from_directory(tokenizer_dir).enc
        return cls(enc, spaceless_pattern, kmax=kmax, sample_texts=sample_texts)

    @property
    def n_vocab(self):
        return self.enc.n_vocab

    def encode(self, text):
        """text -> [(base_id, space_bit, cap_bits[kmax], cap_mask[kmax]), ...]."""
        return surface_encode(text, self.enc, self.sre, self.id2len, self.kmax)

    def decode(self, stream):
        """Inverse of encode (lossless, unicode/byte-split safe)."""
        return surface_decode(stream, self.enc)

    def byte_count(self, stream):
        """Σ base-token bytes + Σ space_bit — the bpb byte denominator (a factored
        leading space counts as its 1 original byte; the case fold is byte-preserving)."""
        return sum(self.id2len[tid] for tid, *_ in stream) + sum(sp for _, sp, *_ in stream)
