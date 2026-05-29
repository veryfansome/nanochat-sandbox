"""
Tokenizer variant: Unigram LM (top-down likelihood-pruned vocab construction).

Tests the construction algorithm axis vs BPE — same pre-tokenization
(SPLIT_PATTERN from nanochat baseline), same SPECIAL_TOKENS (+ `<|unk|>`
which Unigram structurally requires), same corpus stream as
`scripts.tok_train`. The only thing changed is the vocab-construction
algorithm: top-down likelihood pruning (Unigram) rather than bottom-up
greedy-frequency merges (BPE).

Built on HuggingFace tokenizers (`tokenizers.models.Unigram` +
`tokenizers.trainers.UnigramTrainer`). Byte coverage comes from
`initial_alphabet=ByteLevel.alphabet()` — all 256 ByteLevel chars are
guaranteed to be vocab pieces. `<|unk|>` is included because
`UnigramTrainer` requires an `unk_token`, but it should never fire at
runtime (every input byte has a piece). `Unigram(byte_fallback=True)` is
NOT enabled — the initial alphabet makes it unnecessary, and our
trained tokenizer.json has `"byte_fallback": false`.

Saves:
  ~/.cache/nanochat-variants/unigram/tokenizer/tokenizer.json   (HF native)
  ~/.cache/nanochat-variants/unigram/tokenizer/token_bytes.pt   (val/bpb)

Usage:
  uv run python -m wrappers.tok_train_unigram \\
      [--max-chars=... --doc-cap=... --vocab-size=...]

Note: this wrapper does NOT runpy `scripts.tok_train` — that script is
hardcoded to `RustBPETokenizer.train_from_iterator`. We reimplement the
small amount of pre/post-processing here (corpus iteration + token_bytes
write) so the output directory looks identical to what `scripts.tok_train`
would produce, just with `tokenizer.json` in place of `tokenizer.pkl`.
"""

import argparse
import os
import time

import torch
from tokenizers import Tokenizer, Regex
from tokenizers.models import Unigram
from tokenizers.trainers import UnigramTrainer
from tokenizers import pre_tokenizers, decoders

from wrappers._tokenizer_variant_common import (
    setup_variant_base,
    print_downstream_hint,
)


def build_tokenizer():
    """
    Untrained HF Unigram tokenizer + Unigram-extended special-token list.

    Pre-tokenizer chain mirrors `HuggingFaceTokenizer.train_from_iterator`
    in upstream `nanochat/tokenizer.py`: Split(SPLIT_PATTERN) followed by
    ByteLevel, with ByteLevel decoder on the inference side. This keeps
    the input space identical to nanochat's HF-BPE path so the only knob
    being varied is the construction algorithm.
    """
    from nanochat.tokenizer import SPECIAL_TOKENS, SPLIT_PATTERN

    tokenizer = Tokenizer(Unigram())  # empty model; UnigramTrainer fills it
    tokenizer.normalizer = None
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(
            pattern=Regex(SPLIT_PATTERN),
            behavior="isolated",
            invert=False,
        ),
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
    ])
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.post_processor = None
    # Unigram requires an unk token (the model uses unk_id for OOV pieces during
    # Viterbi). Put it first so its id is stable across re-trainings.
    unigram_specials = ["<|unk|>"] + list(SPECIAL_TOKENS)
    return tokenizer, unigram_specials


def train(text_iter, vocab_size: int):
    """Train Unigram. Returns the trained HF Tokenizer."""
    tokenizer, specials = build_tokenizer()
    trainer = UnigramTrainer(
        vocab_size=vocab_size,
        show_progress=True,
        special_tokens=specials,
        unk_token="<|unk|>",
        # initial_alphabet must include the ByteLevel printable-char set so
        # every byte stays representable through the Unigram Viterbi search
        # (the pre-tokenizer maps bytes → these chars before Unigram sees them).
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        # Other UnigramTrainer knobs left at HF defaults:
        #   shrinking_factor=0.75, n_sub_iterations=2, max_piece_length=16.
        # Worth a sweep later if Unigram looks promising at this default.
    )
    tokenizer.train_from_iterator(text_iter, trainer=trainer)
    return tokenizer


def text_iterator(max_chars: int, doc_cap: int):
    """
    Mirror `scripts.tok_train.text_iterator` exactly — same corpus, same
    per-doc crop, same total-char limit. Keeps the BPE-vs-Unigram A/B fair.
    """
    from nanochat.dataset import parquets_iter_batched
    nchars = 0
    for batch in parquets_iter_batched(split="train"):
        for doc in batch:
            if len(doc) > doc_cap:
                doc = doc[:doc_cap]
            nchars += len(doc)
            yield doc
            if nchars > max_chars:
                return


def _build_bytelevel_reverse() -> dict:
    """
    GPT-2 / HF ByteLevel char → byte reverse map. Bytes in printable ranges
    (33–126, 161–172, 174–255) map to themselves; remaining bytes (0–32,
    127–160, 173) map to chars 256+ (in order). Mirrors the helper in
    `tools/eval_tokenizer.py` — small enough that duplication beats a
    wrappers→tools import. Reference: GPT-2 `bytes_to_unicode` in
    https://github.com/openai/gpt-2/blob/master/src/encoder.py.

    Do NOT use `pre_tokenizers.ByteLevel.alphabet()` + `enumerate()` — that
    method returns chars in alphabetized order, not byte-value order, so
    `enumerate` produces a wrong reverse map.
    """
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = list(bs)
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


def write_token_bytes(tokenizer_dir: str, hf_tok: Tokenizer):
    """
    Mirror `scripts.tok_train`'s post-train step: write `token_bytes.pt`
    (per-token UTF-8 byte length, zero for specials) so the trained
    tokenizer is byte-compatible with nanochat's val/bpb eval path.

    For ByteLevel HF tokenizers, byte counts MUST come from the ByteLevel
    reverse-map applied to `id_to_token()`'s output — NOT from
    `decode([tid]).encode("utf-8")`. The decode path returns U+FFFD
    (3 UTF-8 bytes) for any token whose ByteLevel-recovered bytes don't
    form valid UTF-8 in isolation — e.g. the 128 single-byte tokens for
    bytes 0x80–0xFF in the initial alphabet. Using decode would inflate
    token_bytes for those tokens by 2 each and would underestimate
    val/bpb downstream (total_loss_bits / inflated_total_bytes).
    """
    from nanochat.tokenizer import HuggingFaceTokenizer
    wrap = HuggingFaceTokenizer(hf_tok)
    vocab_size = wrap.get_vocab_size()
    special_set = set(wrap.get_special_tokens())
    # Only build the reverse map if the tokenizer actually uses ByteLevel.
    char_to_byte = None
    decoder = hf_tok.decoder
    if decoder is not None and type(decoder).__name__ == "ByteLevel":
        char_to_byte = _build_bytelevel_reverse()
    token_bytes = []
    for tid in range(vocab_size):
        # `id_to_token` returns the ByteLevel-encoded form for normal tokens
        # and the literal text for specials. Use it instead of `decode([tid])`
        # so we can reverse-map deterministically (see docstring).
        tok_str = wrap.id_to_token(tid)
        if tok_str is None or tok_str in special_set:
            token_bytes.append(0)
            continue
        if char_to_byte is not None and all(c in char_to_byte for c in tok_str):
            # Each ByteLevel char represents exactly one byte, so the byte
            # length equals the char count. Avoids materializing the bytes.
            token_bytes.append(len(tok_str))
            continue
        # Non-ByteLevel tokenizer, or a token with chars outside the
        # ByteLevel alphabet (rare; would be an unusual added token, since
        # specials are filtered above). Fall back to UTF-8 encode.
        token_bytes.append(len(tok_str.encode("utf-8")))
    token_bytes = torch.tensor(token_bytes, dtype=torch.int32, device="cpu")
    path = os.path.join(tokenizer_dir, "token_bytes.pt")
    with open(path, "wb") as f:
        torch.save(token_bytes, f)
    print(f"Saved token_bytes to {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Train a Unigram LM tokenizer (HF tokenizers backend)"
    )
    parser.add_argument(
        "--max-chars", type=int, default=2_000_000_000,
        help="Maximum characters to train on (default: 2B, matches scripts.tok_train)",
    )
    parser.add_argument(
        "--doc-cap", type=int, default=10_000,
        help="Per-document character cap (default: 10K, matches scripts.tok_train)",
    )
    parser.add_argument(
        "--vocab-size", type=int, default=32768,
        help="Target vocab size (default: 32768 = 2^15, matches nanochat baseline)",
    )
    args = parser.parse_args()

    variant_base = setup_variant_base("unigram")
    tokenizer_dir = os.path.join(variant_base, "tokenizer")
    os.makedirs(tokenizer_dir, exist_ok=True)

    print(f"==> variant_base:  {variant_base}")
    print(f"==> vocab_size:    {args.vocab_size:,}")
    print(f"==> max_chars:     {args.max_chars:,}")
    print(f"==> doc_cap:       {args.doc_cap:,}")

    t0 = time.time()
    hf_tok = train(text_iterator(args.max_chars, args.doc_cap), args.vocab_size)
    elapsed = time.time() - t0
    print(f"Training time: {elapsed:.2f}s")

    tokenizer_path = os.path.join(tokenizer_dir, "tokenizer.json")
    hf_tok.save(tokenizer_path)
    print(f"Saved tokenizer to {tokenizer_path}")

    write_token_bytes(tokenizer_dir, hf_tok)

    print_downstream_hint(variant_base)


if __name__ == "__main__":
    main()
