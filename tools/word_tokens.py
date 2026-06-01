"""
word_tokens.py — for one or more input words/strings, show how each tokenizer
encodes them. The inverse of `tools.token_contexts` (which goes token→words);
this goes word→tokens.

Use to inspect *why* a per-word firing-count diff moved between two tokenizers:
when `b'less'` lost 10 firings on ` meaningless`, this tool tells you the variant
absorbed ` meaningless` into a single whole-word token instead of `[ meaning][less]`.

Each input word is rendered in BOTH forms (bare and space-prefixed) — GPT-style
BPE trains them as distinct tokens, and the in-corpus dominant form is the
space-prefixed mid-sentence one while the bare form is the sentence-start /
parasite candidate. Leading whitespace in the input is stripped before the two
forms are built, so `" meaningless"` and `"meaningless"` produce the same pair.

Multiple tokenizers (comma-separated) render side-by-side, baseline-first.

Usage:
    # one tokenizer, one or more words
    uv run python -m tools.word_tokens nonetheless meaningless pointless

    # baseline vs manual_merges
    uv run python -m tools.word_tokens \\
        --tokenizers ours,~/.cache/nanochat-variants/manual_merges/tokenizer \\
        meaningless pointless Nonetheless regardless

    # compare against gpt2 / gpt4
    uv run python -m tools.word_tokens --tokenizers ours,gpt2,gpt4 tokenization
"""

import argparse
import sys

from tools.eval_tokenizer import load_tokenizer, _safe_token_bytes


def encode_one(tok, word: str) -> list[int]:
    """Encode a single string, no special tokens. RustBPE + HF both expose this
    via .encode(); pass through unchanged."""
    ids = tok.encode(word)
    # Defensive: encode returns list[int] for str input on both paths.
    assert isinstance(ids, list), f"unexpected encode return type: {type(ids)}"
    return ids


def render_token(tok, tid: int) -> str:
    """Format one token id as `b'..'@rank` using the actual represented bytes
    (handles ByteLevel-encoded HF tokenizers via _safe_token_bytes)."""
    b = _safe_token_bytes(tok, tid)
    if b is None:
        return f"<gap>@{tid}"
    return f"{b!r}@{tid}"


def parse_args():
    p = argparse.ArgumentParser(
        description="Show how a tokenizer (or several) encodes given words.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "words",
        nargs="+",
        help="Words / strings to encode. Leading whitespace is stripped; the "
             "tool always shows both the bare and space-prefixed forms.",
    )
    p.add_argument(
        "--tokenizers", "-t",
        default="ours",
        help="Comma-separated tokenizer specs (built-ins: ours, gpt2, gpt4, gpt4o; "
             "or directory paths). Default: ours. First listed is the reference.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    specs = [s.strip() for s in args.tokenizers.split(",") if s.strip()]
    if not specs:
        print("error: no tokenizers given", file=sys.stderr)
        sys.exit(2)

    toks = [(spec, load_tokenizer(spec)) for spec in specs]
    name_w = max(len(spec) for spec, _ in toks)

    for raw in args.words:
        # Normalize: strip leading whitespace so "  meaningless" and
        # "meaningless" produce the same (bare, space-prefixed) pair.
        bare = raw.lstrip()
        for word in (bare, " " + bare):
            nbytes = len(word.encode("utf-8"))
            print(f"\n=== {word!r}  ({nbytes} bytes) ===")
            for spec, tok in toks:
                ids = encode_one(tok, word)
                rendered = "  ".join(render_token(tok, tid) for tid in ids)
                print(f"  {spec:<{name_w}}  {len(ids):>2}t  {ids}")
                print(f"  {'':<{name_w}}        {rendered}")


if __name__ == "__main__":
    main()
