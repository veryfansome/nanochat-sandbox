"""
Dump a tokenizer's vocab to a flat text file: rank, corpus count, byte length,
and token repr per row.

For our RustBPE tokenizers, `rank` == `id` for non-special tokens (rustbpe
assigns merge ranks 0..N-1 then appends specials at N..N+9). For `gpt2`,
`gpt4`, etc. the same holds. Specials are emitted with count 0.

`count` is the firing count over a held-out ClimbMix sample (val split by
default, doc-level trimmed to --max-chars). Pass `--max-chars 0` to skip
counting entirely (header + rank + bytes only).

Output (tab-separated, header included):

    rank<TAB>count<TAB>byte_len<TAB>repr

Output path defaults to /tmp/{tokenizer-slug}_vocab.txt. Pass --output to
override.

Usage:
    # baseline tokenizer, val-split counts, default 20M char sample
    uv run python -m tools.dump_vocab

    # explicit output path
    uv run python -m tools.dump_vocab --output /tmp/baseline_vocab.txt

    # a variant tokenizer
    uv run python -m tools.dump_vocab \\
        --tokenizer ~/.cache/nanochat-variants/<variant>/tokenizer \\
        --output /tmp/<variant>_vocab.txt

    # skip counting (instant; vocab structure only)
    uv run python -m tools.dump_vocab --max-chars 0

    # sort by count descending instead of rank ascending
    uv run python -m tools.dump_vocab --sort count
"""

import argparse
import os
import sys
from collections import Counter

from tools.eval_tokenizer import load_tokenizer, _safe_token_bytes


def slugify_spec(spec: str) -> str:
    """Pick a filesystem-safe slug for the default output filename."""
    if spec in {"ours", "gpt2", "gpt4", "gpt4o"}:
        return spec
    # Path → basename. But our variant convention is `<variant>/tokenizer/`, so
    # if the basename is literally "tokenizer", climb to the parent dir name —
    # otherwise every variant would slug to the same "tokenizer_vocab.txt".
    path = os.path.expanduser(spec).rstrip("/")
    base = os.path.basename(path)
    if base == "tokenizer":
        parent = os.path.basename(os.path.dirname(path))
        if parent:
            return parent
    return base or "tokenizer"


def encode_corpus_counts(tok, split: str, max_chars: int) -> tuple[Counter, int, int]:
    """Encode `split` until `max_chars` consumed; return per-id counts +
    total tokens + total chars."""
    from tools._corpus_iter import iter_capped_batches

    counts: Counter = Counter()
    total_tokens = 0
    total_chars = 0
    for docs in iter_capped_batches(split, max_chars):
        rows = tok.encode(docs)
        for row in rows:
            counts.update(row)
            total_tokens += len(row)
        total_chars += sum(len(d) for d in docs)
    return counts, total_tokens, total_chars


def parse_args():
    p = argparse.ArgumentParser(
        description="Dump tokenizer vocab + corpus counts to /tmp/*_vocab.txt.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--tokenizer",
        default="ours",
        help="Tokenizer spec: 'ours', 'gpt2', 'gpt4', 'gpt4o', or a directory path. "
             "Default: ours (the trained baseline at $NANOCHAT_BASE_DIR/tokenizer).",
    )
    p.add_argument(
        "--output", "-o",
        default=None,
        help="Output path. Default: /tmp/{tokenizer-slug}_vocab.txt.",
    )
    p.add_argument(
        "--split",
        default="val",
        choices=["train", "val"],
        help="ClimbMix split to count over (default: val).",
    )
    p.add_argument(
        "--max-chars",
        type=int,
        default=20_000_000,
        help="Max chars to tokenize for counts (default: 20M ~ 1-2 min). "
             "Pass 0 to skip counting entirely.",
    )
    p.add_argument(
        "--sort",
        default="rank",
        choices=["rank", "count"],
        help="Sort rows by 'rank' ascending (default) or 'count' descending.",
    )
    return p.parse_args()


def main():
    args = parse_args()

    out_path = args.output or f"/tmp/{slugify_spec(args.tokenizer)}_vocab.txt"
    if not out_path.startswith("/tmp/"):
        print(f"warning: --output {out_path!r} is outside /tmp", file=sys.stderr)

    print(f"Loading tokenizer: {args.tokenizer}")
    tok = load_tokenizer(args.tokenizer)
    V = tok.get_vocab_size()
    special_set = set(tok.get_special_tokens())
    print(f"  vocab_size={V}  num_special={len(special_set)}")

    counts: Counter = Counter()
    total_tokens = 0
    total_chars = 0
    if args.max_chars > 0:
        print(f"Counting token firings on ClimbMix {args.split} (cap {args.max_chars:,} chars)...")
        counts, total_tokens, total_chars = encode_corpus_counts(
            tok, args.split, args.max_chars
        )
        print(f"  tokenized {total_chars:,} chars → {total_tokens:,} tokens")
        print(f"  seen {len(counts):,} distinct ids ({V - len(counts):,} dead)")
    else:
        print("Skipping corpus counts (--max-chars 0).")

    # Build rows
    rows = []
    for tid in range(V):
        b = _safe_token_bytes(tok, tid)
        if b is None:
            # vocab gap (e.g. cl100k_base reserves id 100256)
            rows.append((tid, 0, 0, "<vocab-gap>"))
            continue
        try:
            s = b.decode("utf-8")
            is_special = s in special_set
        except UnicodeDecodeError:
            is_special = False
        # bytes-repr keeps weird whitespace/escape visible and column-safe
        disp = repr(b)
        rows.append((tid, int(counts.get(tid, 0)), len(b), disp + (" [SPECIAL]" if is_special else "")))

    if args.sort == "count":
        rows.sort(key=lambda r: (-r[1], r[0]))
    # else: already in rank order

    # Fixed-width columns. Numeric columns right-aligned; repr is the
    # variable-width tail. Widths are wide enough to cover 32K+ vocab and
    # 7-digit firing counts comfortably.
    col_rank, col_count, col_blen = 8, 12, 10
    header_row = (
        f"{'rank':>{col_rank}}  {'count':>{col_count}}  "
        f"{'byte_len':>{col_blen}}  repr"
    )

    with open(out_path, "w") as f:
        # Header: includes provenance so the file is self-describing
        f.write(f"# tokenizer={args.tokenizer}  vocab_size={V}  num_special={len(special_set)}\n")
        if args.max_chars > 0:
            f.write(
                f"# counts from ClimbMix {args.split}: "
                f"{total_chars:,} chars, {total_tokens:,} tokens, "
                f"{len(counts):,} seen / {V - len(counts):,} dead\n"
            )
        else:
            f.write("# counts: skipped (--max-chars 0)\n")
        f.write(f"# sort={args.sort}\n")
        f.write(header_row + "\n")
        for rank, count, byte_len, disp in rows:
            f.write(
                f"{rank:>{col_rank}}  {count:>{col_count}}  "
                f"{byte_len:>{col_blen}}  {disp}\n"
            )

    print(f"Wrote {len(rows):,} rows to {out_path}")


if __name__ == "__main__":
    main()
