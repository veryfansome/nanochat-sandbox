"""
Per-token usage-delta diagnostic for seed_tokens curation.

Encodes the same held-out corpus (ClimbMix val, by default) with both the
baseline and seed_tokens tokenizers, then tabulates per-token-bytes usage
deltas: baseline tokens whose firing count dropped (displaced by a seed)
and seed tokens whose firing count rose (the seed-mechanism payoff).

The variant vocab is a superset of baseline by construction (same
natural BPE merges + N seeds inserted at mid-training ranks), so nothing
is "lost from vocab" — what changes is encode-time *usage*. A seed that
fires intercepts a composition path baseline would have taken to a
longer single token, fragmenting that span into multiple tokens. This
tool surfaces exactly that ledger:

  - For each token (keyed by its bytes), `count_base` and `count_seed`
    record how many times it appears in the baseline / seed_tokens
    encoding of the corpus.
  - `delta = count_seed - count_base`. Strongly negative ⇒ a baseline
    token that lost most of its usage (a high-cost displacement). Strongly
    positive ⇒ a seed that's actually firing.
  - Net total-tokens regression of seed_tokens equals the sum of
    `-delta` over baselines that lost usage minus the sum of `delta`
    over seeds that fired plus a constant for tokens both share.

The output guides ablation:
  - Drop seeds whose `count_seed` is low (dead seeds; vocab cost without
    encode-time benefit).
  - Drop seeds whose firing strongly correlates with high-count baseline
    losses (anti-productive seeds, blocking longer compositions).

Usage:
    uv run python -m tools.diag_seed_usage \\
        --baseline ~/.cache/nanochat/tokenizer \\
        --variant  ~/.cache/nanochat-variants/seed_tokens/tokenizer \\
        --max-chars 20_000_000 --top-n 50

Roughly matches the eval_tokenizer Tier-2 corpus (ClimbMix val) at 20M
chars by default; bump with --max-chars for a larger sample.
"""

import argparse
import sys
from collections import Counter
from pathlib import Path


def build_bytes_map(tok) -> tuple[dict[bytes, int], dict[int, bytes]]:
    """Recover both bytes→id and id→bytes maps for a tokenizer's mergeable
    vocab (special tokens excluded — their bytes aren't comparable across
    tokenizers and they shouldn't appear in pretraining-corpus encodings).
    """
    enc = tok.enc
    special_set = set(tok.get_special_tokens())
    b2i: dict[bytes, int] = {}
    i2b: dict[int, bytes] = {}
    for tid in range(tok.get_vocab_size()):
        try:
            b = enc.decode_single_token_bytes(tid)
        except Exception:
            continue
        try:
            if b.decode("utf-8") in special_set:
                continue
        except UnicodeDecodeError:
            pass
        b2i[b] = tid
        i2b[tid] = b
    return b2i, i2b


def encode_corpus_count(tok, max_chars: int) -> tuple[Counter, int, int]:
    """Encode ClimbMix val docs until max_chars consumed; return per-id firing
    counts plus total tokens/chars.

    Doc-level trimming via `iter_capped_batches` keeps --max-chars accurate
    to within one doc; both tokenizer encodes consume the same iterator
    deterministically and see the same docs.
    """
    from tools._corpus_iter import iter_capped_batches

    counts: Counter = Counter()
    total_tokens = 0
    total_chars = 0
    for docs in iter_capped_batches("val", max_chars):
        rows = tok.encode(docs)
        for row in rows:
            counts.update(row)
            total_tokens += len(row)
        total_chars += sum(len(d) for d in docs)
    return counts, total_tokens, total_chars


def display_token(b: bytes, max_len: int = 24) -> str:
    """Render token bytes for the diagnostic table. Show as Python bytes
    repr but truncated; bytes-repr keeps weird whitespace/escape visible
    without breaking column alignment."""
    r = repr(b)
    if len(r) > max_len + 2:
        r = r[: max_len + 2] + "..."
    return r


def main(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--baseline", type=Path,
        default=Path("~/.cache/nanochat/tokenizer").expanduser(),
        help="Baseline tokenizer dir",
    )
    p.add_argument(
        "--variant", type=Path,
        default=Path("~/.cache/nanochat-variants/seed_tokens/tokenizer").expanduser(),
        help="Variant tokenizer dir (default: seed_tokens)",
    )
    p.add_argument(
        "--max-chars", type=int, default=20_000_000,
        help="Cap chars consumed from ClimbMix val (default: 20M; matches "
        "the eval_tokenizer Tier-2 sample size)",
    )
    p.add_argument(
        "--top-n", type=int, default=50,
        help="Rows in each delta table (default: 50)",
    )
    args = p.parse_args(argv)

    from nanochat.tokenizer import RustBPETokenizer

    print(f"Loading baseline:  {args.baseline}")
    base = RustBPETokenizer.from_directory(str(args.baseline))
    print(f"Loading variant:   {args.variant}")
    var = RustBPETokenizer.from_directory(str(args.variant))

    print("Indexing vocabs...")
    base_b2i, base_i2b = build_bytes_map(base)
    var_b2i, var_i2b = build_bytes_map(var)
    only_in_var = set(var_b2i) - set(base_b2i)
    only_in_base = set(base_b2i) - set(var_b2i)
    print(
        f"  baseline:  {len(base_b2i):,} mergeable tokens\n"
        f"  variant:   {len(var_b2i):,} mergeable tokens "
        f"(+{len(only_in_var)} unique to variant, "
        f"-{len(only_in_base)} unique to baseline)"
    )
    # If variant is a strict superset of baseline by bytes, every
    # variant-only token IS a seed (no baseline token was displaced from
    # vocab; only encode-time usage shifts). Otherwise — e.g. a different
    # tokenizer like force_merges — variant-only tokens may not be seeds
    # at all, and baseline tokens absent from variant aren't "displaced"
    # in the seed sense. Switch labels accordingly so the report reflects
    # the right mental model. `is_superset` gates the seed-specific
    # wording in the summary + tables below.
    if not only_in_var:
        print(
            "\nNo variant-only tokens — variant adds nothing to baseline. "
            "Nothing to analyze. Did you point --variant at the wrong "
            "tokenizer?",
            file=sys.stderr,
        )
        return
    is_superset = not only_in_base
    if not is_superset:
        print(
            f"\nWARNING: {len(only_in_base):,} baseline tokens are absent "
            f"from variant — variant is NOT a strict superset of baseline "
            f"(e.g., different vocab budget or different mining target). "
            f"Switching labels to generic 'variant-only token' (not "
            f"'seed'); the 'baseline tokens with usage drop' table below "
            f"reflects both displacement AND natural-BPE differences.",
            file=sys.stderr,
        )

    # Vocabulary that gates seed-specific vs. generic phrasing below.
    new_tok = "seed" if is_superset else "variant-only token"
    new_toks = "seeds" if is_superset else "variant-only tokens"
    fired = "fired" if is_superset else "used"
    dead = "dead" if is_superset else "unused"

    print(f"Encoding ClimbMix val ({args.max_chars:,} chars cap)...")
    counts_base, n_base, chars_base = encode_corpus_count(base, args.max_chars)
    counts_var, n_var, chars_var = encode_corpus_count(var, args.max_chars)
    # We re-fetch the corpus per encode; same iterator + same cap → same
    # documents, so chars should match exactly.
    assert chars_base == chars_var, (
        f"corpus mismatch: base saw {chars_base} chars, variant saw {chars_var}"
    )
    # `chars_*` is the codepoint count (from `len(doc)`), not the UTF-8
    # byte count — label as chars/tok, not bytes/tok, to avoid drift on
    # non-ASCII slices. The canonical bytes/token compression metric is
    # in `tools/eval_tokenizer.py`.
    print(
        f"  baseline:  {n_base:,} tokens for {chars_base:,} chars "
        f"({chars_base/n_base:.3f} chars/tok)\n"
        f"  variant:   {n_var:,} tokens for {chars_var:,} chars "
        f"({chars_var/n_var:.3f} chars/tok)\n"
        f"  delta:     {n_var - n_base:+,} tokens "
        f"({(n_var - n_base) / n_base * 100:+.3f}%)"
    )

    # Re-key counts by bytes (the cross-tokenizer identity).
    base_by_bytes = {base_i2b[i]: c for i, c in counts_base.items() if i in base_i2b}
    var_by_bytes = {var_i2b[i]: c for i, c in counts_var.items() if i in var_i2b}

    all_bytes = set(base_by_bytes) | set(var_by_bytes)
    deltas: list[tuple[bytes, int, int]] = [
        (b, base_by_bytes.get(b, 0), var_by_bytes.get(b, 0))
        for b in all_bytes
    ]

    # Categorize.
    blocked = [
        (b, bc, vc) for b, bc, vc in deltas
        if b in base_b2i and bc - vc > 0
    ]  # baseline tokens that lost usage
    blocked.sort(key=lambda r: r[2] - r[1])  # most-negative-delta first

    seed_fired = [
        (b, bc, vc) for b, bc, vc in deltas
        if b in only_in_var and vc > 0
    ]  # genuinely-new seeds with non-zero usage
    seed_fired.sort(key=lambda r: -r[2])

    seed_dead = [b for b in only_in_var if var_by_bytes.get(b, 0) == 0]

    n_total = len(only_in_var)
    n_fired = len(seed_fired)
    fire_total = sum(vc for _, _, vc in seed_fired)
    blocked_loss_total = sum(bc - vc for _, bc, vc in blocked)
    print(
        f"\n{new_toks.capitalize()}: {n_total} added → "
        f"{n_fired} {fired} ({n_fired/n_total*100:.1f}%), "
        f"{len(seed_dead)} {dead} ({len(seed_dead)/n_total*100:.1f}%)"
    )
    drop_attribution = (
        "(seed-mechanism displacement)" if is_superset
        else "(mix of displacement + natural-BPE differences)"
    )
    print(
        f"{new_tok.capitalize()} firings: {fire_total:,} total instances\n"
        f"Baseline loss: {blocked_loss_total:,} instances across "
        f"{len(blocked)} baseline tokens {drop_attribution}"
    )

    # ----- Top-N tables -----
    n = args.top_n

    drop_header = (
        f"Top {n} baseline tokens with LARGEST usage drop"
        + (" (displaced by seeds)" if is_superset
           else " (displacement + natural-BPE differences)")
    )
    print(f"\n=== {drop_header} ===")
    print(f"{'rank':>4}  {'base#':>10}  {'var#':>10}  {'Δ':>10}  "
          f"{'base_id':>8}  {'len':>3}  bytes")
    for i, (b, bc, vc) in enumerate(blocked[:n], 1):
        print(
            f"{i:>4}  {bc:>10,}  {vc:>10,}  {vc - bc:>+10,}  "
            f"{base_b2i[b]:>8}  {len(b):>3}  {display_token(b)}"
        )

    payoff_note = " (seed-mechanism payoff)" if is_superset else ""
    print(f"\n=== Top {n} {new_toks} by usage{payoff_note} ===")
    print(f"{'rank':>4}  {'var#':>10}  {'var_id':>8}  {'len':>3}  bytes")
    for i, (b, bc, vc) in enumerate(seed_fired[:n], 1):
        print(
            f"{i:>4}  {vc:>10,}  {var_b2i[b]:>8}  "
            f"{len(b):>3}  {display_token(b)}"
        )

    cost_note = (" (vocab cost, no benefit)" if is_superset
                 else " (present in variant vocab but never reached at encode time)")
    print(f"\n=== Sample of {min(20, len(seed_dead))} {dead} {new_toks}"
          f"{cost_note} ===")
    for b in sorted(seed_dead, key=lambda x: var_b2i[x])[:20]:
        print(f"  {var_b2i[b]:>8}  {len(b):>3}  {display_token(b)}")


if __name__ == "__main__":
    main()
