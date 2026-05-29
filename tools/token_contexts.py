"""
token_contexts.py — for one or more no-space tokens, show which words each fires
inside and its bare-word %, to classify it for the manual_merges blocking
decision. This is the "check the firings, don't eyeball it" step from
`rustbpe_variants/manual_merges/pairs.py`'s curation methodology, as a one-liner
(we got `red`/`bed`/`led` wrong by guessing — the firing context is the call).

For each token T (given WITHOUT a leading space), over a corpus sample, on the
BASELINE tokenizer by default (its *natural* behavior — what a block would fix):
  - total firings + the pre-tok chunks (words) T actually appears in
  - bare-word %: the share where the word IS T (T standing as the whole word)

bare-word % is the discriminator:
  - high  → T is the real word          → leave it      ("river", e.g. need ~50%)
  - ~0 + T is a dict word → parasite     → block          ("king"/"bed", e.g. led 2%)
  - ~0 + T not a word     → mangle/frag  → block          (e.g. ated, oment)
  - middling → mixed                     → eyeball the word list (e.g. red 11%)

The high/low cutoffs (40% / 8%) are a guide, not gospel — read the word list.
Remember the *no-space* token is what's blocked; the real word usually survives
as the separate space-prefixed ` word` token (so a parasite block is safe).

Usage:
    uv run python -m tools.token_contexts led red ated
    uv run python -m tools.token_contexts --top 20 king thing need oment
    # check against the variant (after blocks) instead of baseline:
    uv run python -m tools.token_contexts \\
        --tokenizer ~/.cache/nanochat-variants/manual_merges/tokenizer ated
"""

import argparse
import os
import sys
from collections import Counter

import regex

from nanochat.tokenizer import RustBPETokenizer
from tools._corpus_iter import iter_capped_batches
from tools.blocked_pairs_report import load_wordlist


def classify(bare_pct, is_word, have_wordlist):
    if bare_pct >= 40:
        return "WHOLE-WORD -> leave (river)"
    if bare_pct < 8:
        if not have_wordlist:
            return "low bare -> block (parasite or mangle)"
        return "PARASITE -> block (king/bed)" if is_word else "MANGLE/FRAGMENT -> block"
    return "MIXED -> eyeball the words"


def formation_routes(enc, token_str):
    """Print the binary splits (L, R) that could merge to form `token_str`, each
    with its operands' ranks, to guide which routes a multi-path kill must cover.
    BPE forms a token by merging two tokens that already exist (lower rank), so:
      - operand absent           -> dead route (can't form the token; no-op block)
      - both operands rank < T    -> LIVE route (forms the token in this tokenizer)
      - an operand rank > T       -> latent (can't form it now — that operand is
                                     newer — but blocking the live routes re-ranks
                                     things, which may open it; recheck after retrain)
    Ranks are this tokenizer's; they shift when you add blocks."""
    tb = token_str.encode("utf-8")
    try:
        rank_t = enc.encode_single_token(tb)
    except Exception:
        return

    def rank_of(b):
        try:
            return enc.encode_single_token(b)
        except Exception:
            return None

    print(f"   formation routes (token rank {rank_t}):")
    for k in range(1, len(tb)):
        lb, rb = tb[:k], tb[k:]
        rl, rr = rank_of(lb), rank_of(rb)
        ls, rs = lb.decode("utf-8", "replace"), rb.decode("utf-8", "replace")
        lr = str(rl) if rl is not None else "absent"
        rrk = str(rr) if rr is not None else "absent"
        if rl is None or rr is None:
            flag = "dead (operand absent)"
        elif rl < rank_t and rr < rank_t:
            flag = "LIVE route (forms it now)"
        else:
            flag = "latent (operand newer than token — recheck after retrain)"
        print(f"      ({ls!r}, {rs!r})  L-rank={lr:<7} R-rank={rrk:<7} -> {flag}")


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("tokens", nargs="+", help="no-space tokens to check, e.g. led red ated")
    p.add_argument(
        "--tokenizer",
        default=os.path.expanduser("~/.cache/nanochat/tokenizer"),
        help="tokenizer to encode with (default: baseline — shows natural behavior).",
    )
    p.add_argument("--split", default="val", help="corpus split (default: val).")
    p.add_argument("--max-chars", type=int, default=8_000_000,
                   help="corpus sample cap (default 8M; bump to 20M for rarer tokens).")
    p.add_argument("--top", type=int, default=12, help="top words to show per token.")
    args = p.parse_args()

    tok = RustBPETokenizer.from_directory(os.path.expanduser(args.tokenizer))
    enc = tok.enc
    pat = regex.compile(enc._pat_str if hasattr(enc, "_pat_str") else enc.pat_str)
    words = load_wordlist()  # may be None (no /usr/share/dict/words)

    # resolve each token to its id, or flag it as not-a-single-token
    tid = {}
    for t in args.tokens:
        try:
            tid[t] = enc.encode_single_token(t.encode("utf-8"))
        except Exception:
            tid[t] = None

    # one corpus pass: count pre-tok chunks (words)
    print(f"scanning {args.max_chars:,} chars of {args.split} via {args.tokenizer} ...",
          file=sys.stderr)
    chunks = Counter()
    for batch in iter_capped_batches(args.split, args.max_chars):
        for doc in batch:
            for m in pat.finditer(doc):
                chunks[m.group()] += 1

    # attribute each word to every target token it contains
    live = {t: i for t, i in tid.items() if i is not None}
    hits = {t: Counter() for t in live}
    for chunk, c in chunks.items():
        ids = set(enc.encode_ordinary(chunk))
        for t, i in live.items():
            if i in ids:
                hits[t][chunk] += c

    for t in args.tokens:
        if tid[t] is None:
            print(f"\n=== {t!r}: not a single token (block would be a no-op) ===")
            continue
        h = hits[t]
        total = sum(h.values())
        if total == 0:
            print(f"\n=== {t!r}: 0 firings in sample (dead — lean block, reclaims a slot) ===")
            formation_routes(enc, t)
            continue
        bare = sum(c for ch, c in h.items() if ch.strip().lower() == t.lower())
        pct = 100 * bare / total
        # Exact membership (not the inflection-stripped _is_word): we want "is T
        # *literally* a word" (king/led/bed) — 'ated' strips to 'at' but isn't a word.
        is_word = bool(words) and t.lower() in words
        verdict = classify(pct, is_word, bool(words))
        wl = "" if words else "  (no wordlist)"
        print(f"\n=== {t!r}: {total} fires | bare-word {bare} ({pct:.0f}%) | "
              f"dict-word={is_word}{wl}  ->  {verdict} ===")
        for ch, c in h.most_common(args.top):
            print(f"   {c:>5}  {ch!r}")
        formation_routes(enc, t)


if __name__ == "__main__":
    main()
