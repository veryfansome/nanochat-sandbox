"""
One-shot offline report for a blocked-pairs tokenizer variant (manual_merges /
blocked_morphemes). Run it after each retrain to get, in a single command:

  1. eval_tokenizer  — Tier-1 compression / structural / probe battery vs baseline
  2. pass_metrics    — corpus deltas vs baseline (compression, dead/mangled tokens,
                       firing-rank buckets); watch `mangled remaining` for cascade
                       convergence (it rises mid-cascade, then falls back below baseline)
  3. dump_vocab      — refresh the variant + baseline vocab dumps (with firing counts)
  4. whole-word count — space-prefixed dictionary-word tokens in baseline vs
                       variant vocab, split into base (direct dict hit = coverage,
                       the number to watch) and inflected (strip-matched; expected
                       to fall as you block inflectional merges). Optional — skipped
                       if /usr/share/dict/words is absent (macOS ships it).
  5. blocked-R diff  — for every distinct R in the variant's `BLOCKED_PAIRS`, the
                       standalone token's firing-count change baseline -> variant.
                       Blocking the `(L, R)` formations forces those words through the
                       clean suffix token, so R's standalone firing rises with the
                       number of pairs blocked into it.
  6. cleanup cands   — blocked pairs that are now no-ops (an operand no longer
                       exists as a token, so the merge can't fire) — safe to drop.
                       State-dependent, so re-derived each run.
  7. latent routes   — blocks that are latent in baseline (an operand outranks the
                       target token, so this route isn't its natural formation).
                       Informational, NOT a prune list — re-ranking can make a
                       latent route live; confirm load-bearing via a leave-out retrain.

Steps 1-3 just shell out to the existing tools (identical output); steps 4-7
(whole-word count + blocked-R diff + cleanup candidates + latent routes) are new,
computed from the step-3 dumps.

Usage:
    # defaults: manual_merges variant vs baseline, dumps to /tmp, 20M-char corpus
    uv run python -m tools.blocked_pairs_report

    # blocked_morphemes (or any blocked-pairs variant)
    uv run python -m tools.blocked_pairs_report \\
        --variant ~/.cache/nanochat-variants/blocked_morphemes/tokenizer \\
        --pairs-module rustbpe_variants.blocked_morphemes.blocks

    # skip the (slower) eval battery; force a fresh baseline dump
    uv run python -m tools.blocked_pairs_report --skip-eval --fresh-baseline

The baseline vocab dump is cached at `<dump-dir>/baseline_vocab.txt` and reused
across runs (it doesn't change); pass --fresh-baseline to regenerate it. Keep
--max-chars identical between the variant and baseline dumps or the diff compares
mismatched corpora.
"""

import argparse
import ast
import importlib
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path


# R values that show up as the RHS of a blocked pair but are NOT suffixes whose
# standalone-token firing we want to track (e.g. the silent-e in ('itat','e')).
# The blocked-R section skips these. Crude denylist on purpose — extend as junk
# R's appear; swap for a word-start-prob gate later if it's worth the corpus pass.
NON_SUFFIX_R = {
    "ess",
    "s",
    "ss",
}


def _run_tool(module_args, label):
    print(f"\n### {label} ###", flush=True)
    # Reuse the existing tools verbatim (same venv python, same output).
    subprocess.run([sys.executable, "-m", *module_args], check=False)


def load_counts(dump_path):
    """Parse a dump_vocab output file -> {token_bytes: firing_count}."""
    counts = {}
    with open(dump_path) as f:
        for line in f:
            if line.startswith("#"):
                continue
            s = line.strip()
            if not s or s.startswith("rank"):
                continue
            parts = s.split(None, 3)
            if len(parts) < 4:
                continue
            try:
                counts[ast.literal_eval(parts[3])] = int(parts[1])
            except (ValueError, SyntaxError):
                pass
    return counts


WORDLIST_PATH = "/usr/share/dict/words"
_INFLECTIONS = ("s", "es", "ed", "ing", "d")


def load_wordlist(path=WORDLIST_PATH):
    """Lowercased word set from a system wordlist, or None if absent. Drives the
    optional whole-word vocab count; skipped when missing (common on minimal
    Linux — tokenizer work is done locally on macOS, where this ships by default)."""
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8", errors="ignore") as f:
        return frozenset(w.strip().lower() for w in f if w.strip())


def _is_word(text, words):
    """text (lowercased) is a whole word if it's in the list, or a light
    inflectional strip of it is: governments->government, runs->run, cities->city."""
    if text in words:
        return True
    for suf in _INFLECTIONS:
        if len(text) > len(suf) + 1 and text.endswith(suf) and text[: -len(suf)] in words:
            return True
    if len(text) > 4 and text.endswith("ies") and text[:-3] + "y" in words:
        return True
    return False


def whole_word_count(dump_path, words):
    """Count *space-prefixed* dictionary-word tokens in a vocab dump, split into:
      base      — the token (sans leading space, lowercased) is *directly* in the
                  list (e.g. ' government') -> vocabulary coverage
      inflected — not a direct hit, but a one-step inflection strip of it is
                  (' governments' -> government) -> forms that blocking inflectional
                  merges is expected to shed
    Returns (base, inflected). Leading-space-required so suffix tokens ('ing','ate')
    aren't counted as their homographic free words."""
    base = inflected = 0
    for tok in load_counts(dump_path):
        try:
            s = tok.decode("utf-8")
        except UnicodeDecodeError:
            continue
        if not s.startswith(" "):
            continue
        w = s[1:].lower()
        if w in words:
            base += 1
        elif _is_word(w, words):  # direct hit handled above, so this is strip-only
            inflected += 1
    return base, inflected


def whole_word_diff(baseline_dump, variant_dump, variant_dir):
    words = load_wordlist()
    print("\n### whole-word vocab tokens (space-prefixed dictionary words) ###")
    if words is None:
        print(f"(skipped — no wordlist at {WORDLIST_PATH})")
        return
    bb, bi = whole_word_count(baseline_dump, words)
    vb, vi = whole_word_count(variant_dump, words)
    print(f"wordlist: {WORDLIST_PATH} ({len(words):,} words)")
    print(f"  base (direct dict hit) : baseline {bb:>6,}  variant {vb:>6,}  diff {vb - bb:>+5,}"
          f"   (coverage — watch this)")
    print(f"  inflected (strip-only) : baseline {bi:>6,}  variant {vi:>6,}  diff {vi - bi:>+5,}"
          f"   (expected to fall as you block inflections)")
    print(f"  total whole words      : baseline {bb + bi:>6,}  variant {vb + vi:>6,}  "
          f"diff {(vb + vi) - (bb + bi):>+5,}")
    _lost_base_words(baseline_dump, variant_dump, words, variant_dir, top=20)


def _lost_base_words(baseline_dump, variant_dump, words, variant_dir, top=20):
    """Base-word tokens (space-prefixed, directly in the dict) that existed in the
    baseline but are gone from the variant — i.e. real words that lost their own
    token to blocking. Sorted by *baseline* firing count (how much fragmentation
    that loss costs), and showing how each now breaks up in the variant, to
    investigate whether anything load-bearing got displaced."""
    base_counts = load_counts(baseline_dump)
    variant_vocab = set(load_counts(variant_dump))
    lost = []
    for tok, c in base_counts.items():
        if not isinstance(tok, bytes):
            continue
        try:
            s = tok.decode("utf-8")
        except UnicodeDecodeError:
            continue
        if s.startswith(" ") and s[1:].lower() in words and tok not in variant_vocab:
            lost.append((s, c))
    lost.sort(key=lambda x: -x[1])

    print(f"  top {top} base words lost (token gone from variant), by baseline firing:")
    if not lost:
        print("    (none)")
        return
    # load the variant tokenizer to show how each lost word now breaks up
    from nanochat.tokenizer import RustBPETokenizer
    enc = RustBPETokenizer.from_directory(os.path.expanduser(variant_dir)).enc
    for s, c in lost[:top]:
        pieces = [enc.decode_single_token_bytes(i).decode("utf-8", "replace")
                  for i in enc.encode_ordinary(s)]
        print(f"    {c:>6}  {s!r:<22} -> {pieces}")


def blocked_r_diff(pairs_module, baseline_dump, variant_dump):
    mod = importlib.import_module(pairs_module)
    blocked = getattr(mod, "BLOCKED_PAIRS", None)
    if not blocked:
        print(f"\n(no BLOCKED_PAIRS in {pairs_module}; skipping blocked-R diff)")
        return
    rs = Counter(R for (_L, R) in blocked)
    skipped = sorted(R for R in rs if R in NON_SUFFIX_R)
    rs = {R: n for R, n in rs.items() if R not in NON_SUFFIX_R}
    base = load_counts(baseline_dump)
    var = load_counts(variant_dump)

    print("\n### blocked-R standalone firing-count diff (baseline -> variant) ###")
    print(f"total blocked pairs: {len(blocked)} | suffixes reported: {len(rs)}"
          + (f" | skipped non-suffix R: {skipped}" if skipped else ""))
    print(f"{'R':>10} {'#blk':>5} | {'baseline':>9} {'variant':>9} {'diff':>9} {'pct':>8}")
    print("-" * 60)
    for R, n in sorted(rs.items(), key=lambda kv: -kv[1]):
        rb = R.encode("utf-8")
        bc, mc = base.get(rb, 0), var.get(rb, 0)
        pct = f"{100 * (mc - bc) / bc:+.0f}%" if bc else ("new" if mc else "—")
        print(f"{R!r:>10} {n:>5} | {bc:>9} {mc:>9} {mc - bc:>+9} {pct:>8}")


def cleanup_candidates(pairs_module, variant_dump):
    """Blocked pairs that can never fire because an operand no longer exists as a
    token in the (retrained) variant — so the block is a no-op and is safe to drop.
    Caveat: state-dependent — an absent operand can reappear after later edits, so
    re-derive this each pass rather than treating it as a permanent delete list."""
    mod = importlib.import_module(pairs_module)
    blocked = getattr(mod, "BLOCKED_PAIRS", None)
    if not blocked:
        return
    vocab = set(load_counts(variant_dump))  # token byte-strings present in the variant
    noop = []
    for (L, R) in blocked:
        miss = []
        if L.encode("utf-8") not in vocab:
            miss.append(f"L={L!r}")
        if R.encode("utf-8") not in vocab:
            miss.append(f"R={R!r}")
        if miss:
            noop.append(((L, R), ", ".join(miss)))

    print("\n### cleanup candidates: no-op blocks (an operand is gone -> can't fire) ###")
    print(f"{len(noop)} of {len(blocked)} blocks can never fire; safe to drop "
          f"(re-derive each pass — an operand can come back)")
    for pair, miss in noop[:60]:
        print(f"  {str(pair):<26} {miss}")
    if len(noop) > 60:
        print(f"  ... +{len(noop) - 60} more")


def load_ranks(dump_path):
    """Parse a dump_vocab output file -> {token_bytes: rank} (the dump's 1st column)."""
    ranks = {}
    with open(dump_path) as f:
        for line in f:
            if line.startswith("#"):
                continue
            s = line.strip()
            if not s or s.startswith("rank"):
                continue
            parts = s.split(None, 3)
            if len(parts) < 4:
                continue
            try:
                ranks[ast.literal_eval(parts[3])] = int(parts[0])
            except (ValueError, SyntaxError):
                pass
    return ranks


def latent_routes(pairs_module, baseline_dump, top=40):
    """Blocked pairs that are *latent in baseline*: the target token T = L+R forms
    in baseline, but this (L, R) route has an operand that outranks T — so it isn't
    T's natural formation path (BPE merges two tokens that already exist, lower rank).
    NOT a prune list: blocking the live routes re-ranks the vocab and can promote a
    latent route to live, so whether one is load-bearing needs a leave-out retrain.
    Surfaced to track which blocks are 'extra' formation-path coverage."""
    mod = importlib.import_module(pairs_module)
    blocked = getattr(mod, "BLOCKED_PAIRS", None)
    if not blocked:
        return
    ranks = load_ranks(baseline_dump)
    latent = []
    for (L, R) in blocked:
        rT = ranks.get((L + R).encode("utf-8"))
        rL = ranks.get(L.encode("utf-8"))
        rR = ranks.get(R.encode("utf-8"))
        if rT is None or rL is None or rR is None:
            continue  # target/operand not a baseline token — a different case, skip
        if rL >= rT or rR >= rT:
            gap = max(rL, rR) - rT  # how far the latent operand formed past the target
            latent.append(((L, R), L + R, rT, rL, rR, gap))
    latent.sort(key=lambda x: -x[5])  # biggest latency gap first

    print("\n### latent-in-baseline routes (operand outranks target -> not its natural formation) ###")
    print(f"{len(latent)} of {len(blocked)} blocks are latent; investigate, DON'T auto-drop "
          f"(re-ranking can make a latent route live — needs a leave-out retrain to confirm)")
    print(f"  (sorted by latency gap = how far the later operand outranks the target)")
    for pair, concat, rT, rL, rR, gap in latent[:top]:
        print(f"  {str(pair):<22} -> {concat!r:<13} target={rT:<6} L={rL:<6} R={rR:<6} (latent by +{gap})")
    if len(latent) > top:
        print(f"  ... +{len(latent) - top} more")


def slugify(tokenizer_dir):
    """`.../<variant>/tokenizer` -> `<variant>`; else the basename."""
    path = Path(tokenizer_dir.rstrip("/"))
    if path.name == "tokenizer" and path.parent.name:
        return path.parent.name
    return path.name or "variant"


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--variant",
        default=os.path.expanduser("~/.cache/nanochat-variants/manual_merges/tokenizer"),
        help="Variant tokenizer dir (default: manual_merges).",
    )
    p.add_argument(
        "--baseline",
        default=os.path.expanduser("~/.cache/nanochat/tokenizer"),
        help="Baseline tokenizer dir.",
    )
    p.add_argument(
        "--pairs-module",
        default="rustbpe_variants.manual_merges.pairs",
        help="Importable module exposing BLOCKED_PAIRS (default: manual_merges).",
    )
    p.add_argument("--dump-dir", default="/tmp", help="Where to write vocab dumps.")
    p.add_argument("--max-chars", type=int, default=20_000_000,
                   help="Corpus cap for pass_metrics + dumps (keep consistent).")
    p.add_argument("--skip-eval", action="store_true", help="Skip eval_tokenizer.")
    p.add_argument("--fresh-baseline", action="store_true",
                   help="Re-dump the baseline vocab even if a cached dump exists.")
    args = p.parse_args()

    variant = os.path.expanduser(args.variant)
    baseline = os.path.expanduser(args.baseline)
    mc = str(args.max_chars)
    variant_dump = os.path.join(args.dump_dir, f"{slugify(variant)}_vocab.txt")
    baseline_dump = os.path.join(args.dump_dir, "baseline_vocab.txt")

    if not args.skip_eval:
        _run_tool(["tools.eval_tokenizer", "--tokenizers", f"ours,{variant}"],
                  "eval_tokenizer")
    _run_tool(["tools.pass_metrics", "--variant", variant,
               "--baseline", baseline, "--max-chars", mc], "pass_metrics")
    _run_tool(["tools.dump_vocab", "--tokenizer", variant,
               "--output", variant_dump, "--max-chars", mc], "dump_vocab (variant)")
    if args.fresh_baseline or not os.path.exists(baseline_dump):
        _run_tool(["tools.dump_vocab", "--tokenizer", baseline,
                   "--output", baseline_dump, "--max-chars", mc], "dump_vocab (baseline)")
    else:
        print(f"\n(reusing cached baseline dump {baseline_dump}; "
              f"--fresh-baseline to refresh)")

    whole_word_diff(baseline_dump, variant_dump, variant)
    blocked_r_diff(args.pairs_module, baseline_dump, variant_dump)
    cleanup_candidates(args.pairs_module, variant_dump)
    latent_routes(args.pairs_module, baseline_dump)


if __name__ == "__main__":
    main()
