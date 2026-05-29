"""
Tokenizer variant: manually-specified merges, applied FIRST (highest priority).

Hand-written counterpart to seed_tokens. Reads `(L, R)` pairs from
`rustbpe_variants/manual_merges/pairs.py`, forms `S = L + R`, and inserts each
into `mergeable_ranks` at its **natural-firing rank** `max(id_L, id_R) + 1`
(clamped to 256). For byte-bigram pairs (single-char operands), that clamp lands
them at rank 256 — i.e. the FIRST merges, so they win the merge race wherever
their operands are adjacent. This is the intended semantics for manual_merges:
the merges are done *first*, not last.

Contrast with the two other forced-merge engines:
  - `force_merges` (rustbpe_force_merges crate, `apply_forced_merges_at_end`):
    appends forced concats at the END (lowest priority). Right when you want to
    force a pair that otherwise *can't* be mergeable (paired with a regex
    carve-out for cross-boundary phrases). Wrong here — appended tokens barely
    fire.
  - `seed_tokens`: same insertion mechanism as this file, but its routes are
    mined and it SKIPS a route whose concat already exists. manual_merges instead
    PROMOTES an existing concat to the front (the whole point is priority).

Base BPE runs on the `rustbpe_force_merges` crate (reused for its `blocked_pairs=`
support; `forced_pairs` stays empty — seeds go in via pure-Python insertion).
BLOCKED_PAIRS forbid specific merges during training. `SPLIT_PATTERN` is the
standard regex minus one tweak — opening delimiters `" ( , “` no longer glue to
the following word (see `apply_patches` / pairs.py) — so pairs are still
within-chunk / sub-word compositions.

Usage:
    uv run python -m wrappers.tok_train_manual_merges
      →  ~/.cache/nanochat-variants/manual_merges/tokenizer/
"""
import runpy
import sys

from wrappers._tokenizer_variant_common import (
    setup_variant_base,
    alias_rustbpe,
    print_downstream_hint,
)


def apply_patches():
    """Install the train_from_iterator shim. Base BPE runs on the
    `rustbpe_force_merges` crate (for `blocked_pairs=` support; `forced_pairs`
    stays empty since seeds go in via pure-Python front-insertion). MANUAL_PAIRS
    are inserted at their natural rank; BLOCKED_PAIRS are forbidden during
    training. Returns MANUAL_PAIRS. Patches SPLIT_PATTERN with one small tweak
    (stop opening-delimiter junk `" ( , “` from gluing to words — see pairs.py)."""
    alias_rustbpe("rustbpe_force_merges")

    import nanochat.tokenizer as tk
    from rustbpe_variants.manual_merges import pairs as _pairs_mod

    # Regex tweak (the one departure from the standard regex — see the note in
    # pairs.py): drop the opening-delimiter junk (" ( , “) from the leading-char
    # word-gluing clause, so '"The', '(x', ',y' etc. don't form delimiter+word
    # tokens. Space / - / . / gluing is untouched. Derived from the live pattern
    # so it tracks upstream changes; asserts the clause is present so a regex
    # change upstream fails loudly instead of silently no-op'ing.
    _GLUE = r"[^\r\n\p{L}\p{N}]?+\p{L}+"
    _GLUE_NEW = r'[^\r\n\p{L}\p{N}"(,“]?+\p{L}+'
    if _GLUE_NEW not in tk.SPLIT_PATTERN:  # idempotent: skip if already patched
        assert _GLUE in tk.SPLIT_PATTERN, (
            "manual_merges: word-gluing clause not found in SPLIT_PATTERN "
            "(upstream regex changed?) — update the tweak in tok_train_manual_merges.py"
        )
        tk.SPLIT_PATTERN = tk.SPLIT_PATTERN.replace(_GLUE, _GLUE_NEW)

    def _patched_train_classmethod(text_iterator, vocab_size):
        import rustbpe  # aliased to rustbpe_force_merges (for blocked_pairs)
        import tiktoken
        from nanochat.tokenizer import SPECIAL_TOKENS

        MANUAL_PAIRS = _pairs_mod.MANUAL_PAIRS
        BLOCKED_PAIRS = _pairs_mod.BLOCKED_PAIRS
        # routes: (L_bytes, R_bytes, S_bytes)
        routes = [(L.encode("utf-8"), R.encode("utf-8"), (L + R).encode("utf-8"))
                  for (L, R) in MANUAL_PAIRS]
        n = len(routes)
        vocab_size_no_special = vocab_size - len(SPECIAL_TOKENS)
        # Slot reservation DISABLED: natural BPE trains to the full vocab and
        # the seeds are added on top. Final vocab = vocab_size + (count of
        # genuinely-new seeds); promoted-existing seeds don't change the count.
        # (With reservation, promoted seeds left their reserved slots empty and
        # the vocab came up short — disabling avoids that.)
        natural_target = vocab_size_no_special
        assert natural_target >= 256

        tokenizer = rustbpe.Tokenizer()
        tokenizer.train_from_iterator(
            text_iterator, natural_target, pattern=tk.SPLIT_PATTERN,
            forced_pairs=[],                # seeds inserted in Python, not here
            blocked_pairs=BLOCKED_PAIRS,     # forbid these merges during training
        )
        pattern = tokenizer.get_pattern()
        mergeable_ranks_list = tokenizer.get_mergeable_ranks()
        natural_ranks = {bytes(k): v for k, v in mergeable_ranks_list}

        ids = [v for _, v in mergeable_ranks_list]
        n_unique = len(set(ids))
        max_natural_id = max(ids)
        if max_natural_id >= n_unique:
            missing = sorted(set(range(max_natural_id + 1)) - set(ids))
            raise ValueError(
                f"pristine rustbpe emitted sparse ids: {n_unique} unique, "
                f"max={max_natural_id}. Missing: {missing[:10]}"
                f"{'...' if len(missing) > 10 else ''}."
            )
        if n_unique < natural_target:
            import warnings
            warnings.warn(
                f"pristine rustbpe filled only {n_unique}/{natural_target} "
                f"tokens. Likely cause: corpus too small.", stacklevel=2,
            )

        # Compute insertion points. Each manual pair lands at max(id_L,id_R)+1
        # (clamped to 256). If S already exists naturally, PROMOTE it: it'll be
        # excluded from the natural stream and re-placed at the insertion point.
        inserts: list[tuple[int, bytes]] = []
        promoted: set[bytes] = set()
        skipped_no_L = skipped_no_R = 0
        for L_b, R_b, S_b in routes:
            id_L = natural_ranks.get(L_b)
            id_R = natural_ranks.get(R_b)
            if id_L is None:
                skipped_no_L += 1
                continue
            if id_R is None:
                skipped_no_R += 1
                continue
            inserts.append((max(max(id_L, id_R) + 1, 256), S_b))
            if S_b in natural_ranks:
                promoted.add(S_b)
        inserts.sort(key=lambda x: x[0])  # stable → MANUAL_PAIRS order within ties

        # Rebuild: byte tokens keep ranks 0..255; manual pairs interleave with
        # natural merges by insertion id; promoted concats are dropped from the
        # natural stream so they appear only at their (earlier) insertion point.
        mergeable_ranks: dict[bytes, int] = {bytes([i]): i for i in range(256)}
        merges_only = sorted(
            ((b, i) for b, i in natural_ranks.items() if i >= 256),
            key=lambda x: x[1],
        )
        out_id = 256
        it = iter(inserts)
        nxt = next(it, None)
        for nat_b, nat_id in merges_only:
            while nxt is not None and nxt[0] <= nat_id:
                if nxt[1] not in mergeable_ranks:
                    mergeable_ranks[nxt[1]] = out_id
                    out_id += 1
                nxt = next(it, None)
            if nat_b in promoted:
                continue  # already (or will be) placed at its insertion point
            mergeable_ranks[nat_b] = out_id
            out_id += 1
        while nxt is not None:
            if nxt[1] not in mergeable_ranks:
                mergeable_ranks[nxt[1]] = out_id
                out_id += 1
            nxt = next(it, None)

        applied = len(inserts)
        front = sum(1 for ip, _ in inserts if ip == 256)
        print(
            f"==> manual merges applied: {applied}/{n} "
            f"(at-front rank-256: {front}; promoted-existing: {len(promoted)}; "
            f"skipped: L={skipped_no_L}, R={skipped_no_R})"
        )

        tokens_offset = len(mergeable_ranks)
        special_tokens = {name: tokens_offset + i for i, name in enumerate(SPECIAL_TOKENS)}
        enc = tiktoken.Encoding(
            name="rustbpe_manual_merges", pat_str=pattern,
            mergeable_ranks=mergeable_ranks, special_tokens=special_tokens,
        )
        return tk.RustBPETokenizer(enc, "<|bos|>")

    tk.RustBPETokenizer.train_from_iterator = classmethod(
        lambda cls, text_iterator, vocab_size: _patched_train_classmethod(text_iterator, vocab_size)
    )
    return _pairs_mod.MANUAL_PAIRS


def main():
    variant_base = setup_variant_base("manual_merges")
    manual = apply_patches()
    from rustbpe_variants.manual_merges.pairs import RECOMMENDED_VOCAB_SIZE

    user_args = sys.argv[1:]
    has_vocab_arg = any(
        a == "--vocab-size" or a.startswith("--vocab-size=") for a in user_args
    )
    if has_vocab_arg:
        effective_vocab = "caller-supplied (see --vocab-size in args below)"
    else:
        user_args = ["--vocab-size", str(RECOMMENDED_VOCAB_SIZE)] + user_args
        effective_vocab = f"{RECOMMENDED_VOCAB_SIZE}"
    from rustbpe_variants.manual_merges.pairs import BLOCKED_PAIRS
    print(f"==> variant_base:    {variant_base}")
    print(f"==> rustbpe module:  {__import__('rustbpe').__file__}")
    print('==> SPLIT_PATTERN:   standard, minus opening-delimiter gluing (" ( , “)')
    print(f"==> manual pairs:    {len(manual)} (seeds @ max(id_L,id_R)+1, byte-bigrams → rank 256)")
    print(f"==> blocked pairs:   {len(BLOCKED_PAIRS)}")
    print(f"==> vocab_size:      {effective_vocab}")
    print(f"==> tok_train args:  {user_args}")
    sys.argv = ["tok_train.py"] + user_args
    runpy.run_module("scripts.tok_train", run_name="__main__")
    print_downstream_hint(variant_base)


if __name__ == "__main__":
    main()
