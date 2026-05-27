"""
Tokenizer variant: data-driven seed tokens via natural-route rank insertion.

Loads `(L_bytes, R_bytes, S_bytes)` routes from
`rustbpe_variants/seed_tokens/routes.py` (mined and curated separately —
see that module's docstring for the current pipeline) and inserts each
seed into `mergeable_ranks` at its **natural-firing rank**: the rank
`max(id_L, id_R) + 1` it would have had if introduced mid-BPE-training.
Natural merges with id ≥ the insertion point shift up by 1 per inserted
seed, preserving order. Tiktoken applies the merges at encode time via
standard rank-based greedy matching.

Why mid-rank insertion (not append-at-end): natural BPE merges that
consume one of a seed's operands have lower rank than an appended seed
and fire first, leaving no `(L, R)` adjacent pair to match. Smoke
witness: seed `(q, q) → qq` appended at rank ~328 got preempted by a
natural ` q` merge at rank ~271. Mid-rank insertion places each seed
where no preempting natural merge can exist yet (both operands still
bare at that point in the BPE merge sequence).

No Rust changes needed. Uses pristine `rustbpe` (sandbox/rustbpe via
runs/build_rustbpe.sh) for normal BPE; seed insertion is pure Python.
The previous YAML-driven Rust crate (constructive merge-chain builder
paying `len(seed) - 1` slots per seed) is preserved in-tree for history
but no longer referenced — data-driven routes use 1 slot per seed.

Usage:
    uv run python -m wrappers.tok_train_seed_tokens
    # → ~/.cache/nanochat-variants/seed_tokens/tokenizer/
"""
import runpy
import sys

from wrappers._tokenizer_variant_common import (
    setup_variant_base,
    print_downstream_hint,
)


def apply_patches():
    """
    Install the train_from_iterator shim. No rustbpe alias — uses pristine.

    The shim looks up SEED_ROUTES *dynamically* from the routes module on
    each call, so tests can swap in synthetic routes without re-installing.
    Returns the currently-loaded SEED_ROUTES for inspection.

    Must be called AFTER setup_variant_base() (so NANOCHAT_BASE_DIR is set)
    and BEFORE any nanochat.tokenizer import (so the patch wins).
    """
    import nanochat.tokenizer as tk
    from rustbpe_variants.seed_tokens import routes as _routes_mod

    def _patched_train_classmethod(text_iterator, vocab_size):
        # `vocab_size` is the TOTAL final vocab including specials (matches
        # nanochat's --vocab-size semantics). We reserve `len(SEED_ROUTES)`
        # slots from the natural BPE budget so the final vocab matches the
        # caller's request when every route applies. If a route can't
        # apply (operand missing, or S already minted naturally) its slot
        # stays empty — final vocab will be slightly smaller. We do not
        # backfill the unused slots; tightness isn't worth the complexity.
        import rustbpe  # pristine
        import tiktoken
        from nanochat.tokenizer import SPECIAL_TOKENS

        SEED_ROUTES = _routes_mod.SEED_ROUTES
        n_routes = len(SEED_ROUTES)
        vocab_size_no_special = vocab_size - len(SPECIAL_TOKENS)
        natural_target = vocab_size_no_special - n_routes
        assert natural_target >= 256, (
            f"natural_target={natural_target} too small after reserving "
            f"{n_routes} seed slots from vocab_size_no_special="
            f"{vocab_size_no_special}"
        )

        tokenizer = rustbpe.Tokenizer()
        tokenizer.train_from_iterator(
            text_iterator, natural_target,
            pattern=tk.SPLIT_PATTERN,
        )
        pattern = tokenizer.get_pattern()
        mergeable_ranks_list = tokenizer.get_mergeable_ranks()
        # Pristine rustbpe emits bytes 0..255 first (ids 0..255), then
        # merges (ids 256..255+n_merges). natural_ranks contains both.
        natural_ranks = {bytes(k): v for k, v in mergeable_ranks_list}

        # Defensive: ids should be dense 0..max. A hole would break the
        # rank-based interleaving below (and indicate a rustbpe regression).
        ids = [v for _, v in mergeable_ranks_list]
        n_unique = len(set(ids))
        max_natural_id = max(ids)
        if max_natural_id >= n_unique:
            missing = sorted(set(range(max_natural_id + 1)) - set(ids))
            raise ValueError(
                f"pristine rustbpe emitted sparse ids: {n_unique} unique, "
                f"max={max_natural_id}. Missing in [0, {max_natural_id}]: "
                f"{missing[:10]}{'...' if len(missing) > 10 else ''}. "
                f"Cannot safely interleave seeds when natural ids have holes."
            )
        if n_unique < natural_target:
            import warnings
            warnings.warn(
                f"pristine rustbpe filled only {n_unique}/{natural_target} "
                f"tokens (256 bytes + {n_unique - 256} merges of "
                f"{natural_target - 256} target). Likely cause: corpus too "
                f"small. Seeds will still be inserted at their natural ranks.",
                stacklevel=2,
            )

        # Compute mid-training-style insertion points. Each seed's rank in
        # the final mergeable_ranks should be `max(id_L, id_R) + 1` — where
        # it would have been minted if introduced mid-BPE. Clamp to 256
        # because ids 0..255 are reserved for byte tokens.
        seed_inserts: list[tuple[int, bytes]] = []
        skipped_no_L = skipped_no_R = skipped_S_exists = 0
        for L_bytes, R_bytes, S_bytes in SEED_ROUTES:
            # Tiktoken merges (L, R) at encode time iff a token has bytes
            # L+R. If S ≠ L+R the seed never fires; guard against corruption.
            assert L_bytes + R_bytes == S_bytes, (
                f"seed route bytes inconsistent: {L_bytes!r} + "
                f"{R_bytes!r} != {S_bytes!r}"
            )
            id_L = natural_ranks.get(L_bytes)
            id_R = natural_ranks.get(R_bytes)
            if id_L is None:
                skipped_no_L += 1
                continue
            if id_R is None:
                skipped_no_R += 1
                continue
            if S_bytes in natural_ranks:
                skipped_S_exists += 1
                continue
            seed_inserts.append((max(max(id_L, id_R) + 1, 256), S_bytes))

        # Interleave seeds with natural merges by insertion id. Stable sort
        # keeps SEED_ROUTES order within ties.
        seed_inserts.sort(key=lambda x: x[0])

        # Build the final dict. Byte tokens keep their identity ranks 0..255
        # (they're the BPE base alphabet — re-numbering them would break the
        # standard "byte i → rank i" convention some downstream tools assume).
        # Merges + seeds occupy 256..256+n_merges+n_applied-1, interleaved by
        # the seed insertion points and their pre-shift natural ids.
        mergeable_ranks: dict[bytes, int] = {bytes([i]): i for i in range(256)}
        merges_only = sorted(
            ((b, i) for b, i in natural_ranks.items() if i >= 256),
            key=lambda x: x[1],
        )
        out_id = 256
        seed_iter = iter(seed_inserts)
        next_seed = next(seed_iter, None)
        for nat_bytes, nat_orig_id in merges_only:
            while next_seed is not None and next_seed[0] <= nat_orig_id:
                mergeable_ranks[next_seed[1]] = out_id
                out_id += 1
                next_seed = next(seed_iter, None)
            mergeable_ranks[nat_bytes] = out_id
            out_id += 1
        # Tail: seeds whose insertion point exceeds every natural merge id.
        while next_seed is not None:
            mergeable_ranks[next_seed[1]] = out_id
            out_id += 1
            next_seed = next(seed_iter, None)

        applied = len(seed_inserts)
        print(
            f"==> seed routes applied: {applied}/{n_routes} "
            f"(skipped: L={skipped_no_L}, R={skipped_no_R}, "
            f"S-exists={skipped_S_exists})"
        )

        tokens_offset = len(mergeable_ranks)
        special_tokens = {
            name: tokens_offset + i for i, name in enumerate(SPECIAL_TOKENS)
        }
        enc = tiktoken.Encoding(
            name="rustbpe_seed_tokens",
            pat_str=pattern,
            mergeable_ranks=mergeable_ranks,
            special_tokens=special_tokens,
        )
        return tk.RustBPETokenizer(enc, "<|bos|>")

    tk.RustBPETokenizer.train_from_iterator = classmethod(
        lambda cls, text_iterator, vocab_size: _patched_train_classmethod(
            text_iterator, vocab_size
        )
    )
    return _routes_mod.SEED_ROUTES


def main():
    variant_base = setup_variant_base("seed_tokens")
    seed_routes = apply_patches()
    from rustbpe_variants.seed_tokens.routes import RECOMMENDED_VOCAB_SIZE

    # Forward caller args; inject --vocab-size only if not already set.
    user_args = sys.argv[1:]
    has_vocab_arg = any(
        a == "--vocab-size" or a.startswith("--vocab-size=")
        for a in user_args
    )
    if has_vocab_arg:
        effective_vocab = "caller-supplied (see --vocab-size in args below)"
    else:
        user_args = ["--vocab-size", str(RECOMMENDED_VOCAB_SIZE)] + user_args
        effective_vocab = (
            f"{RECOMMENDED_VOCAB_SIZE} "
            f"(+{RECOMMENDED_VOCAB_SIZE - 32768} vs nanochat default, "
            "from RECOMMENDED_VOCAB_SIZE)"
        )
    print(f"==> variant_base:    {variant_base}")
    print(f"==> seed routes:     {len(seed_routes)}")
    print(f"==> vocab_size:      {effective_vocab}")
    print(f"==> tok_train args:  {user_args}")
    sys.argv = ["tok_train.py"] + user_args
    runpy.run_module("scripts.tok_train", run_name="__main__")
    print_downstream_hint(variant_base)


if __name__ == "__main__":
    main()
