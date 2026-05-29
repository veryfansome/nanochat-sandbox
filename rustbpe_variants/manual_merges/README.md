# manual_merges

A hand-curated **composite** tokenizer overlay with two independent knobs, both hand-written in [`pairs.py`](pairs.py):

- **`BLOCKED_PAIRS`** — *forbid* specific `(L, R)` merges during BPE training (blocked_morphemes-style), via the `rustbpe_force_merges` crate's `blocked_pairs=` kwarg.
- **`MANUAL_PAIRS`** — *force* specific `(L, R)` concats into the vocab by **inserting them at their natural-firing rank** `max(id_L, id_R) + 1` (clamped to 256) in pure Python (seed_tokens-style natural-rank insertion; single-char byte-bigrams clamp to rank 256 — the "front") — **not** via the crate's `forced_pairs=` / `apply_forced_merges_at_end` append.

Either knob can be empty. Training uses nanochat's pre-tokenization regex with **one small tweak** — the opening-delimiter chars `" ( , “` are dropped from the leading-char word-gluing clause, so junk `delimiter+word` tokens (`"The`, `(x`, `,y`, `“The`) don't form (the `-`/`.`/`/` glued tokens, which form real units like `-based`/`.com`/`/api`, are kept). It is otherwise the standard regex with no carve-out, so every pair — blocked or forced — must still be one the pre-tokenizer keeps within a single chunk (a within-word / sub-word composition). See [`pairs.py`](pairs.py) for the rationale.

> **Heritage vs. current state.** This started as a hand-written flavor of [`force_merges`](../force_merges/) (hence the reused `rustbpe_force_merges` crate). It has since evolved into the composite above: it borrows `force_merges`' crate only for `blocked_pairs=` support, and borrows `seed_tokens`' natural-rank insertion for forcing. The `forced_pairs=` append-at-end mechanism is **not** used here (`forced_pairs=[]` always). The current `pairs.py` exercises **blocking only** — suffix-family `BLOCKED_PAIRS`, 0 `MANUAL_PAIRS` — so right now it behaves as a pure suffix-deglomerator; forced pairs may be added later.

## Mechanism contrast

| | `force_merges` | `seed_tokens` | `manual_merges` |
|---|---|---|---|
| Pair source | mined | mined | **hand-written** in `pairs.py` |
| Forces merges via | `forced_pairs=` append-at-**end** (lowest priority) | insert at natural rank | insert at natural rank (**reuses seed_tokens' mechanism**) |
| Blocks merges via | — | — | `blocked_pairs=` (**reuses force_merges' crate**) |
| Pre-tok regex | carve-out (cross-boundary phrases survive) | standard | **near-standard** — standard minus opening-delimiter gluing (`" ( , “`) |
| Pair scope | cross-pre-tok-boundary phrases | within-chunk compositions | **within-chunk only** (sub-word / morpheme) |
| Rust crate | `rustbpe_force_merges` | none (pure Python) | `rustbpe_force_merges` (for blocking only) |

**The within-chunk consequence:** because the pre-tokenizer splits on word boundaries, a forced pair only fires if its two halves can be adjacent *inside one pre-tok chunk*. So manual_merges targets **within-word compositions** — e.g. forcing `("educ", "ation")` so `education` → `[educ, ation]`, or blocking `("t", "ion")` so the clean `ion` suffix token fires instead of mangled `tion`. Cross-boundary phrases (` of` + ` the`) can't fire here; use `force_merges` for those.

Two natural relationships:
- **Forcing** is the complement of [`blocked_morphemes`](../blocked_morphemes/): blocked_morphemes only *prevents* non-morphemic merges; manual_merges can both prevent (`BLOCKED_PAIRS`) and *force* (`MANUAL_PAIRS`) in one tokenizer.
- The forcing mechanism is identical to [`seed_tokens`](../seed_tokens/) — same rank-insertion engine — but the routes are hand-written, and a pre-existing concat is **promoted** to its insertion rank rather than skipped (the point is priority).

## What it does NOT change

- No Rust crate of its own — reuses `rustbpe_force_merges` (for `blocked_pairs=` only; `forced_pairs` stays empty).
- `SPLIT_PATTERN`: pristine nanochat default **except** the one tweak above — opening-delimiter chars `" ( , “` no longer glue to a following word. No carve-out, no other regex change.

## Layout

```
manual_merges/
├── README.md     # this file
└── pairs.py      # BLOCKED_PAIRS = [(L, R), ...]  (forbid during training)
                  # MANUAL_PAIRS  = [(L, R), ...]  (force via natural-rank insertion)
                  # RECOMMENDED_VOCAB_SIZE
```

## Usage

1. Edit `pairs.py`: add `BLOCKED_PAIRS` to forbid merges and/or `MANUAL_PAIRS` to force them (within-chunk / sub-word only).
2. Smoke: `uv run python -m wrappers.smoke_tok_train_manual_merges`
3. Train: `uv run python -m wrappers.tok_train_manual_merges` → `~/.cache/nanochat-variants/manual_merges/tokenizer/`
4. Eval: see STATUS.md "Evaluating a (re)trained tokenizer variant" — `eval_tokenizer` → `pass_metrics` → `dump_vocab`.

## Vocab-size note

`MANUAL_PAIRS` forcing is additive: the final mergeable vocab grows by the number of forced concats BPE wouldn't have formed itself (promoted-existing concats don't change the count). `BLOCKED_PAIRS` doesn't change the target count — natural BPE just trains to `RECOMMENDED_VOCAB_SIZE` with those merges forbidden. Leave `RECOMMENDED_VOCAB_SIZE` at 32768 to let any forced merges be additive (seed_tokens-style), or set it to `32768 - len(MANUAL_PAIRS)` to hold the final vocab roughly constant by displacing tail BPE merges. See `pairs.py`.
