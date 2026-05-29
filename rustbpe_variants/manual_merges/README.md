# manual_merges

A hand-curated flavor of [`force_merges`](../force_merges/). Same mechanism — force specific `(L, R)` pairs into the BPE vocab via the `rustbpe_force_merges` crate's `forced_pairs=` kwarg + `apply_forced_merges_at_end` — but the pairs are **written by hand** in [`pairs.py`](pairs.py) instead of mined, and training uses nanochat's **standard pre-tokenization regex** (no carve-out).

## How it differs from force_merges

| | `force_merges` | `manual_merges` |
|---|---|---|
| Pair source | mined (`tools/mine_forced_pairs.py`) | hand-written in `pairs.py` |
| Pre-tok regex | patched with a carve-out so cross-boundary phrases (` of the`) survive as one chunk | nanochat **standard** `SPLIT_PATTERN`, unmodified |
| Pair scope | cross-pre-tok-boundary phrases | **within-chunk only** (sub-word / morpheme compositions) |
| Rust crate | `rustbpe_force_merges` | same crate (reused; no new crate) |

**The standard-regex consequence:** because the pre-tokenizer splits on word boundaries, a forced pair can only fire at inference if its two halves can be adjacent *inside one pre-tok chunk*. So manual_merges is for **within-word compositions** — e.g. forcing a morpheme boundary `("educ", "ation")` so `education` tokenizes as `[educ, ation]`. Cross-boundary phrases (` of` + ` the`) won't fire here; use `force_merges` for those.

This is the natural complement to [`blocked_morphemes`](../blocked_morphemes/): blocked_morphemes *prevents* non-morphemic merges; manual_merges *forces* desired ones.

## What it does NOT change

- No Rust crate of its own — reuses `rustbpe_force_merges` (the same `forced_pairs=` machinery).
- No `SPLIT_PATTERN` patch — pristine nanochat default.
- `blocked_pairs` is empty (this variant only forces; to also block, combine lists or use a different variant).

## Layout

```
manual_merges/
├── README.md     # this file
└── pairs.py      # MANUAL_PAIRS = [(L, R), ...] (hand-written) + RECOMMENDED_VOCAB_SIZE
```

## Usage

1. Add `(L, R)` pairs to `MANUAL_PAIRS` in `pairs.py` (within-chunk / sub-word only).
2. Smoke: `uv run python -m wrappers.smoke_tok_train_manual_merges`
3. Train: `uv run python -m wrappers.tok_train_manual_merges` → `~/.cache/nanochat-variants/manual_merges/tokenizer/`
4. Eval: `uv run python -m tools.eval_tokenizer --tokenizers ours,~/.cache/nanochat-variants/manual_merges/tokenizer`

## Vocab-size note

The final mergeable vocab grows by the number of forced concats BPE wouldn't have formed itself. Leave `RECOMMENDED_VOCAB_SIZE` at 32768 to let manual merges be additive (seed_tokens-style), or set it to `32768 - len(MANUAL_PAIRS)` to hold the final vocab roughly constant by displacing tail BPE merges (force_merges-style). See `pairs.py`.
