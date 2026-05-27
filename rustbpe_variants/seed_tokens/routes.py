"""
SEED_ROUTES — canonical seed-token list for the seed_tokens tokenizer.

Each entry: (L_bytes, R_bytes, S_bytes). The wrapper inserts each seed
into mergeable_ranks at rank `max(id_L, id_R) + 1` — the rank it would
have had if introduced mid-BPE-training. Natural merges with id ≥ the
insertion point shift up by 1 per inserted seed. Tiktoken applies the
merge at encode time via standard rank-based greedy matching whenever
L and R are adjacent.

## How candidates are chosen (current canonical)

`tools/mine_seed_compositions.py` — encodes a held-out corpus sample
(ClimbMix val: the last shard of `base_data_climbmix`, the held-out
split returned by `parquets_iter_batched(split="val")`) with the
baseline tokenizer, counts adjacent within-pre-tok-chunk token-pair
occurrences, filters to pairs whose concatenated bytes aren't already
in baseline vocab, ranks by count, emits top-K as
`(L_bytes, R_bytes, S_bytes)`.

Each candidate `(L, R)` is by construction a pair that:
  1. Appears as adjacent baseline tokens in real text → the seed will
     fire wherever those adjacencies occur.
  2. Has a concatenation NOT in baseline vocab → adding the merge fills
     a gap baseline left unfilled (couldn't fit in 32K budget).
  3. Replaces 2 tokens with 1 wherever it fires → net compression
     strictly positive at firing sites.

The "pair IS the route" — no separate route-finder needed, since `id_L`
and `id_R` are already baseline-vocab tokens by construction.

## Within-chunk only

Counting cross-pre-tok-chunk pairs would surface tons of high-frequency
candidates like `(',', ' ')` or `(' ', '1')`, but tiktoken's encoder
cannot merge across pre-tok-chunk boundaries — that's the structural
gap `force_merges` fills with a regex carve-out, a different mechanism
targeting a different gap. So the miner re-pre-tokenizes the corpus
and counts adjacent pairs only within each chunk.

## Previously explored: morpheme mining (failed)

An earlier iteration mined substrings of pre-tok chunks ranked by
distinct-types productivity (`tools/mine_seed_morphemes.py`), then a
route-finder picked operand splits. That approach competed with
baseline's already-efficient short morphemes (`ing`, `ation`, `ate`,
`ize`, ...) and produced net-negative compression at every threshold
tested (see STATUS.md §seed_tokens redesign). The tool is preserved
for reference but no longer the canonical mining path.

## Current canonical

`mined/compositions_climbmix_val_k1750.py` — 1,750 within-chunk
adjacent-pair compositions mined from 20M chars of ClimbMix val.
Offline-validated: **−0.630% ClimbMix val tokens (−28,271)** vs baseline
at +5.3% vocab; 86% of seeds fire (14% dead). K=1,750 chosen at the
per-seed marginal-value plateau-end visible in the K-sweep (full
table + analysis in STATUS.md §seed_tokens). The selection criterion
is deterministic (count desc, max_id asc, S asc), so re-mining
reproduces the same 1,750 routes.

## Re-curating

1. Re-mine: `tools/mine_seed_compositions.py --top-k 1750` →
   `mined/compositions_climbmix_val_k1750.py` (overwrites).
2. Re-train: `uv run python -m wrappers.tok_train_seed_tokens`.
3. Re-eval: `tools.eval_tokenizer --full` vs baseline.
4. If the K choice still looks right, no further action; otherwise
   re-sweep K (`uv run python -m tools.mine_seed_compositions --top-k <K>`
   → train → eval at multiple K values) and update the import below to
   point at the new K's artifact.
"""

from rustbpe_variants.seed_tokens.mined.compositions_climbmix_val_k1750 import SEED_ROUTES

RECOMMENDED_VOCAB_SIZE = 32768 + len(SEED_ROUTES)
