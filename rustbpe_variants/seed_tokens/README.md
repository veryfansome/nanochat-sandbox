# seed_tokens

Data-driven *within-pre-tok-chunk* composition seeds, inserted into the vocab at their **natural-firing rank**. This is the canonical reference for the rank-insertion mechanism that [`manual_merges`](../manual_merges/) also reuses.

## Mechanism: natural-rank insertion (not append-at-end)

Each seed `(L_bytes, R_bytes, S_bytes)` is inserted into the final `mergeable_ranks` at rank `max(id_L, id_R) + 1` — the rank the merge would have had if introduced mid-BPE-training. Natural merges with id ≥ the insertion point shift up by 1 per seed; relative order is preserved. Tiktoken then applies the merge at encode time via standard rank-based greedy matching wherever `L` and `R` are adjacent.

**Why mid-rank, not append-at-end:** a natural BPE merge that would consume one of a seed's operands has a *lower* rank than an appended seed and fires first, leaving no `(L, R)` pair to match. Early-dev witness: seed `(q, q) → qq` appended at rank ~328 was preempted by a natural ` q` merge at rank ~271 and never fired. Mid-rank insertion places each seed where no preempting natural merge can yet exist (both operands are still bare at that point in the BPE sequence). Contrast [`force_merges`](../force_merges/), which *does* append-at-end — correct there because it forces *cross-chunk* pairs that no natural merge competes for.

Pure Python over **pristine `rustbpe`** — no Rust crate, no `SPLIT_PATTERN` carve-out. (A legacy constructive-merge-chain Rust crate exists in this dir but is **unused**, preserved only for history; see "Previously explored" below.)

## How candidates are chosen

`tools/mine_seed_compositions.py` encodes a held-out corpus (ClimbMix val) with the baseline tokenizer, counts adjacent **within-chunk** token-pair occurrences, filters to pairs whose concatenated bytes aren't already in the baseline vocab, ranks by count, and emits the top-K. Each candidate is by construction a pair that (1) fires wherever those adjacencies occur in real text, (2) fills a gap baseline left unfilled under the 32K budget, and (3) replaces 2 tokens with 1 at every firing site — so the pair *is* the route, no separate route-finder needed.

**Within-chunk only:** cross-chunk pairs (`(',', ' ')`, `(' ', '1')`) can't merge under tiktoken's per-chunk encoding — that structural gap is what `force_merges` fills with a regex carve-out, a different mechanism for a different gap.

### Previously explored: morpheme mining (failed)

An earlier iteration mined productive substrings (`tools/mine_seed_morphemes.py`) + a route-finder. It competed with baseline's already-efficient short morphemes (`ing`, `ation`, `ate`, `ize`) and was **net-negative at every threshold**. Tools preserved; no longer the canonical path. Full arc in [`../../STATUS.md`](../../STATUS.md) §seed_tokens redesign.

## Vocab size

`RECOMMENDED_VOCAB_SIZE = 32768 + len(SEED_ROUTES)` — one slot per genuinely-new seed (seeds whose concat already exists are skipped, *not* promoted — the opposite of `manual_merges`). Current canonical K=1,750 → **vocab=34,518**.

## Results

Canonical `mined/compositions_climbmix_val_k1750.py` (1,750 within-chunk compositions from 20M chars of ClimbMix val): **−0.630% ClimbMix val tokens (−28,271)** vs baseline at +5.3% vocab. Offline-positive but small — below the per-task CORE noise floor at d24, so **preserved as a reproducible artifact but not queued for a standalone Lambda run.** Could stack opportunistically with another tokenizer variant.

## Layout

```
seed_tokens/
├── README.md       # this file
├── routes.py       # SEED_ROUTES (imports the canonical mined list) + RECOMMENDED_VOCAB_SIZE
├── mined/          # mined composition lists (compositions_climbmix_val_k1750.py = canonical)
└── src/, Cargo.toml, seed_tokens.yaml  # legacy constructive-merge crate + morpheme list — UNUSED, preserved for history
```

## Usage

1. Smoke: `uv run python -m wrappers.smoke_tok_train_seed_tokens`.
2. Train: `uv run python -m wrappers.tok_train_seed_tokens` → `~/.cache/nanochat-variants/seed_tokens/tokenizer/`. No Rust build needed (pure-Python overlay on pristine `rustbpe`).
3. Eval: see STATUS.md "Evaluating a (re)trained tokenizer variant" — `eval_tokenizer` → `pass_metrics` → `dump_vocab`.
