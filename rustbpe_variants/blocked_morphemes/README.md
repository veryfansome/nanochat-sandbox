# blocked_morphemes

Tokenizer variant that steers BPE away from "mangled-morpheme" tokens — composites of the form `<bound-fragment> + <clean-suffix>` like `sembled` (= `sembl` + `ed`) or `ucation` (= `uc` + `ation`). The hypothesis (see [`../../ideas/tokenizer-variants/README.md`](../../ideas/tokenizer-variants/README.md)) is that at the nanochat-speedrun scale (small model, fixed data), one well-trained `<stem>+<suffix>` 2-token encoding is worth more semantically than one undertrained whole-token specialist, even at the cost of a small compression hit.

## What this variant does NOT change

- No Rust crate of its own. Reuses `rustbpe_force_merges`'s `blocked_pairs=` kwarg.
- No regex carve-out. SPLIT_PATTERN is pristine nanochat default.
- No vocab-size bump. Stays at 32,768 — every blocked-merge slot gets reallocated by BPE to a different (hopefully more morpheme-aligned) merge.

The variant's contribution is purely the **`BLOCKED_PAIRS` list** that prevents specific `(L_bytes, R_bytes)` pair merges during training.

## How the block list is generated

`tools/find_mangled_morphemes.py` reads a reference tokenizer and a corpus sample, then for each vocab token T:

1. **Suffix-strip**: Find the longest match S from a curated English-suffix inventory. Skip if no match, if T == S, or if `T = "" + S`.
2. **Look up L**: Let `L = T[:-len(S)]`. Skip if L isn't a token in the reference vocab.
3. **Bound check**: Compute `word_start_prob(L_id)` = fraction of L's firings that occur at the start of a pre-tok chunk. T is *mangled* iff:
   - `word_start_prob(L_id) < bound_threshold` (default 0.05 — strict), AND
   - `(' ' + L)` is NOT a token in the reference vocab. (If the space-prefixed sibling exists, L is just the mid-word allomorph of a free stem like `' organ'` / `'organ'`; that's a clean stem+suffix combination and we leave it alone.)
4. **Emit block**: For each mangled T, find the `(L', R')` split where both operands exist in the reference vocab and `max(L'_id, R'_id)` is minimal — that's BPE's most-likely formation pair. Add `(L'_bytes, R'_bytes)` to the block list.

The suffix inventory and bound threshold are configurable but defaults reflect the conservative starting point: strict bound threshold catches the obvious cases without sweeping up legitimate stems that happen to be mid-word capable.

## Why this is iterative

A first pass blocks the mangled tokens in the reference tokenizer. After retraining, the new tokenizer may have *new* mangled compositions (BPE's compensation cascade). Re-mine on the new tokenizer; add the survivors; retrain. The loop converges when re-mining produces no new blocks (or when compression cost rises faster than morphological alignment improves).

## Layout

```
blocked_morphemes/
├── README.md                              # this file
├── blocks.py                              # canonical BLOCKED_PAIRS (PASS_1 + PASS_2 + PASS_3)
└── mined/                                 # generator outputs, one per pass (all full-inventory)
    ├── mangled_climbmix_pass1.py          # PASS_1: comprehensive mine of pristine baseline (531)
    ├── mangled_climbmix_pass2.py          # PASS_2: re-mine of PASS_1 tokenizer — survivors (74)
    └── mangled_climbmix_pass3.py          # PASS_3: re-mine of PASS_1+2 tokenizer — converges to 0 (18)
```

## Re-curating

1. Pick a reference tokenizer (baseline for round 1, prior variant for round N+1).
2. Run: `uv run python -m tools.find_mangled_morphemes --reference <tokenizer_dir> --out-py rustbpe_variants/blocked_morphemes/mined/mangled_<corpus>_<reference>.py`
3. Inspect the generated file — every entry is annotated with the target token, suffix matched, and `word_start_prob(L)`.
4. Update `blocks.py` to import from the new mined file.
5. Re-train: `uv run python -m wrappers.tok_train_blocked_morphemes`
6. Eval: `uv run python -m tools.eval_tokenizer --tokenizers ours,~/.cache/nanochat-variants/blocked_morphemes/tokenizer`
