# force_merges

The **adopted default tokenizer.** Forces specific high-frequency *cross-pre-tok-boundary* phrases (` of` + ` the`, `,` + ` and`, ` to` + ` be`) to become single tokens, which standard BPE structurally cannot do because tiktoken processes each pre-tokenization chunk independently.

## Mechanism

Two coupled pieces, both in the `rustbpe_force_merges` crate (this variant's own fork):

1. **Regex carve-out.** `SPLIT_PATTERN` is prepended with an alternation (`FORCED_PAIRS_EXPR`) that matches the forced phrases as single pre-tok chunks, so ` of the` survives as one unit at *both* training and inference. Without this, a merge rule for `(of_token, the_token)` never fires — the two tokens land in different chunks. This is the crux that makes cross-boundary forcing work.
2. **`apply_forced_merges_at_end`.** After normal BPE training completes, each forced concat is **appended** (lowest priority), reusing an existing id when the concat bytes already exist in the vocab. Most of the 122 pairs already exist as natural merges, so they cost **zero new slots**; only genuinely-new concats consume a slot. `blocked_pairs=` (also a crate kwarg) can forbid accidental cross-boundary merges the carve-out would otherwise allow.

Contrast the two sibling forcing variants, which both use the **standard regex** and target **within-chunk** pairs: [`seed_tokens`](../seed_tokens/) inserts at *natural rank* (mid-stream, highest-priority-where-valid); [`manual_merges`](../manual_merges/) does the same with hand-written pairs. force_merges is the only one that appends-at-end and the only one needing a carve-out — because it's the only one forcing pairs BPE *can't* otherwise form.

## Vocab size

`RECOMMENDED_VOCAB_SIZE = 32788` — a hand-set bracket, **+20 over the 32768 nanochat default**. This is *not* a per-pair cost (most pairs reuse existing ids); the +20 is chosen empirically to keep the borderline tail BPE merges that the extra carve-outs would otherwise displace. Re-bracket if the pair set changes substantially.

## Canonical pair list

`pairs.py` holds the frozen literal list — **105 mined FORCED + 15 mined DERIVED + 2 numeric DERIVED = 122 forced merges** (2026-05-26 curation). It is intentionally a flat literal, not a runtime loader. Provenance and regen steps are in the `pairs.py` docstring; full curation history (mining, ablation, the `detect_shadows` shadow-rule fixes) is in [`../../STATUS.md`](../../STATUS.md) §force_merges curation details.

Data-driven curation tooling: `tools/mine_forced_pairs.py --two-pass` (mines cross-boundary bigrams + depth-2 phrases) → `tools/ablate_derived.py` (picks the K cutoff). Subset trainers for ablations: `wrappers/tok_train_force_merges_mined_{forced_subset,derived_subset,ablate}.py`.

## Results

Lambda speedrun on the **2026-05-25 canonical** (the prior 104-pair list at vocab=32768, wandb `7aesostl` vs baseline `qmsi105c`, $115.76): **CORE +5.81%, val/bpb tied (slight edge), ChatCORE +0.62%**, zero compute cost. The current in-tree 122-pair/32788 canonical is locally Tier-1-parity + 0.42% better real-corpus compression, **Lambda-pending**. Caveat: 83% of the CORE delta came from boolq alone — see STATUS.md for the full breakdown and significance discussion.

## Layout

```
force_merges/
├── README.md           # this file
├── pairs.py            # FORCED/BLOCKED/DERIVED pair literals + FORCED_PAIRS_EXPR + RECOMMENDED_VOCAB_SIZE
├── pairs_mined_*.py    # subset definitions for the ablation drivers
├── mined/              # raw + intermediate mining artifacts (provenance / re-curation)
├── Cargo.toml, src/    # the rustbpe_force_merges crate (own fork of pristine rustbpe)
└── ...
```

## Usage

1. Build the crate: `VARIANT=force_merges bash runs/build_rustbpe.sh`.
2. Smoke: `uv run python -m wrappers.smoke_tok_train_force_merges`.
3. Train: `uv run python -m wrappers.tok_train_force_merges` → `~/.cache/nanochat-variants/force_merges/tokenizer/`.
4. Eval: see STATUS.md "Evaluating a (re)trained tokenizer variant" — `eval_tokenizer` → `pass_metrics` → `dump_vocab`.
