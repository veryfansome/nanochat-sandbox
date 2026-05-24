# Tokenizer variants (offline-evaluable) — proposed setup

Status: **proposal / not yet implemented**
Target repo: `nanochat` (Karpathy) — kept **pristine**, never edited.

## Goal

Treat the tokenizer as an experiment surface, not a fixed input. Vary pre-tokenization rules, vocab size, and tokenizer-training corpus mix; evaluate variants **offline** so that GPU time is only spent on candidates that pass cheap filters. The hypothesis: which tokens exist in the vocab shapes what a model of fixed scale can learn — particularly compositional capabilities (arithmetic, code structure, multilingual transfer) where token granularity sets a hard capability ceiling.

## Brain-inspired motivation

Loose mapping to the **representational alphabet**. Brains don't choose their sensory primitives — photoreceptor wavelengths, cochlear frequency bands, somatosensory receptor types are evolutionarily fixed. For language models we *do* pick the atoms: the discrete tokens the network can represent and compose. Different atomic decompositions enable or preclude different compositional reasoning. Digit-by-digit tokenization is the canonical concrete example — lets a model learn positional arithmetic compositionally rather than memorizing each number's identity atomically.

## Why this works as a project track

Two project-specific properties make this tractable here despite tokenization usually being a heavyweight intervention:

1. **The corpus is not pre-tokenized.** Pretraining shards are raw-text parquets (`nanochat/dataset.py:14, 60-65`); tokenization runs inline in the dataloader (`nanochat/dataloader.py:107`). So **changing the tokenizer requires no corpus retokenization** — none of the multi-hour cost typically associated with tokenizer experiments applies.
2. **The speedrun already skips `tok_train` if a tokenizer is cached** (`runs/speedrun.sh:126-131`). So the workflow decouples cleanly: train variants offline, push the `tokenizer.pkl` + `token_bytes.pt` of the winner to `$NANOCHAT_BASE_DIR/tokenizer/` on the Lambda host, and the speedrun jumps straight to pretraining.

GPU minutes are spent only on the A/B that matters. Variant generation and filtering happen on a laptop in parallel with any other track.

## Why it's its own category (not an overlay)

Tokenizer experiments don't touch model code, don't touch the training loop, and don't change CLI args. The artifact is a **preprocessing output** (a trained tokenizer on disk) that the existing pipeline consumes. This is a new overlay-fit category — `preprocessing artifact` — separate from the four model/harness categories in [`../README.md`](../README.md).

## Candidate variants

Roughly ordered cheap-to-rich. None requires nanochat edits — all are `SPLIT_PATTERN` (`nanochat/tokenizer.py:30`) and/or `scripts/tok_train.py` invocation changes.

1. **Digit-rule re-validation against CORE.** Karpathy already swept the `\p{N}{1,2}` clause and reported `{1,2}` best at vocab=32K, *measured on `val/bpb`* (`nanochat/tokenizer.py:27-29`). Re-test `{1}` (Llama-3-style strict digit-per-token) on CORE-arithmetic subscores — `{1}` may lose small bpb but win the downstream metric the project actually cares about. Lowest-overhead variant; just a regex change.
2. **Vocab size × digit-rule grid.** Karpathy's sweet spot is scoped to 32K. Run `{1}` / `{1,2}` / `{1,3}` × `{32K, 64K}` and see whether the optimal rule shifts with vocab budget. Embedding-param cost at 64K is real at d24 (`2 × 64K × 768 ≈ 100M` extra params) — counts against the project's fixed-scale constraint at the upper end of the sweep.
3. **Code-aware splits.** Standard `SPLIT_PATTERN` has no special handling for indentation, operator clustering, or common code structures. Add pre-tokenization rules (e.g., group `    ` → single indentation token, cluster operators) and measure code-domain compression + downstream code-task signal. Bigger regex surgery; the rest of the pipeline is unchanged.
4. **Tokenizer-training corpus mix.** `tok_train.py` consumes the same FineWeb stream as pretraining. Try training the tokenizer on a code-biased or math-biased subset (without changing the *pretraining* corpus) and measure whether the resulting merges shift in useful ways. Lightest possible variant — no regex change, just a `parquets_iter_batched` filter.

## Offline evaluation — the filter (`tools/eval_tokenizer.py`)

The deliverable that makes the whole track work: a single CLI that takes a tokenizer directory and emits a metrics report. Tiers reflect runtime cost.

### Tier 1 — Free (seconds)

- **Per-domain compression** (bytes/token) on a held-out probe battery: English prose (FineWeb val), code (a small Python/JS corpus), math (a curated arithmetic + LaTeX corpus), multilingual (UDHR-style). Uneven compression flags structural problems.
- **Vocab inspection**: longest tokens, token-length distribution, single-byte fraction, frequency-rank vs count (dead/near-dead tokens).
- **Structural reversibility battery**: encode/decode round-trips on hand-picked cases (digit arithmetic, code snippets, contractions, unicode). Catches pre-tokenization bugs.
- **Per-task token-count probes**: `"3 * 17 = 51"`, `"def foo(x):\n    return x"`, `"你好世界"`. These directly cap *what's expressible* in a fixed context.

### Tier 2 — Cheap (minutes)

- **Coverage curves**: fraction of validation corpus covered by top-N tokens. Steep curve = vocab budget concentrated; flat tail = wasted vocab.
- **Information per token**: estimate `H(token | prev token)` from a held-out corpus via simple n-gram counts. Unit-free per-token entropy, comparable across vocabs without training a model. Well-merged tokens reduce locally-predictable redundancy.
- **Bigram-Zipf fit**: how cleanly does the token distribution follow Zipf? Heavy bimodality is a sign the merge process got stuck.

### Tier 3 — Probe model (~50 min on M4 via `runs/runcpu.sh`)

- **d6 probe pretraining** with the candidate tokenizer. Measure overall `val/bpb` and per-domain bpb (math, code). Per the project's `ideas/README.md` "Lessons learned": d6 can't reliably distinguish effects below ~0.05 bpb, so this filter catches **catastrophic regressions** but does not finely rank candidates. Use as a final filter, not a ranker.

### What offline metrics will NOT do

None of the above reliably **predicts** CORE / downstream gain from a full speedrun. They are filters that reject obviously-bad candidates so that GPU A/B sees serious contenders only. Be honest about this in any write-up; do not pick a winner from offline metrics alone.

## Implementation

- `tools/eval_tokenizer.py` — Tier 1 + Tier 2 metrics. Input: a `$NANOCHAT_BASE_DIR/tokenizer/`-style directory. Output: a JSON / Markdown report. Pure Python; standalone (no nanochat training imports beyond `nanochat.tokenizer` and `nanochat.dataset`).
- `wrappers/tok_train_variant.py` — wraps `scripts.tok_train` with overrides for `SPLIT_PATTERN`, vocab size, and tokenizer-training corpus filter. Saves to a *named* tokenizer dir (e.g. `~/.cache/nanochat/tokenizers/digit_1`) so multiple variants coexist.
- A small shell helper or doc page in [`../../runs/`](../../runs/) explaining the offline → GPU workflow:
  1. Train N variants locally with `wrappers.tok_train_variant`.
  2. `tools.eval_tokenizer` on each; drop ~80%.
  3. d6 probe (`OVERLAY= NANOCHAT_BASE_DIR=... bash runs/runcpu.sh`) on 2-3 finalists.
  4. `scp` the winner's `tokenizer.pkl` + `token_bytes.pt` to `$NANOCHAT_BASE_DIR/tokenizer/` on the Lambda host.
  5. `bash runs/speedrun.sh` — skips `tok_train`, picks up the cached tokenizer, proceeds to pretraining A/B.

No edits to `nanochat/`. No edits to `scripts/base_train.py`. The wrapper for `tok_train` is the only piece that mirrors the project's overlay pattern (monkeypatch + `runpy`).

## Expected cost

- **Per variant, offline**: rustbpe training on 2B chars takes a few minutes on M4 (already true; this is what `tok_train.py` defaults to). Offline eval is seconds (Tier 1) to minutes (Tier 2). d6 probe is ~50 min if a variant reaches Tier 3.
- **Per variant, GPU**: only for finalists that go through the full speedrun A/B — same cost as any other A/B in the project.
- **Engineering**: `tools/eval_tokenizer.py` is the biggest piece — maybe 200–400 LOC for a useful first version (compression + vocab inspection + reversibility battery; defer Tier 2 entropy/coverage curves to a follow-up). The variant wrapper is ~30 LOC.

## Combines well with

- **Token-level loss weighting** ([`../token-loss-weighting/README.md`](../token-loss-weighting/README.md)) — both reshape per-token signal but at different layers. Vocab choice determines *which atoms exist*; weighting determines *how much gradient each atom gets*. Genuinely orthogonal — a good A/B is to vary both axes independently.
- **All model-side overlays** — tokenizer-agnostic, so any winning variant carries over to MTP, z-loss, deep supervision, etc. without code changes.
- The track runs **in parallel with the model-overlay sequence** for GPU time. Offline iteration doesn't compete with pretraining experiments; only the final variant's speedrun A/B does.

## Sequencing within this track

1. Build `tools/eval_tokenizer.py` (Tier 1 only).
2. Variant 4 (corpus mix) — cheapest possible: no regex change, just data filter. Establishes the workflow end-to-end.
3. Variant 1 (digit rule re-validation against CORE) — narrowly targeted at a single capability with a clear measurement.
4. Variant 2 (vocab size × digit rule grid) — costs more (multiple speedruns) but maps a real surface.
5. Variant 3 (code-aware splits) — biggest regex surgery; do last.

## Open questions

- Whether `val/bpb` and CORE rank tokenizer variants the same way at d24 scale. If they disagree, the project's primary metric needs revisiting *for this track only* (model-overlay tracks should stay bpb-driven).
- How much per-domain compression imbalance is acceptable. A tokenizer optimized for code may compress prose worse; need a domain-weighted aggregate that matches the pretraining mix.
- Whether Tier 2 entropy / coverage metrics ever decide a tie that Tier 1 doesn't. If not, deprioritize Tier 2.
- Whether the d6 probe is worth running at all given the noise floor — possibly cheaper to skip Tier 3 and run the full speedrun on 2 finalists.
- Interaction with `\p{N}{1,2}` at non-32K vocab sizes — Karpathy's tuning is scoped; variant 2 is the right way to map this.

## References

- Touvron et al. 2024 — Llama 3 paper, digit-by-digit tokenization motivation.
- Karpathy nanochat tokenizer comment block (`nanochat/tokenizer.py:27-29`) — already-explored portion of the digit-rule sweep at vocab=32K.
- Petrov et al. 2023 — "Language Model Tokenizers Introduce Unfairness Between Languages" (per-language compression bias).
- Xue et al. 2022 — ByT5, byte-level alternative (out of scope here but the conceptual contrast worth knowing).
