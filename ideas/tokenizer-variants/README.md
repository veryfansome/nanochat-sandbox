# Tokenizer variants (offline-evaluable)

Status: **active track.** Two live variants, each its own crate: `force_merges` (adopted default, CORE +5.81%) and `auto_tune` (active — held-out-Pareto block-list auto-tuning under the pristine regex). The earlier probes (`seed_tokens`, `unigram`, `blocked_morphemes`, `manual_merges`, `space_digits`) were removed 2026-06-05 as dead directions — findings retained in [`../../STATUS.md`](../../STATUS.md). The catalog + "Prior art" sections below are kept as the design space + the upstream `veryfansome/nanochat` branches the variants were lifted from. See [`../../STATUS.md`](../../STATUS.md) "Tokenizer variants" for current state.

Target repo: `nanochat` (Karpathy) — kept **pristine**, never edited. The Rust tokenizer source (`rustbpe/`) **is in-tree** and modifiable; PyPI wheels are a build artifact, not a black box.

## Goal

Treat the tokenizer as an experiment surface, not a fixed input. Vary the merge-producer (pre-tokenization regex, merge algorithm, seed/forced merges, training corpus) and evaluate variants **offline** so that GPU time is only spent on candidates that pass cheap filters. The hypothesis: which tokens exist in the vocab shapes what a model of fixed scale can learn — particularly compositional capabilities (arithmetic, code structure, multilingual transfer) and signal density per token (forced common-phrase tokens reduce gradient-step count to learn the same content).

## Brain-inspired motivation

Loose mapping to the **representational alphabet**. Brains don't choose their sensory primitives — photoreceptor wavelengths, cochlear frequency bands, somatosensory receptor types are evolutionarily fixed. For language models we *do* pick the atoms: the discrete tokens the network can represent and compose. Different atomic decompositions enable or preclude different compositional reasoning. Digit-by-digit tokenization is the canonical concrete example — lets a model learn positional arithmetic compositionally rather than memorizing each number atomically. Morpheme-aware tokenization is the analog one level up: lets the model compose meaning from sub-word linguistic primitives (Greek/Latin roots, prefixes, suffixes) rather than learning each derivation independently.

## What's actually a knob

The interface between tokenizer training and everything else is a single Python dict: `mergeable_ranks: dict[bytes, int]` (token bytes → merge-priority rank). `RustBPETokenizer.train_from_iterator` produces this dict and stuffs it into a `tiktoken.Encoding`, which is what gets pickled to `tokenizer.pkl` and consumed at inference. **Anything that produces a valid `mergeable_ranks` dict is drop-in.** The merge-producer is a free design choice.

Knobs by tier of access cost:

| Knob | Where | Access cost |
|---|---|---|
| Pre-tokenization regex (`SPLIT_PATTERN`) | `nanochat/tokenizer.py:30` | Trivial — Python string |
| Tokenizer-training corpus | `parquets_iter_batched` filter; `--max_chars`, `--doc_cap` | Easy — wrap or filter |
| Vocab size | `--vocab_size` on `tok_train.py` | Trivial — CLI |
| **Forced / blocked merges**, **seed tokens**, **custom merge scoring** | `rustbpe/src/lib.rs` + thin Python wrapper kwarg | Medium — Rust edit + `maturin develop` |
| Alternative algorithm (Unigram, WordPiece) | Switch tokenizer class at inference; use `HuggingFaceTokenizer` (already a class in `tokenizer.py:39`) | Higher — inference path changes |
| Byte-level vs character-level base alphabet | rustbpe internals | High — fundamental rewrite |

The first four are all reachable in tens to a few hundred lines of code. The Rust source is **single-file** (`rustbpe/src/lib.rs`, ~1000 lines), uses standard crates (`pyo3`, `fancy-regex`, `rayon`, `dary_heap`, `ahash`), and rebuilds incrementally with `maturin develop` in ~10s.

## Prior art (read before reinventing)

Two experimental branches of [`veryfansome/nanochat`](https://github.com/veryfansome/nanochat) implemented merge-producer variants. Both reached working implementations; both were paused for the same reason (inconclusive results given expensive Lambda A/Bs + no domain-expertise-driven priors to guide tuning). The infrastructure is reusable.

### `origin/seed_tokens` — morpheme-seeded BPE

Idea: warm-start BPE with a linguistically-curated seed list (Greek/Latin roots, prefixes, suffixes) so early merges build morpheme-aware tokens rather than purely frequency-driven ones. The branch carried a ~200-morpheme YAML + a Rust constructive-merge-chain builder (`ensure_token`, `compute_common_suffixes`) at vocab=65536.

Why it paused: hard to ground *which* morphemes matter without downstream-task priors; compression didn't move decisively.

**Resolution in this project (2026-05-26)**: ported, then redesigned away from morpheme-mining (net-negative — it competed with baseline's already-efficient short morphemes) toward within-chunk adjacent-pair mining (net-positive but small). Became the `seed_tokens` variant (**removed 2026-06-05** as a dead direction; findings retained in STATUS.md §seed_tokens redesign).

### `origin/force_merges_wip` — forced cross-boundary common-phrase merges

Idea: allow specific high-frequency grammatical phrases ("in the", "of the", "to be", "However,") to become single tokens, despite normally crossing pre-tokenization boundaries.

The crux that makes this work: **a regex carve-out, not a cross-chunk merge**. The naive approach (let BPE count pairs across pre-tok chunks and force-merge the frequent ones) breaks inference because tiktoken processes each pre-tok chunk independently — even if a merge rule for `(in_token, the_token)` exists, it doesn't fire if the two tokens are in different chunks. The branch solves this by **prepending an alternation to the regex that matches the forced phrases as single chunks**:

```python
SPLIT_PATTERN = FORCED_PAIRS_EXPR + r"""|'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+| ?\p{N}{1,2}|..."""
```

Now ` of the` is a single pre-tok chunk at both training and inference. BPE/tiktoken process it as one unit; the forced merge produces a single token consistently. Trailing lookahead `(?=([^a-z]|$))` prevents matching ` of theory` as ` of the` + `ory`.

What's there: hand-picked `FORCED_PAIRS` / `BLOCKED_PAIRS` / `DERIVED_PAIRS` + `apply_forced_merges_at_end` (appends after BPE, reusing an existing id if the concat already exists). A side micro-opt — ` ?` before `\p{N}{1,2}` so leading digits merge with their preceding space — measured **+1.09–1.23% compression for ~220 tokens** (now folded into the in-tree tokenizer; see catalog #2).

Why it paused: large compression gains, but whether they translate to a real CORE/loss win was inconclusive at the available A/B budget — see the methodological problem below. Now the in-tree [`force_merges`](../../rustbpe_variants/force_merges/) adopted default.

### Methodological problem encountered (both branches)

**Compression is not a fair comparison across tokenizers with different vocabs.** A tokenizer that adds " of the" as a token wins compression on any text containing "of the" and is roughly neutral elsewhere. So the headline compression number depends on what's in the test corpus — biased toward the variant whose forced/seed tokens match the test distribution.

This is a real bias, not just noise. Going forward (see "Evaluation"), the eval harness needs to:
- Report compression per-domain (`tools/eval_tokenizer.py` already does this — 20 probes across 8 domains).
- Add probes that *don't* contain the forced/seed phrases — measure compression on neutral text to separate "variant has the right tokens for this corpus" from "variant's general BPE is better."
- For the eventual A/B: use **CORE / val/bpb** on a held-out corpus the tokenizer's training data didn't see, not raw compression.

## Variants — refined catalog

Organized by which knob each touches. Items marked **(adopted)** are the current canonical for this project; items marked **(prior art)** have only branch-level implementation in `veryfansome/nanochat` and haven't been ported here yet.

### Regex-only

1. **Digit-rule re-validation against CORE.** Karpathy tuned `\p{N}{1,2}` on `val/bpb` at 32K vocab (`nanochat/tokenizer.py:27-29`). Re-test `{1}` (Llama-3-style strict digit-per-token) on CORE-arithmetic subscores — may lose small bpb but win the downstream metric. Lowest-overhead variant; just a regex change.
2. **Space-prefix digits (`?\p{N}{1,2}`)** — **(prior art, force_merges_wip)**. Measured +1.09–1.23% compression for ~220 token cost. Lift this micro-optimization out of `force_merges_wip` and merge into baseline regardless of the cross-phrase work — it's a clean small win.
3. **Code-aware splits** — pre-tokenization rules for indentation grouping, operator clustering, common code structures. Bigger regex surgery; defer until simpler variants land.

### Vocab size

4. **Vocab × digit-rule grid.** Karpathy's `{1,2}` sweet spot is scoped to 32K. Run `{1}` / `{1,2}` / `{1,3}` × `{32K, 64K}`. The seed_tokens branch defaulted to 65K — that vocab size is already validated as trainable.

### Training corpus

5. **Tokenizer-training corpus mix.** `tok_train.py` consumes the same ClimbMix stream as pretraining. Try training the tokenizer on a code-biased or math-biased subset (without changing the *pretraining* corpus) and measure whether merges shift in useful ways. Lightest possible variant — no Rust change, just a `parquets_iter_batched` filter.

### Merge-producer (Rust)

6. **Seeded-merge BPE** — **(removed 2026-06-05; offline-positive but too small for standalone Lambda — findings retained)**. Two iterations tested. First, morpheme-mining via the prior-art YAML+constructive-merge approach: net-negative on ClimbMix val (the surfaced morphemes competed with baseline's already-efficient short morphemes). Second, reframed the mining target to **within-pre-tok-chunk adjacent-token-pair bigrams from baseline encoding output** — pairs baseline knows are common but couldn't fit in the 32K budget; the pair IS the route, no separate route-finder needed. Insertion is mid-rank (each seed at `max(id_L, id_R) + 1`, pure Python). Canonical: K=1,750 at the per-seed-marginal plateau-end, vocab=34,518. Offline result: **−0.63% ClimbMix val tokens at +5.3% vocab**. The magnitude doesn't predict a clear CORE win at d24; preserved but not queued for Lambda alone. See STATUS.md §seed_tokens redesign for the full empirical arc, K-sweep, isolation control vs extended-BPE-at-same-vocab, K_switch sweep, and six takeaways.
7. **Forced common-phrase merges** — **(adopted; current default tokenizer)**. Cross-pre-tok-chunk forced merges via regex carve-out; Rust crate at `rustbpe_variants/force_merges/` with `apply_forced_merges_at_end` + `blocked_pairs` hook. Data-driven curation via `tools/mine_forced_pairs.py --two-pass` + `tools/ablate_derived.py`; current canonical: 105 mined FORCED + 15 mined DERIVED + 2 numeric DERIVED = 122 pairs at vocab=32788. Lambda speedrun on the 2026-05-25 canonical: **CORE +5.81%, val/bpb tied, ChatCORE +0.62%**. See STATUS.md §force_merges for full curation provenance.
8. **`min_frequency` filter for normal BPE merges.** Skip merges below a count threshold, reallocate the saved vocab slots to other merges (or just produce a smaller vocab). Not currently exposed by rustbpe. ~30 LOC change.
9. **Train-big-ship-small (vocab pruning).** Train at vocab=64K, drop the bottom-K by validation-set usage, ship a 32K tokenizer. Dead-token fraction → near zero by construction; the question is whether the pruned tokens were genuinely useless. Combine with `tools/eval_tokenizer.py --full` to pick the pruning threshold.
10. **Branch-entropy merge scoring.** Penalize merges whose merged token has many high-frequency continuations (token captures less context-conditioning information). Hypothesis: better-tuned per-token information content → better next-token prediction headroom. Speculative; lowest priority.
14. **Blocked morphemes** — **(removed 2026-06-05; mechanism continues as `auto_tune`)**. Forbade mangled `<stem>+<suffix>` merges during BPE training via a `blocked_pairs` cascade, so suffix firing-mass concentrates on clean standalone stems instead of mangled merges. This `blocked_pairs=` approach is exactly what the live `auto_tune` variant now uses (under the pristine regex, with the held-out-Pareto block-list audit). (Numbered 14–15 — appended after the catalog's existing 1–12; an earlier #13 was removed.)
15. **Manual merges** — **(removed 2026-06-05)**. Composite of the blocking mechanism (14) and the seed mechanism (6): `BLOCKED_PAIRS` forbid merges *and* hand-written `MANUAL_PAIRS` inserted at their natural rank, under the standard regex (within-chunk pairs only). Its hand-curated block list was the original seed for the auto_tune audit.

### Alternative algorithm (path 2: HF tokenizers)

11. **Unigram LM tokenizer** (SentencePiece-style). Different inference algorithm; would route through `HuggingFaceTokenizer`. Bigger change but tests the "is BPE itself optimal" question. **Implemented + offline-evaluated (2026-05-28): Unigram compresses 12% worse than BPE at vocab=32K on climbmix; mechanism + per-domain breakdown + caveats in [`unigram-vs-bpe.md`](unigram-vs-bpe.md). No downstream LM eval yet.**
12. **WordPiece** — same shape as 11.

## Evaluation — what changes given prior art

The eval harness is already built: [`../../tools/eval_tokenizer.py`](../../tools/eval_tokenizer.py). It computes:

- **Per-domain compression** on 20 probes across 8 domains (English prose, code, math, science, 9 languages, structured/JSON, whitespace, emoji).
- **Vocab inspection** — bytes/tok stats, single-byte fraction, **digit-token analysis by length** (directly answers Karpathy's `\p{N}{1,2}` question), longest tokens, vocab gaps.
- **Structural round-trip battery** (19 PASS/FAIL tests, catches inference inconsistencies — exactly the class of bug the cross-boundary work needed to verify).
- **Task probes** (19 capability-relevant strings — arithmetic at multiple digit lengths, code idioms, URLs, dates, named entities).
- **Coverage curve** (Tier 2 `--full`) — top-N cumulative share + dead-token count.

`eval_tokenizer.py` is the Tier-1 filter; the full offline loop pairs it with `tools/pass_metrics.py` (corpus deltas vs baseline) and `tools/dump_vocab.py` (vocab inspection) — see STATUS.md "Evaluating a (re)trained tokenizer variant".

Adjustments needed given the prior-art lessons:

- **Add "no-forced-phrase" task probes** to separate "this tokenizer wins because it has the right merge for this probe" from "this tokenizer's general BPE is better." E.g., a probe corpus deliberately written without any of `FORCED_PAIRS`.
- **The structural battery is the right safety net** for the cross-boundary regex trick — if any round-trip test fails, the regex carve-out has an unintended interaction with the rest of the pattern. Currently the battery passes for the cached `\p{N}{1,2}` tokenizer; new regex prefixes need to keep it passing.
- **Compression alone is insufficient as a winner-selection signal.** The d6 probe (Tier 3, via `runs/runcpu.sh`) is the cheapest way to ground compression deltas in actual loss. If a variant wins compression but ties or loses d6 bpb, it's not a winner.

## Implementation paths

The sandbox now vendors Karpathy's pristine rustbpe at [`../../rustbpe/`](../../rustbpe/) (lifted from upstream commit `9467d83`) and provides a parallel-crate overlay pattern at [`../../rustbpe_variants/`](../../rustbpe_variants/) — each variant is its own Rust crate producing a uniquely-named Python module, so multiple variants coexist in one venv. Build with `bash runs/build_rustbpe.sh` (pristine) or `VARIANT=<name> bash runs/build_rustbpe.sh`. See [`../../SETUP.md`](../../SETUP.md) "Tokenizer variants" for the full workflow and [`../../rustbpe_variants/README.md`](../../rustbpe_variants/README.md) for the add-a-variant recipe.

Three implementation paths in priority order:

1. **Parallel-crate Rust variant** — copy `sandbox/rustbpe/` to `sandbox/rustbpe_variants/<name>/`, rename the package + `#[pymodule]`, add variant logic, build, then write `wrappers/tok_train_<name>.py` that aliases `sys.modules['rustbpe']` to your variant before `runpy`ing `scripts.tok_train`. The path for merge-producer variants whose logic needs to run **inside** BPE training (8, 9, 10) and for the `force_merges` adopted-canonical (7) which uses `apply_forced_merges_at_end` + a `blocked_pairs` hook. The variant-7 crate at `rustbpe_variants/force_merges/` is the worked example; see `rustbpe_variants/README.md` for the add-a-variant recipe.

2. **Python-only overlay** — sufficient for regex (variants 1, 2, 3), corpus mix (variant 5), and any post-training adjustment of `mergeable_ranks` (variant 6 / `seed_tokens` falls here). `wrappers/tok_train_<name>.py` monkeypatches what it needs (e.g. `nanochat.tokenizer.SPLIT_PATTERN`, or replaces `RustBPETokenizer.train_from_iterator` to insert seeds post-natural-BPE) and `runpy`s `scripts.tok_train`. No Rust build needed; uses pristine `rustbpe`. **The current `seed_tokens` canonical uses this path** despite being a merge-producer variant — its mid-rank insertion into `mergeable_ranks` is pure Python; the prior Rust constructive-merge crate in `rustbpe_variants/seed_tokens/` is unused but preserved for history.

3. **HuggingFace `tokenizers` library** — for non-BPE algorithms (variants 11, 12). Switches inference path away from tiktoken; bigger change. Use the existing `HuggingFaceTokenizer` class in `nanochat/tokenizer.py:39` as the inference-side hook.

The sandbox project's overlay discipline applies: don't edit `nanochat/` master and don't edit `sandbox/rustbpe/` (vendored pristine). All variant work lives in `sandbox/rustbpe_variants/<name>/` (Rust changes) and `wrappers/tok_train_<name>.py` (Python wiring).

## Combines well with

- **[Token-level loss weighting](../token-loss-weighting/README.md)** — both reshape per-token signal but at different layers. Vocab choice determines *which atoms exist*; weighting determines *how much gradient each atom gets*. Genuinely orthogonal — a good A/B is to vary both axes independently.
- **All model-side overlays** — tokenizer-agnostic, so any winning variant carries over to MTP, z-loss, deep supervision, etc. without code changes.
- The track runs **in parallel with the model-overlay sequence** for GPU time. Offline iteration doesn't compete with pretraining experiments; only the final variant's speedrun A/B does.

## Sequencing within this track

1. **Use `tools/eval_tokenizer.py` immediately** on the cached tokenizer to establish baseline measurements. **Done.**
2. **Variant 2 (space-prefix digits)** — smallest possible Rust-free change, prior-art-validated compression win. **Done** — folded into `force_merges`'s `SPLIT_PATTERN`, so the standalone `space_digits` variant exists but running both is redundant.
3. **Variant 1 (digit rule re-validation against CORE)** — narrowly targeted at a single capability; needs Lambda time for the A/B but is cheap to set up.
4. **Variant 5 (corpus mix)** — establishes the offline workflow end-to-end without touching Rust.
5. **Variant 7 (forced merges)** — **Done and adopted**. Data-driven curation via `tools/mine_forced_pairs.py --two-pass` + `tools/ablate_derived.py`; Lambda speedrun on the 2026-05-25 canonical landed CORE +5.81% / val/bpb tied / ChatCORE +0.62%; in-tree canonical re-curated 2026-05-26 with corrected shadow rule (Lambda-pending for the new canonical). See the "For forced merges" item under Open questions below and STATUS.md §force_merges for full details.
6. **Variant 6 (seeded merges)** — **Done offline; variant removed 2026-06-05; not queued for standalone Lambda** (−0.63% ClimbMix val tokens, too small for a clear d24 CORE win). Full arc in catalog #6 + STATUS.md §seed_tokens redesign.
7. **Variants 4 (vocab grid), 8 (min_frequency), 9 (vocab pruning)** — narrower questions, do as targeted follow-ups.
8. **Variants 10 (branch entropy), 11/12 (non-BPE algorithms)** — speculative, defer.

## Open questions

- Whether `val/bpb` and CORE rank tokenizer variants the same way at d24 scale. If they disagree, the project's primary metric needs adjustment for this track.
- How to disentangle "tokenizer has the right merges for the test corpus" from "tokenizer's general BPE is better." See "Methodological problem" — needs neutral-corpus probes.
- For forced merges: which phrases to force. **Resolved by data-driven curation (2026-05-25, re-curated 2026-05-26 with corrected shadow rule).** Replaced the ~70-pair hand-picked list with `tools/mine_forced_pairs.py --two-pass` + `tools/ablate_derived.py`; current canonical at [`rustbpe_variants/force_merges/pairs.py`](../../rustbpe_variants/force_merges/pairs.py) (105 mined FORCED + 15 mined DERIVED + 2 numeric DERIVED = 122 forced merges, vocab=32788 — +20 over the nanochat default to absorb the extra carve-outs without displacing borderline BPE merges). Achieves **−6.9% total tokens vs the baseline nanochat tokenizer** on a 15M-char training-corpus sample; Lambda speedrun on the prior 2026-05-25 canonical (104 pairs at vocab=32768) yielded **CORE +5.81%, val/bpb tied, ChatCORE +0.62%**. Full construction provenance, per-pair attribution table, vocab-cost bracketing experiment, and reusable infrastructure live in [`STATUS.md`](../../STATUS.md) §force_merges. To re-curate against a different corpus: (1) regenerate the mining artifacts under `rustbpe_variants/force_merges/mined/`, (2) rerun `ablate_derived` to find the new K cutoff, (3) **paste the new pair tuples into `pairs.py`** (it's intentionally a flat literal list, not a runtime loader — see the module docstring for the regen command), (4) **re-validate the `_NUMERIC_DERIVED` hand-adds + the `RECOMMENDED_VOCAB_SIZE` constant** against the new corpus + probe battery.
- For seed tokens: **resolved by within-chunk composition mining (2026-05-26)** — replaced the intuition-driven morpheme-list approach with mined within-chunk adjacent-pair bigrams (K=1,750 canonical, −0.63% offline, too small for standalone Lambda). Full empirical arc, K-sweep, and controls in catalog #6 + STATUS.md §seed_tokens redesign.
- Whether to combine forced merges + seed tokens — they target **different structural gaps**: `force_merges` captures cross-pre-tok-chunk pairs (BPE structurally cannot see them; addressed via regex carve-out), while `seed_tokens` captures within-chunk pairs that BPE saw at training time but couldn't fit in the 32K budget (addressed via mid-rank insertion). Mechanically orthogonal. Untested as a stack; worth an offline A/B if it bundles into another tokenizer experiment cheaply.

## References

- Touvron et al. 2024 — Llama 3 paper, digit-by-digit tokenization motivation.
- Karpathy nanochat tokenizer comment (`nanochat/tokenizer.py:27-29`) — already-explored portion of the digit-rule sweep at vocab=32K.
- Petrov et al. 2023 — "Language Model Tokenizers Introduce Unfairness Between Languages" (per-language compression bias).
- Xue et al. 2022 — ByT5, byte-level alternative (out of scope here but the conceptual contrast worth knowing).
- See [`papers.md`](papers.md) for end-to-end reads of Zouhar et al. 2023 (BPE optimality theory), PathPiece (Schmidt et al. 2024 — compression vs downstream at 350M–2.4B params), GreedTok (Lim et al. 2025 — partition-cover construction + 1B-param LM result + code review), and Train-It-And-Forget-It (Sawada & Goyal 2025 — merge-list-free BPE inference).
- Prior art branches: [`origin/seed_tokens`](https://github.com/veryfansome/nanochat/tree/seed_tokens) and [`origin/force_merges_wip`](https://github.com/veryfansome/nanochat/tree/force_merges_wip) in `veryfansome/nanochat`.
