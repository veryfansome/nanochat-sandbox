# Unigram vs BPE on climbmix (vocab=32K)

Standalone results document. Records what we learned from a controlled comparison of two vocabulary-construction algorithms — Byte-Pair Encoding (greedy bottom-up merges) and Unigram LM (top-down likelihood pruning) — on the same corpus, same vocab size, same `SPLIT_PATTERN` regex (the construction algorithms operate on the same byte content; see Setup below for the small backend-level differences). Self-contained: no prerequisites beyond standard tokenizer terminology.

**Date**: 2026-05-28.
**Eval**: offline (per-domain compression + vocab inspection). **No downstream LM training was run.** This document reports tokenizer-side findings only; conclusions about LM-side effects are explicitly out of scope.

## Setup

| | BPE (baseline) | Unigram |
|---|---|---|
| Library | `rustbpe` (Karpathy's Rust BPE) | HuggingFace `tokenizers` (`models.Unigram`) |
| Inference | merge replay | Viterbi (likelihood-based) |
| Corpus | climbmix, 200M chars (10% of nanochat default) | identical |
| Vocab size | 32768 | 32768 |
| Pre-tokenization | nanochat's `SPLIT_PATTERN` (regex split) on raw bytes | nanochat's `SPLIT_PATTERN` + ByteLevel byte-to-char remap |
| Special tokens | nanochat's 9 specials | nanochat's 9 + `<|unk|>` (Unigram structurally requires one) |
| Training time | 7.17s | 27.32s |
| Training hyperparameters | `rustbpe` defaults | HF UnigramTrainer defaults (`shrinking_factor=0.75`, `n_sub_iterations=2`, `max_piece_length=16`) |
| Byte coverage | `rustbpe` operates on raw bytes | `initial_alphabet=ByteLevel.alphabet()` (all 256 ByteLevel chars in vocab); `byte_fallback` on the Unigram model is **off** — initial alphabet makes it unnecessary |

The construction algorithm is the primary variable; both sides share the same `SPLIT_PATTERN` regex, corpus, vocab size, and the 9 nanochat special tokens (Unigram has one extra `<|unk|>` that its trainer structurally requires). The byte-coverage layer differs by necessity: `rustbpe` operates on raw bytes natively, whereas HF tokenizers' Unigram model operates on Unicode codepoints and needs ByteLevel to bridge to byte-level granularity. ByteLevel is a byte-for-char bijection, so the two backends see the same underlying byte content through different surface representations — fair comparison, but see "Generalization across BPE backends" in the caveats section below.

## Headline result

**BPE compresses 12% better than Unigram overall** on the offline probe battery (`tools/eval_tokenizer.py` — 20 probes across 7 domains).

| | Total bytes | Total tokens | bytes/token |
|---|---:|---:|---:|
| BPE | 2,746 | 1,257 | **2.185** |
| Unigram | 2,746 | 1,428 | 1.923 |

Per-probe sign test: BPE wins 13, Unigram wins 6, 1 tie (p ≈ 0.084, one-sided; two-sided ≈ 0.167).

## Per-domain breakdown

**BPE substantially beats Unigram** on:

| Domain | BPE bytes/token | Unigram bytes/token | BPE advantage |
|---|---:|---:|---:|
| arithmetic | 2.04 | 1.38 | +48% |
| code_python | 2.81 | 2.01 | +40% |
| json | 2.23 | 1.65 | +35% |
| science | 3.47 | 2.75 | +26% |
| math_latex | 2.13 | 1.71 | +25% |
| modern_prose | 3.54 | 2.88 | +23% |
| code_javascript | 2.58 | 2.11 | +22% |

**Unigram modestly beats BPE** on:

| Domain | BPE bytes/token | Unigram bytes/token | Unigram advantage |
|---|---:|---:|---:|
| multi_arabic | 1.20 | 1.49 | +24% |
| multi_japanese | 1.25 | 1.48 | +18% |
| multi_chinese | 1.10 | 1.18 | +7% |
| multi_russian | 1.74 | 1.84 | +6% |
| whitespace | 2.70 | 2.84 | +5% |
| multi_korean | 1.13 | 1.18 | +4% |

Spanish, French, German, and Hindi pattern with English in direction (BPE wins on all of them) but at larger magnitudes — 18–36% vs English's 8% (Spanish 36%, French 22%, German 20%, Hindi 18%). Spanish, French, and German are Latin scripts and don't fit the non-Latin multi-byte-per-character pattern that gave Unigram its CJK/Arabic/Russian advantage. Hindi (Devanagari, non-Latin, multi-byte-per-character) *does* fit that script profile but BPE still wins on it — see the counterexample analysis in Mechanism 3 below.

## Why the difference exists (mechanism)

The per-domain pattern has three explanations:

### 1. Unigram doesn't reward category coverage

**The clearest diagnostic**: BPE produces 100 two-digit-number tokens (covering ~all of 00–99); Unigram produces **zero**.

| Token length | BPE digit-only count | Unigram digit-only count |
|---|---:|---:|
| 1 | 10 | 10 |
| 2 | 100 | 0 |

BPE's per-step greedy frequency criterion rewards merging *any* digit pair that's common, because each merge step picks the single highest-frequency pair. Across many similar low-individual-frequency pairs (e.g., "37", "42", "85"), this accumulates a full category of two-digit tokens.

Unigram's likelihood EM evaluates each piece independently against the corpus distribution. A specific 2-digit token like "37" has a low marginal likelihood gain — it appears in many contexts but not predictably enough to dominate other candidate pieces. So pruning systematically eliminates the entire category.

**Consequence**: probes with multi-digit literals regress sharply on Unigram — arithmetic +48%, code_python +40%, JSON +35%, code_javascript +22%. The smaller code_javascript regression suggests the JavaScript probe has fewer numeric literals than the Python one; the arithmetic probe is pure numbers and pays the largest cost.

### 2. `max_piece_length=16` caps Unigram's long tokens

BPE's longest 3 tokens are 32-byte punctuation runs (`--------------------------------`, `................................`, `________________________________`). Its 4th-longest is 24B (`……………………`, 8 ellipses). Unigram's longest tokens cap at 16 bytes — 8 of its top-10 are content words (` characteristics`, ` recommendations`, ` electromagnetic`, ` straightforward`, ` differentiation`, `neurotransmitter`, ` experimentation`, ` Characteristics`); 2 are punctuation runs at the 16B cap (`----------------`, `................`).

This is **not** a vocab-quality issue — Unigram also allocates to punctuation patterns, just truncated to 16B. But losing the longer 32B punctuation tokens costs a small amount of compression on text containing extended runs (document separators, code formatting).

Raising `max_piece_length` to 32 in UnigramTrainer would close this specific gap. Not tested.

### 3. The CJK reversal

Unigram modestly wins on CJK / Russian / Arabic. Possible mechanism (untested): for non-Latin scripts where each character takes 2–3 UTF-8 bytes, the per-character likelihood gain of a long token is higher than for Latin script. Unigram's likelihood criterion is more sensitive to high-per-piece information content, so it allocates more multi-character CJK tokens at the same vocab budget.

The magnitude is modest (4–24%), insufficient to compensate for the larger losses on English/code in a mixed-corpus regime.

(Whitespace also tilts Unigram by +5% — a different, untested mechanism. Plausibly Unigram learns a few specific multi-whitespace pieces that BPE doesn't allocate, but this isn't part of the script-based pattern above.)

**Counterexample**: Hindi (Devanagari, 3 bytes per character) fits the "non-Latin, multi-byte-per-character" shape but BPE wins on it by 18%, not the other way around. Plausible reason (untested): less Hindi than CJK / Cyrillic / Arabic in climbmix, so at the same vocab budget neither tokenizer allocates many Hindi-specific multi-char pieces and a different mechanism dominates. The mechanism above describes the modal case; Hindi shows it's not universal.

## Tokens look qualitatively similar

Despite the compression gap, the *content* of the two vocabularies is more similar than different. Top-10 longest non-punctuation tokens:

**BPE**: ` telecommunications`, ` interdisciplinary`, ` multidisciplinary`, ` neurotransmitters`, ` responsibilities`, …

**Unigram**: ` characteristics`, ` recommendations`, ` electromagnetic`, ` straightforward`, ` differentiation`, `neurotransmitter`, ` experimentation`, ` Characteristics`, …

Both algorithms are picking up multi-syllable English content words. The differences are at the *middle* of the rank distribution (which 32K-th token is included), not at the top.

## Vocab inspection summary

| Metric | BPE | Unigram |
|---|---:|---:|
| Vocab size | 32768 | 32768 |
| Non-special tokens | 32759 | 32758 |
| Bytes/token mean | 6.57 | 7.02 |
| Bytes/token median | 6 | 7 |
| Bytes/token max | 32 | 16 |
| Single-byte fraction | 0.8% | 0.8% |
| Digit-only tokens | 110 (max len 2) | 10 (max len 1) |

## What this experiment does NOT tell us

- **Downstream LM performance.** No LM was trained. The 12% compression gap could translate to a downstream regression, neutral, or even a downstream win — none of these are determinable from compression alone. Independent literature (Schmidt et al. 2024, *Tokenization Is More Than Compression*) found at 350M params / 32K vocab on The Pile that Unigram with likelihood segmentation slightly *outperformed* BPE downstream (49.2 vs 48.8 avg accuracy). At their scale and corpus, the construction algorithm's compression delta (whatever its sign) didn't predict the downstream ranking — consistent with their broader finding that Pearson(CTC, accuracy) = 0.241 across all 18 tokenizers they tested. Compression and downstream rank can decouple.
- **Generalization across vocab sizes.** Tested only at 32K. UnigramTrainer's `max_piece_length=16` may bind less tightly at larger vocabs; relative rankings may shift.
- **Generalization across corpora.** Tested only on climbmix's web-crawl mix. A heavier English-text corpus (Wikipedia, arXiv) might shrink the gap; a heavier multilingual corpus might invert it.
- **Generalization across UnigramTrainer hyperparameters.** Used HF defaults. `shrinking_factor`, `n_sub_iterations`, and `max_piece_length` are not swept.
- **Generalization across BPE backends.** Compared `rustbpe` BPE to HF Unigram. HF BPE backend was not used. nanochat's HF-BPE path (the `HuggingFaceTokenizer.train_from_iterator` setup) might tokenize differently from `rustbpe` at the margins (different merge tie-breaking, etc.).

## What it does tell us

- At vocab=32K on a web-crawl corpus, **the choice of construction algorithm has a meaningful impact on offline compression** (~12% on overall bytes/token across 20 probes). This isn't a rounding error.
- The two algorithms allocate vocab budget **systematically differently**, not just randomly differently. The category-coverage failure (zero two-digit number tokens) is a *deterministic property* of Unigram's EM, not noise — would reproduce on any rerun.
- **Compression-only eval is not a free pass.** Different domains diverge by 20–50%; aggregate numbers hide structure. Per-domain reporting is essential.
- **BPE's greedy-per-step frequency is not obviously suboptimal**, despite the well-known "early merges block better later merges" critique. On compression specifically, at this scale and corpus, BPE comes out ahead of Unigram's globally-optimized likelihood criterion.

## Reusable artifacts produced

- `wrappers/tok_train_unigram.py` — drop-in Unigram trainer matching `scripts.tok_train`'s corpus/CLI surface. Same `--max-chars`, `--doc-cap`, `--vocab-size` flags. Saves `tokenizer.json` (HF native) + `token_bytes.pt` (val/bpb metadata) to `~/.cache/nanochat-variants/unigram/tokenizer/`.
- `wrappers/smoke_tok_train_unigram.py` — 8-check smoke battery (build, vocab-size sanity, special-token presence, save/load roundtrip, encode/decode lossless on prose/code/CJK/accents/emoji/URL, `_safe_token_bytes` correctness, `token_bytes.pt` structure, single-byte-token byte count for high bytes 0x80–0xFF). Runs in <10s with no real data.
- `tools/eval_tokenizer.py` extension — `load_tokenizer` auto-detects HF format (`tokenizer.json`) vs BPE format (`tokenizer.pkl`) and dispatches accordingly. Existing BPE tokenizer paths unchanged.
- `tools/eval_tokenizer.py:_safe_token_bytes` — handles ByteLevel byte recovery correctly via the canonical GPT-2 `bytes_to_unicode` reverse map. **Important gotcha**: do NOT use `pre_tokenizers.ByteLevel.alphabet()` to build the reverse map by `enumerate()` — that method returns chars in *alphabetized order*, not byte-value order, and produces a wrong mapping. The correct mapping is computed from first principles in `_build_bytelevel_reverse()`.

## Reproduction

```bash
# Fetch the corpus shards if not cached.
uv run python -m nanochat.dataset -n 8

# Train baseline BPE on 200M chars (also caches at ~/.cache/nanochat/tokenizer/).
uv run python -m scripts.tok_train --max-chars=200000000 --vocab-size=32768

# Train Unigram on the same corpus.
uv run python -m wrappers.tok_train_unigram --max-chars=200000000 --vocab-size=32768

# Compare.
uv run python -m tools.eval_tokenizer \
    --tokenizers ours,~/.cache/nanochat-variants/unigram/tokenizer
```

Wall clock on M1 (16GB): data fetch ~30s, BPE training ~7s, Unigram training ~27s, eval ~60s. Total ~2 minutes once shards are local.
