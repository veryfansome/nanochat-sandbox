# Surface factoring

Separate token **content** (which word) from **surface form** (spacing, casing). Factor orthogonal, predictable, vocab-duplicating surface features out of token identity into per-token side-channel bits predicted by a cheap auxiliary head. The base vocabulary becomes a lexicon of content atoms; the surface bits handle rendering. Two validated features: **leading space** and **first-letter capitalization**.

**Status (2026-06-26): Surface = space + case factoring on the BASELINE vocab (32768, K_max=1, no force_merges).** Offline **+4.90% tokens**; **no model A/B yet** — surface-only vs baseline is the open factoring-isolation test. The earlier fm+space+case **TRIPLE** (K_max=3, +11.11%) was a premature stacking — it confounded the d24 A/B (couldn't separate "factoring is inert" from "factoring is redundant with phrases") and forced the K_max=3 cap machinery — so it was **rolled back: DEPRECATED, d24 checkpoint abandoned**. That work is in the [Deprecated appendix](#deprecated-the-fmspacecase-triple-dead-end) below (real lesson, not the current result). Surface is now SURFACE-ONLY: a crate-free tokenizer (base rustbpe, spaceless pattern + case-folded corpus, no forced pairs) built by `wrappers/tok_train_surface.py`.

This started as "space_factoring" (factor only the leading space) and generalized once capitalization measured as an equally strong second feature. The name is **surface factoring**; space is the first instance, case the second.

## Why this fits the project

Under the fixed-scale / fixed-data constraint, a tokenizer that compresses better (fewer tokens per byte) means the model sees more underlying text per fixed token budget — the same sample-efficiency lever that `force_merges` exploits (see [`../tokenizer-variants/README.md`](../tokenizer-variants/README.md) and STATUS.md). Surface factoring reclaims the ~24% of vocab that BPE wastes on space/case duplicates (the classic GPT-2 4-way ` The`/`The`/` the`/`the` redundancy) and reinvests it in longer merges.

**Overlay fit: combined preprocessing-artifact + model-overlay + harness-change.** Unlike the other tokenizer variants (pure preprocessing artifacts that the existing pipeline consumes via cache, no GPU until the A/B), surface factoring ALSO needs (a) a model head to predict the surface bits and (b) a harness change so the dataloader emits the bit targets and the loss includes the head. This is the most cross-cutting idea in the catalog — it touches the tokenizer, the model, training, and eval.

## The idea

A "surface feature" F is a property of a token's rendered form that is orthogonal to its lexical identity. Factoring F out collapses near-duplicate vocab entries `{x, F(x)}` into one base token `x` plus a per-token bit, freeing slots and pooling statistics. A feature is worth factoring iff ALL of:

1. **Reversible / unambiguous** — applying F to (base, bit) exactly recovers the original. No information loss, no ambiguity.
2. **Cheaply predictable** — F has low conditional entropy given context, so the aux head is cheap.
3. **Causes real vocab duplication** — many `{x, F(x)}` pairs exist, so reclaiming slots is worthwhile.
4. **Leaves the base token useful** — F carries no lexical meaning the base token needs (the model still disambiguates from context).

By this bar, the worthwhile set is `{leading space, first-letter case}`. Ruled out: **trailing punctuation** (the GPT-4 regex already separates `word`+`.`, so ~no duplication), **morphology** (plurals/suffixes — semantic, ambiguous, high-entropy, base token loses meaning), **diacritics** (`café`/`cafe` — lexical, rare in English), **all-caps/acronyms** (a different whole-token feature, minor, and `NASA` must NOT fold). The [multi-space "count" extension](#dead-ends) was measured and is a dead end on this corpus.

**Capitonym caveat — case partially violates criterion 4.** Titlecase folding collapses capitonyms (`Apple`/`apple`, `March`/`march`, `Polish`/`polish`, `May`/`may`) into one base token, erasing a genuine lexical case distinction — the same "lexical, not surface" objection that rules out diacritics. Case is tolerated where diacritics aren't because (a) it's recovered by the cap bit + context, and the model disambiguates `Apple`-the-company from context anyway, exactly as it must for any lowercase homograph; (b) it's frequent enough in English to be worth the ~+2% compression; (c) all-caps capitonyms (`US`/`us` → `US` stays, not titlecase) don't fold. So case is a deliberate *partial* violation, not clean orthogonality.

## Mechanisms

### Leading space

Modify the pretokenization regex so the leading space is NOT glued onto words/punct; it falls through to `\s+` and is isolated as its own pretoken, then absorbed as a bit. BPE trains on space-stripped pretokens, so ` the` and `the` both become `the` (pooled), freeing the space-prefixed slots.

```
# nanochat baseline SPLIT_PATTERN (nanochat/tokenizer.py:30):
'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,2}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+

# SPACELESS_PATTERN (tools/space_factor_compression.py): space excluded from the word
# leading-char class ([... ]), leading " ?" dropped from the punct alt, number kept as
# \p{N}{1,2} (no digit-space glue — the space before a number is also a bit):
'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N} ]?+\p{L}+|\p{N}{1,2}|[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+
```

Held-out token count = `raw_encode_count - bit_count`, where `bit_count` = number of single spaces immediately followed by a non-whitespace char (regex ` (?=\S)`) = the lone space adjacent to each content token, which the regex isolates as a single ` ` pretoken (one token) that the bit absorbs. This is NOT "subtract all space tokens": multi-space runs split so the (N-1) leading spaces stay as tokens and only the adjacent one is a bit, so indentation is not over-credited. Validated against a slow pretoken-level simulation (exact match).

**Digit-space caveat (important, already known):** ~28% of the standalone spaceless gain (+0.87% of +3.08%) is NOT the space-factoring idea — it is the ` ?\p{N}{1,2}` "space-prefix-digit" regex tweak (baseline's `\p{N}{1,2}` has no leading-space capture, so ` 42` wastes a token on the space). This is the removed `space_digits` probe, already folded into `force_merges`' SPLIT_PATTERN (see `rustbpe_variants/force_merges/pairs.py:217`, measured +1.09–1.23% there). The genuine space-factoring gain vs a digit-fixed baseline is **+2.23%** — that is the surface-only space contribution. (force_merges is a separate sibling tokenizer that carries the same digit fix; surface no longer stacks on it.)

### First-letter capitalization

Lowercase the first letter of **titlecase** words (`\p{Lu}\p{Ll}*` — first upper, rest lower) and record a per-token cap bit. The titlecase rule is the crux: it cleanly folds sentence-start words and simple proper nouns (`The`→`the`, `London`→`london`, `Café`→`café`, `Łódź`→`łódź`) and leaves everything else untouched because none are titlecase — acronyms (`NASA`), camelCase (`getElementById`), `iPhone`, `macOS`, `McDonald`, `OpenAI`. Verified lossless on all of these.

```
# fold detection (tools/case_factor_compression.py): match uppercase-initial word-starts,
# fold only if the whole run is titlecase AND the fold is reversible + byte-length-preserving:
_FOLD_RE = regex.compile(r"(?<!\p{L})\p{Lu}\p{L}*")   # uppercase-initial, not mid-word (excludes camelCase)
_TITLE   = regex.compile(r"\p{Lu}\p{Ll}*")            # whole run must be titlecase to fold
```

Unlike the space bit, the cap bit is a **free annotation** — folding lowercases a letter in place, it does not add or remove a token, so there is NO token subtraction for case. The compression gain comes purely from the denser case-folded vocab (pooled `The`+`the` → longer merges). Held-out token count = `encode_count(fold_case(text))`, no subtraction.

## Results

All compression/ladder numbers on held-out ClimbMix val (the pinned last shard `shard_06542`, disjoint from train), 20,031,540 chars / 20,060,719 bytes, bytes-per-token (higher = better compression). (One exception: the Bit-predictability table is measured on a 15M-char sub-sample via `case_factor_analysis`; bits/byte is ~scale-invariant.) Reference `baseline_repro` has a **byte-identical merge table / vocab / split-pattern** to the cached canonical tokenizer (`~/.cache/nanochat/tokenizer`) — verified equal `_mergeable_ranks` + `_pat_str`; only the `.pkl` serialization metadata differs — so this is an A/B against the real baseline, not an approximation.

### Vocab duplication (baseline tokenizer, 32,759 non-special tokens)

| feature | tokens duplicated | exact-duplicate pairs (reclaimable) |
|---|---:|---:|
| leading space (` X` tokens) | 22,257 (67.9%) | 3,986 (12.2%) |
| first-letter case (titlecase) | 6,721 (20.5%) | 4,972 (15.2%) |
| space × case (all variants, ≥2 per base; 4,736 such families) | — | 7,745 slots (23.6%) |

Capitalization duplicates MORE vocab than spaces (15.2% vs 12.2%). The joint `{x, " "+x, Cap, " "+Cap}` families reclaim ~24% of the vocab.

### Bit predictability (conditional entropy of the surface bit given the previous token)

| feature | marginal P(set) | conditional cost |
|---|---:|---:|
| leading space | 0.769 | 0.073 bits/byte (0.354 bit/tok) |
| first-letter case | 0.0996 | 0.0525 bits/byte (0.247 bit/tok) |

The case bit is CHEAPER than the space bit (capitalization is rarer). Capitalized tokens split ~49.4% sentence-start (trivially predictable from preceding `. `/`\n`; H≈0.60 bit/tok in that regime) and ~50.6% mid-sentence proper nouns (lexical; H≈0.22 bit/tok). The space bit's marginal/context-free upper bound is 0.161 bits/byte — much higher; the 0.073 figure is the realistic conditional. (These are gross head costs; see [the currency caveat](#caveats).)

### Compression ladder (the headline)

Surface-only: space + case factoring on the baseline vocab (32768). The headline is **`space + case` at +4.90%**; the chain is monotonic (each row strictly better than the one above).

| scheme | vocab | tokens | bytes/tok | vs baseline |
|---|---:|---:|---:|---:|
| baseline | 32768 | 4,269,541 | 4.6986 | — |
| baseline + digit-fix (`space_digits`) | 32768 | 4,232,409 | 4.7398 | +0.87% |
| caseless | 32768 | 4,181,656 | 4.7973 | +2.06% |
| spaceless | 32768 | 4,138,046 | 4.8479 | +3.08% |
| **space + case (surface-only)** | 32768 | 4,060,174 | 4.9409 | **+4.90%** |

**Per-row provenance:** baseline / spaceless / `space + case` from `space_factor_compression.py` + `case_factor_compression.py`; `caseless` from `case_factor_compression.py`; `baseline + digit-fix` from `digitfix_remeasure.py`. (The deprecated fm-family / triple rows — `+4.34%` through the `+11.11%` triple — are in the [Deprecated appendix](#deprecated-compression-ladder-rows-the-1111-triple).)

### Rendering (losslessness)

Every surface-only scheme verified by full encode → (tokens, surface bits) → decode → original round-trip: **2000/2000 held-out docs lossless** for space, case, and space+case, plus hand-crafted adversarial batteries (~20 cases each: CJK/emoji/unicode, multi-space, tabs, camelCase, acronyms, `iPhone`/`McDonald`, `Łódź`, etc.). With **K_max=1** each token owns at most one word-start: it carries a leading-space bit plus `cap_bits[1]`/`cap_mask[1]` (mask=1 for a word-initial token, 0 for a mid-word continuation or non-word token); rendering re-applies the leading space, then the unicode-correct first-letter `.upper()` if an owned cap bit is set. (The deprecated triple's *phrase* tokens carried internal spaces and a per-word cap *pattern* — those losslessness subtleties are in the [Deprecated appendix](#triple-rendering-subtleties-phrase-internal-spaces--per-word-cap-patterns).)

## Scripts

The reproducible offline harness. All run with `uv run python -m tools.<name>` from the sandbox dir, no GPU — **but they need the ClimbMix data cache** (each calls nanochat's `parquets_iter_batched`): `uv run python -m nanochat.dataset -n 8` downloads ~8 train shards + the always-pinned val shard. They then train rustbpe tokenizers (~35–75s each at 2B train chars; the bootstrap makes `rustbpe`, `nanochat.*` importable). **Pass `--train-chars 2000000000` explicitly** — `space_factor_compression.py` defaults to **500M** (the case scripts default to 2B; `digitfix` hardcodes 2B); running script 1 bare with `--save-dir` would save 500M-trained reference tokenizers and silently skew every downstream step. Saved reference tokenizers live under `~/.cache/nanochat-variants/{baseline_repro,spaceless_repro}/tokenizer`; `baseline_repro` has a **byte-identical merge table / vocab / split-pattern** to the cached canonical (equal `_mergeable_ranks` + `_pat_str`; only `.pkl` metadata differs).

**Canonical surface-only builder:** `wrappers/tok_train_surface.py` — trains + saves the production surface-only tokenizer (vocab 32768, K_max=1): base rustbpe (NO force_merges crate), `SPLIT_PATTERN = tools.stack_fm_spaceless.SPACELESS_BODY`, case-folded corpus, `forced_pairs=[]` → `~/.cache/nanochat-variants/surface_only/tokenizer/` (+ `token_bytes.pt`). This is the tokenizer the model A/B uses.

| script | what it does | key invocation |
|---|---|---|
| `tools/space_factor_compression.py` | Space A/B: trains baseline-pattern + spaceless-pattern tokenizers on identical data, measures compression, the conditional space-bit cost, and a fast-vs-slow bit-count validation. **Run this first** (saves the reference tokenizers). | `uv run python -m tools.space_factor_compression --train-chars 2000000000 --val-chars 20000000 --validate --compare-cached --save-dir ~/.cache/nanochat-variants` |
| `tools/digitfix_remeasure.py` | Isolates the `space_digits` digit-fix from the idea-specific space gain (baseline → digit-fixed baseline → spaceless). Reproduces +0.87% digit-fix, +2.23% idea-specific. **Requires the saved reference tokenizers from script 1.** | `uv run python -m tools.digitfix_remeasure` |
| `tools/case_factor_analysis.py` | Cheap potential analysis for capitalization: vocab case-duplication, space×case families, and the cap-bit conditional entropy + sentence-start/proper-noun split. No training. | `uv run python -m tools.case_factor_analysis --val-chars 15000000` |
| `tools/case_factor_compression.py` | Case + joint space+case compression (titlecase fold), with the lossless space+cap round-trip. Defines `fold_case`. **This pair produces the +4.90% headline.** | `uv run python -m tools.case_factor_compression --train-chars 2000000000 --val-chars 20000000` |
| `tools/stack_fm_spaceless.py` | **`SPACELESS_BODY` (the live surface-only pattern) lives here** — reusable. **DEPRECATED-triple-only otherwise:** `build_config(pairs)` and the force_merges + space stack (`--all-pairs` = 122; default = 69) build the fm-pair tokenizer the rolled-back triple needed (reuses the `rustbpe_force_merges` crate). Surface-only does NOT use the pair build — only `SPACELESS_BODY`. | (pattern import only) — `from tools.stack_fm_spaceless import SPACELESS_BODY` |
| `tools/stack_triple.py` | **DEPRECATED (triple-only).** The full force_merges(122) + space + case stack: folds the corpus and the phrases, handles per-word-position caps. Produced the rolled-back +11.11% headline. Kept for the appendix's history. | (deprecated) |

Reusable building blocks worth knowing when extending: `space_factor_compression.SPACELESS_PATTERN` and `corpus_byte_and_bit_stats`; `case_factor_compression.fold_case`; `stack_fm_spaceless.SPACELESS_BODY` (the live surface-only pattern). To extend to a new surface feature: add a vocab-duplication + bit-entropy analysis (like `case_factor_analysis.py`), then a fold/strip + compression A/B (like `case_factor_compression.py`), confirm the 4 factorability criteria, and verify lossless round-trip before believing any compression number.

## Architecture

(Model side — built; see [Direction](#direction).) The model predicts, per position, head-1 = the base content token (as today) and a **surface head** = the surface bits for that token. With **surface-only the surface label is a space bit plus `cap_bits[1]`/`cap_mask[1]`** (K_max=1 — each token owns at most one word-start; a word-initial token has cap_mask=1, a mid-word continuation or non-word token cap_mask=0, conditioned just like the space bit). Decisions, both grounded in the offline findings:

- **Two binary surface heads (a space head + a cap head), NOT a flat categorical.** A "space × case = 4-class softmax" is only well-defined for single-word tokens; the two-binary-head split is what generalizes cleanly to multi-word tokens (the deprecated triple). For surface-only the cap head is a single binary bit per token. (A *joint* 4-class head would have bought ~nothing on the loss anyway: space↔case co-firing at sentence starts is mostly *explained by context*, so the residual `I(space; case | context, token)` is small.)
- **The cap bit is per-OCCURRENCE — occurrence-active masks.** (This supersedes an earlier "fixed token-local mask" idea, which is ASCII-only: once a token can byte-split a multi-byte letter — or the same id appears word-start vs mid-word — the owned word-start is context-dependent.) A token **owns** the word-start whose first byte falls in its span; a mid-word continuation owns NONE (`mask=0`). The encoder fills `cap_mask` + `cap_bits` per occurrence from the **full folded text's** char-level word-starts; **decode operates on the reconstructed string** — proper unicode `.upper()` of the word-start char, NOT per-token byte ops — so multi-byte folds (`Łódź`→`łódź`, `Ελληνικά`) and byte-split tokens round-trip (validated in the smoke). The cost: at **generation** the mask must be derived from running context (the word-starts the emitted token owns given the decoded prefix), which a fixed mask would have avoided — a deliberate trade for unicode correctness. (`wrappers/_surface_ref.py`'s `_char_word_starts`/`surface_encode`/`surface_decode` are the reference impls.)
- **Condition the surface head on head-1's chosen token (sequential), not parallel — this is the key efficiency decision.** A naïve parallel head (predict surface bits from the same hidden state as the token, independently) pays `I(word; feature | context)` — the word↔surface correlation baseline's joint token captures for free — measured at ~0.03–0.07 bits/byte net penalty in the space adversarial verification. Conditioning the surface head on the emitted base token recovers `H(feature | word, context)`, and the gain **differs by feature**: **case** becomes ~free (the word near-determines its capital — `london` is lowercase, `London` was the folded one), while **space** drops substantially but *not* to zero (the word reveals whether it's a word-start vs a mid-word continuation, but the leading space is still partly context-determined — empirical residual ~0.03 bits/byte honest, down from the ~0.073 context-only figure). So sequential conditioning makes case ~free and roughly halves the space cost, but a residual space cost is irreducible — and it's one both baseline and the factored model pay (baseline inside head-1's `" the"`/`"the"` choice, the factored model in the surface head), so it's relocated, not net-new. Sharpest finding from the adversarial verification.
- **Retained K>1 multi-slot machinery — the DEPRECATED-triple generalization, not the live design.** The overlay also implements what multi-word *phrase* tokens forced: per-position `cap_bits[K_max]` + `cap_mask[K_max]`, `K_max` = max word-starts a single token can own (`compute_kmax`), **slot-specific** cap embeddings (`cap_emb[slot, bit]`, e.g. `Embedding(2·K_max, d)` indexed by `slot·2 + bit`), and a `_cap_compose` step. Slot-specific (not a single shared 0/1) embedding is *required* in the K>1 case because a shared masked *sum* is **not injective** over cap patterns (`[1,0]` and `[0,1]` collapse, erasing *which* word of a phrase was capitalized — `Of the` vs `of The`). With surface-only **K_max=1 all of this collapses to one cap bit** — the space-and-cap embedding pair plus the input composition `wte[base] + space_emb[space_bit] + cap_emb[cap_bit]`; the multi-slot path is retained only as a generalization. **Loss** is `BCE(space) + masked-BCE(cap over valid slots)`. Both heads are token-conditioned (previous bullet). See the [0.2% multi-cap finding](#triple-rendering-subtleties-phrase-internal-spaces--per-word-cap-patterns) (a deprecated-triple artifact).

## Training integration

How the tokenizer is used in the training loop (the concrete shape of the "harness change"). At training time the only tokenizer operation is **encode** — decode/rendering is generation-only — and the encode is deterministic from text, so training is fully teacher-forced and parallel with no model-in-the-loop and no autoregressive feedback.

**The shape, per position (the whole change in one sentence).** Every position in the context window holds a pair `(base_token_id, surface_label)`, and the model both *sees* and *predicts* that pair. It is **fed both as input** — the surface label is added to the token embedding, so the trunk sees `(token, surface)` at every position, not just at the output — and this is what keeps the model information-equivalent to baseline (baseline's `" The"` token already carries "new word, capitalized" into the trunk; here that same signal arrives via the surface embedding instead of being baked into the token id). And it **predicts the next position's pair**: head-1 → the next base token id (as today), the surface head → the next surface label. So the entire architectural change from a standard next-token LM is: one extra input embedding and one extra small output head, both riding alongside the token at every position. The rest of this section is the mechanical detail of that.

**Data flow.** nanochat's dataloader (`tokenizing_distributed_data_loader_with_state_bos_bestfit`) today calls `tokenizer.encode(doc_batch, prepend=bos)` → one token-id list per doc, best-fit-packs into rows of length `T+1`, slices `inputs = row[:-1]`, `targets = row[1:]`, and the step is `loss = model(x, y)`. The surface tokenizer's encode instead emits **two per-token-aligned streams**: the base content id and the surface label `(space_bit, cap_bit)` (for surface-only; `(space_bit, cap_pattern)` under the deprecated triple). The dataloader packs both with the same best-fit packing/BOS/shift (they're aligned, so this is mechanical), yielding `base_x, base_y, surface_x, surface_y` — four tensors where `surface_y[t]` is the *next* token's surface.

**Model consumption (teacher-forced, parallel).**
- **Input embedding = `wte[base_x] + surface_emb[surface_x]`.** Load-bearing: the trunk must *see* the surface form, or it loses the sentence-boundary signal baseline's combined `" The"` token carries for free. With it added, the trunk has the same *information* as baseline (the surface is recoverable) — though composed **additively** (`" The"` and `"the"` are forced to differ by exactly `surface_emb`) rather than as baseline's free, independent learned vector per surface variant. Only the *output* is split. **Value embeddings are surface-conditioned (BUILT, not just a caveat):** nanochat *also* injects per-layer value embeddings keyed by token id (`value_embeds[i](idx)`, `nanochat/gpt.py`). Keyed on the *base* id alone they'd be surface-blind (baseline's `" The"`/`"the"` have distinct ones) — a partial loss of info-equivalence beyond the `wte` pathway. The overlay therefore **conditions the value embeddings on surface too** — `ve = base_ve + space_emb' + cap_emb'` per ve-layer (the chosen full-info-equivalence path), smoke-validated (overlay check B2). So both the `wte` *and* the value pathways see the surface; info-equivalence with baseline is complete.
- **Heads off `h_t`:** head-1 (content) predicts `base_y[t]` via the existing `lm_head`; the **surface heads** (a space head + a cap head, per [Architecture](#architecture) — `surface_x`/`surface_y` are the structured `(space_bit, cap_bits[K_max], cap_mask[K_max])` label; for surface-only `K_max=1` so it's just a space bit + a cap bit, not a scalar) predict `surface_y[t]`'s space bit and (masked) cap bit, conditioned on the next base token (`base_y[t]`, teacher-forced) — `surface_head(h_t, wte[base_y[t]])` — which is what makes case near-free.
- **Loss — TRAINING and EVAL are different.** *Training* objective (mean reduction): `content CE + λ·(space BCE + cap BCE)`, where λ is a gradient-balance hyperparameter. *Eval* — the `model(x, y, loss_reduction='none')` path that `nanochat/loss_eval.py:33` calls for bpb (and the CORE likelihood path) — must return the **unweighted joint** per-token NLL `content NLL + space NLL + cap NLL` (**no λ** — λ is a training knob, not bits), with **BOS/special/ignored targets masked for BOTH** the content and surface terms. Then bpb needs both ends fixed: **numerator** = both heads' bits (a one-head bpb undercounts — the MTP/zloss eval-path lesson); **byte denominator** = `token_bytes[base_y] + space_bit_y` (the base token's bytes OMIT the factored leading space — 1 byte per set space bit; the case fold is byte-preserving), else the denominator is short ~one byte per space-preceded token and bpb is inflated.

The predicted bits feed back only at *generation* (emit base token → predict its surface bits → render → feed both into the next step). Training stays exactly as parallel as standard nanochat; the deltas are: a surface embedding on the input, a small surface head, the extra loss term, and a dual-stream dataloader. This is the "harness change" (patched `base_train.py`) listed in [Direction](#direction).

**Throughput — the hot path is single-threaded Python; it had a quadratic bug (now fixed), is fine for d6, and is the open risk for a fast GPU run.** The *actual* training hot path is `surface_dataloader._encode_doc → SurfaceTokenizer.encode → surface_codec.surface_encode` (per-pretoken `encode_ordinary` + an O(tokens) cap walk), **not** `bench_surface_encode.py`'s `opt_surface_encode`, which is a lossy *timing stand-in* (it drops every single-space token). The numbers below were **measured on the deprecated production triple tokenizer** (per-pretoken `encode_ordinary` + a multi-word cap walk; surface-only's encode is the same shape and simpler — a single-word cap is one bit, no multi-word walk). 128-doc batch, `overlay/surface_codec.surface_encode`, single thread:

| encoder | tok/sec (1 thread) | vs nanochat batched encode |
|---|---:|---:|
| surface_encode — original (O(tokens·word-starts) cap scan) | ~72k | ~29× slower |
| surface_encode — **two-pointer cap walk (current)** | **~265k** | ~8× slower |
| nanochat `encode_ordinary_batch` (num_threads=4, reference) | ~2.1M | — |

The original had an **O(tokens·word-starts) quadratic** in the cap-ownership scan (`[o for o in ws_off if lo<=o<hi]` per token — 69% of encode time). A two-pointer walk over the sorted word-starts fixed it (**3.7×**, surfaces preserved — all surface smokes still pass). The remaining cost is balanced single-threaded Python: per-pretoken `encode_ordinary` ~24%, the pre-tokenize regex ~21%, `char_word_starts` ~18%, the emit/derive loop ~20%.

**Is it fast enough?** For **d6** (model ~24k tok/sec) 265k is ~11× the consumption rate, so the dataloader is no longer the bottleneck. But the **first d6 runcpu — with the quadratic still in — ran ~2× slower than a stock runcpu** (jagged 4–24k tok/sec, the synchronous-refill-stall signature); the two-pointer fix is expected to largely close that (re-run to confirm). For a **fast multi-GPU d24 speedrun** the single-process Python encode at ~265k tok/sec/rank is the open risk — levers, in order: (a) batch the per-pretoken `encode_ordinary` into one `encode_ordinary_batch(num_threads)` per refill (parallelizes the ~24%); (b) **background-prefetch** the tokenization to overlap it with GPU compute (hides the whole latency — the robust fix); (c) Rust-ify the fold + cap walk only if (a)+(b) fall short. The earlier "~70× over requirement, no Rust needed" verdict was based on the lossy stand-in and is **retracted** — benchmark the real path before the GPU run. The production encoder already keeps the O(tokens) shape with the validated lone-space rule (absorb a single space as a bit only when the next token is non-whitespace content), so the algorithmic shape is right; the gap is constant factors + single-threadedness.

**Gotchas.** For surface-only K_max=1, so `surface_y` is a fixed-width `(space_bit, cap_bits[1], cap_mask[1])` — no variable-length label (cap_mask=0 for a token that owns no word-start: a mid-word continuation or non-word token). (Under the deprecated triple a *phrase* token's surface label was a per-word cap *pattern*, not one bit — the [0.2% multi-cap finding](#triple-rendering-subtleties-phrase-internal-spaces--per-word-cap-patterns).) Surface bits must stay aligned with base tokens through the best-fit packer and the BOS prepend (BOS gets a null surface label) — a place to get an off-by-one.

### Pre-A/B validation (smoke ladder — rungs 0–4 done)

The model-side contract above is validated by self-contained smokes that prove it's implementable and the pieces fit, **before** running a real A/B. They use a tiny reference impl (`wrappers/_surface_ref.py`: a small spaceless+case-folded tokenizer trained inline, the slot-based encoder/decoder, and a tiny `MiniSurfaceGPT`), run in seconds with no data/GPU, and follow the repo's `smoke_<idea>.py` convention. The reference impl and several checks exercise the **K>1 (phrase-token) path** — the retained deprecated-triple generalization; surface-only collapses these to `K_max=1`, but the K>1 smokes still validate the machinery's correctness.

`wrappers/smoke_surface_factoring.py` (rungs 0–3, **13/13 PASS**):

| rung | check | validates |
|---|---|---|
| 0a | encode→decode round-trip lossless incl. unicode (`Łódź`, `Ελληνικά`) | the dual-stream contract is lossless + unicode-safe |
| 0b | Σ base_bytes + Σ space_bit == original bytes | the bpb byte-denominator fix (review P1.2) |
| 0c | `{of the, Of The, Of the, of The}` → `{00,11,10,01}` cap values; 1 word = 1 owned slot, continuations own 0 | per-occurrence cap values + occurrence-active mask (K>1 path) |
| 0d | `SurfaceTokenizer` wrapper: `encode` == codec, `decode` round-trips, `byte_count` == bytes | the production tokenizer wrapper agrees with the codec |
| 1a | loss == content CE + λ(space BCE + masked cap BCE) | loss decomposition |
| 1b | flipping masked-out cap targets leaves cap loss unchanged | masking actually masks |
| 1c | `wte+pos+space_emb+Σ slot-specific masked cap_emb`; zeros → `wte+pos` exactly | input-embedding composition |
| 1c2 | `Of the` (`[1,0]`) ≠ `of The` (`[0,1]`) in the trunk input | slot-specific cap embedding is injective, K>1 path (review P1) |
| 1d | naive GPT subclass RAISES; proper subclass passes | param-accounting overrides (review P1.3) |
| 2 | forward+backward; grads reach every surface param | nothing disconnected |
| 3a | surface-off content CE == baseline, **Δ=0.0** | reduces-to-baseline (machinery non-disruptive) |
| 3b | bpb denominator over 8 texts == original bytes | denominator invariant at batch scale |
| 3c | overfit-a-batch: loss 6.9→0.24, space/cap acc 1.00 | the heads actually learn the surface bits |

`wrappers/smoke_surface_factoring_eval.py` (rung 4, **3/3 PASS**):

| rung | check | validates |
|---|---|---|
| 4a | joint NLL ranks true seq below content- AND surface-corrupted variants | surface-aware CORE-style scoring (review P2.5) |
| 4b-i | incremental token-by-token decode reconstructs the original | generation-side decode path |
| 4b-ii | autoregressive sample (base→conditioned space+cap) → valid UTF-8 | generation plumbing (review P2.5) |

`wrappers/smoke_surface_factoring_overlay.py` — the REAL overlay (`overlay/surface_factoring.py`) against nanochat's `GPT` (**5/5 PASS**):

| check | validates |
|---|---|
| A | `num_scaling_params` / `estimate_flops` / `setup_optimizer` all build with the surface params (the assert-coverage overrides work on the real GPT) |
| B | dual-stream forward + backward; gradients reach the active surface params (input embs + heads) |
| B2 | surface value-embed wiring: `ve = base_ve + space + cap` (both labels condition the value embedding) |
| C | **reduces-to-baseline: overlay content logits == vanilla GPT, max\|Δ\|=0.0** — the copied forward faithfully reproduces `GPT.forward` |
| D | loss contract: eval (`loss_reduction='none'`) is the per-token, **λ-free** joint NLL; training is λ-weighted |

`wrappers/smoke_surface_harness.py` — the data→context→model→bpb path on synthetic docs (**4/4 PASS**):

| check | validates |
|---|---|
| 1 | `_encode_doc`: the four BOS-prepended streams round-trip to the text |
| 2 | the loader yields aligned base `(inputs, targets)` + sets the surface context; a packed row decodes back to its document (packing + shift + handoff) |
| 3 | the real overlay consumes a loader batch *via the out-of-band context* (finite joint loss, grads to the surface head) |
| 4 | `surface_evaluate_bpb` runs with the `token_bytes[base_y] + space_y` denominator |

**Retired vs. remaining.** The *correctness / "do the pieces fit"* risk is retired at every level: contract (rungs 0–4 on the tiny reference), real-integration (`overlay/surface_factoring.py` bit-identical to vanilla GPT with surface off), and **end-to-end harness** (the d6 runcpu trained cleanly through a real training loop — that run was the deprecated triple, see the [Deprecated appendix](#downstream-runcpu-d6--the-deprecated-triple-model-run), but the harness validation stands and carries over to the simpler surface-only `K_max=1` tokenizer). **Built + validated (29 checks across 5 suites):** the overlay (surface-conditioned value embeds + heads + train/eval loss), the codec + `SurfaceTokenizer`, the harness (`wrappers/train_surface_factoring.py` + `overlay/surface_dataloader.py` + `surface_context.py` + `runs/runcpu_surface.sh`), and the CORE scorer (`wrappers/smoke_surface_factoring_core.py`, rung 4c). The production **surface-only** tokenizer (vocab 32768, K_max=1) is **cache-generated on demand** by `wrappers/tok_train_surface.py` (not a committed artifact). **Remaining before the GPU A/B:** the **surface-aware generation path** — generation with context-derived masks (the Engine streams token ids only) and surface-aware `chat_sft`/`chat_cli` for ChatCORE. (The CORE eval path is already BUILT + was validated end-to-end on the **deprecated triple's** d24 CORE eval — `overlay/surface_core_eval.py` joint-likelihood scoring, `wrappers/base_eval_surface.py`, and `overlay/surface_checkpoint.py` K_max-aware load via the persisted `surface_config`; it's tokenizer-agnostic, so it carries to surface-only, which has no run of its own yet.) Plus the [throughput](#training-integration) follow-ups for a fast multi-GPU run (batch-encode / background-prefetch). One known generation cost stands: occurrence-active masks must be derived from running context at generation (the eval smoke approximates the *input* mask; decode recomputes ownership exactly, so correctness holds).

## Caveats

- **The token reduction doesn't *mechanically* lower bpb — whether it lowers bpb in the fixed-compute regime is the A/B question.** The +4.90% is a *sequence-length / sample-efficiency* gain (fewer tokens per byte → more text per fixed token budget). bpb is normalized per *byte*, so unlike bits-per-*token* it doesn't drop just because there are fewer tokens; in the *infinite-capacity* limit it's invariant to a lossless re-tokenization. But under this project's *fixed-compute* constraint bpb is NOT invariant — a denser, cleaner tokenization can be easier for a fixed-capacity model to predict (that finite-capacity effect IS the sample-efficiency bet, and is why force_merges' compression produced a CORE win). Once both heads are counted everything is one currency (bits/byte): the open question is whether the lower *content* bpb from denser training outweighs the surface head's *added* bpb (~0.1 bits/byte gross for space+case, less after sequential conditioning). That net is the A/B question.
- **Surface-only has NO model A/B yet — both bpb and CORE are UNVERIFIED.** There is no surface-only training run. The only end-to-end model evidence so far — a d6 runcpu bpb lead — was on the **deprecated triple** ([appendix](#downstream-runcpu-d6--the-deprecated-triple-model-run)) and is abandoned; it does NOT carry over as a surface-only result (different tokenizer, vocab, K_max, and the d24 follow-up showed the d6 bpb lead didn't even replicate at depth for the triple). So for surface-only: the offline +4.90% compression stands, but *both* training-time bpb and CORE (the capability target) remain to be measured by the clean surface-only vs baseline A/B.
- **Gains are corpus-mix-dependent.** Measured on ClimbMix (prose-heavy). Code-heavy or more-prose corpora would shift the numbers. The [multi-space extension](#dead-ends) is useless here precisely because ClimbMix has ~100% single-space gaps.

### Dead ends

- **Multi-space "count" bit** (extend the space bit to encode N consecutive spaces): +0.003% on ClimbMix — useless, because ~100% of leading-space gaps are a single space. Only helps indentation-heavy raw code, which ClimbMix is not. Don't pursue unless the target corpus is source code (and even then it can't encode tabs).
- **Stacking force_merges + surface factoring (the "triple").** Building surface on top of the force_merges phrase vocab (fm122 + space + case, +11.11%, K_max=3) was a premature, greedy stack: it confounded the d24 A/B (couldn't separate "factoring is inert" from "factoring is redundant with force_merges' phrase merges") and forced the K_max=3 multi-slot cap machinery. Rolled back; see the [Deprecated appendix](#deprecated-the-fmspacecase-triple-dead-end). The clean experiment is surface-only vs baseline.

## Direction

The decisive next experiment is the **clean factoring-isolation A/B**: **surface-only** (space + case on the baseline vocab, vocab 32768, K_max=1) vs the **vanilla baseline**, with the dual-head model, at d24 on a Lambda speedrun, measuring CORE / val_bpb / ChatCORE. This isolates whether factoring space + case out of token identity helps, hurts, or is inert — the question the deprecated triple A/B couldn't answer (it confounded factoring with force_merges' phrase merges). Optionally run **force_merges as a SEPARATE sibling arm** for reference (NOT stacked). It is a real build, sequenced:

1. **Production tokenizer wrapper — ✅ BUILT.** `overlay/surface_codec.py` (the unicode-safe dual-stream `surface_encode`/`surface_decode`/`compute_kmax`, tokenizer-agnostic) + `overlay/surface_tokenizer.py` (`SurfaceTokenizer`: `from_directory` load, `encode`/`decode`, `byte_count` for the bpb denominator). Regression-tested in `wrappers/smoke_surface_factoring.py` (`check_surface_tokenizer`). The production **surface-only** tokenizer (vocab 32768, K_max=1) is **cache-generated on demand** by `wrappers/tok_train_surface.py` (case-folded corpus + spaceless pattern + NO forced pairs → saved to `~/.cache/nanochat-variants/surface_only/tokenizer/` with `token_bytes.pt`); `runs/runcpu_surface.sh` runs it before training. It is reproducible from the corpus, not a committed/prebuilt file — a fresh machine regenerates it (a few minutes).
2. **Model overlay** (`overlay/surface_factoring.py`) — **✅ BUILT + smoke-validated** (`wrappers/smoke_surface_factoring_overlay.py`, 5 checks: param accounting, forward/backward, surface-value-embed wiring, **reduces-to-baseline bit-identical to vanilla GPT (max|Δ|=0.0)**, and the train-vs-eval loss contract). `surface_emb` + (slot-specific) `cap_emb` on the input, **surface-conditioned per-layer value embeddings** (the chosen full-info-equivalence path), and the space + cap heads conditioned on the next base token. Overrides `num_scaling_params`/`setup_optimizer`/`estimate_flops`/`init_weights`. Reads `SURFACE_KMAX`/`SURFACE_LAMBDA` from env (instance attrs, not GPTConfig fields) — for surface-only `SURFACE_KMAX=1`. Follow the overlay conventions in [`../../overlay/README.md`](../../overlay/README.md) (hyperparameters as instance attributes, not GPTConfig fields). **Smoke must verify the *joint* eval, NOT vanilla-GPT eval-path equivalence** — this overlay's eval path is deliberately different, so the standard "eval returns plain token CE" check (from zloss/MTP) does NOT apply. Smoke instead: `bpb = content CE + surface CE`, the byte-denominator fix (factored spaces added back), round-trip losslessness, and BOS/special targets carrying masked/null surface labels. **This overlay ADDS parameters** (`surface_emb`, the heads), so unlike the pure-loss overlays it must override the methods that `assert` exact coverage — `num_scaling_params` (`gpt.py:345`; `assert total == sum(p.numel() ...)` at `:364`) and `setup_optimizer` (`gpt.py:374`; the param-group-count assert) — plus `estimate_flops` (`gpt.py:317`) for FLOP accounting, `init_weights`, and the optimizer LR groups. It's closer to an architecture-subclass overlay than a pure-loss one. **Checkpoint/load — ✅ BUILT:** `surface_emb`/heads are sized by `K_max` (tokenizer-derived; 1 for surface-only), and stock `checkpoint_manager.py` strict-loads a vanilla `GPT` (so a surface checkpoint would NOT load there — unexpected keys, `K_max` unknown). Handled: `wrappers/train_surface_factoring.py` persists the surface metadata (`K_max`, λ) in the checkpoint meta's `surface_config`, and `overlay/surface_checkpoint.py` (`surface_build_model`) rebuilds `SurfaceFactoringGPT` from it on load — the surface-aware `base_eval_surface` uses this path.
3. **Harness — ✅ training + CORE-eval paths BUILT (generation/ChatCORE still TODO).** Instead of copying `base_train.py`, the harness keeps it pristine and patches around it: `wrappers/train_surface_factoring.py` swaps in `SurfaceFactoringGPT`, the surface dataloader, and `surface_evaluate_bpb`, then runpys upstream `base_train`. The four surface streams flow out-of-band through `overlay/surface_context.py` (the dataloader stashes the current batch; the patched `forward` reads it — safe because the upstream loop strictly alternates `next()`→`model()`). `overlay/surface_dataloader.py` best-fit-packs all four streams in lockstep and yields the base `(inputs, targets)` upstream expects; `surface_evaluate_bpb` fixes the byte denominator to `token_bytes[base_y] + space_y`. `wrappers/tok_train_surface.py` trains+saves the surface-only tokenizer (case-folded corpus + spaceless pattern, NO forced pairs) into the standard dir so `get_tokenizer`/`get_token_bytes` resolve unpatched. Smoke: `wrappers/smoke_surface_harness.py` (4 checks — packing+alignment, context handoff, overlay-consumes-batch, surface bpb) on synthetic docs; run it via `runs/runcpu_surface.sh`. For surface-only **`K_max=1`** (no multi-word phrase tokens). **CORE eval — ✅ BUILT:** `overlay/surface_core_eval.py` (surface-aware prompt encoding + joint-likelihood scoring), `wrappers/base_eval_surface.py` (surface-aware `base_eval`), and `overlay/surface_checkpoint.py` (rebuilds `SurfaceFactoringGPT` from the persisted `K_max`/`surface_config`) — validated end-to-end on the deprecated triple's d24 CORE eval (tokenizer-agnostic, so it carries to surface-only). **Still TODO** (the generation half): the Engine streams token ids only, so generation needs surface-aware generate+decode, and `chat_sft`/`chat_cli` need surface-aware variants for ChatCORE. `runcpu_surface.sh` skips `base_eval` because it's a CPU mechanism check (CORE is expensive on CPU), not because the path is missing. Likewise `train_surface_factoring.py` disables base_train's in-training `core_metric` + sampling (`--core-metric-every=-1 --sample-every=-1`): both would run the standard scorer/generate on the surface model OOD (base-only inputs, no surface decode), so their live WandB curves would mislead — the valid surface CORE is the post-training `base_eval_surface`. **Crucially, the A/B metrics also need surface-aware eval + generation:** CORE/ChatCORE aren't meaningful until those paths are surface-aware — budget for them, not just train + val-bpb.
4. **A/B** — same pinned nanochat commit + seed; baseline = **vanilla baseline** (clean factoring isolation), variant = **surface-only**; optionally **force_merges as a separate sibling arm** for reference (NOT stacked). Do NOT speedrun until the overlay + harness pass smoke and the lead is worth the GPU cost (see the [no-premature-speedrun discipline](../../STATUS.md)).

Decisions settled (2026-06): **value_embeds = surface-conditioned from the start** (full info-equivalence — built); **the A/B target is surface-only** (space + case on the baseline vocab, K_max=1) — the fm+space+case triple was rolled back. Still open: **λ** (`SURFACE_LAMBDA`=0.5, untuned) and **exposure/scale** as deliberate variables; whether to add a **space-only** ablation to isolate each feature's contribution; and whether to run the **force_merges** sibling arm for reference.

### Cheap λ/exposure triage (paired content-bpb)

The cheapest reliable signal for triaging a surface-only knob change (λ, exposure) without a full d24 run (~$90/arm) is a **PAIRED Δ(content-only bpb)** — candidate arm vs a *once-baked* surface-only control at identical commit / seed / data-shard-order (both see the same batches; only the surface params differ). Pairing drops the noise floor to ~0.001–0.002 bpb; `wrappers/content_bpb.py` computes it. It is a **KILL filter, NOT a promote signal** (content-bpb is empirically decoupled from CORE here), so it cheaply rejects a knob change that failed its own premise; promotion still needs the MC-CORE robust trio. **Validity is direction-specific:** the **within-surface** paired comparison (surface-λ_test vs surface-λ_control) is *valid* — the spacing/casing offloading cancels in the pairing. Do **not** compare content-bpb *across tokenizers* (e.g. surface-vs-fm): a surface content stream offloads spacing/casing so it encodes LESS — its lower content bpb is accounting, not better modeling; the clean cross-tokenizer signal is the **JOINT** bpb. The full triple-era protocol (the once-baked λ=0.5 triple control, the run-tomorrow λ=0.25 triage, the MC-CORE gate specifics) is in the [Deprecated appendix](#cheap-ablation-protocol-triage-a-lever-for-10-of-a-full-ab).

## Fast follow

Cheap, offline, no-GPU refinements to run on the surface-only tokenizer **before** committing to the model-A/B build. These are token-*quality* follow-ups (junk-merge removal, whole-word coverage) that complement the compression results above — token quality (real whole words, fewer dead/junk tokens) plausibly matters as much downstream as raw bytes/token, and these reuse the existing `auto_tune` held-out-block audit machinery. None of these are done yet.

1. **Run a `--propose` junk-merge pass against the surface-only tokenizer.** `heldout_fixpoint_fm`'s firing-percentile-33 mining + held-out-Pareto gate was only a *marginal* compression lever on its own, but it is **excellent at surfacing unexpected junk merges** — it ranks blocks that yield whole words, improve compression, or reduce dead tokens. Run the orchestrated `--propose` loop against a **surface-only context** (the spaceless `SPACELESS_BODY` pattern + the case-folded corpus, NO forced pairs). **Requires extending the driver:** `tools/heldout_fixpoint_cached.py` currently has `--fm-context` (force_merges pattern + 122 pairs at vocab 32890); add a `--surface-context` that swaps in the spaceless pattern (`stack_fm_spaceless.SPACELESS_BODY`) and case-folded corpus (`case_factor_compression.fold_case`) at vocab 32768 with no forced pairs. Then drive it per the audit README (`rustbpe_variants/auto_tune/audit/heldout_fixpoint_fm/README.md`): orchestrated `--propose` → hand-judge each round by the whole-word/realness criteria → `--commit '["L","R"]'`, with the **hard semantic veto** (never block a legitimate token — `ob+by` shatters hobby/lobby). **Never** run the bare autonomous loop — it auto-commits junk. Goal: a surface-specific block list, the way `force_merges_combined` is the force_merges-specific one.

2. **Consider blocking punctuation-prefixed word tokens at the regex level.** The baseline (and current spaceless) word alternative captures one leading punct char — `[^\r\n\p{L}\p{N} ]?+\p{L}+` — producing tokens like `.com`, `(the`, `"word`. A prior (informal) experiment found a *few* are frequent and useful (`.com`) but *most* are junk, and the aggregate frequency is likely too low to justify a dedicated surface feature (the way space/case earn their bit). The cheap alternative is a regex change: drop the leading-punct capture (`[^\r\n\p{L}\p{N} ]?+\p{L}+` → `\p{L}+`) so the punct splits off as its own token and these tokens never form. There is a small compression cost (the frequent `.com`-class tokens become two tokens), but it may be worth it for **token quality** (fewer junk tokens, lower dead-token count, cleaner word atoms). **Evaluate:** measure the bytes/token cost and the dead-token / whole-word-coverage change of the `\p{L}+`-word variant vs the current spaceless word pattern (a one-line edit to the word-alt substring `[^\r\n\p{L}\p{N} ]?+\p{L}+` in `space_factor_compression.py`'s `SPACELESS_PATTERN` and `stack_fm_spaceless.py`'s `SPACELESS_BODY` — which the surface-only builder pulls in).

3. **Compare whole-word token coverage, baseline vs surface-only.** The audit's primary token-quality metric is whole-word coverage — how many vocab tokens are real dictionary whole words with standalone usage (`tools/pass_metrics.py` `whole_word_counts`, `tools/word_realness.py`, `tools/blocked_pairs_report.py` `_is_word`). Surface factoring pools `The`+`the` / ` the`+`the` and frees ~24% of vocab, so the freed slots plausibly form *more* complete-word tokens (denser merges) — open question whether the difference is notable. **Caveat that needs handling first:** `whole_word_counts` keys on **space-prefixed** dict words (` word`), which the surface tokenizer doesn't have (spaces are factored out, and titlecase is folded). Adapt the metric to count **base-form** whole words (no leading space, case-folded comparison) for an apples-to-apples baseline-vs-surface count, then report the delta alongside the bytes/token ladder.

## References

- Related: [`../tokenizer-variants/README.md`](../tokenizer-variants/README.md) (force_merges, auto_tune — a separate sibling tokenizer track, not stacked with surface), [`../../STATUS.md`](../../STATUS.md) (force_merges downstream result, tokenizer-eval workflow), [`../../overlay/README.md`](../../overlay/README.md) (overlay implementation lessons).
- force_merges crate: `rustbpe_variants/force_merges/` (`pairs.py` for the 122 forced phrases + the `space_digits` regex note; `src/lib.rs` for the `block_leading/trailing_space` + forced-merge training) — used by the sibling force_merges track and the deprecated triple; surface-only is crate-free (base rustbpe).

---

## Deprecated: the fm+space+case triple (dead-end)

**This whole section is abandoned history, kept for the lesson — not the current surface result.** Surface factoring *used to* be implemented as a STACK on the `force_merges` phrase vocab: `force_merges` (cross-word PHRASE merges) + space-factoring + case-factoring = the "triple" tokenizer (vocab 32890, K_max=3, +11.11% tokens). That stacking was premature/greedy: it **confounded the d24 A/B** (it couldn't separate "factoring is inert" from "factoring is redundant with phrases") and forced the K_max=3 multi-slot cap machinery. It was **rolled back**; surface is now [surface-only](#the-idea) (space + case on the baseline vocab, K_max=1). The d24 triple checkpoint is abandoned. The findings below are real history of the *deprecated triple* — they do NOT describe surface-only, which has **no model A/B yet**.

### Preliminary d24 A/B findings

A 3-arm GPU A/B (8×H100, d24, same nanochat commit `dc54a1a`, **default λ=0.5, 5569 steps**) — **vanilla baseline**, **force_merges**, **surface (force_merges+space+case = the triple)** — scored on CORE + bpb. Read as *preliminary history of the deprecated triple*: the verdict was **"tied with force_merges — no net benefit, but no harm — at the as-tested config," not "worse" and not "doesn't work."**

**What was clean.** *bpb is tied:* triple-surface **0.7141** vs force_merges **0.7137** (raw-byte basis, apples-to-apples) — the d6 surface bpb lead (−0.021) **did not replicate at d24** (a small-model effect the bigger model erases on its own). *CORE ladder:* vanilla **0.2604** → force_merges **0.2758** → surface **0.2114 (buggy eval) → 0.2773 (corrected)**. Corrected, **surface ties force_merges** (0.2773 vs 0.2758, +0.0015 = within single-seed noise) and beats vanilla; force_merges' own d6 +5.81% over vanilla holds at d24. Multiple-choice/schema CORE was tied throughout. So on every fair axis (bpb + MC + the corrected LM) the triple is a **wash vs force_merges — no benefit, no harm.**

**The apparent deficit was 100% an eval bug, not the model.** The buggy surface headline (0.2114) was dragged down entirely by the **language-modeling tasks**: the LM scorer ran the surface trunk **base-only — out-of-distribution** (no surface input embeddings, which the trunk saw at every training step). An inspection of real examples confirmed it (OLD base-only predictions garbage — `"a"`, `"what05"`; fed the surface input streams, coherent/exact — `"boston"`, `"2005"`). Fixed in `overlay/surface_core_eval.py`. **The fix scores a *lossless* exact-match** — content + space + cap must all match = exact rendered-text match, identical in spirit to a standard tokenizer whose tokens *are* the text (a content-only match is lenient on space/case and not apples-to-apples; `lossless_lm_match` requires all three streams). The corrected LM tasks **recovered to fm parity** — squad 0.06→**0.45** (fm 0.44), lambada 0.18→**0.47** (fm 0.48), coqa 0.16→**0.32** (fm 0.32), wikidata-QA 0.13→**0.48** (fm 0.47) — lifting surface CORE 0.2114→**0.2773**. So surface ≈ fm; the +11% compression nets to zero capability change, not a loss.

**Why no benefit on the clean axes — and why it wasn't called final.** A whole-word analysis (refuting an earlier hand-wave) shows the triple-surface tokenizer **is** a genuinely denser/cleaner content tokenizer (+4.7% words per fixed window; whole-word-hit +2.86%, 5.2× the base→fm jump; 77.5% genuine content densification; case has *negative* compression = a free annotation). **But the gain is the wrong shape for capability at a fixed budget:** it sits in **rare long single-word atoms** (undertrained embeddings) with **zero new phrase merges** (69 = 69 vs force_merges' 122) — and force_merges' CORE win came precisely from *frequent, reused, compositional* phrase merges. De-duplication is **moot for the frequent words** that drive capability: fm's busiest surface-variant of each already captures a **median 92.3%** of its occurrences, so collapsing variants buys only ~1.08× more exposure. ("Denser ≠ more learnable" at fixed budget; the glue-vs-content tension reduces to "real-but-undertrained vs ~free," which one d24 run can't separate.) **This was a confounded read** — and is exactly why the stack was rolled back to surface-only, where factoring is no longer entangled with the phrase merges.

**Untested levers (the verdict was premature without these):** **`SURFACE_LAMBDA` (λ=0.5)** — the content-vs-surface capacity split, never tuned; **exposure/scale** — 5569 steps undertrains the rare atoms; **lighter value-embeds** — the heaviest part of the 3-channel objective; **phrase-steered budget** — make surface's reclaimed slots buy fm-style high-frequency *phrase* merges instead of rare long words. Caveats: single seed; cross-tokenizer token-count normalization. (λ and exposure carry forward as the open levers for the surface-only A/B; the [Cheap λ/exposure triage](#cheap-λexposure-triage-paired-content-bpb) above is the rescoped, surface-only version of the protocol below.)

#### Cheap ablation protocol (triage a lever for <10% of a full A/B)

A multi-agent design pass settled the question "what's the minimum to know if a knob change is promising, without a full d24 run (~$90/arm) + full 22-task CORE." The results (triple-era anchors; the still-valid *within-surface* paired methodology is rescoped to surface-only in [Cheap λ/exposure triage](#cheap-λexposure-triage-paired-content-bpb)):

**Cheapest reliable signal = a PAIRED Δ(content-only bpb)** — candidate arm vs a *once-baked* λ=0.5 control at identical commit / seed / data-shard-order (both see the same batches; only the surface params differ). Pairing drops the noise floor to ~0.001–0.002 bpb (vs the 0.001–0.005 unpaired band); `wrappers/content_bpb.py` computes it.

**It is a KILL filter, NOT a promote signal.** content-bpb is empirically *decoupled from CORE* here — the λ=0.5 arm already had a content-bpb edge and produced ZERO CORE gain — so it cheaply REJECTS a knob change that failed its own premise, but promotion to a full run still needs the bug-free **MC-CORE robust trio** (arc_easy / piqa / hellaswag via `surface_core_eval.py`'s MC/schema argmin path; skip the ~28-min bigbench_language_identification).

**content-bpb's validity flips with the comparison.** surface-vs-fm content-bpb is *confounded* (different tokenizers — surface's content stream offloads spacing/casing, so a lower content bpb is accounting, not better modeling; the clean cross-tokenizer signal is the JOINT bpb, which tied 0.7141≈0.7137). But surface-λ_test vs surface-λ_control is *valid* — the offloading cancels in a paired within-surface comparison. Don't compare content-bpb across tokenizers.

**Per knob:**
- **λ** — content-bpb is mandatory (joint bpb is confounded by exactly what λ trades: content_CE vs surface_BCE). KILL at K≈500 if Δ(content-bpb) ≥ −0.003; promote-screen at K≈1850 (⅓) if ≤ −0.005, sign-stable over the cooled tail, joint Δ ≤ +0.002, AND MC-trio Δ ≥ +0.01.
- **Exposure — NOT truncatable.** Its hypothesis is that rare long-word atoms mature *late*, so any short run falsely KILLS it. Instead read the rare-atom-subset content-bpb SLOPE on the EXISTING checkpoints (0 new steps); if still descending vs a frequent-atom subset, the only valid confirmation is a *longer/larger* run — never a shorter one.
- **Value-embeds** — joint bpb is OK here (doesn't redefine the split) + a Δ_content/Δ_surface decomposition; K≈500 Tier-A kill if content regresses.

**Two discipline points:** (1) use a genuinely shorter cooled schedule (`--num-iterations=K --warmdown-ratio=0.65`), NOT a step-K snapshot of the full 5569 schedule (near-full-LR, reorders after cooldown). (2) Scale guard: the d6→d24 collapse was a DEPTH effect — pin depth at 24; pairing + the right axis + reading the SIGN (not magnitude or an extrapolated asymptote) handle the rest.

**Run-tomorrow (λ=0.25 triage, ~$16):** bake the λ=0.5 control + a λ=0.25 candidate, both `--num-iterations=500 --warmdown-ratio=0.65`, then `content_bpb.py` paired on the two cooled checkpoints; KILL if Δ(content-bpb) ≥ −0.003. ~$8/arm (control reused across knobs) → all three knobs screenable for less than one full A/B (~3–6× cheaper than deciding via full A/Bs).

### Deprecated compression ladder rows (the +11.11% triple)

The fm-family and triple rows, relocated from the headline ladder. Same held-out ClimbMix val basis as the surface-only ladder.

| scheme | vocab | tokens | bytes/tok | vs baseline |
|---|---:|---:|---:|---:|
| baseline | 32768 | 4,269,541 | 4.6986 | — |
| force_merges (69 word-phrases only) | 32837 | 4,084,148 | 4.9118 | +4.34% |
| fm(69) + space | 32837 | 3,989,534 | 5.0283 | +6.56% |
| force_merges (122 full) | 32890 | 3,973,768 | 5.0483 | +6.93% |
| fm(122) + space | 32890 | 3,878,996 | 5.1716 | +9.15% |
| **fm(122) + space + case (triple)** | 32890 | 3,795,152 | 5.2859 | **+11.11%** |

`fm(122)` reproduces adopted `force_merges` (+6.93% ≈ STATUS.md's recorded −6.9% tokens). Vocab is additive per force_merges convention (32768 + #phrases). **Across the two ladders the token column is not globally monotonic** — surface-only `space + case` (4,060,174) shows fewer tokens than the `fm(69)` row (4,084,148) because they're different families; each within-family chain IS monotonic. **Provenance:** `fm(69)+space` from `stack_fm_spaceless.py` (default), `fm(122)+space` from `--all-pairs`; the triple from `stack_triple.py`.

### Orthogonality / stacking

The features reclaim different slices of duplication, so gains compose (deltas multiply): e.g. baseline → fm+space → triple is 0.909 × 0.978 = 0.889 = −11.11%.

| feature | standalone | stacked on space | on fm+space |
|---|---:|---:|---:|
| space (idea-specific, vs digit-fixed) | +2.23% | — | +2.39% (on fm122) |
| case | +2.06% | +1.88% | +2.16% |

Capitalization contributes ~+2% on **every** base it stacks on — the signature of a genuinely orthogonal feature. (This orthogonality is *why* surface-only retains both space and case — but the cross-fm column is the deprecated stacking, not the surface-only target.)

### Triple rendering subtleties (phrase-internal spaces + per-word cap patterns)

The triple's phrase tokens forced two losslessness subtleties that surface-only (K_max=1, single-word tokens) does not have. Every triple scheme was still verified by full encode → decode round-trip: **2000/2000 held-out docs lossless** for fm+space and the triple, plus the adversarial batteries (the `of the`/`Of The`/`OF THE` triple, overlapping phrases, etc.).

- **Phrase tokens keep INTERNAL spaces** — ` of the` → token `of the` (internal space in the token bytes) + one leading-space bit; rendering re-applies only the leading space. `block_trailing_space`/`block_leading_space` in the force_merges crate guarantee internal spaces attach as leading to the following sub-word, so reconstruction is exact even when a phrase doesn't fully merge.
- **Phrase tokens carry a per-WORD cap pattern, not a single bit** — case-folding the corpus folds *titlecase* words inside phrases, so a 2-word phrase token (`of the` from `Of The`) can carry 2 capitals (titlecase examples: `Is The`, `At The`, `More Than`). Note all-caps (`OF THE`) does NOT fold — it stays uppercase content carrying zero cap bits — so the multi-cap occurrences are the titlecase-multi-word subset, an upper bound of **~0.2% of phrase occurrences**. It's unavoidable because the folded `of the` is byte-identical regardless of *which subset of its words were titlecased* (`Of the`, `of The`, `Of The` all collapse to `of the`), so the carve-out can't distinguish them — hence the cap side-channel is per-word-position within a token. The round-trip represents caps as a per-token offset LIST (`tools/stack_triple.py` `encode_triple`/`decode_triple`); ordinary tokens carry a length-1 list, phrase tokens up to N. (This per-word-cap machinery — `cap_bits[K_max]`, slot-specific embeddings, `_cap_compose` — is the K>1 path retained in the overlay but inert at surface-only K_max=1.)

### Downstream (runcpu d6 — the deprecated-triple model run)

The first end-to-end model run (2026-06-25, `runs/runcpu_surface.sh`, d6 / 5000 iters / single seed / M4, 109.8 min) trained the full overlay on real data and measured `val/bpb`. It was the **deprecated triple** tokenizer (vocab 32890, K_max=3). It is a *mechanism check + a controlled training-time bpb A/B* — NOT a capability verdict (CORE is disabled in runcpu) — and the d24 follow-up showed this d6 bpb lead **did not replicate at depth**, so it does NOT transfer to surface-only.

| d6 run (nanochat `dc54a1a`, 5000 iters, batch 16384, seq 512) | min val bpb | Δ vs baseline |
|---|---:|---:|
| d6_baseline (vocab 32768) | 1.1654 | — |
| d6_force_merges (adopted; CORE +5.81%) | 1.1590 | −0.0064 |
| **d6_surface_factoring (triple, vocab 32890, K_max=3, λ=0.5)** | **1.1378** | **−0.0276** |

The triple posted the lowest bpb of every archived d6 run — ~4× force_merges' own bpb gain over baseline. bpb descended smoothly 3.21 → 1.1378 with no instability from the joint loss. The comparison was sound on its own basis (same byte basis via `surface_evaluate_bpb`, same commit/config, no likelihood leak). **But** it was a *compression* metric (CORE was never measured), the lead **did not replicate at d24**, and the tokenizer it measured is abandoned — which is why surface-only is treated as having **no model A/B yet**. Artifacts: `results/d6_surface_factoring/` + checkpoint at `~/.cache/nanochat-variants/surface_triple/base_checkpoints/` (deprecated).
