# Surface factoring

Separate token **content** (which word) from **surface form** (spacing, casing). Factor orthogonal, predictable, vocab-duplicating surface features out of token identity into per-token side-channel bits predicted by a cheap auxiliary head. The base vocabulary becomes a lexicon of content atoms; the surface bits handle rendering. Two validated features: **leading space** and **first-letter capitalization**.

**Status (2026-06-23): offline-validated, downstream-unverified.** The tokenizer-level compression and lossless rendering are measured and solid (see [Results](#results)). The full surface-factored tokenizer (force_merges + space + case) compresses **+11.11%** vs the nanochat baseline on held-out ClimbMix val. Whether that translates to a downstream CORE/bpb gain is **not yet tested** — it needs a dual/multi-head model A/B (a real overlay + harness build), which is the gate (see [Direction](#direction)). All numbers here are reproducible offline with no GPU via the [scripts](#scripts).

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

**Digit-space caveat (important, already known):** ~28% of the standalone spaceless gain (+0.87% of +3.08%) is NOT the space-factoring idea — it is the ` ?\p{N}{1,2}` "space-prefix-digit" regex tweak (baseline's `\p{N}{1,2}` has no leading-space capture, so ` 42` wastes a token on the space). This is the removed `space_digits` probe, already folded into `force_merges`' SPLIT_PATTERN (see `rustbpe_variants/force_merges/pairs.py:217`, measured +1.09–1.23% there). The genuine space-factoring gain vs a digit-fixed baseline is **+2.23%**. When stacking on force_merges (which already has the digit fix), space contributes only this idea-specific ~+2.3%.

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

| scheme | vocab | tokens | bytes/tok | vs baseline |
|---|---:|---:|---:|---:|
| baseline | 32768 | 4,269,541 | 4.6986 | — |
| baseline + digit-fix (`space_digits`) | 32768 | 4,232,409 | 4.7398 | +0.87% |
| caseless | 32768 | 4,181,656 | 4.7973 | +2.06% |
| spaceless | 32768 | 4,138,046 | 4.8479 | +3.08% |
| space + case | 32768 | 4,060,174 | 4.9409 | +4.90% |
| force_merges (69 word-phrases only) | 32837 | 4,084,148 | 4.9118 | +4.34% |
| fm(69) + space | 32837 | 3,989,534 | 5.0283 | +6.56% |
| force_merges (122 full) | 32890 | 3,973,768 | 5.0483 | +6.93% |
| fm(122) + space | 32890 | 3,878,996 | 5.1716 | +9.15% |
| **fm(122) + space + case (triple)** | 32890 | 3,795,152 | 5.2859 | **+11.11%** |

`fm(122)` reproduces adopted `force_merges` (+6.93% ≈ STATUS.md's recorded −6.9% tokens). Vocab is additive per force_merges convention (32768 + #phrases). **Rows are grouped by family (no-fm, then fm), so the token column is not globally monotonic** — `space + case` (4,060,174) shows fewer tokens than the later `fm(69)` row (4,084,148) because they're different families; each within-family chain IS monotonic. **Per-row provenance:** baseline / spaceless / `space + case` from `space_factor_compression.py` + `case_factor_compression.py`; `caseless` from `case_factor_compression.py`; `baseline + digit-fix` from `digitfix_remeasure.py`; `fm(69)+space` from `stack_fm_spaceless.py` (default), `fm(122)+space` from `--all-pairs`; the triple from `stack_triple.py`.

### Orthogonality / stacking

The features reclaim different slices of duplication, so gains compose (deltas multiply): e.g. baseline → fm+space → triple is 0.909 × 0.978 = 0.889 = −11.11%.

| feature | standalone | stacked on space | on fm+space |
|---|---:|---:|---:|
| space (idea-specific, vs digit-fixed) | +2.23% | — | +2.39% (on fm122) |
| case | +2.06% | +1.88% | +2.16% |

Capitalization contributes ~+2% on **every** base it stacks on — the signature of a genuinely orthogonal feature.

### Rendering (losslessness)

Every scheme verified by full encode → (tokens, surface bits) → decode → original round-trip: **2000/2000 held-out docs lossless** for space, case, space+case, fm+space, and the triple, plus hand-crafted adversarial batteries (~20 cases each: phrases at boundaries, overlapping phrases, CJK/emoji/unicode, multi-space, tabs, camelCase, acronyms, `iPhone`/`McDonald`, the `of the`/`Of The`/`OF THE` triple, `Łódź`, etc.).

Two losslessness subtleties handled:
- **Phrase tokens keep INTERNAL spaces** — ` of the` → token `of the` (internal space in the token bytes) + one leading-space bit; rendering re-applies only the leading space. `block_trailing_space`/`block_leading_space` in the force_merges crate guarantee internal spaces attach as leading to the following sub-word, so reconstruction is exact even when a phrase doesn't fully merge.
- **Phrase tokens carry a per-WORD cap pattern, not a single bit** — case-folding the corpus folds *titlecase* words inside phrases, so a 2-word phrase token (`of the` from `Of The`) can carry 2 capitals (titlecase examples: `Is The`, `At The`, `More Than`). Note all-caps (`OF THE`) does NOT fold — it stays uppercase content carrying zero cap bits — so the multi-cap occurrences are the titlecase-multi-word subset, an upper bound of **~0.2% of phrase occurrences**. It's unavoidable because the folded `of the` is byte-identical regardless of *which subset of its words were titlecased* (`Of the`, `of The`, `Of The` all collapse to `of the`), so the carve-out can't distinguish them — hence the cap side-channel is per-word-position within a token. The round-trip represents caps as a per-token offset LIST (`tools/stack_triple.py` `encode_triple`/`decode_triple`); ordinary tokens carry a length-1 list, phrase tokens up to N.

### Downstream (runcpu d6 — the first model A/B)

The first end-to-end model run (2026-06-25, `runs/runcpu_surface.sh`, d6 / 5000 iters / single seed / M4, 109.8 min) trained the full overlay on real data and measured `val/bpb`. It is a *mechanism check + a controlled training-time bpb A/B* — NOT a capability verdict (CORE is disabled in runcpu).

| d6 run (nanochat `dc54a1a`, 5000 iters, batch 16384, seq 512) | min val bpb | Δ vs baseline |
|---|---:|---:|
| d6_baseline (vocab 32768) | 1.1654 | — |
| d6_force_merges (adopted; CORE +5.81%) | 1.1590 | −0.0064 |
| **d6_surface_factoring (triple, vocab 32890, K_max=3, λ=0.5)** | **1.1378** | **−0.0276** |

Surface factoring posts the **lowest bpb of every archived d6 run** — ~4× force_merges' own bpb gain over baseline (and force_merges' smaller edge earned CORE +5.81%). bpb descended smoothly 3.21 → 1.1378 with no instability from the joint loss. The run-to-run spread among similar d6 tokenizers is ~0.001–0.005, so −0.028 is well clear of it (the project's blanket "<0.05 bpb ≈ noise" heuristic is conservative for noisy regularizer tweaks; tokenizer deltas reproduce far tighter — see the cluster above).

**Why the comparison is sound (all three checked):**
- **Same basis** — `surface_evaluate_bpb` divides the *joint* per-token NLL (content + space + cap, λ-free) by `token_bytes[base_y] + space_y`. Verified on the production tokenizer + 300 real val docs: 300/300 lossless and `Σ(base_bytes + space_bit)` == actual text bytes *exactly* (807,596 = 807,596). So it measures the same bits-per-original-byte as nanochat's stock bpb.
- **Same commit + config** — `d6_baseline` / `d6_force_merges_r194` are at the identical nanochat commit (`dc54a1a`) and training config; only the tokenizer differs (the experiment).
- **No likelihood leak** — the surface heads condition on the *true* next base token at eval, i.e. the chain rule `P(base|ctx)·P(surface|base,ctx)`, the correct joint likelihood (not teacher-forcing leakage).

**Caveats:** single seed; **bpb ≠ CORE** (the real capability target, unmeasured here — needs the surface-aware eval path); λ=0.5 untuned; +0.15% params (122 vocab + 15,364 surface). **Read:** a strong, controlled bpb lead over the *adopted* force_merges baseline — clears the "worthwhile lead" bar for the GPU A/B, pending the surface-aware CORE path. Artifacts: `results/d6_surface_factoring/` (`train.log`, `bpb_trajectory.txt`, report, `meta.json`) + checkpoint at `~/.cache/nanochat-variants/surface_triple/base_checkpoints/`.

## Scripts

The reproducible offline harness. All run with `uv run python -m tools.<name>` from the sandbox dir, no GPU — **but they need the ClimbMix data cache** (each calls nanochat's `parquets_iter_batched`): `uv run python -m nanochat.dataset -n 8` downloads ~8 train shards + the always-pinned val shard (the 2B-char train budget needs ~8; see [`../../SETUP.md`](../../SETUP.md)). They then train rustbpe tokenizers (~35–75s each at 2B train chars; the bootstrap makes `rustbpe`, `rustbpe_force_merges`, `nanochat.*` importable). **Pass `--train-chars 2000000000` explicitly** — `space_factor_compression.py` defaults to **500M** (the case/stack scripts default to 2B; `digitfix` hardcodes 2B); running script 1 bare with `--save-dir` would save 500M-trained reference tokenizers and silently skew every downstream step that loads them. Saved reference tokenizers live under `~/.cache/nanochat-variants/{baseline_repro,spaceless_repro}/tokenizer` (created by script 1's `--save-dir`); `baseline_repro` has a **byte-identical merge table / vocab / split-pattern** to the cached canonical (equal `_mergeable_ranks` + `_pat_str`; only `.pkl` metadata differs).

| script | what it does | key invocation |
|---|---|---|
| `tools/space_factor_compression.py` | Space A/B: trains baseline-pattern + spaceless-pattern tokenizers on identical data, measures compression, the conditional space-bit cost, and a fast-vs-slow bit-count validation. **Run this first** (saves the reference tokenizers). | `uv run python -m tools.space_factor_compression --train-chars 2000000000 --val-chars 20000000 --validate --compare-cached --save-dir ~/.cache/nanochat-variants` |
| `tools/digitfix_remeasure.py` | Isolates the `space_digits` digit-fix from the idea-specific space gain (baseline → digit-fixed baseline → spaceless). Reproduces +0.87% digit-fix, +2.23% idea-specific (token-count framing; the script's own docstring quotes the equivalent bytes/token framing +0.96% / +3.18%). **Requires the saved reference tokenizers from script 1.** | `uv run python -m tools.digitfix_remeasure` |
| `tools/case_factor_analysis.py` | Cheap potential analysis for capitalization: vocab case-duplication, space×case families, and the cap-bit conditional entropy + sentence-start/proper-noun split. No training. | `uv run python -m tools.case_factor_analysis --val-chars 15000000` |
| `tools/case_factor_compression.py` | Case + joint space+case compression (titlecase fold), with the lossless space+cap round-trip. Defines `fold_case`. | `uv run python -m tools.case_factor_compression --train-chars 2000000000 --val-chars 20000000` |
| `tools/stack_fm_spaceless.py` | force_merges + space stack. `build_config(pairs)` constructs the spaceless+carve-out pattern and the leading-space-stripped forced pairs. `--all-pairs` = full 122; default = 69 word-phrases. Reuses the `rustbpe_force_merges` crate. | `uv run python -m tools.stack_fm_spaceless --all-pairs --train-chars 2000000000 --val-chars 20000000` |
| `tools/stack_triple.py` | The full triple: force_merges(122) + space + case. Folds the corpus and the phrases, handles per-word-position caps in the round-trip. Produces the +11.11% headline. | `uv run python -m tools.stack_triple --train-chars 2000000000 --val-chars 20000000` |

Reusable building blocks worth knowing when extending: `space_factor_compression.SPACELESS_PATTERN` and `corpus_byte_and_bit_stats`; `case_factor_compression.fold_case`; `stack_fm_spaceless.build_config` (turns a forced-pair list into pattern + transformed pairs + vocab); the force_merges crate API `rustbpe_force_merges.Tokenizer().train_from_iterator(iterator, vocab_size, buffer_size=8192, pattern=None, blocked_pairs=None, forced_pairs=None, block_trailing_space=False, block_leading_space=False)` (see `rustbpe_variants/force_merges/src/lib.rs`) — note the positional is `vocab_size`, `block_*_space` default **False** (pass `True` explicitly for the phrase-internal-space handling), and `blocked_pairs=` is what the [Fast-follow](#fast-follow) block-list work uses. The forced-pair transform for spaceless/case: strip ONE leading space from the LEFT operand only (`(' of',' the')`→`('of',' the')`), which preserves `strip_lead(a)+b == strip_lead(a+b)` for word AND punct pairs; for the triple also `fold_case` both operands.

To extend to a new surface feature: add a vocab-duplication + bit-entropy analysis (like `case_factor_analysis.py`), then a fold/strip + compression A/B (like `case_factor_compression.py`), confirm the 4 factorability criteria, and verify lossless round-trip before believing any compression number.

## Architecture

(Model side — not yet built.) The model predicts, per position, head-1 = the base content token (as today) and a **surface head** = the surface bits for that token. Two decisions, both grounded in the offline findings:

- **Two binary surface heads (a space head + a per-word-start cap head), NOT a flat categorical.** A "space × case = 4-class softmax" is only well-defined for single-word tokens; a phrase token spans N words and needs N cap bits — a variable output dimension — so the softmax is undefined for it. Use a **space head** (binary) and a **cap head** (binary) applied at each word-start the token covers. (A *joint* 4-class head would have bought ~nothing on the loss anyway: space↔case co-firing at sentence starts is mostly *explained by context*, so the residual `I(space; case | context, token)` is small — the reason to split into two binary heads is the phrase-token output shape, not an MI argument.)
- **Both cap *values* AND cap *slots* (the mask) are per-OCCURRENCE — occurrence-active masks.** (This supersedes an earlier "fixed token-local mask" idea, which is ASCII-only: once a token can byte-split a multi-byte letter — or the same id appears word-start vs mid-word — the owned word-starts are context-dependent.) A token **owns** the word-starts whose first byte falls in its span; a mid-word continuation owns NONE (`mask=0`). The encoder fills `cap_mask` + `cap_bits` per occurrence from the **full folded text's** char-level word-starts; **decode operates on the reconstructed string** — proper unicode `.upper()` of the word-start char, NOT per-token byte ops — so multi-byte folds (`Łódź`→`łódź`, `Ελληνικά`) and byte-split tokens round-trip (validated in the smoke). The cost: at **generation** the mask must be derived from running context (the word-starts the emitted token owns given the decoded prefix), which a fixed mask would have avoided — a deliberate trade for unicode correctness. (`wrappers/_surface_ref.py`'s `_char_word_starts`/`surface_encode`/`surface_decode`, and `tools/stack_triple.py`'s `encode_triple`, are the reference impls — the latter unicode-validated 2000/2000.)
- **The implementable tensor contract.** Per position the surface target is `space_bit` (1) plus `cap_bits[K_max]` + `cap_mask[K_max]`, `K_max` = max word-starts a single token can own (tokenizer/corpus-derived; ≤3 for the 122-phrase set, computed once via the encoder, e.g. `compute_kmax`). The encoder fills both `cap_mask` (occurrence-active) and `cap_bits` per occurrence. **Input embedding** composes them: `wte[base] + space_emb[space_bit] + Σ_{valid slots i} cap_emb[i, cap_bits[i]]` — the cap embedding is **slot-specific** (`cap_emb[slot, bit]`, e.g. `Embedding(2·K_max, d)` indexed by `slot·2 + bit`), NOT a single shared 0/1 embedding. A shared-embedding masked *sum* is **not injective** over cap patterns (`[1,0]` and `[0,1]` collapse, erasing *which* word of a phrase was capitalized and breaking "same information as baseline" for asymmetric-cap phrase tokens like `Of the` vs `of The`); slot-specific embeddings keep it injective. This is the concrete form of the `surface_emb[surface_x]` shorthand used in [Training integration](#training-integration). **Loss** is `BCE(space) + masked-BCE(cap over valid slots)`. Both heads are token-conditioned (next bullet). See the [0.2% multi-cap finding](#rendering-losslessness).
- **Condition the surface head on head-1's chosen token (sequential), not parallel — this is the key efficiency decision.** A naïve parallel head (predict surface bits from the same hidden state as the token, independently) pays `I(word; feature | context)` — the word↔surface correlation baseline's joint token captures for free — measured at ~0.03–0.07 bits/byte net penalty in the space adversarial verification. Conditioning the surface head on the emitted base token recovers `H(feature | word, context)`, and the gain **differs by feature**: **case** becomes ~free (the word near-determines its capital — `london` is lowercase, `London` was the folded one), while **space** drops substantially but *not* to zero (the word reveals whether it's a word-start vs a mid-word continuation, but the leading space is still partly context-determined — empirical residual ~0.03 bits/byte honest, down from the ~0.073 context-only figure). So sequential conditioning makes case ~free and roughly halves the space cost, but a residual space cost is irreducible — and it's one both baseline and the factored model pay (baseline inside head-1's `" the"`/`"the"` choice, the factored model in the surface head), so it's relocated, not net-new. Sharpest finding from the adversarial verification.

## Training integration

How the tokenizer is used in the training loop (the concrete shape of the "harness change"). At training time the only tokenizer operation is **encode** — decode/rendering is generation-only — and the encode is deterministic from text, so training is fully teacher-forced and parallel with no model-in-the-loop and no autoregressive feedback.

**The shape, per position (the whole change in one sentence).** Every position in the context window holds a pair `(base_token_id, surface_label)`, and the model both *sees* and *predicts* that pair. It is **fed both as input** — the surface label is added to the token embedding, so the trunk sees `(token, surface)` at every position, not just at the output — and this is what keeps the model information-equivalent to baseline (baseline's `" The"` token already carries "new word, capitalized" into the trunk; here that same signal arrives via the surface embedding instead of being baked into the token id). And it **predicts the next position's pair**: head-1 → the next base token id (as today), the surface head → the next surface label. So the entire architectural change from a standard next-token LM is: one extra input embedding and one extra small output head, both riding alongside the token at every position. The rest of this section is the mechanical detail of that.

**Data flow.** nanochat's dataloader (`tokenizing_distributed_data_loader_with_state_bos_bestfit`) today calls `tokenizer.encode(doc_batch, prepend=bos)` → one token-id list per doc, best-fit-packs into rows of length `T+1`, slices `inputs = row[:-1]`, `targets = row[1:]`, and the step is `loss = model(x, y)`. The surface tokenizer's encode instead emits **two per-token-aligned streams**: the base content id and the surface label `(space_bit, cap_pattern)`. The dataloader packs both with the same best-fit packing/BOS/shift (they're aligned, so this is mechanical), yielding `base_x, base_y, surface_x, surface_y` — four tensors where `surface_y[t]` is the *next* token's surface.

**Model consumption (teacher-forced, parallel).**
- **Input embedding = `wte[base_x] + surface_emb[surface_x]`.** Load-bearing: the trunk must *see* the surface form, or it loses the sentence-boundary signal baseline's combined `" The"` token carries for free. With it added, the trunk has the same *information* as baseline (the surface is recoverable) — though composed **additively** (`" The"` and `"the"` are forced to differ by exactly `surface_emb`) rather than as baseline's free, independent learned vector per surface variant. Only the *output* is split. **Value-embedding caveat:** nanochat *also* injects per-layer value embeddings keyed only by token id (`value_embeds[i](idx)`, `nanochat/gpt.py`). Under factoring these key on the *base* id, so surface variants **share** value embeddings (baseline's `" The"`/`"the"` have distinct ones) — the value pathway is surface-blind, a *partial* loss of info-equivalence beyond the main `wte` pathway. To fully match baseline, condition the value embeddings on surface too (e.g. add a surface term), or accept the value pathway doesn't see surface.
- **Heads off `h_t`:** head-1 (content) predicts `base_y[t]` via the existing `lm_head`; the **surface heads** (a space head + a per-word-start cap head, per [Architecture](#architecture) — `surface_x`/`surface_y` are the structured `(space_bit, cap_bits[K_max], cap_mask[K_max])` label, not a scalar) predict `surface_y[t]`'s space bit and masked cap bits, conditioned on the next base token (`base_y[t]`, teacher-forced) — `surface_head(h_t, wte[base_y[t]])` — which is what makes case near-free.
- **Loss — TRAINING and EVAL are different.** *Training* objective (mean reduction): `content CE + λ·(space BCE + cap BCE)`, where λ is a gradient-balance hyperparameter. *Eval* — the `model(x, y, loss_reduction='none')` path that `nanochat/loss_eval.py:33` calls for bpb (and the CORE likelihood path) — must return the **unweighted joint** per-token NLL `content NLL + space NLL + cap NLL` (**no λ** — λ is a training knob, not bits), with **BOS/special/ignored targets masked for BOTH** the content and surface terms. Then bpb needs both ends fixed: **numerator** = both heads' bits (a one-head bpb undercounts — the MTP/zloss eval-path lesson); **byte denominator** = `token_bytes[base_y] + space_bit_y` (the base token's bytes OMIT the factored leading space — 1 byte per set space bit; the case fold is byte-preserving), else the denominator is short ~one byte per space-preceded token and bpb is inflated.

The predicted bits feed back only at *generation* (emit base token → predict its surface bits → render → feed both into the next step). Training stays exactly as parallel as standard nanochat; the deltas are: a surface embedding on the input, a small surface head, the extra loss term, and a dual-stream dataloader. This is the "harness change" (copied `base_train.py`) listed in [Direction](#direction).

**Throughput — measured, NOT a bottleneck (no Rust needed).** The surface encode (case-fold + bit extraction + cap-label mapping) is heavier than plain tiktoken, but plain tiktoken is so fast that even the dragged version far exceeds the dataloader requirement. Benchmark (`tools/bench_surface_encode.py`, 10M val chars; tok/sec are machine/load-dependent — clean re-runs vary ~6%, CPU contention with a concurrent train drops headroom to ~50× — so these are ~1–2 sig figs, not bit-reproducible):

| encoder | tok/sec | vs plain | vs per-rank req (~36k) |
|---|---:|---:|---:|
| plain tiktoken (today) | ~12M | 1× | ~330× |
| **surface-opt** (production Python) | ~2.5M | ~5× slower | **~70×** |
| surface-naive (`encode_triple` reference) | ~0.7M | ~18× slower | ~19× |

The dataloader must sustain ~291k tok/sec total = **36,400 tok/sec/rank** (each rank tokenizes its 1/8 shard, prefetched behind the ~1.8s step — this requirement *is* exact). Surface-opt does ~2.5M tok/sec — **~70× over requirement**, hiding entirely behind the prefetch (~16k-token micro-batch encodes in ~6ms vs a ~100ms+ GPU micro-step). The dominant cost is single-threaded Python (`fold_case` + the bit/cap walk, ~80% of the time ≈ ~3M tok/sec isolated, still ~80× the requirement). Benchmarked on `spaceless_repro` + `fold_case` as a stand-in for the triple (encode is ~vocab-independent and the cap walk is O(tokens), so it represents the triple's cost). **Conclusion: Rust-ification is unnecessary** — it would only matter if training throughput rose ~50× or on a single CPU-bound rank.

- **Use the O(tokens) *shape* of `opt_surface_encode`, but with `encode_triple`'s validated space rule.** `stack_triple.py`'s `encode_triple` (per-pretoken `encode_ordinary` + per-token `decode_single_token_bytes`) is the 18× one — lossless but do NOT put it in the hot path. `bench_surface_encode.py`'s `opt_surface_encode` is the fast O(tokens) shape (one `encode_ordinary_batch`, then a single pass mapping cap byte-offsets to tokens via a precomputed `id→bytelen` table) — **but as written it is a *timing* stand-in, not lossless**: it drops *every* single-space token, which over-collapses multi-space runs that didn't merge and loses trailing spaces. The production encoder must keep the O(tokens) shape but absorb a space token as a bit **only when the next token starts non-whitespace content** (the validated rule — `encode_triple` / the ` (?=\S)` bit-count in `space_factor_compression.py`). Same complexity, so the throughput verdict stands; just don't ship `opt_surface_encode` verbatim.

**Gotchas.** The surface label for a phrase token is a per-word cap *pattern*, not one bit (the [0.2% multi-cap finding](#rendering-losslessness)), so `surface_y` carries a small variable label there and the head predicts it. Surface bits must stay aligned with base tokens through the best-fit packer and the BOS prepend (BOS gets a null surface label) — a place to get an off-by-one.

### Pre-A/B validation (smoke ladder — rungs 0–4 done)

The model-side contract above is validated by two self-contained smokes that prove it's implementable and the pieces fit, **before** building the production overlay/dataloader/harness or running `runcpu.sh` (rung 5). They use a tiny reference impl (`wrappers/_surface_ref.py`: a ~320-vocab spaceless+case-folded+2-phrase tokenizer trained inline, the slot-based encoder/decoder, and a tiny `MiniSurfaceGPT`), run in seconds with no data/GPU, and follow the repo's `smoke_<idea>.py` convention.

`wrappers/smoke_surface_factoring.py` (rungs 0–3, **12/12 PASS**):

| rung | check | validates |
|---|---|---|
| 0a | encode→decode round-trip lossless incl. unicode (`Łódź`, `Ελληνικά`) | the dual-stream contract is lossless + unicode-safe |
| 0b | Σ base_bytes + Σ space_bit == original bytes | the bpb byte-denominator fix (review P1.2) |
| 0c | `{of the, Of The, Of the, of The}` → `{00,11,10,01}` cap values; 1 word = 1 owned slot, continuations own 0 | per-occurrence cap values + occurrence-active mask |
| 1a | loss == content CE + λ(space BCE + masked cap BCE) | loss decomposition |
| 1b | flipping masked-out cap targets leaves cap loss unchanged | masking actually masks |
| 1c | `wte+pos+space_emb+Σ slot-specific masked cap_emb`; zeros → `wte+pos` exactly | input-embedding composition |
| 1c2 | `Of the` (`[1,0]`) ≠ `of The` (`[0,1]`) in the trunk input | slot-specific cap embedding is injective (review P1) |
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

**Retired vs. remaining.** The *correctness / "do the pieces fit"* risk is retired — both at the contract level (rungs 0–4 on the tiny reference) and now at the **real-integration** level: `overlay/surface_factoring.py` is built and bit-identical to vanilla GPT with surface off, and the `overlay/surface_codec.py` + `overlay/surface_tokenizer.py` dual-stream tokenizer round-trips losslessly (incl. unicode). **Built:** the overlay (surface-conditioned value embeds + heads + train/eval loss), the production codec/tokenizer class, and all three smoke suites (21 checks). **Remaining before the A/B:** (i) persist the production-vocab (32890) triple tokenizer artifact so `SurfaceTokenizer.from_directory` can load it; (ii) the **harness** — a copied `base_train.py` with a dual-stream dataloader + the joint loss, a surface-aware `evaluate_bpb`/CORE, and surface-aware `base_eval`/`chat_*` wrappers (these still build vanilla GPT + strict-load); (iii) **rung 5** `runcpu.sh` (d6 mechanism check); then (iv) the **Lambda A/B** (quality). One known generation cost stands: occurrence-active masks must be derived from running context at generation time (the eval smoke approximates the *input* mask from intrinsic word-starts; decode recomputes ownership exactly, so correctness holds).

## Caveats

- **The token reduction doesn't *mechanically* lower bpb — whether it lowers bpb in the fixed-compute regime is the A/B question.** The +11.11% is a *sequence-length / sample-efficiency* gain (fewer tokens per byte → more text per fixed token budget). bpb is normalized per *byte*, so unlike bits-per-*token* it doesn't drop just because there are fewer tokens; in the *infinite-capacity* limit it's invariant to a lossless re-tokenization. But under this project's *fixed-compute* constraint bpb is NOT invariant — a denser, cleaner tokenization can be easier for a fixed-capacity model to predict (that finite-capacity effect IS the sample-efficiency bet, and is why force_merges' compression produced a CORE win). Once both heads are counted everything is one currency (bits/byte): the open question is whether the lower *content* bpb from denser training outweighs the surface head's *added* bpb (~0.1 bits/byte gross for space+case, less after sequential conditioning). That net is the A/B question.
- **Training-time bpb is now downstream-validated; CORE is still UNVERIFIED.** The d6 runcpu A/B ([Downstream](#downstream-runcpu-d6--the-first-model-ab)) shows the lowest bpb of any d6 run (−0.0276 vs baseline, −0.0212 vs the adopted force_merges), on a verified same-byte-basis at the same commit/config. But that is a *prediction/compression* metric; **CORE — the capability target — remains unmeasured** (disabled in runcpu, and the surface-aware `core_eval`/`base_eval` path isn't built). force_merges' bpb edge did convert to CORE +5.81%, so the bpb lead is encouraging, but whether surface factoring adds CORE on top is still the open question the GPU A/B must answer.
- **Gains are corpus-mix-dependent.** Measured on ClimbMix (prose-heavy). Code-heavy or more-prose corpora would shift the numbers. The [multi-space extension](#dead-ends) is useless here precisely because ClimbMix has ~100% single-space gaps.

### Dead ends

- **Multi-space "count" bit** (extend the space bit to encode N consecutive spaces): +0.003% on ClimbMix — useless, because ~100% of leading-space gaps are a single space. Only helps indentation-heavy raw code, which ClimbMix is not. Don't pursue unless the target corpus is source code (and even then it can't encode tabs).

## Direction

The decisive next experiment is a **model A/B**: `force_merges` vs `force_merges + surface-factoring` (the +11.11% triple tokenizer) with the dual-head model, at d24 on a Lambda speedrun, measuring CORE / val_bpb / ChatCORE. This is what settles whether the compression converts to capability. It is a real build, sequenced:

1. **Production tokenizer wrapper — ✅ BUILT.** `overlay/surface_codec.py` (the unicode-safe dual-stream `surface_encode`/`surface_decode`/`compute_kmax`, tokenizer-agnostic) + `overlay/surface_tokenizer.py` (`SurfaceTokenizer`: `from_directory` load, `encode`/`decode`, `byte_count` for the bpb denominator). Regression-tested in `wrappers/smoke_surface_factoring.py` (`check_surface_tokenizer`). *Remaining for this step:* train + SAVE the production-vocab (32890) triple tokenizer artifact (via `tools/stack_triple.py`'s build) so `from_directory` has something to load — the class is done, the artifact isn't persisted yet.
2. **Model overlay** (`overlay/surface_factoring.py`) — **✅ BUILT + smoke-validated** (`wrappers/smoke_surface_factoring_overlay.py`, 5 checks: param accounting, forward/backward, surface-value-embed wiring, **reduces-to-baseline bit-identical to vanilla GPT (max|Δ|=0.0)**, and the train-vs-eval loss contract). `surface_emb` + slot-specific `cap_emb` on the input, **surface-conditioned per-layer value embeddings** (the chosen full-info-equivalence path), and the space + per-word-start cap heads conditioned on the next base token. Overrides `num_scaling_params`/`setup_optimizer`/`estimate_flops`/`init_weights`. Reads `SURFACE_KMAX`/`SURFACE_LAMBDA` from env (instance attrs, not GPTConfig fields). Follow the overlay conventions in [`../../overlay/README.md`](../../overlay/README.md) (hyperparameters as instance attributes, not GPTConfig fields). **Smoke must verify the *joint* eval, NOT vanilla-GPT eval-path equivalence** — this overlay's eval path is deliberately different, so the standard "eval returns plain token CE" check (from zloss/MTP) does NOT apply. Smoke instead: `bpb = content CE + surface CE`, the byte-denominator fix (factored spaces added back), round-trip losslessness, and BOS/special targets carrying masked/null surface labels. **This overlay ADDS parameters** (`surface_emb`, the heads), so unlike the pure-loss overlays it must override the methods that `assert` exact coverage — `num_scaling_params` (`gpt.py:345`; `assert total == sum(p.numel() ...)` at `:364`) and `setup_optimizer` (`gpt.py:374`; the param-group-count assert) — plus `estimate_flops` (`gpt.py:317`) for FLOP accounting, `init_weights`, and the optimizer LR groups. It's closer to an architecture-subclass overlay than a pure-loss one. **Checkpoint/load:** `surface_emb`/heads are sized by `K_max` (tokenizer-derived), and `checkpoint_manager.py` builds a vanilla `GPTConfig`/`GPT` from the saved config and **strict-loads** the state dict — so a surface checkpoint will NOT load into vanilla GPT (unexpected keys; `K_max` unknown). Persist surface metadata (`K_max`, λ, surface config) in the checkpoint meta — or derive it deterministically from the exact tokenizer artifact — and make the load path build the `SurfaceGPT`.
3. **Harness — ✅ training path BUILT (eval/generation path still TODO).** Instead of copying `base_train.py`, the harness keeps it pristine and patches around it: `wrappers/train_surface_factoring.py` swaps in `SurfaceFactoringGPT`, the surface dataloader, and `surface_evaluate_bpb`, then runpys upstream `base_train`. The four surface streams flow out-of-band through `overlay/surface_context.py` (the dataloader stashes the current batch; the patched `forward` reads it — safe because the upstream loop strictly alternates `next()`→`model()`). `overlay/surface_dataloader.py` best-fit-packs all four streams in lockstep and yields the base `(inputs, targets)` upstream expects; `surface_evaluate_bpb` fixes the byte denominator to `token_bytes[base_y] + space_y`. `wrappers/tok_train_surface.py` trains+saves the triple tokenizer (folded corpus + spaceless pattern + folded forced pairs) into the standard dir so `get_tokenizer`/`get_token_bytes` resolve unpatched. Smoke: `wrappers/smoke_surface_harness.py` (4 checks — packing+alignment, context handoff, overlay-consumes-batch, surface bpb) on synthetic docs; run it via `runs/runcpu_surface.sh` (the d6 mechanism check, real data). `K_max=3` is derived from the forced pairs (3-word phrases like "some of the" exist). **Still TODO** (the eval/generation half): `core_eval.py`'s `forward_model` is plain-token next-token CE, so CORE/ChatCORE need surface-aware prompt encoding + joint-likelihood scoring; the Engine streams token ids only, so generation needs surface-aware generate+decode; and `base_eval`/`chat_sft`/`chat_cli` build vanilla GPT + strict-load, so each needs a surface-aware variant (build `SurfaceFactoringGPT` from persisted `K_max`). `runcpu_surface.sh` skips `base_eval` for exactly this reason. **Crucially, the A/B metrics also need surface-aware eval + generation**, which the doc's training focus undersold: `core_eval.py`'s `forward_model` computes next-token CE over plain token ids, so CORE/ChatCORE need surface-aware prompt encoding + joint-likelihood scoring across both heads; and the Engine streams token ids only, so free-form generation / ChatCORE need surface-aware generate + decode. CORE/ChatCORE aren't meaningful until those paths are surface-aware — budget for them, not just train + val-bpb. The standard downstream wrappers (`base_eval`, `chat_sft`, `chat_cli`) likewise build vanilla GPT + strict-load + tokenize plain ids, so each needs a surface-aware variant (build `SurfaceGPT` from the persisted metadata; surface-aware encode/score/generate/decode).
4. **A/B** — same pinned nanochat commit + seed; baseline = force_merges (already adopted), variant = force_merges + surface. Do NOT speedrun until the overlay + harness pass smoke and the lead is worth the GPU cost (see the [no-premature-speedrun discipline](../../STATUS.md)).

Decisions settled (2026-06): **value_embeds = surface-conditioned from the start** (full info-equivalence — built); **phrase set = full triple (fm122 + space + case, +11.11%)** is the A/B target. Still open: whether to *also* run a space-only or space+case ablation to isolate each feature's contribution; and the production-tokenizer artifact + the harness (steps 1-tail and 3) before runcpu.

## Fast follow

Cheap, offline, no-GPU refinements to run on the surface-factored tokenizer **before** committing to the model-A/B build. These are token-*quality* follow-ups (junk-merge removal, whole-word coverage) that complement the compression results above — token quality (real whole words, fewer dead/junk tokens) plausibly matters as much downstream as raw bytes/token, and these all reuse the existing `auto_tune` held-out-block audit machinery. None of these are done yet.

1. **Re-check the `(',00')`-style ghost merges under surface factoring.** The `force_merges` numeric DERIVED pairs (`('00','0')`, `(',','000')`) cause `(',','00')` to *form* as a token but it never *fires* — pure junk. The `heldout_fixpoint_fm` audit (`rustbpe_variants/auto_tune/audit/heldout_fixpoint_fm/`, lineage round 51 / ledger `','+'00'`) found that **blocking** it ejected the dead `,00` token and surfaced real words (`appear`, `clear`, `pear`, `pearance`) for **~505 fewer tokens on held-out shards** (net cov 0 / comp −505.5 / dead −1.8) — small but free. Digits don't case-fold and the numeric pairs carry into the triple unchanged, so `(',00')` almost certainly still forms in the surface tokenizer. **Verify:** dump the triple vocab (`tools/dump_vocab.py`) and grep for `,00` / check its firing count; if it's still a dead/non-firing token, block it (it's already in the force_merges_combined 126-block set — confirm it survives the surface-context fold). Likely a free win that carries straight over.

2. **Run a `--propose` pass against the surface-factored tokenizer.** `heldout_fixpoint_fm`'s firing-percentile-33 mining + held-out-Pareto gate was only a *marginal* compression lever on its own, but it is **excellent at surfacing unexpected junk merges** — it ranks blocks that yield whole words, improve compression, or reduce dead tokens, the exact mechanism that caught `(',00')`. Run the orchestrated `--propose` loop against a **surface context** (the analog of `--fm-context`, but with `SPACELESS_PATTERN` + the case fold + the folded forced pairs from `tools/stack_triple.py`). **Requires extending the driver:** `tools/heldout_fixpoint_cached.py` currently has `--fm-context` (force_merges pattern + 122 pairs + block predicates at vocab 32890); add a `--surface-context` that swaps in the spaceless+folded-phrase pattern and case-folded corpus (reuse `stack_fm_spaceless.build_config` + `case_factor_compression.fold_case`). Then drive it per the audit README (`rustbpe_variants/auto_tune/audit/heldout_fixpoint_fm/README.md`): orchestrated `--propose` → hand-judge each round by the whole-word/realness criteria → `--commit '["L","R"]'`, with the **hard semantic veto** (never block a legitimate token — `ob+by` shatters hobby/lobby). **Never** run the bare autonomous loop — it auto-commits junk. Goal: a surface-specific block list, the way `force_merges_combined` is the force_merges-specific one.

3. **Consider blocking punctuation-prefixed word tokens at the regex level.** The baseline (and current spaceless) word alternative captures one leading punct char — `[^\r\n\p{L}\p{N} ]?+\p{L}+` — producing tokens like `.com`, `(the`, `"word`. A prior (informal) experiment found a *few* are frequent and useful (`.com`) but *most* are junk, and the aggregate frequency is likely too low to justify a dedicated surface feature (the way space/case earn their bit). The cheap alternative is a regex change: drop the leading-punct capture (`[^\r\n\p{L}\p{N} ]?+\p{L}+` → `\p{L}+`) so the punct splits off as its own token and these tokens never form. There is a small compression cost (the frequent `.com`-class tokens become two tokens), but it may be worth it for **token quality** (fewer junk tokens, lower dead-token count, cleaner word atoms). **Evaluate:** measure the bytes/token cost and the dead-token / whole-word-coverage change of the `\p{L}+`-word variant vs the current spaceless word pattern (a one-line edit to the word-alt substring `[^\r\n\p{L}\p{N} ]?+\p{L}+` in `space_factor_compression.py`'s `SPACELESS_PATTERN` and `stack_fm_spaceless.py`'s `SPACELESS_BODY` — `stack_triple.py` pulls its pattern from the latter via `build_config`).

4. **Compare whole-word token coverage, baseline vs surface.** The audit's primary token-quality metric is whole-word coverage — how many vocab tokens are real dictionary whole words with standalone usage (`tools/pass_metrics.py` `whole_word_counts`, `tools/word_realness.py`, `tools/blocked_pairs_report.py` `_is_word`). Surface factoring pools `The`+`the` / ` of the` and frees ~24% of vocab, so the freed slots plausibly form *more* complete-word tokens (denser merges) — open question whether the difference is notable. **Caveat that needs handling first:** `whole_word_counts` keys on **space-prefixed** dict words (` word`), which the surface tokenizer doesn't have (spaces are factored out, and titlecase is folded). Adapt the metric to count **base-form** whole words (no leading space, case-folded comparison) for an apples-to-apples baseline-vs-surface count, then report the delta alongside the bytes/token ladder.

## References

- Related: [`../tokenizer-variants/README.md`](../tokenizer-variants/README.md) (force_merges, auto_tune — the phrase-merging partner this stacks with), [`../../STATUS.md`](../../STATUS.md) (force_merges downstream result, tokenizer-eval workflow), [`../../overlay/README.md`](../../overlay/README.md) (overlay implementation lessons).
- force_merges crate: `rustbpe_variants/force_merges/` (`pairs.py` for the 122 forced phrases + the `space_digits` regex note; `src/lib.rs` for the `block_leading/trailing_space` + forced-merge training).
