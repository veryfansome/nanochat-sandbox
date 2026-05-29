# BPE-dropout (subword-segmentation regularization)

Status: **design — not started.** Candidate training-dynamics lever for sample efficiency under fixed scale & data. Emerged from the `blocked_morphemes` tokenizer exploration (see [`../tokenizer-variants/README.md`](../tokenizer-variants/README.md) + STATUS.md) as a cleaner attack on the same target: undertrained subword atoms in a small model. (`blocked_morphemes` itself was removed 2026-06-05; its block-the-mangled-merge mechanism continues as the `auto_tune` variant. The conceptual comparisons below still hold — read `blocked_morphemes` as "the block-list approach.")

Target repo: `nanochat` (Karpathy) — kept **pristine**, never edited.

## Goal

Make the model's subword atoms (especially the pieces inside rare/long tokens) better-trained, and make the model robust to segmentation, **without** changing model scale, training data, the tokenizer vocabulary, or inference-time compression.

## What it is

Standard BPE encoding is deterministic: apply the learned merges greedily by rank until none remain. **BPE-dropout** (Provilkov et al., ACL 2020) makes encoding *stochastic during model pretraining*: at each encode, every candidate merge is independently skipped with probability `p`. The same word therefore appears segmented many different ways across training examples:

```
"unrelated"  →  [unrelated]            (p effectively 0 for this occurrence)
             →  [un, related]          (top merge dropped once)
             →  [un, re, lated]        (dropped twice)
```

The vocabulary and the merge list are **unchanged** — only *which* merges get applied on a given occurrence is randomized. At inference, `p = 0` (standard deterministic BPE), so the model gets the efficient canonical segmentation. It is pure training-time regularization / data augmentation in segmentation space.

Three phases, to keep the terminology straight:
1. **Tokenizer training** (BPE merge-learning): unchanged, run once, frozen.
2. **Model pretraining**: dropout lives here — stochastic encode per occurrence.
3. **Inference / eval**: `p = 0`, deterministic.

Published result: +up to 3 BLEU on MT vs plain BPE, +0.9 vs unigram-LM subword regularization (Kudo 2018). The follow-on "curse of tokenization" work (arXiv 2406.11687) finds scaling mitigates tokenization brittleness but doesn't eliminate it, and names BPE-dropout as a working mitigation — which argues it should matter *more* at nanochat's small scale, where there's the least model capacity to compensate.

## Why it fits the project thesis

The constraint is "fixed scale & data; gains from objective/dynamics, not bigger/more." BPE-dropout extracts more signal (multiple segmentation views) from the *same* tokens of data — a sample-efficiency lever in training dynamics, adding no data and no parameters. It directly attacks the concern behind `blocked_morphemes`: rare tokens are undertrained because they fire rarely. Dropout trains their pieces by stochastically exposing them, **without** removing the efficient whole-token from the vocabulary.

### Relationship to `blocked_morphemes` — two attacks on one target

| | `blocked_morphemes` | BPE-dropout |
|---|---|---|
| What changes | the **vocabulary** (delete mangled tokens) | the **segmentation** (randomized at pretrain time) |
| Rare token `ished` | deleted; always `[ish, ed]` | kept; *sometimes* `[ish, ed]`, sometimes `[ished]` |
| Trains the common pieces | by forcing them (deterministic) | by stochastic exposure (regularization) |
| Inference compression cost | **+0.145% forever** (measured) | **zero** (inference = standard BPE) |
| Dead/undertrained tokens | *rises* (+0.45pp — planted stems sit idle) | should *fall* (pieces get extra firings) |
| Vocabulary | fought / changed | untouched |

They are **composable** (dropout on top of the blocked vocab) but worth testing independently first. BPE-dropout is the cleaner lever for the *atom-training* goal specifically, because it keeps inference compression and doesn't strand dead vocab slots.

## Why this is NOT the "regularization/dropout" the catalog ruled out

[`README.md`](../README.md) "Ruled out" lists "regularization/dropout (not in an overfitting regime)." That refers to **weight/activation dropout** as an *overfitting* cure — and nanochat's single-ish-pass pretraining isn't overfitting-bound, so that's correctly out of scope. BPE-dropout is a different mechanism with a different target: it is **segmentation-space data augmentation** aimed at (a) training subword atoms that are otherwise starved of gradient and (b) tokenization robustness/compositionality — not at reducing a train/val generalization gap. The "not overfitting" objection doesn't apply; the relevant question is sample efficiency of atom training, not overfitting control.

## Design / implementation in nanochat

**The enabling fact:** nanochat tokenizes **on-the-fly from raw-text parquet**, not from pre-tokenized shards. `nanochat/dataloader.py` (`refill_buffer`) calls `tokenizer.encode(doc_batch, prepend=bos, num_threads=...)` live during training (`dataloader.py:107`); the parquet files store the raw `text` column. So injecting dropout means supplying a dropout-capable `encode` — **no shard regeneration, no data-format change.** This is much cheaper than a typical pre-tokenized pipeline.

**Overlay shape (no copied harness):**
- Implement a dropout encode: pre-tokenize with the existing regex, then for each chunk run greedy rank-merge BUT skip each candidate merge with probability `p`. tiktoken's vocab `mergeable_ranks` *is* the ordered merge list (rank = merge order), so the standard BPE-dropout algorithm maps directly onto it.
- Monkeypatch `RustBPETokenizer.encode` (and/or the dataloader's encode call) before `runpy`-ing `scripts.base_train`, exactly like the model overlays patch `nanochat.gpt.GPT`. Read `p` from an env var (e.g. `BPE_DROPOUT_P`), store as an instance attribute (same attribute-not-config-field discipline as z-loss).

**The one wrinkle — train-split-only gating.** The same `tokenizer` object feeds the train loader, the val loader (val/bpb), and CORE prompt encoding. Dropout must apply to **train encoding only** — val/bpb and CORE must use deterministic `p = 0`, or the metrics are corrupted (a dropout-segmented val set is not comparable to baseline). Cleanest gating: patch the dataloader so `split == "train"` uses the dropout encode and everything else uses plain encode (monkeypatch `tokenizing_distributed_data_loader_with_state_bos_bestfit`, no file edit). This is the single piece that reaches past a pure `tokenizer.encode` patch.

**Throughput risk.** The baseline encode uses tiktoken's fast multi-threaded Rust path; a Python dropout-encode loses that and could become the dataloader bottleneck (encoding runs on CPU, prefetched concurrently with the GPU step at `base_train.py:518`, so *some* slowdown hides under GPU time at d20/8×A100, but a slow pure-Python encoder may not fully hide). Mitigations, in order of effort: (a) tight NumPy/Cython dropout-encode; (b) a `rustbpe` variant that implements dropout in Rust (mirrors `force_merges`/`blocked_morphemes` crate pattern); (c) bridge to HuggingFace `tokenizers`, which has native fast BPE-dropout (`dropout=`) — at the cost of a vocab-format conversion + dependency. Start with (a)/Python for the mechanism, measure the throughput hit, escalate only if it dominates.

## Expected cost

- **Inference**: zero — `p = 0` at eval/inference, identical to baseline tokenizer.
- **Training throughput**: TBD — depends on the encode implementation (see throughput risk). Measure tok/sec vs baseline in the smoke / d6 run before committing to a full speedrun.
- **Sample-efficiency caveat (the real cost).** At a fixed *token* budget, dropout makes training sequences longer (fragmented), so each step covers less *unique text*. For multi-epoch MT (where the result was proven) this is just the regularizer; for nanochat's mostly-single-pass pretraining it partly spends tokens on re-segmentations instead of new text. Whether the atom-training/regularization payoff beats the less-unique-text cost is **the** empirical question. The dropout rate `p` is the knob — low `p` (≈0.05–0.1) gives the regularization signal with little fragmentation overhead.

## Testing / A-B

- Smoke first: verify (a) dropout encode round-trips (decode∘encode = identity for any dropout sample), (b) `p=0` reproduces standard BPE exactly, (c) train-split gating — val/CORE encode is deterministic, (d) varied segmentations actually occur at `p>0`. No data, <5s.
- Same pinned nanochat commit + same seed as baseline.
- **Measure bits-per-byte (or per-char), NOT per-token loss.** Dropout changes fertility (tokens/word) during training; per-token CE can move for mechanical reasons. (The MorphBPE paper, arXiv 2502.00894, reports *per-token* CE gains alongside a fertility increase — a confound to avoid.) Also watch CORE, but treat per-token CORE deltas with the same caution and lean on bpb as the compression-invariant measure.
- d6/M4 only verifies the mechanism survives a real loop (per [`README.md`](../README.md) "Lessons learned"); real claims need a Lambda speedrun.

## Open questions

- Dropout rate `p` — sweep ≈ {0.05, 0.1, 0.2}. Higher `p` = more regularization but more fragmentation / less unique text per token budget.
- Single-pass payoff: does the regularization help when data isn't revisited? May want to pair with a small number of epochs, but that changes the data budget — keep within the fixed-data constraint.
- Interaction with nanochat's BOS-bestfit cropping (~35% tokens cropped at T=2048): dropout lengthens docs, shifting the crop distribution — check it doesn't interact badly with packing.
- Compose with `blocked_morphemes` vocab? Test independently first; revisit only if both show signal.
- English is the least-favorable language for morphology-aware tokenization (MorphBPE: English alignment-F1 gains far smaller than Hungarian/Arabic). BPE-dropout's benefit is robustness/regularization rather than morpheme alignment per se, so it's less language-gated — but temper effect-size expectations for an English corpus at small scale.

## References

- Provilkov, Emelianenko, Voita 2020 — *BPE-Dropout: Simple and Effective Subword Regularization*, ACL 2020 (arXiv 1910.13267). The method; +≤3 BLEU on MT.
- Kudo 2018 — *Subword Regularization*, the unigram-LM predecessor BPE-dropout improves on by ≈0.9 BLEU.
- *Tokenization Falling Short: On Subword Robustness in LLMs*, 2024 (arXiv 2406.11687). Curse of tokenization; scaling mitigates but doesn't eliminate; BPE-dropout as mitigation.
- MorphBPE, 2025 (arXiv 2502.00894) — referenced for the per-token-vs-bits-per-byte confound and the English-is-least-favorable caveat.
