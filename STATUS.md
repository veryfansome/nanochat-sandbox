# Status

Working state of the project — implementation progress, current focus, sequencing, next concrete steps. **Not auto-loaded** as agent context; read explicitly when you need to know "where are we right now?". Update this file as work lands.

## Implementation

| Idea | Designed | Implemented | Smoke ✓ | Real-data ✓ |
|------|:--------:|:-----------:|:-------:|:-----------:|
| [z-loss](ideas/zloss/README.md) | ✓ | ✓ (naive default; fused autograd opt-in via `ZLOSS_FUSED=1`) | ✓ (a/b/c/d/e) | **✓ d24/8xA100 — real positive: CORE +5.15%, val/bpb cost ≈ 0 (artifact-corrected)** |
| [MTP](ideas/mtp/README.md) | ✓ | ✓ (naive only; shared unembedding; k=3 default) | ✓ (a/b/c) | **✓ d24/8xA100 — net negative: val/bpb +8.78%, CORE "win" one-task; deprioritized** |
| [Deep supervision](ideas/deep-supervision/README.md) | ✓ | — | — | — |
| [Token-level loss weighting](ideas/token-loss-weighting/README.md) | ✓ | — | — | — |
| [Differential attention](ideas/diff-attention/README.md) | ✓ | — | — | — |
| [Online data selection + batch-size tuning](ideas/online-data-selection/README.md) | ✓ | — | — | — |
| [Adaptive sequence length / batch size](ideas/adaptive-schedule/README.md) | ✓ | — | — | — |
| [Layer-wise LR / staged maturation](ideas/layerwise-lr/README.md) | ✓ | — | — | — |
| [Tokenizer variants](ideas/tokenizer-variants/README.md) | ✓ | partial — `tools/eval_tokenizer.py` harness done; pristine `sandbox/rustbpe/` vendored; three variants ported (`space_digits` [Py-only], `force_merges` [Rust], `seed_tokens` [Rust]) | ✓ all three (smoke_tok_train_{space_digits,force_merges,seed_tokens}) | — |
| [Non-backprop (DFA → block-local)](ideas/non-backprop/README.md) | ✓ (research track) | — | — | — |

## d24 trio results (8xA100 40GB, ~$337 wall-clock)

Sequential trio on a single Lambda instance; baseline included SFT (subsequently `USE_SFT` defaulted to 0 — workflow tweak, not a hardware constraint). All three share the same nanochat commit, seed, and hardware envvars (`USE_FP8=0 WINDOW_PATTERN=L`). MTP needed `DEVICE_BATCH_SIZE=4` (vs 8 for the others) to fit 40GB HBM with k=3 aux softmaxes.

| | baseline | zloss | mtp |
|---|---:|---:|---:|
| wandb | `qmsi105c` | `06731h2o` | `b35xnu0d` |
| `DEVICE_BATCH_SIZE` | 8 | 8 | **4** (k=3 softmaxes pushed B=8 to OOM) |
| val/bpb (final) | 0.71609 | 0.72009* | 0.77898 |
| CORE (`base_eval`, full) | 0.250488 | **0.263387** | 0.258635 |
| tok/sec | 291,210 | 288,483 | 260,517 |
| total time | 5h33m | 5h36m | 6h13m |

*zloss val/bpb is inflated by ~0.004 due to an eval-path bug (since fixed) — see below. mtp's batch-size asymmetry preserves total optimization signal (`base_train` holds total batch in tokens constant via grad-accum) but worth noting in any writeup.

### Headline verdicts

- **zloss is a real positive result and should be adopted.** CORE +5.15% with 16 wins / 5 losses / 1 tie across 22 tasks (one-sided sign test dropping the tie: p ≈ 0.013); val/bpb cost is ≈ 0 once the bug is corrected; throughput cost <1%. The CORE gain concentrates on robust high-N tasks (piqa +3.9pp on N=1838, commonsense_qa +3.4pp, lambada +1.7pp, coqa +2.0pp).

- **MTP at our k=3, α=0.3, shared-unembedding config is a net loss at d24.** val/bpb regression got *worse* at scale (+4.17% at d6 → +8.78% at d24), not better as Gloeckle et al. predict. CORE "+3.25%" is essentially one task (boolq +0.111; excluding it, mean Δ across 21 tasks is +0.003 — pure noise). Sign test p ≈ 0.19, not significant. Compute cost +12%. Three rescue paths exist (α=0.1 + k=1 single-aux-head, DeepSeek-MTP architecture, scale to d30+) but none are next-up.

### Eval-path bug in zloss (now fixed)

The original `ZLossGPT.forward` added the scalar z-loss term to the CE tensor on every code path, including `loss_reduction='none'` (the path `nanochat/loss_eval.py` uses for val/bpb). That inflated val/bpb by a constant offset of `z_coeff·E[lse²] / mean_bytes / ln 2` ≈ 0.004 — exactly the originally-observed regression. Fix: `forward` now branches on `loss_reduction != 'mean'`. **Smoke check `(e)` was added to enforce this contract** — the lesson was already documented in [`overlay/README.md`](overlay/README.md), but text-only lessons aren't enforcement. MTP got the contract right because it was written *after* the lesson landed; zloss missed it because the rule postdated its first implementation. The d24 CORE result for zloss is unaffected by this bug — CORE is computed by a separate pipeline.

### Earlier d6/M4 results (kept for reference)

- **`d6_baseline`** (wandb `qdoniwrj`) — `val/bpb` 1.16534. Pure baseline reference.
- **`d6_zloss`** (wandb `szesl4yg`) — `val/bpb` 1.17007 (within noise). tok/sec ~7% under baseline. CORE 0.0361.
- **`d6_zloss_fused`** (wandb `de562k89`, killed at ~300 steps) — Python fused-autograd path is **numerically correct** (bit-exact loss + ~2e-8 grad agreement vs naive) but **~32% slower than baseline** in pure PyTorch. Default flipped to naive; opt in via `ZLOSS_FUSED=1`.
- **`d6_mtp`** (wandb `bcv9920m`) — `val/bpb` 1.21398 (+4.17%). Showed the d24 regression's signature already at toy scale.

Compare any subset (CORE now shows both wandb mid-training subsample and authoritative `base_eval` from local `eval.csv`):

```bash
uv run python -m tools.compare_runs d24_baseline d24_zloss d24_mtp
```

## Suggested sequencing (updated post-d24)

1. **z-loss is settled — adopt as the default for future speedruns.** Single open question worth ablating: turn off the existing logit-softcap (`gpt.py:472`) with z-loss on; the hard cap may now be unnecessary. Cheap d24 ablation, ~$100.
2. **Deep supervision** — depth-axis complement to MTP; folds into the same `MTPGPT` subclass for one combined experiment. Token weighting's **RHO-Loss variant** can fold in here too (shares `forward` surface; needs a reference model that can be added once for both).
3. **Token-level loss weighting** (entropy / focal variants) — pure-loss overlay, no extra model, independent of MTPGPT line. Cheap parallel track.
4. **Differential attention** — independent architecture A/B.
5. **Adaptive sequence length (Part A of adaptive-schedule)** — harness change, but the model needs no edits and Part A is the rare idea that should make the speedrun **faster on wall-clock** at the same final loss. Cheap-ish entry into the harness-change tier.
6. **Online data selection + adaptive batch size (Part B of adaptive-schedule)** — bigger, harness-touching; pursue if earlier results justify it. Both share the copied-`base_train.py` pattern, so fold into a single forked harness. Online data selection is complementary to token weighting (sample-level vs. token-level signal shaping), so worth A/B-ing alone, weighting alone, and combined.

MTP rescue probes (α=0.1+k=1 first) only happen if a follow-up idea raises a specific question that an MTP variant can answer. Don't revisit MTP for its own sake at d24.

**[Tokenizer variants](ideas/tokenizer-variants/README.md)** is a **parallel track**. Infrastructure is now end-to-end ready: `tools/eval_tokenizer.py` harness validated; pristine rustbpe vendored at `sandbox/rustbpe/` (Karpathy@9467d83) with `runs/build_rustbpe.sh` for local builds; parallel-crate variants pattern in `sandbox/rustbpe_variants/`. **Three variants ported** from `veryfansome/nanochat` prior art:

- **`space_digits`** (Python-only) — adds optional leading space before the digit clause in `SPLIT_PATTERN`. Wrapper: `wrappers/tok_train_space_digits.py`. Smoke: `wrappers/smoke_tok_train_space_digits.py`. Uses PyPI rustbpe; no Rust build needed.
- **`force_merges`** (Rust) — forced cross-boundary common-phrase merges (` of the`, `, and`, etc.) via regex carve-out. Crate: `rustbpe_variants/force_merges/` (`rustbpe_force_merges` module); canonical pair list at `pairs.py` (87 mined FORCED + 15 mined DERIVED + 2 numeric DERIVED = 104, vocab=32768). Build: `VARIANT=force_merges bash runs/build_rustbpe.sh`. On a 15M-char climbmix sample: **−6.5% total tokens vs baseline** (3.27M → 3.06M), trading ~237 rare-content BPE merges (` Staphylococcus`, ` philosophers`, inflected forms) for ~237 high-frequency phrasal merges (` of the`, `, and the`, `. The`, ` can be`).

### How `pairs.py` was constructed (2026-05-25)

Replaced a ~70-pair hand-picked list (lifted from `veryfansome/nanochat@force_merges_wip`) with a data-driven list derived end-to-end from corpus statistics + Tier-1 ablation. Tooling preserved for re-curation; intermediate exploration variants (v0–v5) were consolidated away.

1. **Two-pass corpus mining** (`tools/mine_forced_pairs.py --two-pass`, 20 climbmix row groups, ~60M chars). Pass 1 mines cross-boundary bigrams from the baseline pre-tokenization with a closed-class grammatical-role filter (LHS = punctuation/preposition/conjunction/auxiliary; RHS = determiner/auxiliary/pronoun/conjunction/capitalized-sentence-starter). Pass 2 re-mines bigrams whose LHS is a pass-1 concatenation — recovers depth-2 phrases like `(" of the", " most")`, `(" can be", " a")` that single-pass can't see. Artifacts at `rustbpe_variants/force_merges/mined/forced_pairs_climbmix_20rg{,_twopass}.py`.

2. **Tier-1 sanity check** against the hand-picked baseline (79.6% pair coverage at top-150). Single-pass mining matches/beats hand-picked on English prose; loses `currency` by 1 token because numeric DERIVED pairs (`("00","0")`, `(",","000")`) fail the function-word filter. Added back as hand-additions.

3. **Kitchen-sink (all 22 DERIVED, "v2")** improves `modern_prose` (+1.8%) but regresses `english` (−2.1%, +1 token). Token-diff diagnosis: lost the `"izards"` BPE merge (used in "wizards") because the extra carve-outs consumed vocab budget.

4. **Per-pair DERIVED ablation** to identify the cutoff:
   - *Cumulative* (`tools/ablate_derived.py --k-values 0,5,10,15,20`): `modern_prose` jumps at K=5→10 (pair #6 helps); `english` regresses at K=15→20.
   - *Drill-down* (K=15..20): the regression appears in a single step (K=15→16); all later K transitions flat.
   - *Single-pair attribution* (`INCLUDE_INDICES`, K=15 + only one of pairs #17–20): **all four individually displace `izards`**. The regressor is "whichever pair you add 16th" — the threshold is the vocab budget, not a specific pair.

5. **Curation rule**: stop at the K where Tier-1 starts to regress. Final = 15 mined DERIVED + 2 numeric DERIVED. Pairs #16–20 dropped despite real corpus frequency (count=794–1,031 each in 60M sample); they're past what the 32K vocab budget accommodates cheaply.

6. **Vocab-cost bracketing** — would a small vocab bump rescue pairs #16–20?

   | variant                                | `english` | `izards` | `acclim` | `ampton` | `uran` |
   |----------------------------------------|:-------:|:-:|:-:|:-:|:-:|
   | Final (15 DERIVED, vocab=32768)        | 4.21 bpt | ✓ | ✓ | ✓ | ✓ |
   | +5 vocab, 20 DERIVED                   | **4.12 (regressed)** | ✗ | ✗ | ✗ | ✗ |
   | +10 vocab, 20 DERIVED                  | 4.21 (recovered) | ✓ | ✓ | ✓ | ✓ |
   | +20 vocab, 20 DERIVED                  | 4.21 | ✓ | ✓ | ✓ | ✓ |

   Each forced merge has an effective **~2× vocab cost**: 1 reservation slot for the carve-out token, plus ~1 BPE-priority shift since each carve-out also pushes its punctuation-less variant (` but the`, ` If the`, ` and other`, ` be a`) into the top-budget BPE merges. +5 vocab gives BPE 5 extra slots but needs ~10 to absorb 5 new derived pairs, so 4 borderline merges get displaced. Rule of thumb: **for N derived pairs, budget ~2N extra vocab**. The +10 candidate ("v5") beat the K=15 final by only 1,114 tokens out of 3.06M on real corpus (0.036%) — not worth the constraint deviation. Final stays at 32K.

### Per-pair DERIVED attribution (top-20 mined)

| rank | pair | verdict | evidence |
|---:|---|:-:|---|
| 1–5 | `(', and', ' the')`, `('. If', ' you')`, `('. It', ' is')`, `(', it', ' is')`, `('. This', ' is')` | KEEP | Tier-1 silent |
| **6** | **`(', but', ' it')`** | **KEEP** | +0.070 bpt `modern_prose` (` this`+`, but`+` it` → ` this`+`, but it`) |
| 7–15 | `(', which', ' is')`, `('. There', ' are')`, `(' of the', ' most')`, `('. In', ' the')`, `('. They', ' are')`, `(', and', ' it')`, `(', and', ' a')`, `('. In', ' this')`, `(' of the', ' following')` | KEEP | Tier-1 silent |
| **16–20** | **`(', but', ' the')`, `('. If', ' the')`, `(', and', ' other')`, `(' can be', ' a')`, `(' to be', ' a')`** | **DROP** | each individually displaces ` w`+`izards` → ` w`+`iz`+`ards` (−0.088 bpt english) |

### Reusable infrastructure

- `tools/mine_forced_pairs.py` — corpus miner; closed-class role filter, single-pass + `--two-pass`, shadow detection. Outputs `pairs.py`-compatible artifacts.
- `tools/ablate_mined.py` (env `MINED_TOP_K`) — cumulative ablation over the FORCED list.
- `tools/ablate_derived.py` (env `DERIVED_TOP_K`) — cumulative ablation over the DERIVED list.
- `wrappers/tok_train_force_merges_mined_{forced_subset,derived_subset,ablate}.py` + matching `pairs_mined_*.py` modules — FORCED / DERIVED / arbitrary-`INCLUDE_INDICES` subset trainers invoked by the ablation drivers.
- Per-K eval JSONs at `results/{mined,derived}_ablation/`.
- Mining artifacts at `rustbpe_variants/force_merges/mined/`.

To re-curate against a different corpus: regenerate the mining artifacts, re-run `ablate_derived` to find the new K cutoff, update `_KEEP_DERIVED_K` in `pairs.py`. The 2× rule still applies.
- **`seed_tokens`** (Rust) — morpheme-seeded BPE with constructive merge-chain builder. Crate: `rustbpe_variants/seed_tokens/` (`rustbpe_seed_tokens` module). Data: `rustbpe_variants/seed_tokens/seed_tokens.yaml` (~200 morphemes). Wrapper + smoke under `wrappers/`. Build: `VARIANT=seed_tokens bash runs/build_rustbpe.sh`.
- **`case_marker`** (designed only, not yet implemented) — lossless casing collapse via input preprocessing: cased words → `<|cap|>` / `<|allcaps|>` marker tokens + lowercased base. Pure-Python (~200-300 LOC); no Rust work. Orthogonal to all merge-producer variants. Catalog entry: `ideas/tokenizer-variants/README.md` variant 13. Decision-gated on force_merges d6 outcome.

`force_merges` was data-driven curated 2026-05-25 and the resulting tokenizer is currently in a Lambda speedrun (`d24_force_merges` vs `d24_baseline` `qmsi105c`). `space_digits` and `seed_tokens` smoke-pass at d6/local scale but haven't been trained on the full corpus yet — same eval workflow (`tools/eval_tokenizer.py` → speedrun A/B) applies.

The [non-backprop LLM](ideas/non-backprop/README.md) is a **separate research track**, not part of this capability-tuning sequence. Pursue independently.

## Next concrete steps

1. **Optional zloss val/bpb re-eval** — the d24 zloss run's val/bpb is inflated by ~0.004 due to the eval-path bug (since fixed). Either re-run for a citable number (~$80, one Lambda spin) or document the corrected value analytically (`0.71609 + noise`, matching baseline — derivable from the fix mechanics). The CORE win is unaffected; this is bookkeeping.

2. **Add smoke check (d) "eval-path equivalence" to MTP's smoke** — `mtp.py` honors the contract, but `wrappers/smoke_mtp.py` doesn't verify it. Copy the pattern from `wrappers/smoke_zloss.py` check (e). Cheap insurance against a future regression.

3. **Build the next overlay** — per sequencing, **deep supervision** (folds into the existing `MTPGPT` subclass, scaffolding already in place) or **token-level loss weighting** (entropy/focal — pure-loss overlay, no extra model). Deep supervision is the more natural next step; token weighting is a viable parallel cheap track if you want two signals in flight.

4. **Tokenizer-variants track** — `force_merges` is in a Lambda speedrun against `d24_baseline`. `space_digits` and `seed_tokens` are next: train each on the full corpus (no GPU), run `tools/eval_tokenizer.py` for Tier-1 comparison, pick survivors for a follow-up Lambda A/B.

5. After that: differential attention, then online data selection if the earlier results justify the harness-touching investment.
