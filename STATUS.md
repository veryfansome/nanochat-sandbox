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
| [Tokenizer variants](ideas/tokenizer-variants/README.md) | ✓ | `tools/eval_tokenizer.py` harness + pristine `sandbox/rustbpe/` vendored. **Two live variants, each its own crate:** `force_merges` (data-driven curated, adopted default) and `auto_tune` (`rustbpe_auto_tune`; blocking-only under the pristine regex + corpus-cached held-out-Pareto block-list audit — **fixpoint converged (r194, 129 blocked pairs); d24 A/B: CORE +5.59%, val/bpb tied (single seed, borderline)**). Earlier probes (`space_digits`, `seed_tokens`, `blocked_morphemes`, `manual_merges`, `unigram`) removed 2026-06-05 as dead directions — findings retained in the sections below + git history. | ✓ (smoke_tok_train_{force_merges,auto_tune}) | **✓ d24/8xA100 — `force_merges` real positive: CORE +5.81%, val/bpb tied (slight edge), ChatCORE +0.62%; `auto_tune` r194: CORE +5.59%, val/bpb tied — single seed, sign-test p≈0.095, needs 2nd seed**; `seed_tokens` was offline-only (−0.63% ClimbMix val tokens at +5.3% vocab; too small to queue for Lambda), since removed |
| [Surface factoring](ideas/surface_factoring/README.md) | ✓ | **surface-only** (space+case on the baseline vocab, 32768, K_max=1, no force_merges) BUILT: offline tools (`tools/{space_factor_compression,case_factor_compression,case_factor_analysis}.py`) + overlay/harness `overlay/surface_{factoring,codec,tokenizer,dataloader,context,lambda}.py` + `wrappers/{train_surface_factoring,tok_train_surface}.py` + `runs/runcpu_surface.sh`; 25 smoke checks (`wrappers/smoke_surface_factoring{,_eval,_overlay,_harness}.py`). (`tools/stack_triple` builds the DEPRECATED triple — kept for reference only.) | surface-only vs baseline A/B (clean factoring-isolation test); λ/exposure levers untested | surface-only offline **+4.90% tokens** (space+case); **no model A/B yet**. The earlier fm+space+case **TRIPLE** d24 A/B (surface TIES fm, a wash) is a **DEPRECATED dead-end** — stacking force_merges with surface factoring was premature and confounded the read; its d24 checkpoint is abandoned. See [the idea doc](ideas/surface_factoring/README.md). |
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

- **`force_merges` (data-driven curated tokenizer) is a real positive result and is adopted.** CORE +5.81% / val/bpb tied (slight persistent edge) / ChatCORE +0.62%, zero compute cost. Unlike MTP, the sign test is significant (p ≈ 0.039) and val/bpb is favorable, so it isn't a single-task fluke despite boolq carrying 83% of the CORE delta. Full breakdown, and the 2026-05-25 speedrun (104 pairs @ 32768) vs in-tree (122 pairs @ 32890 additive, Lambda-pending) canonical distinction, in §d24 force_merges results + §force_merges curation details below.

- **`auto_tune` (held-out-Pareto block-list tokenizer, r194) is a strong-but-borderline positive on a single-seed d24 A/B.** CORE +5.59% / val/bpb tied (slight persistent edge −0.0002) / zero compute cost, against the *same* baseline as force_merges and nearly matching it (+5.81%) — while keeping the **pristine** regex (0/48706 boundary-crossing ⇒ bpb exactly comparable, no prefix caveat). The catch vs force_merges: the sign test is **not** significant (14/7/1, p ≈ 0.095) and ~79% of the CORE delta is boolq + agi_lsat. A second seed is the cheap next step before adopt/reject. Full breakdown in §d24 auto_tune results below.

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

## d24 force_merges results (8xA100 40GB, $115.76 single-run)

Single-instance run on 2026-05-26 with the **2026-05-25 canonical** (87 mined FORCED + 15 mined DERIVED + 2 numeric DERIVED = 104, vocab=32768 — the version curated under the buggy `detect_shadows`; current in-tree canonical is the post-shadow-fix 122-pair list at vocab=32890 additive, see §force_merges curation details). Same nanochat commit, seed, depth=24, hardware envvars as the baseline `qmsi105c` (USE_FP8=0, WINDOW_PATTERN=L, DEVICE_BATCH_SIZE=8). Differs from `qmsi105c` in just two ways: tokenizer + USE_SFT=1 (added because tokenizer changes touch downstream user-facing tasks; first SFT data for any tokenizer experiment).

| | baseline (`qmsi105c`) | force_merges (`7aesostl`) | Δ |
|---|---:|---:|---:|
| val/bpb (final) | 0.71609 | 0.71588 | **−0.00021** (slight edge, persistent throughout training) |
| **CORE (`base_eval`, full)** | 0.250488 | **0.265034** | **+0.014546 (+5.81%)** |
| **ChatCORE (after SFT)** | 0.3560 | **0.3582** | **+0.0022 (+0.62%)** |
| tok/sec | 291,210 | 291,161 | tied (no compute cost) |
| total time | 5h33m | 5h33m (+SFT 1h36m) | identical training; SFT adds ~30% |
| LM boundary-crossing % | — | 0.82% (398/48706) | flag for direct LM-task prefix-comparability |

**Sign test (per-task CORE)**: 15 wins, 6 losses, 1 tie out of 22 tasks → p ≈ 0.039 (significant). Wins are distributed (lambada +0.045, piqa +0.029, arc_challenge +0.029, hellaswag_zeroshot +0.020, arc_easy +0.011, squad +0.019, agi_eval_lsat_ar +0.060, plus 8 smaller). Losses concentrate in small bigbench tasks (copa, dyck_languages, cs_algorithms, operators).

**boolq dominance caveat (same flag the MTP analysis raised)**: 83% of the CORE headline comes from boolq alone (baseline −0.273 → force_merges −0.007 = +0.266 of the +0.014546 total). Excluding boolq: mean Δ across 21 tasks = +0.0026 (~+1%). Unlike MTP, the rest of the picture rules out single-task fluke:
- val/bpb is favorable (MTP regressed +8.78%)
- sign test is significant at p≈0.039 (MTP was p≈0.19)
- ChatCORE carries the win downstream

**val/bpb trajectory** held the lead at every measured step (from −0.030 at step 250 to −0.00021 at final step 5568). Persistent direction; never crossed back.

Reproduce: `uv run python -m tools.compare_runs d24_baseline force_merges`

## d24 auto_tune (r194) results (8xA100 40GB, base-only single run, ~5.5 h)

Single-instance run 2026-06-14 with the **r194 block list** (129 mangled stem+suffix pairs from the held-out-Pareto fixpoint; vocab=32768 — **no** bump, pristine `SPLIT_PATTERN`). Same nanochat commit (`dc54a1a`), seed, depth=24, and hardware envvars (USE_FP8=0, WINDOW_PATTERN=L, DEVICE_BATCH_SIZE=8) as **the same baseline** (`d24_baseline`, CORE 0.250488) that force_merges used — so the two tokenizer variants are directly comparable. Differs from baseline in exactly one thing: the tokenizer (USE_SFT=0, base-model only — no ChatCORE this run).

| | baseline (`d24_baseline`) | auto_tune r194 (`d24_autotune_r194`) | Δ |
|---|---:|---:|---:|
| val/bpb (final) | 0.7161 | 0.7159 | **−0.0002** (slight edge, persistent) |
| **CORE (`base_eval`, full)** | 0.250488 | **0.2645** | **+0.0140 (+5.59%)** |
| compute | — | — | tied — vocab=32768 (no bump) ⇒ byte-identical model, zero throughput cost |
| LM boundary-crossing % | — | **0.0% (0/48706)** | pristine regex ⇒ bpb is exactly prefix-comparable (no caveat, unlike force_merges' 0.82%) |

**Headline (+5.59% CORE, bpb tied) nearly matches force_merges (+5.81%)** — two independent tokenizer interventions, same shape (CORE up, bpb flat: the win is downstream-task, not per-byte prediction). **But the significance is weaker than force_merges and should be stated plainly:**
- **Sign test**: 14 wins / 7 losses / 1 tie → p ≈ **0.095** — *not* significant at 0.05 (force_merges was 0.039).
- **Concentration**: boolq carries **49%** of the CORE delta, agi_eval_lsat_ar (small-N LSAT) another ~30% — ~79% from two noisy tasks. Excluding boolq, mean centered Δ over 21 tasks is **+0.0003** (~the same small residual as force_merges).
- **Direction is right on the robust high-N tasks** (all small but positive: arc_challenge, piqa, winograd, arc_easy, hellaswag_zeroshot, lambada, winogrande); the main loss is copa −0.05 (100-example, noise-scale).
- **Single seed.** A second seed is the cheap next step to move this from "strong-but-borderline" to adopt/reject. Worth the confirmation: the persistent bpb edge + pristine-regex cleanliness + the headline matching force_merges all point positive.

Reproduce: `uv run python -m tools.compare_runs d24_baseline d24_autotune_r194`

## d24 combined (force_merges + r194) results — PROVISIONAL (ran on wrong vocab size)

Single-instance base-only run 2026-06-15: the **combined** tokenizer = force_merges (122 forced cross-boundary phrases) + r194's 129 mangled-morpheme blocks + the `block_trailing_space` predicate. Same nanochat commit (`dc54a1a`), seed, depth=24, hardware (USE_FP8=0, WINDOW_PATTERN=L) as `d24_baseline`. **DEVICE_BATCH_SIZE=4** (not 8): the non-2^15 vocab fell off torch.compile's fused-cross-entropy path on the 40GB A100 and OOM'd at DBS=8; DBS=4 holds the total batch constant via grad-accum, so the result is unaffected (wall-clock only). FP8 must be off (A100 has no fp8 HW — `USE_FP8=0`); a run that lost that flag crashed in a triton fp8e4nv compile.

| | baseline | force_merges | auto_tune r194 | **combined** |
|---|---:|---:|---:|---:|
| CORE (base_eval, full) | 0.2505 | 0.2650 (+5.8%) | 0.2645 (+5.6%) | **0.2671 (+6.6%)** |
| val/bpb (final) | 0.7135 | 0.7133 | 0.7132 | 0.7132 |

**Additivity — strongly sub-additive.** Combined beats the better single arm (force_merges) by only **+0.0021 CORE (~0.8%, noise-level)**; the naive sum of the two single gains would be +0.0285 (→0.279). Stacking the morpheme blocks onto the forced phrases added essentially nothing measurable — the "two orthogonal ~+5.7% tokenizer wins compound toward +10%" hypothesis does **not** hold here. Per-task the combined is "best" on ~12/22 tasks, but most of those margins are single-seed noise (several sit in the bucket where force_merges and auto_tune *disagree* in sign); on the cross-arm-consistent tasks it only trades wins with force_merges.

### Asterisk: this ran on the suboptimal 32788 tokenizer — the conclusions above are provisional

The combined was meant to run at **vocab 32890** = 32768 **+122** — the forced phrases sitting on *top* of the full base vocab, specified explicitly so they would **not displace auto_tune's blocked-morpheme merges** (and validated in the 32768+122 verification diff: 99.6% base overlap with auto_tune). The miss: `RECOMMENDED_VOCAB_SIZE` was never bumped off **force_merges-alone's** bracketed **32788** (+20 — the minimal bump that recovers *force_merges'* Tier-1 probe tokens; correct for force_merges by itself, see §force_merges curation details, but **not** for the combined). So the bake + this d24 run used 32788, and the 122 forced phrases **displaced ~102 base merges** (base merges 32,657 vs baseline's 32,759) — exactly the displacement of auto_tune's work that 32890 was meant to avoid. Consequence (20M ClimbMix-val battery, `tools/blocked_pairs_report`): the combined **kept** auto_tune's mangled reduction (−92) and suffix-firing redistribution **but lost most of its whole-word coverage gain** — auto_tune **+101** dict-word tokens → combined-32788 **+25**.

So today's combined CORE — and the sub-additivity / boolq-interference reads below — are **on a tokenizer we would now replace, and must be re-confirmed on the corrected 32890 build before they are load-bearing.** The 32890 build was validated offline (coverage recovery below) but **not shipped**: `force_merges` was reverted to its clean original — 122 forced phrases, `BLOCKED_PAIRS=[]` (trailing-space now handled by the `block_trailing_space` crate predicate, not an enumerated list), `RECOMMENDED_VOCAB_SIZE=32788` (since re-baked to **32890 additive** — see §force_merges curation details) — to keep the combined experiment out of it. The combined-at-32890 is a **next-step candidate**, re-derivable as `force_merges` + the auto_tune audit's r194 block list at `RECOMMENDED_VOCAB_SIZE=32890`. This run's exact 32788 tokenizer is preserved at `~/.cache/nanochat-variants/force_merges_32788/`.

**The 32890 re-bake recovers coverage only partially:** +25 → **+60** (total whole-words −7 → +45), with mangled (−92) and compression (−6.9%) unchanged. The original coverage loss splits ~half/half: ~half was the vocab-budget mistake (fixed by 32890), ~half is **structural** — the carve-out regex (the ` ?\p{N}` space-digit tweak + the phrase carve-outs) plus the trailing-space rule reshape which base merges form, an irreducible cost of the forced-phrase mechanism (matches the ~130-token diff vs auto_tune in the verification diff). So even the corrected combined is **not** a clean `auto_tune ⊕ force_merges`.

### boolq mechanism — probe-confirmed; corrects an earlier wrong read

Most of every tokenizer arm's CORE headline is boolq (force_merges ~83%, auto_tune ~50%, combined ~32%). An earlier analysis read this as "the model always answers *no* and never reads the passage — the gain is just debiasing-to-chance, a fragile artifact." **That was wrong**, from assuming boolq's CORE random baseline is 50%. It is **62** (majority/always-yes rate; `eval_bundle/eval_meta_data.csv`), so raw accuracies are baseline **51.6%**, force_merges **61.7%**, auto_tune **57.4%**, combined **56.1%** — all above the 50% coin-flip. The per-example probe (`tools/boolq_probe.py --ablate swap`, removed in the 2026-06 cleanup; recoverable from git) on the combined model confirmed it: pred-yes 0.40 (a residual *no*-lean, not always-no) and **swap-delta +0.08** (a wrong passage costs 8 points) → **the model genuinely uses the passage.** The boolq/CORE gains are real reading improvements, not calibration artifacts.

boolq is the extreme of the sub-additivity: combined raw **56.1% < force_merges 61.7%** (and < auto_tune 57.4%) — combining partially *undid* force_merges' boolq gain. That's a representational interaction, **not** a tokenization-boundary artifact: a binding test shows the r194 blocks fire ~identically under the pristine and carve-out regexes (121/129 bind under both; only 8 go inert under the carve-out).

**Next:** seeded d24 of the **32890** combined vs baseline, plus ≥2 seeds to settle force_merges-vs-combined (within noise at one seed). Until then treat "combining is sub-additive / force_merges-alone captures the win" as provisional.

## Suggested sequencing (updated post-d24)

1. **z-loss and `force_merges` are both adopted — stack them as the default for future speedruns.** No model-side interaction (z-loss is a loss-term overlay, force_merges is a tokenizer artifact), both are essentially-free wins (~zero compute cost), so the next default speedrun runs `OVERLAY=zloss` *with* the canonical `force_merges` tokenizer in place. Single open question worth ablating: turn off the existing logit-softcap (`gpt.py:472`) with z-loss on; the hard cap may now be unnecessary. Cheap d24 ablation, ~$100.
2. **Deep supervision** — depth-axis complement to MTP; folds into the same `MTPGPT` subclass for one combined experiment. Token weighting's **RHO-Loss variant** can fold in here too (shares `forward` surface; needs a reference model that can be added once for both).
3. **Token-level loss weighting** (entropy / focal variants) — pure-loss overlay, no extra model, independent of MTPGPT line. Cheap parallel track.
4. **Differential attention** — independent architecture A/B.
5. **Adaptive sequence length (Part A of adaptive-schedule)** — harness change, but the model needs no edits and Part A is the rare idea that should make the speedrun **faster on wall-clock** at the same final loss. Cheap-ish entry into the harness-change tier.
6. **Online data selection + adaptive batch size (Part B of adaptive-schedule)** — bigger, harness-touching; pursue if earlier results justify it. Both share the copied-`base_train.py` pattern, so fold into a single forked harness. Online data selection is complementary to token weighting (sample-level vs. token-level signal shaping), so worth A/B-ing alone, weighting alone, and combined.

MTP rescue probes (α=0.1+k=1 first) only happen if a follow-up idea raises a specific question that an MTP variant can answer. Don't revisit MTP for its own sake at d24.

**[Tokenizer variants](ideas/tokenizer-variants/README.md)** is a **parallel track**. Infrastructure is now end-to-end ready: `tools/eval_tokenizer.py` harness validated; pristine rustbpe vendored at `sandbox/rustbpe/` (Karpathy@9467d83) with `runs/build_rustbpe.sh` for local builds; parallel-crate variants pattern in `sandbox/rustbpe_variants/`.

**Evaluating a (re)trained tokenizer variant** — the standard offline loop, all `vs baseline` (`~/.cache/nanochat/tokenizer`):

```bash
# 0. smoke, then (re)train → ~/.cache/nanochat-variants/<variant>/tokenizer/
uv run python -m wrappers.smoke_tok_train_<variant>
uv run python -m wrappers.tok_train_<variant>

# 1. Tier-1 filter: compression battery, structural round-trips, task probes
uv run python -m tools.eval_tokenizer \
    --tokenizers ours,~/.cache/nanochat-variants/<variant>/tokenizer

# 2. corpus deltas vs baseline: compression, dead/mangled tokens, firing-rank buckets
uv run python -m tools.pass_metrics \
    --variant ~/.cache/nanochat-variants/<variant>/tokenizer

# 3. inspect the vocab (rank, corpus firing count, byte_len, repr) — throwaway dump
uv run python -m tools.dump_vocab \
    --tokenizer ~/.cache/nanochat-variants/<variant>/tokenizer \
    --output /tmp/<variant>_vocab.txt
```

These are **offline filters** — they do NOT predict CORE/SFT gain (see `ideas/tokenizer-variants/README.md` "Methodological problem encountered"). A favorable read gates a GPU A/B, it doesn't replace one.

**Live variants** (each its own crate; see [`rustbpe_variants/README.md`](rustbpe_variants/README.md)):

- **`force_merges`** (Rust, **adopted default**) — forced cross-boundary common-phrase merges (` of the`, `, and`, etc.) via regex carve-out. Crate: `rustbpe_variants/force_merges/` (`rustbpe_force_merges` module); canonical pair list at `pairs.py` (105 mined FORCED + 15 mined DERIVED + 2 numeric DERIVED = 122, vocab=32890 additive). Build: `VARIANT=force_merges RELEASE=1 bash runs/build_rustbpe.sh`. On a 15M-char climbmix sample: **−6.9% total tokens vs baseline** (3.27M → 3.05M). Lambda speedrun (2026-05-25 canonical): **CORE +5.81%, val/bpb tied, ChatCORE +0.62%** ($115.76, see §d24 force_merges results). Construction provenance + per-pair attribution + vocab-cost mechanism in the `force_merges curation details` subsection below. (Its ablation tooling — `tools/ablate_*`, `wrappers/tok_train_force_merges_mined_*` — was removed 2026-06-05; the result + `pairs.py` are kept, tooling recoverable from git.)
- **`auto_tune`** (Rust crate `rustbpe_auto_tune`, **fixpoint converged r194 → 129 blocked pairs; d24 A/B CORE +5.59%, single seed — see §d24 auto_tune results**) — blocking-only under the **pristine** `SPLIT_PATTERN` (no forced merges, no vocab bump); curated `BLOCKED_PAIRS` of mangled stem+suffix merges in `rustbpe_variants/auto_tune/pairs.py`. Forked from `force_merges` + `load_corpus`/`train_from_cached` so the block-list audit trains many block-lists over one cached corpus (~30s/candidate vs ~174s). Driver: `tools/heldout_fixpoint_cached.py` (held-out-Pareto greedy fixpoint). Wrapper: `wrappers/tok_train_auto_tune.py`. See [`rustbpe_variants/auto_tune/`](rustbpe_variants/auto_tune/) and its `audit/`.

**Removed 2026-06-05** (dead research directions; findings retained here + in git history): `space_digits` (leading-space-before-digits regex tweak — subsumed by force_merges' `SPLIT_PATTERN`); `seed_tokens` (within-chunk composition seeds at mid-rank — offline-positive −0.63% val tokens but below the d24 noise floor; see §seed_tokens redesign); `blocked_morphemes` (the `blocked_pairs=` mangled-merge cascade — precursor to auto_tune's mechanism); `manual_merges` (composite blocking + hand-written merges); `unigram` (Unigram-LM-vs-BPE comparison track).

The [non-backprop LLM](ideas/non-backprop/README.md) is a **separate research track**, not part of this capability-tuning sequence. Pursue independently.

### force_merges curation details

> **Update (2026-06-15):** the **+20 bracket / vocab=32788** discussed throughout this subsection was **superseded** — `RECOMMENDED_VOCAB_SIZE` is now **32890, fully additive** (32768 + 122). The `block_leading_space` crate predicate frees the natural-bigram slots the carve-outs used to displace (the same fix killed the "cannibalization" of forced phrases by natural bigrams), so the displacement-bracketing experiment below — and its `+20` conclusion — is now **historical**: the 122 forced phrases are bolted on, displacing nothing.

#### How `pairs.py` was constructed (2026-05-25)

Replaced a ~70-pair hand-picked list (lifted from `veryfansome/nanochat@force_merges_wip`) with a data-driven list derived end-to-end from corpus statistics + Tier-1 ablation. Tooling preserved for re-curation; intermediate exploration variants (v0–v5) were consolidated away.

1. **Two-pass corpus mining** (`tools/mine_forced_pairs.py --two-pass`, 20 climbmix row groups, ~60M chars). Pass 1 mines cross-boundary bigrams from the baseline pre-tokenization with a hand-curated continuation whitelist (LHS leans closed-class — prepositions, conjunctions, auxiliaries, punctuation — plus discourse markers + `one`/`two`; RHS leans the same way plus common open-class verbs that empirically show up in glue phrases — `make`, `find`, `want`, `need`). Pass 2 re-mines bigrams whose LHS is a pass-1 concatenation — recovers depth-2 phrases like `(" one of", " the")`, `(" can be", " a")` that single-pass can't see. Artifacts at `rustbpe_variants/force_merges/mined/forced_pairs_climbmix_20rg{,_twopass}.py`; the pre-shadow-fix two-pass output is preserved as `forced_pairs_climbmix_20rg_twopass_pre_shadow_fix.py` for historical lineage.

2. **Tier-1 sanity check** against the hand-picked baseline (79.6% pair coverage at top-150). Single-pass mining matches/beats hand-picked on English prose; loses `currency` by 1 token because numeric DERIVED pairs (`("00","0")`, `(",","000")`) fail the continuation-whitelist filter (numeric RHS). Added back as hand-additions.

3. **Kitchen-sink (all 22 DERIVED, "v2")** improves `modern_prose` (+1.8%) but regresses `english` (−2.1%, +1 token). Token-diff diagnosis: lost the `"izards"` BPE merge (used in "wizards") because the extra carve-outs consumed vocab budget.

4. **Per-pair DERIVED ablation** to identify the cutoff:
   - *Cumulative* (`tools/ablate_derived.py --k-values 0,5,10,15,20`): `modern_prose` jumps at K=5→10 (pair #6 helps); `english` regresses at K=15→20.
   - *Drill-down* (K=15..20): the regression appears in a single step (K=15→16); all later K transitions flat.
   - *Single-pair attribution* (`INCLUDE_INDICES`, K=15 + only one of pairs #17–20): **all four individually displace `izards`**. The regressor is "whichever pair you add 16th" — the threshold is the vocab budget, not a specific pair.

5. **Curation rule (2026-05-25)**: pick the smallest no-regression cutoff under the Tier-1 probe battery. K=15 + 2 numeric DERIVED → 104 total forced merges at vocab=32768. This was the canonical that produced the Lambda speedrun result above; pair #16+ were dropped because they pushed total forced merges past the 32K-vocab `izards`-class threshold (one-token swings on 198–216-byte probes — near the noise floor).

6. **Re-curation (2026-05-26)** after Codex reviews surfaced two bugs in the miner's `detect_shadows`:
   - *Case 2 "regex precedence" overflag.* Original rule flagged `(c, d)` as shadowed by `(a, b)` when `pa == b`. Wrong — e.g. it dropped `(' one', ' of')` as "shadowed by ` of the`," but at pos 0 of " one of the" the regex actually matches ` one of` first; ` of the` would only start at pos 4. Removed.
   - *Chained false-shadowing.* Even with case-1 only (`pb == a`), the rule added every prior pair to `seen` regardless of whether the prior was itself shadowed — so a shadowed prior could spuriously shadow later candidates. Witness in the pre-fix artifact: `(' to', ' do')` at [`mined/forced_pairs_climbmix_20rg_twopass_pre_shadow_fix.py:84`](rustbpe_variants/force_merges/mined/forced_pairs_climbmix_20rg_twopass_pre_shadow_fix.py) marked "SHADOWED by `(' is', ' to')`," but `(' is', ' to')` itself is shadowed on the line above. Fixed: only un-shadowed (emitted) priors participate.

   Re-running with both fixes produces **+18 pass-1 entries** (`(' one', ' of')`, `(' that', ' is')`, `(' have', ' to')`, `(' However', ',')`, etc.) and a re-ordered top-20 DERIVED — `(' one of', ' the')`, `('. However', ',')`, `(' some of', ' the')`, `(', as', ' well')` rise; the 2026-05-25 canonical's `(' of the', ' most')`, `(', and', ' other')`, `(' can be', ' a')`, `(' to be', ' a')` drop out of the top-15.

7. **Vocab-cost bracketing** to fit the larger pair set:

   | variant                                  | `english` | `izards` | ` acclim` | `ampton` | `uran` |
   |------------------------------------------|:-------:|:-:|:-:|:-:|:-:|
   | 2026-05-25 canonical (104 fm, vocab=32768) | 4.21 bpt | ✓ | ✓ | ✓ | ✓ |
   | Re-curated (122 fm, vocab=32768)         | **4.12 (regressed)** | ✗ | ✗ | ✗ | ✗ |
   | **Re-curated (122 fm, vocab=32788, +20)**| **4.21 (recovered)** | ✓ | ✓ | ✓ | ✓ |
   | Re-curated (122 fm, vocab=32848, +80)    | 4.21 | ✓ | ✓ | ✓ | ✓ |

   **+20 vocab is the smallest bump that recovers all 4 known borderline merges**. Cost: ~10K params at d24 (~0.001% of model size, well below numerical noise). Beyond +20, returns diminish sharply (only 0.01% additional corpus compression per +20 vocab). Even at +80, 8 specific canonical tokens (the `(' of the', ' most')`-family carve-outs from the prior curation) never come back — they were demoted past K=15 in the re-mined ranking and aren't recoverable just by adding vocab. This is a *vocab-content swap*, not pure addition.

   An earlier "2× vocab cost per forced pair" heuristic (extrapolated from a single v5 data point) is the wrong model entirely: forced pairs mostly reuse existing ids (`apply_forced_merges_at_end`), so there is no per-pair vocab price — the +20 bracket is set by merge *displacement* (the swap above), not by pair count. Don't treat any "slots per pair" figure as a rule.

8. **Promoted 2026-05-26**: re-curated list (105 mined FORCED + 15 mined DERIVED + 2 numeric DERIVED = 122 forced merges, now at vocab=32890 additive — see the update note at the top of this subsection) is the in-tree canonical. The Lambda speedrun result above used the 2026-05-25 canonical; whether the local +0.42% real-corpus gain translates to a CORE/val_bpb improvement at d24 scale is the open question — the next default speedrun (`zloss + force_merges` stacked) will tell us.

#### Per-pair DERIVED attribution

Steps 1–5 above describe the methodology, applied first to the 2026-05-25 mining output. The table below shows the **post-shadow-fix re-mined ordering** (current canonical's source) — the K=15 cutoff and the methodology are unchanged from steps 1–5; only which specific pairs land in each bucket differs. The 2026-05-25 ordering had `(', but', ' it')` at rank #6 and the boundary at K=15 too; in the new ranking, that helper slipped to #8.

| rank | pair | verdict | evidence |
|---:|---|:-:|---|
| 1–5 | `(' one of', ' the')`, `(', and', ' the')`, `('. If', ' you')`, `('. It', ' is')`, `(', it', ' is')` | KEEP | Tier-1 silent |
| **6** | **`('. However', ',')`** | **KEEP** | enabled by un-shadowed `(' However', ',')` in pass-1 |
| 7 | `('. This', ' is')` | KEEP | Tier-1 silent |
| **8** | **`(', but', ' it')`** | **KEEP** | +0.070 bpt `modern_prose` (helper, same pair as 2026-05-25 canonical's rank #6 — slipped to rank 8 in new ordering) |
| 9–15 | `(', which', ' is')`, `(' some of', ' the')`, `('. There', ' are')`, `(', as', ' well')`, `('. In', ' the')`, `('. They', ' are')`, `(', and', ' it')` | KEEP | Tier-1 silent at this K |
| **16–20** | **`(', and', ' a')`, `('. In', ' this')`, `(', but', ' the')`, `(' of the', ' following')`, `('. If', ' the')`** | **DROP** | each individually pushes total forced merges past the `izards`-class vocab-budget threshold (same boundary mechanism as 2026-05-25 curation, just at a different K within the new ranking) |

#### Implementation hygiene notes

- **Ablate-driver caches** are keyed by an 8-char hash of all tokenizer-defining inputs (mining artifact + subset module + `pairs.py` + wrapper + `src/lib.rs` + upstream `nanochat` HEAD). Any edit invalidates automatically; `--force` bypasses for `.so`-rebuilt-outside-tree edge cases. Implemented in `tools/_ablate_common.py` (shared by both ablate drivers).
- **Pre-shadow-fix tokenizer preserved** at `~/.cache/nanochat-variants/force_merges_pre_shadow_fix_104pairs/tokenizer/` — the exact tokenizer used in speedrun `7aesostl`. The matching mining output is `mined/forced_pairs_climbmix_20rg_twopass_pre_shadow_fix.py`.

#### Curation infrastructure (removed 2026-06-05)

force_merges is frozen as a kept result, so its mining + ablation tooling was removed (recoverable from git): `tools/mine_forced_pairs.py` (corpus miner, continuation-whitelist filter, two-pass + shadow detection), `tools/ablate_{mined,derived}.py` (cumulative FORCED/DERIVED ablation), `tools/_ablate_common.py`, and `wrappers/tok_train_force_merges_mined_{forced_subset,derived_subset,ablate}.py` (subset trainers). The kept artifacts remain: `rustbpe_variants/force_merges/pairs.py`, the `pairs_mined_*.py` modules, mining outputs under `force_merges/mined/`, and per-K eval JSONs at `results/{mined,derived}_ablation/`. To re-curate (re-mine → re-ablate for the K cutoff → paste tuples into `pairs.py`'s `_MINED_FORCED`/`_MINED_DERIVED` → re-validate the `_NUMERIC_DERIVED` hand-adds against the probe battery → re-bracket `RECOMMENDED_VOCAB_SIZE`), restore the tooling from git history.

## seed_tokens redesign (2026-05-26)

> **Variant removed 2026-06-05** (dead direction; code in git history). Findings below retained as research record — notably the cross-chunk-vs-within-chunk gap analysis and the offline→CORE translation caveat, which inform the auto_tune work.

Replaced a YAML-driven constructive-merge-chain approach (from `origin/seed_tokens` prior art) with a data-driven mining pipeline + mid-rank insertion in pure Python. Two mining iterations tested: morpheme-mining (failed, net-negative) and within-chunk composition mining (succeeded, K=1,750 canonical). The successful variant delivers a small offline win but the magnitude doesn't predict a clear downstream gain; canonical is preserved but not queued for Lambda alone.

### Mechanism: mid-rank insertion

Each seed `(L_bytes, R_bytes, S_bytes)` is inserted into the final `mergeable_ranks` at rank `max(id_L, id_R) + 1` — the rank the merge would have had if introduced mid-BPE-training. Natural merges with id ≥ the insertion point shift up by 1 per inserted seed; relative order preserved. Tiktoken applies the merge at encode time via standard rank-based greedy matching whenever L and R are adjacent.

**Why mid-rank, not append-at-end**: a natural BPE merge that would consume one of a seed's operands has lower rank than an appended seed and fires first, leaving no `(L, R)` adjacent pair to match. Smoke witness during early development: seed `(q, q) → qq` appended at rank ~328 got preempted by a natural ` q` merge at rank ~271 and never fired. Mid-rank insertion places each seed where no preempting natural merge can yet exist (both operands are still bare at that point in the BPE sequence). Each seed consumes exactly one vocab slot (final vocab = baseline + N_seeds). force_merges uses a different mechanism — `apply_forced_merges_at_end` appends each forced concat *after* normal BPE training, reusing an existing id whenever the concat already exists in the vocab (so most of the 122 pairs cost zero new slots). Its vocab bump is therefore not a per-pair cost: `RECOMMENDED_VOCAB_SIZE=32890` is fully additive (32768 + 122 forced phrases) — the old +20 bracket at 32788 was retired once `block_leading_space` freed the natural-bigram slots the carve-outs displaced (see §force_merges curation details).

### What we tried

1. **Port the prior-art constructive-merge-chain approach** (YAML morpheme list + Rust merge-chain builder from `origin/seed_tokens`). Smoke passed; first real-data eval was net-negative.
2. **Diagnose the failure** with a per-token usage-delta tool comparing baseline vs variant firing counts. The seeds were displacing high-utility short baseline morphemes (`ing`, `ation`, `ate`, `ize`, ...) — competing with rather than complementing baseline's natural morphology.
3. **Substring-blocker filter** as salvage on the morpheme list (drop candidates whose bytes contain a high-firing baseline token as substring). Narrowed damage but never reached parity; the conceptual mismatch was structural.
4. **Reframe the mining target**: instead of mining linguistic morphemes, mine **within-pre-tok-chunk adjacent-token-pair bigrams from baseline encoding output**. Each pair is, by construction, a composition baseline knows is common but couldn't fit in the 32K budget — every firing replaces 2 tokens with 1, so net-positive at the firing site.
5. **K-sweep** to find the per-seed-marginal plateau-end → K=1,750 canonical (vocab=34,518, +1,750 over baseline).
6. **Isolation control**: trained pure-BPE at the same final vocab size with no seeds, to separate vocab-size effects from seed-mechanism effects.
7. **K_switch sweep**: varied the BPE-vs-seed boundary at fixed final vocab to test whether earlier mining (more seeds, fewer natural BPE merges) would help. It did not.

### Headline results (ClimbMix val, 20M chars)

| variant | vocab | tokens | Δ vs baseline | per added slot |
|---|---:|---:|---:|---:|
| baseline | 32,768 | 4,489,413 | — | — |
| morpheme-mining canonical (469 routes) | 33,237 | 4,498,454 | **+9,041 (+0.20% worse)** | — |
| extended BPE @ same vocab (no seeds) | 34,518 | 4,472,633 | −16,780 (−0.37%) | 9.6 t/slot |
| **composition-mining canonical (K=1,750)** | **34,518** | **4,461,142** | **−28,271 (−0.63%)** | **16.2 t/slot** |
| K_switch=20K (earlier mining, more seeds) | 34,510 | 4,537,216 | +47,803 (+1.07% worse) | regression |
| force_merges (reference, different gap) | 32,788 | 4,179,430 | **−309,983 (−6.91%)** | ~15,500 t/slot |

The K-sweep identified K=1,750 as the per-seed-marginal plateau-end. K=1,500–1,750 averaged ~13 tokens/seed (plateau), K=1,750→2,000 declined to 10/seed, K=2,000→2,250 went actively anti-productive at −8.7/seed (51% dead in that cohort; the live seeds collectively cost ~17 tokens each by intercepting longer natural compositions). Larger K eventually beats K=1,750 in absolute terms (K=3,000 saves 36,165) but only by walking through the anti-productive K=2,000–2,250 dip; K=1,750 is the cleaner stop condition.

The K_switch sweep tested whether earlier mining (more seeds, fewer natural BPE merges, same final vocab) would help. Monotonically hurt — at K_switch=20K, swapping ~12,800 natural BPE merges for seeds caused a +47,803-token regression vs the K_switch=32,768 canonical.

### Takeaways

1. **Mining target dominates over insertion mechanism.** Both morpheme and composition pipelines used identical wrapper + identical mid-rank insertion. One was net-negative, the other net-positive. The candidate set matters more than refining the mechanic.

2. **The seed mechanism does mechanism-specific work, but the magnitude is small.** At the same vocab cost, seeds beat extended BPE by 11,491 tokens — a 1.69× per-slot efficiency gain. Not just vocab growth in disguise. The advantage is mid-rank insertion: seeds at `max(id_L, id_R) + 1` fire reliably; the same compositions at tail rank (extended BPE) get preempted by lower-rank natural merges that consume their operands.

3. **The seed mechanism is a TAIL-EXTENDER, not a mid-budget REPLACEMENT.** BPE's iterative count-update selection in the middle of its distribution outperforms snapshot mining; the seed mechanism's value materializes specifically at the budget boundary, where BPE's iterative dynamics are weak (low counts, narrow advantages) AND extended-BPE's tail-rank placement is a structural handicap. There's no benefit to mining earlier than where BPE naturally ends.

4. **force_merges and seed_tokens target structurally different gaps.** force_merges captures cross-pre-tok-chunk pairs (BPE structurally cannot see them — addressed via regex carve-out). seed_tokens captures within-chunk pairs (BPE saw them but couldn't fit). Per-slot returns differ by ~three orders of magnitude (15,500 vs 16 t/slot). The cross-chunk gap is just much larger than the within-chunk leftover, so within-chunk seeding plays a smaller game on smaller stakes.

5. **Offline → CORE translation is uncertain at small magnitudes.** Applying force_merges's offline-to-CORE multiplier (−6.9% offline → +5.81% CORE) to seed_tokens predicts ~+0.5% CORE — likely below the per-task variance floor at d24 scale. The result is real but its magnitude doesn't predict a clear downstream win.

6. **Determinism in mining pipelines requires explicit work.** Bugs surfaced during reviews: undefined sort tiebreaks (`Counter.most_common()` first-seen order leaking into vocab rank), top-K slicing through same-count buckets, firings caches reusing across tokenizer rebuilds, per-batch char caps rounding to row-group boundaries. Each was invisible at single-run scale but would have made re-mining non-reproducible. Explicit tiebreaks + content fingerprints + doc-level cap accuracy belong in every mining pipeline from day one.

### Decision

Canonical (K=1,750, vocab=34,518) is preserved as a complete, reproducible artifact but **not queued for Lambda alone**. At ~$120/run the predicted downstream win sits below the test's resolution. If a future tokenizer variant earns Lambda time, seed_tokens can stack opportunistically for ~zero marginal compute cost — but is not worth running on its own merits.

### Open questions

- **Stack with force_merges**: untested. They target orthogonal gaps (cross-chunk vs within-chunk), use mechanically independent insertion (regex carve-out vs rank insertion), and have no obvious interference path. Worth an offline A/B if it bundles into another tokenizer experiment cheaply; a standalone test is hard to justify given the small expected magnitude.
- **Iterative-selection at mid-rank**: extended BPE picks merges by iterative count update (better selection); seed_tokens places at mid-rank (better placement). A hybrid — extended-BPE's merge set placed at natural-firing ranks instead of tail ranks — would isolate which advantage matters more. Untested.

## Next concrete steps

1. **Optional zloss val/bpb re-eval** — the d24 zloss run's val/bpb is inflated by ~0.004 due to the eval-path bug (since fixed). Either re-run for a citable number (~$80, one Lambda spin) or document the corrected value analytically (`0.71609 + noise`, matching baseline — derivable from the fix mechanics). The CORE win is unaffected; this is bookkeeping.

2. **Add smoke check (d) "eval-path equivalence" to MTP's smoke** — `mtp.py` honors the contract, but `wrappers/smoke_mtp.py` doesn't verify it. Copy the pattern from `wrappers/smoke_zloss.py` check (e). Cheap insurance against a future regression.

3. **Build the next overlay** — per sequencing, **deep supervision** (folds into the existing `MTPGPT` subclass, scaffolding already in place) or **token-level loss weighting** (entropy/focal — pure-loss overlay, no extra model). Deep supervision is the more natural next step; token weighting is a viable parallel cheap track if you want two signals in flight.

4. **Tokenizer-variants track** — `force_merges` is the adopted default (122-pair canonical at vocab=32890 additive). **`auto_tune`**: the held-out-Pareto block-list **fixpoint** (`tools/heldout_fixpoint_cached.py`) has **converged** (r194, 129 blocked pairs under the pristine regex; anti-churn cooldown + transactional net-commit were added mid-run to break a limit cycle — see `auto_tune/audit/README.md`). Fresh-shard generalization (`tools/generalization_check_cached.py`, two never-used 10-shard sets): coverage (+111 words) and dead-token reduction generalize; the gate's compression gain does **not** (≈ neutral on fresh shards) — the run traced a coverage-vs-compression frontier, so the promotion target is a prefix on that frontier, not necessarily the full r194 set. Next:
   - **d24 A/B done** (§d24 auto_tune results): CORE +5.59%, val/bpb tied — a strong-but-borderline single-seed positive (sign test p≈0.095; ~79% of the delta is boolq + agi_lsat). **Next: a 2nd seed** to confirm before adopting; then promote the chosen prefix into `auto_tune/pairs.py` → canonical retrain (`wrappers/tok_train_auto_tune.py`) → `tools/blocked_pairs_report`. (val/bpb tied means the post-r170 compression cost was downstream-invisible → the full r194 looks promotable, not just an earlier frontier knee.) Deferred candidates (the ~52 coverage-flat keeps + the LOG'd set) remain a later pass.
   - **Stacked A/B (decisive)**: `zloss` + the best tokenizer (force_merges canonical and/or the auto_tune block list) in one d24 run vs `d24_baseline`. Offline tokenizer metrics are proxies; the speedrun is the real verdict. ~$120.

5. After that: differential attention, then online data selection if the earlier results justify the harness-touching investment.
