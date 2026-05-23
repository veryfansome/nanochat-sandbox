# Online data selection (+ batch-size tuning) — proposed setup

Status: **proposal / not yet implemented**
Target repo: `nanochat` (Karpathy) — kept **pristine**, never edited.

## Goal

Improve **sample efficiency** — capability gained per training token — without changing model scale or the dataset.

Important framing: nanochat is tuned for **compute / wall-clock efficiency** (the speedrun), which actively trades sample efficiency away. So part of the work here is *undoing speedrun compromises*, not only adding machinery. This doc covers two sample-efficiency levers; both are measured as **loss per token trained on**, not loss per wall-clock second.

---

## Part A — Batch size / critical-batch tuning *(do this first)*

### What it is

Above the **critical batch size**, large batches are compute/wall-clock efficient but **sample-inefficient**: each extra token buys sub-linear per-step progress, so more total tokens are needed to reach a given loss. Leaderboard run #3 ("bump total batch size to 1M tokens") improved *wall-clock* — and very likely cost sample efficiency to get it.

If the metric is loss-per-token, **go smaller** on batch size and retune the LR.

### Why it fits the setup

**No overlay, no code edit at all** — `--total-batch-size` is a CLI arg on `scripts/base_train.py`. This is the cheapest experiment on the whole ideas list; run it first.

### How to test

- Sweep `--total-batch-size` downward from the speedrun default.
- **Retune the LR for each batch size** (smaller batch → smaller LR; use a scaling rule or a short sweep) — otherwise the A/B is unfair.
- Plot `val_bpb` vs **tokens consumed**, not vs wall-clock.
- Same pinned nanochat commit + same seed across runs.

### Cost / caveats

- Smaller batch = more optimizer steps = **more wall-clock** and lower GPU utilization. That is the trade: you pay wall-clock to buy sample efficiency.
- There is a floor — too small and gradient noise dominates and the GPU is starved. Find the knee, don't just minimize.
- Interacts with `--device-batch-size` and the derived grad-accumulation steps.

---

## Part B — Online data selection

### What it is

Score training examples *during* training and spend gradient updates on the **most informative** ones — skipping tokens already learned, noisy, or low-learnability. Variants, by cost:

- **Selective backprop** *(cheap)* — overselect a larger batch, forward-pass to get per-example loss, backprop only the top fraction.
- **RHO-LOSS** *(more setup)* — select by *reducible holdout loss* (`training loss − reference-model loss`); needs a small holdout/reference model. Picks points "learnable, worth learning, not yet learnt."
- **Learnability-based** — select where loss is both high and *decreasing*; needs per-example loss history / bookkeeping.

This is **selection from the fixed dataset**, not a change to the dataset — the corpus is unchanged; only *which tokens get gradient updates* changes.

### Why it needs a harness change (not a pure overlay)

nanochat's training step is `loss = model(x,y); loss.backward()` (`base_train.py:510-518`). Online selection restructures that step: overselect → score with a forward pass → subset → real forward+backward on the subset. That edits the loop, so it needs a **copied `base_train.py`** (per the overlay plan: copy the script once into `wrappers/`, diff against upstream periodically to pull harness improvements).

### nanochat hooks that help

- `GPT.forward(..., loss_reduction='none')` already exists (`gpt.py:416`, used at the `F.cross_entropy` call ~`gpt.py:477`) — so **per-token loss for scoring is available for free**, no model change needed.
- Overselection draws from the existing `tokenizing_distributed_data_loader_*` in `nanochat/dataloader.py`.

### Cost / caveats

- Scoring forward pass adds overhead: a forward is ~1/3 of a train step; overselecting 2× and scoring adds roughly +2/3 of a forward per step.
- It **streams more raw data** (cheap I/O) to do the same number of updates — fine for a large web corpus, and still "selection," not data modification.
- The sample-efficiency win must beat the scoring overhead — judge on loss-per-token-trained-on, and also report raw tokens streamed.
- Interacts with Part A: selection inherently overselects then subsets, so the *effective* backprop batch is a subset — tune the two together.

### How to test

- Same pinned commit + seed as baseline.
- Metric: `val_bpb` / CORE vs **tokens trained on**; also log tokens streamed.
- Smoke-test wiring with the repo's tiny config (`--depth=4 --num-iterations=20`).
- Start with **selective backprop** (cheapest variant) before RHO-LOSS.

### Open questions

- Overselection ratio and keep-fraction.
- Scoring signal: raw loss vs. reducible loss vs. learnability.
- Whether a reference model (RHO-LOSS) is worth the extra setup at this scale.
- Per-token vs. per-sequence selection granularity.

---

## Suggested order

1. **Part A** (batch-size tuning) — config-only, free, run first.
2. **Part B** (online data selection) — bigger, harness-touching; pursue if A and MTP justify further sample-efficiency investment. Begin with selective backprop.

## References

- McCandlish et al. 2018 — "An Empirical Model of Large-Batch Training" (critical batch size).
- Mindermann et al. 2022 — RHO-LOSS, "Prioritized Training on Points that are Learnable, Worth Learning, and Not Yet Learnt."
- Selective backprop / online batch selection (prior line of work).
