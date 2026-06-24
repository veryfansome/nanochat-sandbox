# force_merges_combined

**force_merges + 126 mined block-list pairs** — a candidate tokenizer for the speedrun A/B. **Not adopted**; kept as a separate variant so the adopted default ([`force_merges`](../force_merges/README.md)) stays untouched until an A/B shows the 126 blocks actually help downstream.

> Handoff doc — captures full state as of **2026-06-23** so this can be resumed cold. How the 126 were produced and how to **resume or extend** the audit (driver command, per-round judging methodology, tool inventory): [`../auto_tune/audit/heldout_fixpoint_fm/README.md`](../auto_tune/audit/heldout_fixpoint_fm/README.md). Round-by-round log: its `ledger.md`.

## What it is

Byte-for-byte identical to `force_merges` — same `rustbpe_force_merges` crate, carve-out `SPLIT_PATTERN`, 122 `FORCED_PAIRS`, and `RECOMMENDED_VOCAB_SIZE = 32890` — **except** for an added `BLOCKED_PAIRS` list of **126** `(L, R)` pairs forbidden from merging during BPE training. `pairs.py` re-exports the four force_merges symbols and defines only the 126 blocks, so the two tokenizers differ by exactly those blocks — which makes every force_merges-vs-combined A/B a clean isolation of their effect.

The 126 are the **hand-curated output of a held-out-Pareto block-list audit** (`../auto_tune/audit/heldout_fixpoint_fm/`). The audit mines candidate mangled-merge `(L,R)` pairs, retrains a tokenizer with each one blocked, and keeps a pair only if blocking it improves a held-out Pareto gate (coverage ↑; compression/dead within tolerance) under a **net-commit** test: trial-commit → settle the leave-one-out ejections it triggers → gate the *net* delta (so a pair that looks good alone but displaces better existing blocks is rejected). Over 255 rounds it committed 126 blocks, each a hand-judged pick from the gate-passing board — the loop is Claude-orchestrated, applying a semantic veto + realness review on top of the statistical gate (see the [run README](../auto_tune/audit/heldout_fixpoint_fm/README.md)).

**Where it stopped is a *judged* stop, not a driver fixpoint.** The driver declares convergence only when a round yields zero gate-passing keeps *and* an empty cooldown bench (`tools/heldout_fixpoint_cached.py`); that was never reached — the audit state is still `running`. At round 256, 1,344 candidates still pass the per-candidate gate, but the board is topped by hard semantic vetoes — `ob+by` (+3 / −15.1; shatters hobby/lobby) and `s+hip` (+2 / −5.6; shatters the `-ship` morpheme) — with only small candidates below them, which is *why* the loop was manually halted at 126. So treat the 126 as a **validated, hand-curated candidate set** (which is all the A/B needs), not a proven optimum. Two routes could still extend coverage: (a) a few small free-on-compression gate-passers on the round-256 board were never individually net-commit-tested — resume and judge them toward a true 0-keep fixpoint (run README); and (b) the committed blocks banked a large compression surplus (−891.6/shard, below) that was never spent — coverage can be bought by relaxing the tolerance (*Open directions* #1).

Crucially, the audit ran **in the force_merges context** (the driver's `--fm-context`: the same carve-out `SPLIT_PATTERN`, 122 forced phrases, and vocab 32890 this variant ships), so the 126 are tuned to the tokenizer they live in. That's the key difference from the earlier `combined` (the "r194" combined, STATUS.md §"d24 combined"): its 129 blocks were mined under the **pristine** regex and then bolted onto force_merges at the wrong vocab (32788, which displaced ~100 base merges) — a mismatch that erased most of the coverage gain and left it strongly **sub-additive** (combined beat force_merges by only ~+0.0021 CORE, noise-level). Re-deriving the blocks in-context is the bet this variant exists to test.

## Mechanism

A blocked pair `(L, R)` forbids the BPE merge `L·R`. Blocking a *mangled* merge — one that produced a bound, non-word fragment token (e.g. ` c`+`ateg` → `categ`) — forces the affected words to re-tokenize through a different route, which typically surfaces a whole word atomically (`category`) in place of a non-morphemic split. The crate already accepts `blocked_pairs=`; this variant just passes the 126.

## Provenance & validation

- **Gate tally** (10 held-out shards, cumulative over the 126): covΔ **+52**, compΔ **−1152.6**, deadΔ **−49.5**.
- **Fresh-shard generalization** (20 never-selected shards, via `tools/generalization_check_cached.py --fm-context`): compΔ **−891.6/shard (77% of the gate tally; 20/20 shards improved)**, deadΔ **−41.3 (83%; 20/20)**, covΔ **+52** (vocab-exact). Generalizes cleanly off the gate shards.
- **Baked & verified:** `~/.cache/nanochat-variants/force_merges_combined/tokenizer/` — all 126 blocks confirmed live (0 no-op, 0 latent) via `tools/blocked_pairs_report.py`.

## What this variant actually changes: coverage, not compression

*All figures in this section reproduce via `tools/compare_tokenizers.py` (`uv run python -m tools.compare_tokenizers`), which encodes baseline/force_merges/combined over the 30 held-out shards and reports compression, whole-word firing coverage, and morpheme-atom firing.*

Compression is a **red herring** here — it's the axis the audit explicitly discounted, and the 126 blocks barely move it:

| tokenizer | tokens (30 shards) | vs baseline |
|---|---|---|
| baseline | 64,032,974 | — |
| force_merges | 59,571,912 | −6.97% (30/30) |
| **combined** | 59,557,226 | the 126 add only **−14,686 = −0.025%** over force_merges |

The real effect is a shift in the **firing distribution** (combined vs force_merges, 30 held-out shards — a clean isolation, since the tokenizers are otherwise identical):

**Whole-word coverage that fires:** 18,080 → 18,132 = **+52 distinct whole-word tokens fire**, matching the audit's covΔ +52 **1:1** (so the held-out coverage gain is real, not a metric artifact). Of those: **+44 real used words** (`Allah`, `fisherman`, `reagents`, `compensated`, `heroin`, `protests`, `conspicuous`, `Protecting` …; **6,769 firing events**, all with high standalone usage), **+10 artifact fragments** (`kang`/`afric`/`wob`/`jur`/`cros`/`mont`/`Ens`/`Jur` — standalone ≤4, ~1,400 events; left-halves exposed when a mangle is blocked), **−2 displaced** (`Andre`, `shap`). **Net +42 real whole-words become atomic tokens** with their own embedding, at 18% artifact collateral.

**Morpheme atoms** (firing summed over `tools/find_mangled_morphemes.py`'s `SUFFIXES`/`PREFIXES` inventories): **suffix firing −5,777 (−0.92%)** — `es`/`ed`/`ing`/`ation`/`able` all decompose *less*; **prefix firing +586 (+0.08%)**, ~flat. The ~5,777 lost suffix firings ≈ the ~6,769 the 44 new whole-words gained: **one shift seen twice — whole-word consolidation, not morpheme decomposition.** `compensated`, split as `compens`+`ated` under force_merges, is now one token, so its suffix decomposes less. Not a concern — the drops are <1% of each atom's firing (`ing` still fires 49K×), so the morpheme atoms stay heavily trained.

**Direction note:** this consolidation is a *consequence of how the 126 were mined* — a firing-percentile cut that happens to reroute rare merge tokens toward whole words. A morpheme-oriented mine (`find_mangled_morphemes --mode suffix/prefix/both`; the hypothesis in [`../auto_tune/README.md`](../auto_tune/README.md): "stem+suffix → better-trained semantic atoms") targets a **different** axis — cleaner suffix/prefix firing. Different, but **not necessarily opposed**: the firing-percentile mine *traded* suffix firing for whole words here, yet nothing makes that trade intrinsic. A block whose mangled token has a clean stem+suffix route but no whole-word route would sharpen morpheme firing *without* costing coverage. Whether one can gain morpheme cleanliness while holding (or adding) whole words is open and worth exploring — see *Open directions*.

> Metric caveat: the audit's Rust `evaluate_many` compression runs ~2× the real baked-tokenizer (tiktoken) compression under the carve-out pattern — a constant per-shard offset that cancels inside the audit's rankings but inflates absolute magnitudes (e.g. −891.6/shard `evaluate_many` ≈ −489/shard baked). Use baked-tokenizer numbers for any absolute compression claim.

## Status & next steps (resume here)

1. **Speedrun A/B — pending** (paid Lambda; user launches). Three arms:
   - **baseline** — already have downstream data; no rerun.
   - **force_merges** — needs a rerun: the current in-tree build (122-pair, vocab 32890, `block_leading_space`) has **no downstream data yet** — the last Lambda run was the older 104-pair/32768 build (see [`../force_merges/README.md`](../force_merges/README.md) "Results"). This is the honest mid-point for the comparison.
   - **force_merges_combined** — this variant.
   - **Key read:** combined vs force_merges, on *capability* (CORE/ChatCORE), not compression — does the in-context re-derivation of the 126 (see "What it is") beat the sub-additive r194 combined?
   - Mechanics: tokenizer-swap A/B — `OVERLAY` empty, the variant enters via its pre-baked tokenizer, per-arm `MODEL_TAG`/`WANDB_RUN` (never `"dummy"`). See [`../../runs/README.md`](../../runs/README.md).
2. **Commit before the run** (currently untracked): this variant's `pairs.py` + `README.md`; `wrappers/tok_train_force_merges_combined.py`; `tools/harvest.py` + `tools/compare_tokenizers.py`; the modified `tools/heldout_fixpoint_cached.py` + `tools/generalization_check_cached.py`; and the audit record `../auto_tune/audit/heldout_fixpoint_fm/{README.md,checkpoint.json,ledger.md,lineage.jsonl,.gitignore}` (the `.gitignore` keeps per-run backups/scratch out of the commit).
3. **STATUS.md** — add the hand-curated 126-block result + this variant (not yet done).
4. The compression, firing-distribution, and morpheme analyses above are reproducible via **`tools/compare_tokenizers.py`** — `uv run python -m tools.compare_tokenizers` runs all three over the 30 held-out shards (`--shards N` for a quick pass; `--ref`/`--cmp` to pick the isolation pair; `--tokenizer NAME=DIR` to add a candidate). It encodes each tokenizer once and reports per-shard compression, whole-word firing coverage (born/died classified real vs artifact by standalone usage), and morpheme-atom firing.

## Open directions (not yet pursued)

Two leads the analysis above surfaces, both *beyond* the immediate A/B:

1. **Spend the banked compression for more coverage.** The 126 is a validated hand-curated point, and its committed blocks banked ~−891/shard of compression (fresh shards) that was never spent. Relaxing the gate's compression tolerance would draw some of that down to admit coverage-positive-but-compression-*costly* blocks (the move the r194 set made wholesale) — a different axis from the ≈zero-cost free wins the audit prioritized. A tolerance sweep would map the coverage↔compression frontier and let us pick a point that buys more whole-word coverage at a chosen compression cost. This is *distinct* from simply finishing the harvest: the run was hand-stopped with free-on-compression gate-passers still on the round-256 board (see "What it is" route (a)), so the 126 is not even the free-win endpoint, let alone a frontier optimum.
2. **A complementary morpheme mine.** A `find_mangled_morphemes --mode suffix/prefix/both` mine targets cleaner suffix/prefix firing — a *different* axis from these blocks, but not necessarily an opposed one (see the Direction note). Open question: can it sharpen morpheme atoms while *holding or adding* whole-word coverage? If so it layers on top of the 126 rather than replacing them. Worth a small exploratory mine to find out.

## Usage

1. Build the (shared) crate: `VARIANT=force_merges bash runs/build_rustbpe.sh`.
2. Bake: `uv run python -m wrappers.tok_train_force_merges_combined` → `~/.cache/nanochat-variants/force_merges_combined/tokenizer/`.
3. Verify blocks live: `uv run python -m tools.blocked_pairs_report --tokenizer ~/.cache/nanochat-variants/force_merges_combined/tokenizer`.
4. Eval: STATUS.md "Evaluating a (re)trained tokenizer variant" — `eval_tokenizer` → `pass_metrics` → `dump_vocab`.

No dedicated smoke wrapper — `wrappers/smoke_tok_train_force_merges.py` covers the shared crate/pattern; the only delta here is the 126-pair literal.

## Layout

```
force_merges_combined/
├── README.md   # this file (handoff)
└── pairs.py    # re-exports force_merges' SPLIT_PATTERN / FORCED_PAIRS / FORCED_PAIRS_EXPR /
                # RECOMMENDED_VOCAB_SIZE, and defines the 126 BLOCKED_PAIRS (the only delta)
```

No crate of its own — reuses `../force_merges/` (the `rustbpe_force_merges` fork). Wrapper: `wrappers/tok_train_force_merges_combined.py`.
