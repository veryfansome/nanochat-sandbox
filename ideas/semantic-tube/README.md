# Semantic Tube Prediction (STP)

Status: **implemented; smoke passing (incl. under `torch.compile`); not yet A/B'd.** Pure model-side overlay (forward/loss) + a hidden-state hook. **Key caveat: STP is published as a *fine-tuning* regularizer, not a pretraining one — see "Regime mismatch" below. This is the load-bearing open question, not a detail.**
Target repo: `nanochat` (Karpathy) — kept **pristine**, never edited.

Paper: Huang, LeCun, Balestriero, *Semantic Tube Prediction: Beating LLM Data Efficiency with JEPA*, [arXiv 2602.22617](https://arxiv.org/abs/2602.22617) (ICML 2026). Code: [github.com/galilai-group/llm-jepa](https://github.com/galilai-group/llm-jepa) (`stp.py`, `--linear=random_span`).

## Components

- [`../../overlay/stp.py`](../../overlay/stp.py) — `STPGPT(GPT)`. Training path (`loss_reduction='mean'`) returns `CE + STP_COEFF · (1 − mean cos(h_t−h_r, h_r−h_s))` over randomly sampled triples `s<r<t`; eval path (`loss_reduction != 'mean'`) returns plain CE. Hidden states captured via a forward pre-hook on `lm_head`.
- [`../../wrappers/train_stp.py`](../../wrappers/train_stp.py) — patches `nanochat.gpt.GPT` then `runpy.run_module("scripts.base_train")`. Does **not** patch `GPTConfig`.
- [`../../wrappers/smoke_stp.py`](../../wrappers/smoke_stp.py) — verifies (a) forward math, (b) config-portability, (c) monkeypatch, (d) hook fidelity + STP gradient reaches the trunk, (e) eval-path CE purity, (f) `torch.compile` equivalence (the training forward is compiled — base_train.py:246), (g) span RNG stays live under compile, (h) the STP_DIAG instrument + held-out stash + log enrichment, (i) the diag stash commits under compile. Passes in a few seconds with no data.

Run the smoke: `uv run python -m wrappers.smoke_stp` (from a worktree, prefix `PYTHONPATH=<path-to-nanochat>` since the `.pth` bootstrap resolves `../nanochat` relative to the main checkout, not the worktree).

Env knobs (read at `__init__`, stored as instance attributes — never `GPTConfig` fields): `STP_COEFF` (λ, default `0.02`), `STP_SPANS` (triples sampled per row, default `1`), `STP_MAX_SPAN` (span window width, default `256`; set ≥ `sequence_len` for whole-row sampling), `STP_DIAG` (`0`/`1`, default `0` — the diagnostic instrument, see below). Capture them in `MODEL_TAG` (e.g. `d24_stp_002`).

## Diagnostics (`STP_DIAG=1`)

Because a benefit should surface in **CORE, not val/bpb** (STP shapes representations, it doesn't optimize next-token likelihood — bpb is expected to be a wash or slightly worse), the loss curve alone can't tell "working" from "collapsing." `STP_DIAG=1` adds a geometric instrument that distinguishes three states, computed on **held-out** (val) batches and logged onto base_train's existing `val/bpb` step:

- `stp/tube_cos` — **the primary straightening readout.** Cos of two consecutive length-*d* velocity segments at span *d* ≈ `STP_MAX_SPAN`/3 — the deterministic, equal-span analog of STP's actual objective (`1 − cos` over random triples) at the scale STP operates on. Rises toward 1 as long-range trajectories straighten. Deterministic (no RNG → can't perturb training, reproducible across evals).
- `stp/vel_cos` — the *adjacent*-position velocity cosine (span 1). A more-local, weaker straightening proxy; keep it as the short-scale reference but read `tube_cos` for "is STP acting."
- `stp/hidden_var` — mean per-dim variance of the (post-norm) hidden. → 0 under *constant* collapse (only a fall toward 0 is informative — it's ~unit-bounded by RMSNorm).
- `stp/eff_rank` — participation ratio of the hidden-covariance eigenvalues. → 1 under *directional* collapse; large when representations span many dimensions.
- `stp/hidden_rms` — trivially ~1.0 (post-norm capture); sanity only.

The reading that means "STP works, healthily": `stp/tube_cos` climbing while `stp/eff_rank` and `stp/hidden_var` hold steady. Straightening *with* `eff_rank`→1 or `hidden_var`→0 is the collapse failure the paper never stress-tested from scratch. **All of this needs the matched `STP_COEFF=0` baseline to attribute anything to STP** — the same diagnostics log on a `STP_COEFF=0` run (the eval-path stash is independent of the coefficient), giving a clean control that differs only in the STP gradient.

Mechanics (no harness edit): the overlay stashes the held-out hidden on the eval path (`STP_DIAG=1` only); `wrappers/train_stp.py` patches `wandb.init` so the run's `.log` enriches the `val/bpb` entry via `overlay.stp.enrich_log_data`. Diagnostics are computed in **eager** (outside the compiled forward) and cost nothing when off. Enable with `STP_DIAG=1` alongside the other env vars.

## Goal

Add a JEPA-style geometric prior on the model's own hidden-state trajectory, on top of the standard next-token loss, to improve sample efficiency — without changing model scale or training data. No paired views required (unlike LLM-JEPA / VL-JEPA), so it applies to a plain next-token stream, which is what makes it explorable in nanochat at all.

## What it is

STP rests on the **Geodesic Hypothesis**: as a model processes a token sequence, its hidden state traces a path that is *locally linear* — a near-straight geodesic on a smooth semantic manifold. STP adds a loss that keeps that path straight: consecutive velocity vectors should point the same way, confining the trajectory to a "tube" around the geodesic.

Total loss (the paper uses `γ=1`):

```
L = L_NTP + λ · L_STP
```

### (1) Loss form — pinned

For three token indices `s < r < t` drawn along the sequence, with last-layer hidden states `h_s, h_r, h_t`:

```
L_STP = 1 − cos( h_t − h_r ,  h_r − h_s )
```

`h_r − h_s` and `h_t − h_r` are two consecutive **velocity** (difference) vectors; the loss minimizes the angle between them, enforcing collinearity. Note the indices are **arbitrary spans, not adjacent tokens** (`t` need not be `r+1`) — STP straightens the trajectory over long ranges, not just between neighbors. The repo carries ablation variants under the same code: `--linear=curvature` (a second-difference `‖h_{t+1}−2h_t+h_{t-1}‖` penalty), `--linear=e2e` (the two-view LLM-JEPA path), and `random_span_mask` (mask-recover). The paper's headline method is `random_span`, the form above.

### (2) Which layer — pinned

**Final layer only.** In the repo, `random_span_layer` defaults to `-1` — HF's `hidden_states[-1]`, i.e. the post-final-norm state that feeds the LM head. Configurable in principle (any layer), but all headline results use the last.

nanochat analog: the input to `self.lm_head` — the tensor `x` right after the final `x = norm(x)` (`nanochat/gpt.py:465`), consumed at `gpt.py:469`. See "Overlay design" for how to grab it.

### (3) Anti-collapse — pinned (and this is the risk)

**There is no explicit anti-collapse mechanism.** No variance/covariance regularizer, no stop-gradient, no EMA target, no normalization beyond the cosine's inherent scale-invariance. The paper argues collapse is prevented by:

- **The next-token loss** — `L_NTP` keeps hidden states discriminative (distinct tokens must stay separable to be predicted), so they can't all fold to a constant.
- **An ODE-uniqueness argument** — if the token-to-token map is smooth, trajectories from distinct prompts can't intersect (Picard–Lindelöf), preserving diversity.

This holds in the paper because it always starts from a **well-trained model** (see below): representations are already meaningful and `L_NTP` is already low and informative. Whether the same argument survives **pretraining from random init** — where early `L_NTP` is high and uninformative and there's no explicit guard — is untested. A straightening term has a trivial degenerate minimum (drive every velocity to one shared direction, or shrink the hidden-state variance), and cosine-only loss does nothing to stop the latter. **This is the first thing to check on a from-scratch run.**

### Hyperparameters — pinned

- `λ` typically in `[0.01, 0.08]`; per-dataset values: NL-RX-SYNTH 0.02, NL-RX-TURK 0.04, GSM8K 0.005, Spider 0.04, NQ-Open 0.16, HellaSwag 0.02. Optional linear warmup of `λ` from `min_lbd` to `λ`.
- One span per sequence (`random_span_times=1`), indices sampled among non-padding tokens; several sampling schemes (non-uniform default, uniform-over-spans, draw-both).
- Reduction: `L_STP = 1 − mean(cos)` over the batch (optional length-weighted mean).

### The 16× claim — the paper's own headline, unverified here

"With `L_NTP + L_STP`, accuracy matches full-dataset standard fine-tuning using only 1/16 of the training data" — **on NL-RX-SYNTH** (natural-language → regex, a narrow synthetic task), fine-tuning Llama-3 1B, with consistent-but-smaller gains reported at 3B/8B and on other task datasets (NL-RX-TURK, GSM8K, Spider, NQ-Open, HellaSwag). Treat 16× as benchmark-specific and as the paper's claim, not a result to expect in our regime. There is independent commentary on STP's loss-unbiasedness worth reading before over-investing (arxiviq / ding.ee writeups surfaced in search).

## Regime mismatch — the central question for us

**STP as published is a supervised fine-tuning method** (`stp.py` fine-tunes pretrained instruct models on task datasets; the repo doc states it "applies during fine-tuning phases, not pretraining"). The Geodesic Hypothesis is a claim about the geometry of an *already-trained* representation space. The sandbox's center of gravity is the **base_train speedrun** — pretraining from scratch. These are different regimes, and STP's evidence lives entirely in the first.

Two ways to explore it, and they answer different questions:

- **A — SFT overlay (`scripts.chat_sft`).** Faithful to the paper's actual claim. Reproduce first: does the mechanism transfer to nanochat's tokenizer / model / task data at all? Lower risk (starts from a trained base, so the anti-collapse argument applies), but off the sandbox's main A/B rail and doesn't directly answer "does this improve pretraining sample efficiency."
- **B — pretraining overlay (`scripts.base_train`).** The strategically interesting test and the natural overlay slot, but a genuine extrapolation beyond the paper: collapse risk is highest here, and the geodesic prior may fight the model while it's still forming representations.

Recommended sequencing (consistent with "reproduce before trusting" and "no premature speedrun"): **A before B.** Confirm the mechanism reproduces in our stack on the paper's own regime before spending a pretraining speedrun on the riskier extrapolation. Do not treat B's outcome as validation-or-refutation of the paper — B is a new hypothesis the paper never tested.

## Why it fits the overlay

Pure loss change — no harness edit, rides the monkeypatch + `runpy` wrapper, same category as z-loss / MTP / deep supervision. **One difference from z-loss:** z-loss works off the `logits` that `GPT.forward` already returns; STP needs the *hidden states*, which `forward` does **not** return. So the overlay grabs them via a forward pre-hook rather than reimplementing `forward` (still zero harness edits — see below).

## Overlay design (as implemented)

- **Grab the hidden state with a hook.** A forward pre-hook registered on `self.lm_head` in `__init__` stashes its input (`args[0]`, shape `(B, T, C)`, the post-final-norm hidden) into `self._stp_hidden`. The overlay's `forward` calls `super().forward(idx, targets=None)` to get logits (the hook fires during that call), then computes CE from the logits and `L_STP` from `self._stp_hidden`. No copy of nanochat's `forward`; upstream trunk changes stay transparent. **Verified to work under `torch.compile`** (base_train compiles the model at :246; smoke `(f)` shows compiled == eager loss/grad to ~1e-9), so the set-in-hook / read-in-forward pattern survives Dynamo.
- **Local span window (the document-boundary fix).** nanochat packs many documents per `T≈2048` row, so uniform-over-row triples would almost all straddle document boundaries — connecting unrelated text, which the geodesic hypothesis says nothing about (the paper never hits this; it fine-tunes short single examples). `_sample_triples` draws `s<r<t` within a bounded window of width `STP_MAX_SPAN` (default 256), keeping spans local and more likely intra-document — also more faithful to the *local* linearity claim. This is a heuristic, not true boundary-respecting masking (it does not read BOS positions); set `STP_MAX_SPAN ≥ sequence_len` to recover whole-row sampling. Packed rows have no padding, so every position is a real trajectory point (no attention-mask gating needed). `STP_SPANS` (default 1, the paper's per-sequence count) controls triples per row.
- **Hyperparameters via env, not config.** `STP_COEFF`/`STP_SPANS`/`STP_MAX_SPAN` are read from the environment at `__init__` and stored as plain instance attributes — **not** `GPTConfig` fields. Adding a config field pollutes `meta_*.json` and crashes downstream loaders (`scripts.base_eval` / `chat_sft` / `chat_cli`) with `TypeError: GPTConfig.__init__() got an unexpected keyword argument`. The saved checkpoint stays config-identical to baseline (the hook and `_stp_hidden` are not in `state_dict`); capture λ via `MODEL_TAG` and wandb. Smoke `(b)` enforces this.
- **Eval-path purity.** On the eval path (`loss_reduction != 'mean'`, used by `nanochat/loss_eval.py` for `val/bpb`), `forward` returns **plain CE only** — adding `L_STP` there inflates bpb and breaks baseline comparability (this exact bug rode zloss through a whole d24 trip). The guard also drops STP when `stp_coeff == 0` or the window is too short to form a triple. Smoke `(e)` verifies `loss_reduction='none'`/`'sum'` match vanilla GPT exactly.

## Expected cost

**Cheaper than z-loss/MTP.** STP touches hidden states `(B, T, C)`, not logits `(B, T, V)` — so it avoids the "materialize `(B,T,V)` twice" memory trap entirely (`C ≪ V`). The extra work is a handful of gathers and one cosine over the sampled spans, plus holding the already-live `(B,T,C)` hidden for backward. Expect near-zero throughput cost; confirm on runcpu. Inference path unchanged (no hook contribution when `targets is None`).

## Smoke coverage

`wrappers/smoke_stp.py` verifies: (a) forward math — `CE + coeff·(1−mean cos)` over the same sampled triples, RNG-controlled; (b) config-portability — env-read instance attrs, no extra `GPTConfig` fields; (c) monkeypatch substitutes the class; (d) hook fidelity — the lm_head pre-hook captures exactly the post-norm hidden (Δ=0 vs a manual forward) and the STP term moves trunk gradients (coeff-off-vs-on delta on `wte`); (e) eval-path equivalence — nonzero λ, `loss_reduction='none'`/`'sum'` match vanilla GPT exactly; (f) `torch.compile` — with a deterministic single-triple config (T=3, window=3), compiled loss/grad equal eager (skips gracefully if the inductor CPU toolchain is absent). All pass in a few seconds with no data.

## Testing / A-B

- Same pinned nanochat commit + same seed as baseline; `runcpu.sh` maps `OVERLAY=stp` → `wrappers/train_stp.py` and tags the run `d6_stp` automatically.
- **First:** mechanism check on `runs/runcpu.sh` (d6) — confirms the overlay survives a real training loop and, critically, **watch for collapse** via the `STP_DIAG` instrument (`stp/vel_cos` should climb while `stp/hidden_var` and `stp/eff_rank` hold). runcpu can't resolve a real capability delta (< ~0.05 bpb is noise), but it *can* catch collapse and wiring bugs. Run it with a real wandb name and diagnostics on:
  ```bash
  WANDB_RUN=d6_stp_002 STP_COEFF=0.02 STP_DIAG=1 MODEL_TAG=d6_stp_002 OVERLAY=stp bash runs/runcpu.sh
  ```
- **Then:** the regime decision above — reproduce on an SFT-like setup (A) before committing a pretraining speedrun (B).
- Metrics: **CORE is the primary "does it help" channel** (representation quality → downstream capability), read at large-N tasks and calibrated to ~fm-scale; **val/bpb expected to be a wash or slightly worse** (STP doesn't optimize likelihood) — watch it only for a large regression. Plus the `STP_DIAG` collapse probe. Sweep `λ ∈ {0.005, 0.02, 0.08}` per the paper's range. Run the real A/B with `STP_DIAG=0` for clean throughput; use a paired `STP_DIAG=1` run for the diagnostic curves.

## Open questions

- **Collapse under from-scratch pretraining** — does CE alone hold, or is an explicit variance/covariance guard needed? (Highest-priority check; the paper never faced this.)
- **Regime** — does the effect exist at all in pretraining, or only in fine-tuning where the paper measured it?
- **Document-boundary spans** in packed sequences — v1 mitigates with a bounded `STP_MAX_SPAN` window (heuristic locality); does true boundary-respecting masking (read BOS positions) matter, and is 256 the right width? Sweep `STP_MAX_SPAN`.
- **Layer** — v1 is last-layer only (the paper's `random_span_layer=-1`). Does an intermediate layer help (cf. deep supervision)? Not yet implemented — would hook a transformer block instead of `lm_head`.
- **λ and warmup** — the paper's `[0.01, 0.08]` is fine-tuning-calibrated; the pretraining sweet spot is unknown, and a warmup (small λ early, when representations are unformed) may be the natural mitigation for the collapse risk.

## References

- Huang, LeCun, Balestriero 2026 — *Semantic Tube Prediction* ([2602.22617](https://arxiv.org/abs/2602.22617)), ICML 2026. Code: [galilai-group/llm-jepa](https://github.com/galilai-group/llm-jepa).
- Related in `../../JEPA.md` (sibling `jepa/`): §7 (LLM-JEPA, VL-JEPA — the paired-view predecessors STP drops the pairing from) and Temporal Straightening ([2603.12231](https://arxiv.org/abs/2603.12231), the video-planning analog of the same straightening idea).
