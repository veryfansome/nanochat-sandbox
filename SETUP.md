# Overlay setup

How to bootstrap this overlay project from scratch, add a new overlay, and run it against an unmodified `nanochat` checkout. Per-idea READMEs in `ideas/` cover *what* each overlay does and why; this doc covers *how* the machinery is wired.

The goal is that `nanochat/` is **never edited** — upstream improvements stay `git pull`-able forever. All experiment code lives here.

For project orientation (what's implemented vs. proposed, next concrete steps, constraints not to relitigate), see [`README.md`](README.md).

## Layout

```
pyproject.toml           # nanochat-sandbox: deps + hatchling build (overlay/, wrappers/)
.venv/                   # uv-managed virtualenv
.gitignore               # .venv, __pycache__, results, *.pyc
SETUP.md                 # this file
ideas/                   # design docs (one folder per experiment)
overlay/
  __init__.py            # prepends ../nanochat to sys.path on first import
  zloss.py               # ZLossGPT subclass
  <new_idea>.py
wrappers/
  __init__.py            # imports overlay → triggers sys.path bootstrap
  train_zloss.py         # patches nanochat.gpt + runpy.run_module(scripts.base_train)
  smoke_zloss.py         # subclass + monkeypatch verification (no data, no training)
  train_<new_idea>.py
  smoke_<new_idea>.py
runs/
  setup.sh               # one-shot provisioning (uv + nanochat clone + .venv + .pth + smoke)
  speedrun.sh            # full GPU pipeline; swaps base_train → wrappers.train_$OVERLAY
  runcpu.sh              # CPU/macOS smoke run (depth=6, 5000 iters)
  lambda.sh              # Lambda Cloud REST wrapper: launch / bootstrap / ssh / terminate
  build_rustbpe.sh       # build vendored rustbpe (or a variant); see "Tokenizer variants" below
tools/
  compare_runs.py        # pull wandb metrics for N runs; side-by-side table + trajectory
  eval_tokenizer.py      # offline tokenizer eval harness (Tier 1/2 metrics, multi-tokenizer compare)
  pass_metrics.py        # tokenizer corpus deltas vs baseline (compression, dead/mangled tokens, firing buckets)
  dump_vocab.py          # dump a tokenizer's vocab (rank, corpus count, byte_len, repr) for inspection
  blocked_pairs_report.py # one-shot post-retrain report for a blocked-pairs variant: eval+pass_metrics+dumps+blocked-R firing diff
  token_contexts.py      # which words a no-space token fires in + bare-word % -> river/parasite/mangle call before blocking
rustbpe/                 # PRISTINE vendored Karpathy rustbpe (commit 9467d83); built via build_rustbpe.sh
rustbpe_variants/        # parallel-crate variants; each produces a uniquely-named Python module
results/                 # per-run logs / checkpoints / metadata (gitignored)
```

Assumed sibling layout: `nanochat/` sits at `../nanochat` relative to this directory. Both live under the same parent.

## Quickstart — `runs/setup.sh`

On any fresh machine (Lambda GPU node, local dev box, anywhere):

```bash
bash runs/setup.sh                 # GPU default
EXTRA=cpu bash runs/setup.sh       # CPU / macOS
```

Idempotent. It installs uv, clones nanochat as a sibling if missing, runs `uv sync --extra $EXTRA`, drops the `.pth` bootstrap into the venv (see below), and runs the z-loss smoke test. After a successful run, `nanochat.*` and `scripts.*` are importable from any `uv run python` invocation.

Env vars: `NANOCHAT_GIT` (default `https://github.com/karpathy/nanochat.git`), `NANOCHAT_COMMIT` (default tip), `EXTRA` (`gpu` / `cpu`).

### Manual bootstrap (alternative)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
cd <this dir>
uv sync --extra cpu     # or --extra gpu
echo "import overlay" > "$(uv run python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')/_overlay_bootstrap.pth"
uv run python -m wrappers.smoke_zloss
```

The `echo > ...pth` step is what `setup.sh` does in [4/5]. Skip it and only the `wrappers.*` modules will be able to import nanochat — plain `uv run python -m nanochat.dataset` will fail.

## Full pipeline — `runs/speedrun.sh` / `runs/runcpu.sh`

`runs/speedrun.sh` mirrors upstream `../nanochat/runs/speedrun.sh` exactly — data → tokenizer → pretrain → eval → SFT → report — with one swap: `base_train` becomes `wrappers.train_$OVERLAY` when `OVERLAY` is set.

```bash
OVERLAY=zloss WANDB_RUN=zloss bash runs/speedrun.sh         # overlay
WANDB_RUN=baseline bash runs/speedrun.sh                    # baseline (upstream)
```

Env vars: `OVERLAY` (empty = upstream baseline), `MODEL_TAG` (default `d${DEPTH}_${OVERLAY:-baseline}` — keeps baseline and overlay A/B runs from overwriting each other's checkpoints), `NPROC` (default 8), `DEPTH` (default 24), `WANDB_RUN` (default `dummy`), `NANOCHAT_BASE_DIR` (default `~/.cache/nanochat`), `FORCE_RETRAIN_TOKENIZER` (default 0 = skip if cached; upstream's `scripts/tok_train.py` retrains unconditionally, so we add a guard to avoid wasting tokenizer-training time across A/B runs).

`runs/runcpu.sh` is the same shape, tuned for CPU/macOS (depth=6, 5000 iters, ~30 min on an M3 Max). Not a real comparison — the cheap end-to-end check that the overlay survives real training iterations on real data. **Deletes the trained checkpoint by default** after `base_eval` completes (~500 MB savings per run; wandb has the curves, the report has the metrics — pass `KEEP_CHECKPOINT=1` if you want to keep it for resume / chat / further eval).

```bash
OVERLAY=zloss bash runs/runcpu.sh
```

### Lambda / fresh-instance flow

For Lambda Cloud, `runs/lambda.sh` wraps the REST API to provision, bootstrap, SSH, and tear down. No CLI install needed — just `curl` + `jq` + `ssh` + `rsync`. From your local machine:

```bash
export LAMBDA_API_KEY=...           # https://cloud.lambda.ai/api-keys
export LAMBDA_SSH_KEY=mykey         # name of an SSH key already uploaded to Lambda
                                    # (not a local file path)
export WANDB_API_KEY=...            # optional; if set, `bootstrap` will pipe it
                                    # to the host over stdin and run `wandb
                                    # login` there so training auto-authenticates

# 1. launch an 8xH100 (default; --type / --region to override). Auto-picks a
#    region with capacity. Polls until active; final stdout line is "<id> <ip>".
read ID IP < <(bash runs/lambda.sh launch | tail -1)

# 2. rsync this sandbox/ to the instance, run setup.sh, and (if WANDB_API_KEY
#    is set) seed wandb credentials on the host
bash runs/lambda.sh bootstrap "$ID"

# 3. SSH in
bash runs/lambda.sh ssh "$ID"

# 4. on the instance, launch the run in tmux so it survives SSH disconnects
OVERLAY=zloss WANDB_RUN=zloss tmux new -s sr "cd ~/sandbox && bash runs/speedrun.sh 2>&1 | tee runs/sr.log"
# Ctrl-B D to detach; `tmux attach -t sr` to reconnect.

# 5. when done — IMPORTANT, billing is per hour:
bash runs/lambda.sh terminate "$ID"
```

Other subcommands: `types [--available]`, `list`, `status <id>`. Run `bash runs/lambda.sh` (no args) to see the full usage.

For an A/B, do step 4 in a second tmux session under a different `WANDB_RUN`. `MODEL_TAG` is auto-derived from `OVERLAY` so the on-disk checkpoint dirs don't collide.

If you'd rather provision manually (web dashboard) then SSH in and run `bash ~/sandbox/runs/setup.sh` yourself, that works too — `lambda.sh` is just the curl wrapper, not a requirement.

## How the sys.path bootstrap works

`overlay/__init__.py` runs once on first import and prepends `../nanochat` to `sys.path`. Because `wrappers/__init__.py` imports `overlay`, any `python -m wrappers.X` invocation triggers the bootstrap before any `from nanochat... import ...` line in `X` ever runs.

The **`.pth` file** dropped by `runs/setup.sh` at `.venv/lib/pythonX.Y/site-packages/_overlay_bootstrap.pth` (contents: `import overlay`) extends this to *every* Python invocation in the venv. Python's `site.py` reads `.pth` files on startup and executes any line starting with `import`, so the overlay bootstrap fires before any user code. Net effect: `nanochat.*` and `scripts.*` are importable from *any* `uv run python`, not only the `wrappers/X.py` modules. That's what lets `runs/speedrun.sh` invoke `python -m nanochat.dataset`, `python -m scripts.tok_train`, etc. directly.

This is also why `runpy.run_module("scripts.base_train")` in `wrappers/train_<idea>.py` finds the upstream training script — `scripts/` is a namespace package alongside `nanochat/` in the upstream checkout, and both are reachable via the bootstrap.

## Why we don't `pip install` nanochat

Tempting first instinct: declare nanochat as a path dep in `pyproject.toml` and let uv install it editable. **This fails out of the box.** `nanochat/pyproject.toml` does not declare `[build-system]` or `[tool.setuptools] packages`, and setuptools' auto-discovery refuses because the repo has multiple top-level dirs (`nanochat/`, `dev/`, `runs/`):

```
error: Multiple top-level packages discovered in a flat-layout: ['dev', 'runs', 'nanochat'].
```

Fixing that requires editing `nanochat/pyproject.toml` — violating the pristine-checkout rule. The sys.path bootstrap is the workaround.

**Trade-off**: nanochat's runtime deps must be mirrored into `pyproject.toml` (see the `dependencies` list there). Keep loosely in sync when bumping the pinned nanochat commit — if upstream adds or upgrades a dep, mirror it here.

## Adding a new overlay

For an idea that's a **pure model-side change** (forward/loss only — MTP, z-loss, deep supervision), the recipe is three small files. Use z-loss as the template.

### 1. The model subclass — `overlay/<idea>.py`

```python
import os
from nanochat.gpt import GPT

class MyIdeaGPT(GPT):
    def __init__(self, config):
        super().__init__(config)
        self.my_hyperparam = float(os.environ.get("MY_HYPERPARAM", "0.1"))

    def forward(self, idx, targets=None, kv_cache=None, loss_reduction='mean'):
        # Inference: defer to base
        if targets is None:
            return super().forward(idx, targets=None, kv_cache=kv_cache, loss_reduction=loss_reduction)
        # Training: call super with targets=None to get logits, compute your loss yourself
        logits = super().forward(idx, targets=None, kv_cache=kv_cache)
        ...
        return loss
```

### Hyperparameter placement: attribute, not config field

The template above stores `my_hyperparam` as a plain Python attribute, **NOT** as a field on a `GPTConfig` subclass. This is deliberate and important.

`scripts/base_train.py` saves the model config to `meta_*.json` via `asdict(model.config)`. Any field on a `GPTConfig` subclass ends up in that JSON. A fresh Python process that loads the checkpoint with vanilla `GPTConfig(**kwargs)` — `scripts.base_eval`, `scripts.chat_sft`, `scripts.chat_cli`, anything downstream — then crashes with `TypeError: GPTConfig.__init__() got an unexpected keyword argument 'your_field'`. Keeping overlay hyperparameters off the config makes the saved checkpoint byte-identical to a baseline run and fully portable to the entire upstream toolchain. (z-loss hit this bug in a real run; the smoke test now has a check `(b)` that flags any overlay leaking fields into the config.)

**Use a config field only if the hyperparameter affects model *shape*** — i.e. the model literally cannot be reconstructed without it (number of extra heads, layer-band count, etc.). In that case you must *also* provide either:

- a sibling eval wrapper (`wrappers/eval_<idea>.py` that patches `GPTConfig` and `runpy`s `scripts.base_eval`), repeated for every upstream tool you want to use, or
- a pre-load shim that strips your fields from `meta_*.json` before vanilla tools touch it.

When in doubt, use the attribute pattern. Loss-only and training-dynamics knobs always fit this pattern. The trade-off is the coefficient isn't recorded in the checkpoint — capture it via `MODEL_TAG` (e.g. `d24_zloss_1e4`) and wandb config.

### Loss-shape gotcha: don't materialize `(B, T, V)` twice

If your overlay computes anything off `logits` *in addition to* the standard CE — auxiliary heads, z-loss-style regularizers, anything that touches the unembedding output — be aware that PyTorch's autograd holds saved-for-backward tensors per op. Two ops on `logits` means two `(B, T, V)` tensors live across the backward pass: the softmax `F.cross_entropy` saves internally + whatever your second op needs. This is **invisible at GPU scale** (you've got 80–141 GB HBM and the trunk dominates the time cost), but it caused observable swap memory pressure and ~7–10% throughput loss when z-loss was first written naively and run on an M4 mini.

The fix is a small custom `torch.autograd.Function` that takes `logits` as input, computes both loss terms from one `log_softmax`, and saves *only* `log_probs` for backward (an analytic backward derives both gradient components from that one saved tensor). See `overlay/zloss.py`'s `_FusedCEZLoss` for a worked example; it's ~60 lines and reduces backward-time `(B, T, V)` memory by ~50% relative to the naive two-op path. A smoke `(d)` check compares fused vs naive numerics on every smoke run to catch regressions.

This matters more for overlays that touch logits multiple ways (MTP's `k` heads, deep supervision's intermediate heads). The same fused-autograd pattern carries over: compute everything in one pass, save just `log_probs`, write the analytic backward.

### 2. The wrapper — `wrappers/train_<idea>.py`

```python
import runpy
import nanochat.gpt as g
from overlay.<idea> import MyIdeaGPT

g.GPT = MyIdeaGPT
# Do NOT patch g.GPTConfig — see "Hyperparameter placement" above. Patching
# GPTConfig pollutes the saved checkpoint with overlay-only fields, which
# breaks base_eval / chat_sft / chat_cli when they load it.

runpy.run_module("scripts.base_train", run_name="__main__")
```

This is the entire integration — `base_train.py`'s `from nanochat.gpt import GPT` resolves to the patched class at its import time.

### 3. The smoke test — `wrappers/smoke_<idea>.py`

Four required checks + one conditional (no data, no training):

- **(a) Forward-correctness**: build a tiny baseline and a subclass with identical weights, run forward on random tokens, assert the difference equals what your math says it should.
- **(b) Config-portability**: `asdict(my_idea_model.config).keys()` equals `asdict(baseline.config).keys()` — confirms no overlay-only fields leak into the saved checkpoint. **This would have caught a real bug in z-loss.** Skip only if your overlay genuinely needs to change the config (see hyperparameter-placement section above).
- **(c) Monkeypatch**: after `g.GPT = MyIdeaGPT`, assert `from nanochat.gpt import GPT` resolves to `MyIdeaGPT`.
- **(d) Fused/naive numerical equivalence** *(only if your overlay has a custom `torch.autograd.Function` fast path)*: assert loss bit-exact and gradients agree to ~1e-4 vs the naive two-op formulation. See `wrappers/smoke_zloss.py` check `(d)` for the template.
- **(e) Eval-path equivalence — REQUIRED for any overlay that adds to the loss tensor**: with a nonzero overlay coefficient, `MyIdeaGPT(..., loss_reduction='none')` and `MyIdeaGPT(..., loss_reduction='sum')` must match vanilla GPT exactly. The eval path (`nanochat/loss_eval.py`'s `evaluate_bpb`) calls the model with `loss_reduction='none'` to compute per-token bpb and expects baseline-comparable numbers. **Aux losses must be confined to `loss_reduction='mean'`** (the training path) — any contribution on the eval path inflates `val/bpb` and makes overlay runs non-comparable to baseline. See `wrappers/smoke_zloss.py` check `(e)` for the template. (This is not optional — z-loss had this bug for an entire d24 trip despite the rule being documented in `overlay/README.md`. The smoke is enforcement; the docs are not.)

`wrappers/smoke_zloss.py` is the working template — copy it and adapt the math check + the overlay-coefficient name. Always run the smoke before any real training.

### Overrides that may also be needed

Depending on the idea, also override on the subclass:

- `init_weights()` — if you added new params (extra heads, feedback matrices).
- `setup_optimizer()` — to include new params in the right groups.
- `estimate_flops()` — so MFU / tok/sec logging stays honest.

## Running a custom overlay

### Single device (CPU / MPS / 1 GPU)

```bash
uv run python -m wrappers.train_zloss \
    --depth=4 --max-seq-len=512 --device-batch-size=1 --eval-tokens=512 \
    --core-metric-every=-1 --total-batch-size=512 --num-iterations=20
```

The flags after `-m wrappers.train_zloss` are passed straight through to `scripts.base_train`'s argparse via `sys.argv` — `runpy.run_module` inherits the parent process's argv unchanged. **Do not add a separate argparse to the wrapper** — it would conflict with upstream's.

### Distributed (torchrun)

```bash
uv run torchrun --standalone --nproc_per_node=8 -m wrappers.train_zloss -- --depth=26 ...
```

Note the `--` separator (upstream convention): everything after `--` is forwarded to the script's argparse instead of being interpreted by torchrun.

torchrun spawns one Python process per rank; each independently imports `wrappers.train_zloss`, which patches its own `nanochat.gpt` module before invoking `scripts.base_train`. The patch is per-process, which is exactly what's needed.

### Idea-specific hyperparameters → env vars

Pass them as env vars to avoid argparse conflicts with the upstream parser:

```bash
Z_LOSS_COEFF=1e-4 uv run python -m wrappers.train_zloss --depth=4 ...
```

The config subclass reads the env var in `__post_init__`. Document the env-var name in the idea's README.

## Checkpoint and log location

`scripts/base_train.py` writes under `get_base_dir()` — by default `~/.cache/nanochat/` (settable via `NANOCHAT_BASE_DIR`). That keeps generated artifacts out of both the `../nanochat/` checkout (pristine) and this directory (gitignored).

Per-run results live there under `base_checkpoints/<model_tag>/`. Recommend setting `--model-tag` so baseline and overlay runs don't collide: `--model-tag=d4_zloss_smoke` vs. `--model-tag=d4_baseline`.

**Per-run archive** — `runs/runcpu.sh` and `runs/speedrun.sh` both auto-archive the durable run artifacts to `sandbox/results/$MODEL_TAG/` after `base_eval` finishes: the eval CSV, the base-model report markdowns, the checkpoint `meta.json`, the nanochat commit hash, and an `ENV` file with the relevant env vars + timestamp. This is necessary because upstream's eval CSV path is *step-keyed*, not *model_tag-keyed* (`base_model_<step>.csv`), and `nanochat.report reset` wipes `report/` at the start of every run — without the archive, the only durable record of a run's metrics besides wandb gets clobbered by the next run.

wandb logging: `--run=<wandb_run_name>` enables it; `--run=dummy` (the default) skips it. `runs/runcpu.sh` and `runs/speedrun.sh` both set `WANDB_DIR="$NANOCHAT_BASE_DIR"` so wandb's local cache lands under `~/.cache/nanochat/wandb/` alongside the other generated artifacts, rather than dropping a `wandb/` dir into the sandbox.

## A-B discipline

Same pinned nanochat commit + same seed for the baseline (no overlay) and the overlay run — otherwise upstream changes confound the comparison. `runs/runcpu.sh` and `runs/speedrun.sh` both auto-record the nanochat commit + env into `results/$MODEL_TAG/NANOCHAT_COMMIT` / `ENV` (see the "Per-run archive" note above).

## Comparing runs — `tools/compare_runs.py`

After two or more runs finish, pull their metrics from wandb and compare them side by side:

```bash
# A/B (first run is the reference for deltas)
uv run python -m tools.compare_runs d6_baseline d6_zloss

# N-way (e.g. a coefficient sweep)
uv run python -m tools.compare_runs d24_baseline d24_zloss_1e4 d24_zloss_1e3 d24_zloss_1e5

# JSON for piping into other tools / spreadsheets
uv run python -m tools.compare_runs --json d6_baseline d6_zloss > results/ab.json
```

Default project is `nanochat` (matching `base_train.py:100`); entity is auto-discovered from `~/.netrc`. Override with `--project` / `--entity` if needed. Uses the wandb credentials you already set with `wandb login`.

What it pulls (extend `SUMMARY_FIELDS` / `TRAJECTORY_FIELDS` at the top of the file):
- Summary scalars: `val/bpb` (final), `train/loss` (final), `core_metric` (if logged — disabled in runcpu by default), `train/tok_per_sec`, `train/mfu`, `total_training_time`, `total_training_flops`.
- Trajectory: `val/bpb` per eval step, with deltas vs the reference run.
- `train/loss` summary: count, max, min, last-5 mean — for quick spike-frequency checks.

If a metric isn't logged by `base_train.py` it shows as `—` rather than failing. To add a new metric, append a `(wandb_key, display_name, format_spec)` tuple to `SUMMARY_FIELDS`.

## Syncing upstream

```bash
git -C ../nanochat pull
# Then diff to check whether nanochat's runtime deps changed:
diff <(grep -E '^\s*"' ../nanochat/pyproject.toml | head -30) \
     <(grep -E '^\s*"' pyproject.toml | head -30)
```

If upstream's `nanochat.gpt` API changed (rare for the public-facing `GPT.forward` / `GPTConfig` / `setup_optimizer` surface, common for internal helpers), the overlay subclasses may need a touch-up. Run the smoke after every upstream pull — it catches API breakage fast.

## Worked end-to-end example — z-loss

What's currently checked in and verified:

- **`overlay/zloss.py`** — `ZLossGPT(GPT)` overriding `forward()` to add `z_loss_coeff · mean(logsumexp(logits)²)` over non-ignored target positions; `z_loss_coeff` is read from the `Z_LOSS_COEFF` env var at `__init__` and held as an instance attribute (no `GPTConfig` subclass — saved checkpoints load with vanilla `GPT`).
- **`wrappers/train_zloss.py`** — six lines: patch `nanochat.gpt.GPT`, `runpy.run_module("scripts.base_train")`. Does **not** patch `GPTConfig`.
- **`wrappers/smoke_zloss.py`** — verifies (a) forward math matches `baseline + z_coeff · E[lse²]` to ~1e-7, (b) the model's config has no overlay-only fields (checkpoint stays portable to upstream tools), (c) the monkeypatch substitution works. Passes in <5 seconds with no data.

Design rationale: `ideas/zloss/README.md`.

## For ideas that need a *copied* `base_train.py` (harness changes)

Some ideas restructure the training loop itself — online data selection (per-token scoring + subsetting), per-depth LR scheduling (Variant B), the non-backprop track. The patch + `runpy` trick does not cover those.

For those, the recipe is to **copy `../nanochat/scripts/base_train.py` once into `wrappers/train_<idea>.py`** and edit your copy. You lose auto-sync of the harness, but the harness changes far less often than the model. Keep an upstream diff handy:

```bash
diff ../nanochat/scripts/base_train.py wrappers/train_<idea>.py
```

Pull harness improvements by hand-merging the diff after each upstream sync.

## Tokenizer variants — `rustbpe/` + `rustbpe_variants/`

For the tokenizer-experiment track (see [`ideas/tokenizer-variants/README.md`](ideas/tokenizer-variants/README.md)) we vendor Karpathy's `rustbpe` Rust crate at [`rustbpe/`](rustbpe/) — pristine, lifted from upstream commit `9467d83` (the last pristine version before user mods on the `seed_tokens` / `force_merges_wip` branches). Variants live in [`rustbpe_variants/<name>/`](rustbpe_variants/) as fully independent crates producing uniquely-named Python modules so multiple variants coexist in one venv.

**Default state:** the venv has PyPI `rustbpe` from `uv sync`. This is functionally identical to our vendored pristine (both build from the same source). Build the vendored copy only when you want to modify the Rust source for experiments.

**Build commands:**
```bash
# Build pristine vendored rustbpe → overrides PyPI rustbpe in venv
bash runs/build_rustbpe.sh

# Build a variant (must exist at rustbpe_variants/<name>/ — e.g. force_merges, auto_tune)
VARIANT=auto_tune bash runs/build_rustbpe.sh

# Release-mode build (slower compile, faster runtime; for actual training)
RELEASE=1 bash runs/build_rustbpe.sh
```

**Reverting to PyPI:** `uv sync` reinstalls the PyPI version, overriding any locally-built copy. So the default-after-setup state is always PyPI; build commands are explicitly scoped to variant work.

**Prereqs:** Rust toolchain (`rustup` from https://rustup.rs/). `maturin` auto-installs into the sandbox venv on first build. Incremental rebuilds in dev mode complete in ~5–10s.

See [`rustbpe_variants/README.md`](rustbpe_variants/README.md) for the variant-overlay pattern (Cargo / pyproject / `#[pymodule]` renames; matching `wrappers/tok_train_<name>.py` to alias `sys.modules['rustbpe']` before invoking `scripts.tok_train`).

## Quick reference

| Action | Command |
|--------|---------|
| Provision (fresh machine, manual) | `bash runs/setup.sh` (or `EXTRA=cpu bash runs/setup.sh`) |
| Provision Lambda instance + bootstrap | `read ID IP < <(bash runs/lambda.sh launch \| tail -1); bash runs/lambda.sh bootstrap "$ID"` |
| Terminate Lambda instance | `bash runs/lambda.sh terminate <id>` |
| Compare runs (A/B from wandb) | `uv run python -m tools.compare_runs RUN_A RUN_B [RUN_C ...]` |
| Evaluate tokenizer(s) offline | `uv run python -m tools.eval_tokenizer [--tokenizers ours,gpt2,gpt4,/path/...] [--full]` |
| Tokenizer corpus deltas vs baseline | `uv run python -m tools.pass_metrics --variant ~/.cache/nanochat-variants/<v>/tokenizer` |
| Dump a tokenizer's vocab | `uv run python -m tools.dump_vocab --tokenizer <dir> --output /tmp/<v>_vocab.txt` |
| Post-retrain report (blocked-pairs variant) | `uv run python -m tools.blocked_pairs_report` (defaults to auto_tune) |
| Classify a token before blocking it | `uv run python -m tools.token_contexts led red ated` (firing context + bare-word %) |
| Build vendored rustbpe (pristine) | `bash runs/build_rustbpe.sh` |
| Build a rustbpe variant | `VARIANT=<name> bash runs/build_rustbpe.sh` |
| Verify wiring | `uv run python -m wrappers.smoke_zloss` |
| Full GPU run | `OVERLAY=zloss bash runs/speedrun.sh` |
| CPU smoke run | `OVERLAY=zloss bash runs/runcpu.sh` |
| Launch in tmux | `OVERLAY=zloss tmux new -s sr "bash runs/speedrun.sh 2>&1 \| tee runs/sr.log"` |
| Custom single-device | `uv run python -m wrappers.train_<idea> --depth=4 ...` |
| Custom distributed | `uv run torchrun --standalone --nproc_per_node=N -m wrappers.train_<idea> -- --depth=24 ...` |
| Pass idea hyperparam | `MY_VAR=value bash runs/speedrun.sh` |
| Sync upstream | `git -C ../nanochat pull && uv run python -m wrappers.smoke_<idea>` |
