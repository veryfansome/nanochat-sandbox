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

`runs/runcpu.sh` is the same shape, tuned for CPU/macOS (depth=6, 5000 iters, ~30 min on an M3 Max). Not a real comparison — the cheap end-to-end check that the overlay survives real training iterations on real data.

```bash
OVERLAY=zloss bash runs/runcpu.sh
```

### Lambda / fresh-instance flow

After SSH'ing to a fresh GPU instance with this directory cloned to `~/sandbox`:

```bash
# 1. provision (one-shot, idempotent)
bash ~/sandbox/runs/setup.sh

# 2. launch in tmux so it survives SSH disconnects
OVERLAY=zloss WANDB_RUN=zloss tmux new -s sr "cd ~/sandbox && bash runs/speedrun.sh 2>&1 | tee runs/sr.log"
```

In tmux: Ctrl-B then D to detach (training continues). Reconnect later with `tmux attach -t sr`.

For an A/B, run the baseline in a second tmux session under a different `WANDB_RUN` and `--model-tag`.

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
from dataclasses import dataclass
import os
from nanochat.gpt import GPT, GPTConfig

@dataclass
class MyIdeaConfig(GPTConfig):
    my_hyperparam: float = 0.0  # populated from env in __post_init__

    def __post_init__(self):
        if self.my_hyperparam == 0.0:
            self.my_hyperparam = float(os.environ.get("MY_HYPERPARAM", "0.1"))

class MyIdeaGPT(GPT):
    def forward(self, idx, targets=None, kv_cache=None, loss_reduction='mean'):
        # Inference: defer to base
        if targets is None:
            return super().forward(idx, targets=None, kv_cache=kv_cache, loss_reduction=loss_reduction)
        # Training: call super with targets=None to get logits, compute your loss yourself
        logits = super().forward(idx, targets=None, kv_cache=kv_cache)
        ...
        return loss
```

Important convention: idea-specific hyperparameters get **defaults** on the config subclass (read from env vars in `__post_init__`), so upstream's `GPTConfig(sequence_len=..., ...)` constructor call in `base_train.py` keeps working without passing your new fields.

### 2. The wrapper — `wrappers/train_<idea>.py`

```python
import runpy
import nanochat.gpt as g
from overlay.<idea> import MyIdeaGPT, MyIdeaConfig

g.GPT = MyIdeaGPT
g.GPTConfig = MyIdeaConfig

runpy.run_module("scripts.base_train", run_name="__main__")
```

This is the entire integration — `base_train.py`'s `from nanochat.gpt import GPT, GPTConfig` resolves to the patched classes at its import time.

### 3. The smoke test — `wrappers/smoke_<idea>.py`

Two checks (no data, no training):
- **Forward-correctness**: build a tiny baseline and a subclass with identical weights, run forward on random tokens, assert the difference equals what your math says it should.
- **Monkeypatch**: after `g.GPT = MyIdeaGPT`, assert `from nanochat.gpt import GPT` resolves to `MyIdeaGPT`.

`wrappers/smoke_zloss.py` is the working template — copy it and adapt the math check. Always run the smoke before any real training.

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

wandb logging: `--run=<wandb_run_name>` enables it; `--run=dummy` (the default) skips it.

## A-B discipline

Same pinned nanochat commit + same seed for the baseline (no overlay) and the overlay run — otherwise upstream changes confound the comparison. Record the commit hash in `results/<run>/META`:

```bash
git -C ../nanochat rev-parse HEAD > results/<run>/META
```

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

- **`overlay/zloss.py`** — `ZLossGPT(GPT)` overriding `forward()` to add `z_loss_coeff · mean(logsumexp(logits)²)` over non-ignored target positions; `ZLossConfig(GPTConfig)` with `z_loss_coeff` defaulted from `Z_LOSS_COEFF`.
- **`wrappers/train_zloss.py`** — six lines: patch `nanochat.gpt.GPT`/`GPTConfig`, `runpy.run_module("scripts.base_train")`.
- **`wrappers/smoke_zloss.py`** — verifies forward-correctness (loss diff matches the z-term to ~1e-7) and the monkeypatch substitution. Passes in <5 seconds with no data.

Design rationale: `ideas/zloss/README.md`.

## For ideas that need a *copied* `base_train.py` (harness changes)

Some ideas restructure the training loop itself — online data selection (per-token scoring + subsetting), per-depth LR scheduling (Variant B), the non-backprop track. The patch + `runpy` trick does not cover those.

For those, the recipe is to **copy `../nanochat/scripts/base_train.py` once into `wrappers/train_<idea>.py`** and edit your copy. You lose auto-sync of the harness, but the harness changes far less often than the model. Keep an upstream diff handy:

```bash
diff ../nanochat/scripts/base_train.py wrappers/train_<idea>.py
```

Pull harness improvements by hand-merging the diff after each upstream sync.

## Quick reference

| Action | Command |
|--------|---------|
| Provision (fresh machine) | `bash runs/setup.sh` (or `EXTRA=cpu bash runs/setup.sh`) |
| Verify wiring | `uv run python -m wrappers.smoke_zloss` |
| Full GPU run | `OVERLAY=zloss bash runs/speedrun.sh` |
| CPU smoke run | `OVERLAY=zloss bash runs/runcpu.sh` |
| Launch in tmux | `OVERLAY=zloss tmux new -s sr "bash runs/speedrun.sh 2>&1 \| tee runs/sr.log"` |
| Custom single-device | `uv run python -m wrappers.train_<idea> --depth=4 ...` |
| Custom distributed | `uv run torchrun --standalone --nproc_per_node=N -m wrappers.train_<idea> -- --depth=24 ...` |
| Pass idea hyperparam | `MY_VAR=value bash runs/speedrun.sh` |
| Sync upstream | `git -C ../nanochat pull && uv run python -m wrappers.smoke_<idea>` |
