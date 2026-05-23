# nanochat-sandbox

Overlay experiments on top of Karpathy's [nanochat](https://github.com/karpathy/nanochat), kept as a separate project so nanochat itself stays pristine and `git pull`-able.

## What this is

Explore ideas to improve nanochat's **capability** and **sample efficiency** under the constraint that **model scale and training dataset are fixed**. Gains must come from the training objective, architecture, or training dynamics — not from "bigger model" or "more data."

Every experiment is a small overlay: a `GPT` subclass plus a wrapper that monkeypatches `nanochat.gpt.GPT` and `runpy`s the upstream training script. Nanochat is never edited; loaded via a sys.path bootstrap. Every upstream improvement is one `git pull ../nanochat` away.

## Where to look

| You want to… | Read… |
|--------------|-------|
| Know what's done, in flight, and next | [`STATUS.md`](STATUS.md) |
| Set up on a fresh machine, run smoke, launch a full speedrun | [`SETUP.md`](SETUP.md) |
| Pick what to work on; browse the idea catalog | [`ideas/README.md`](ideas/README.md) |
| Read the design for a specific idea | `ideas/<idea>/README.md` |
| See an implemented overlay's pattern | [`overlay/zloss.py`](overlay/zloss.py), [`wrappers/train_zloss.py`](wrappers/train_zloss.py), [`wrappers/smoke_zloss.py`](wrappers/smoke_zloss.py) |

## Layout

```
README.md        # this file (auto-loaded as AGENTS.md / CLAUDE.md via symlink)
SETUP.md         # provisioning, running, adding overlays — mechanics
STATUS.md        # implementation progress + currently in flight + next steps
ideas/           # design docs, one folder per experiment (+ index)
overlay/         # model subclasses (one .py per implemented idea)
wrappers/        # Python entrypoints: patch nanochat.gpt + runpy base_train
runs/            # shell pipelines: setup.sh, speedrun.sh, runcpu.sh, lambda.sh
tools/           # analysis utilities: compare_runs.py (wandb A/B), ...
results/         # per-run logs / metadata (gitignored)
pyproject.toml   # nanochat-sandbox project (nanochat NOT installed as a pkg)
```

Sibling layout assumed: `../nanochat/` next to `sandbox/`. Never edited; loaded via the `.pth` bootstrap dropped by `runs/setup.sh`.

## Constraints (settled — don't relitigate)

- **Scale & data are fixed.** No bumping `--depth`, no adding tokens, no new datasets.
- **nanochat stays pristine.** Loaded via sys.path bootstrap; deliberately NOT pip-installed (its pyproject doesn't declare `[build-system]`; setuptools auto-discovery fails on the flat layout). See SETUP.md "Why we don't pip install nanochat".
- **Out of scope** (see [`ideas/README.md`](ideas/README.md) "Ruled out" for rationale): knowledge distillation, RLVR, test-time compute, optimizer swaps, attention/norm micro-tweaks, MLA, µP, regularization/dropout.

## Conventions

- **Docs**: Markdown prose unwrapped (one line per paragraph/bullet).
- **A/B discipline**: same pinned nanochat commit + same seed for baseline vs overlay. `--model-tag` is auto-derived in `runs/*.sh` to `d${DEPTH}_${OVERLAY:-baseline}` so runs don't clobber each other.
- **Idea-specific hyperparameters**: passed as env vars (e.g. `Z_LOSS_COEFF=1e-4`), read at `__init__` and stored as an instance attribute on the overlay subclass — **not** added to `GPTConfig`, since that breaks downstream load via `scripts.base_eval` / `scripts.chat_sft` / `scripts.chat_cli`. See SETUP.md "Hyperparameter placement: attribute, not config field".
- **Smoke before training**: every new overlay gets a `wrappers/smoke_<idea>.py` that verifies (a) the loss math, (b) config-portability (no overlay-only fields leak into the saved checkpoint), (c) the monkeypatch substitution. Runs in <5 seconds with no data. Always run before any real training.
- **Prefer existing tools over inline scripts**: e.g. use `tools/compare_runs.py` for wandb comparisons. Extend the tool when it's missing a feature, don't bypass it.

## Glossary

- **overlay**: a model subclass (e.g. `overlay/zloss.py`) that changes the loss/forward of `GPT`.
- **wrapper**: a Python entrypoint (e.g. `wrappers/train_zloss.py`) that monkeypatches `nanochat.gpt.GPT` and `runpy`s the upstream `scripts.base_train`.
- **harness**: the upstream `scripts/base_train.py` training loop. Most overlays don't touch it; a few (online data selection, non-backprop, layer-wise-LR variant B) need a copied version diffed against upstream.
- **`.pth` bootstrap**: a one-line file (`import overlay`) dropped in `.venv/site-packages/` by `runs/setup.sh`. Python's `site.py` reads it on startup, which prepends `../nanochat` to `sys.path`, so `nanochat.*` and `scripts.*` are importable from any `uv run python`.
