# nanochat-sandbox

Overlay experiments on top of Karpathy's [nanochat](https://github.com/karpathy/nanochat), kept as a separate project so nanochat itself stays pristine and `git pull`-able.

## What this is

Explore ideas to improve nanochat's **capability** and **sample efficiency** under the constraint that **model scale and training dataset are fixed**. Gains must come from the training objective, architecture, or training dynamics — not from "bigger model" or "more data."

Every experiment is a small overlay: a `GPT` subclass plus a wrapper that monkeypatches `nanochat.gpt.GPT` and `runpy`s the upstream training script. Nanochat is never edited; loaded via a sys.path bootstrap. Every upstream improvement is one `git pull ../nanochat` away.

## Where to look

| You want to… | Read… |
|--------------|-------|
| Set up on a fresh machine, run smoke, launch a full speedrun | [`SETUP.md`](SETUP.md) |
| Pick what to work on next; see the curated sequencing | [`ideas/README.md`](ideas/README.md) |
| Read the design for a specific idea | `ideas/<idea>/README.md` |
| See the pattern implemented end-to-end | [`overlay/zloss.py`](overlay/zloss.py), [`wrappers/train_zloss.py`](wrappers/train_zloss.py), [`wrappers/smoke_zloss.py`](wrappers/smoke_zloss.py) |

## Layout

```
SETUP.md         # provisioning, running, adding overlays — mechanics
README.md        # this file
ideas/           # design docs, one folder per experiment (+ index)
overlay/         # model subclasses (one .py per implemented idea)
wrappers/        # Python entrypoints: patch nanochat.gpt + runpy base_train
runs/            # shell pipelines: setup.sh, speedrun.sh, runcpu.sh
results/         # per-run logs / metadata (gitignored)
pyproject.toml   # nanochat-sandbox project (nanochat NOT installed as a pkg)
```

Sibling layout assumed: `../nanochat/` next to `sandbox/`. Never edited; loaded via the `.pth` bootstrap dropped by `runs/setup.sh`.

## Current state

| Idea | Designed | Implemented | Smoke ✓ | Real-data ✓ |
|------|:--------:|:-----------:|:-------:|:-----------:|
| [z-loss](ideas/zloss/README.md) | ✓ | ✓ | ✓ | — |
| [MTP](ideas/mtp/README.md) | ✓ | — | — | — |
| [Deep supervision](ideas/deep-supervision/README.md) | ✓ | — | — | — |
| [Differential attention](ideas/diff-attention/README.md) | ✓ | — | — | — |
| [Online data selection + batch-size tuning](ideas/online-data-selection/README.md) | ✓ | — | — | — |
| [Layer-wise LR](ideas/layerwise-lr/README.md) | ✓ | — | — | — |
| [Non-backprop (DFA → block-local)](ideas/non-backprop/README.md) | ✓ (research track) | — | — | — |

## Next concrete steps

1. **Validate z-loss on real data.** The mechanism is smoke-verified locally; the next milestone is a real training run. Cheap: `OVERLAY=zloss bash runs/runcpu.sh` (~30 min on M3 Max, single CPU). Real: `OVERLAY=zloss bash runs/speedrun.sh` on a multi-GPU node. Compare `val_bpb` and CORE to a baseline run on the same pinned nanochat commit + seed.
2. **Build the next overlay.** Per [`ideas/README.md`](ideas/README.md) sequencing, **MTP** is the highest-leverage next item (capability + sample efficiency, pure model-side). Follow SETUP.md's "Adding a new overlay" recipe — three files plus a smoke. Compose with z-loss into a single `MTPGPT` subclass.
3. Then per the sequencing: deep supervision (folds into MTP subclass), differential attention, online data selection. Non-backprop is a separate research track.

## Constraints (settled — don't relitigate)

- **Scale & data are fixed.** No bumping `--depth`, no adding tokens, no new datasets.
- **nanochat stays pristine.** Loaded via sys.path bootstrap; deliberately NOT pip-installed (its pyproject doesn't declare `[build-system]`; setuptools auto-discovery fails on the flat layout). See SETUP.md "Why we don't pip install nanochat".
- **Out of scope** (see [`ideas/README.md`](ideas/README.md) "Ruled out" for rationale): knowledge distillation, RLVR, test-time compute, optimizer swaps, attention/norm micro-tweaks, MLA, µP, regularization/dropout.

## Conventions

- **Docs**: Markdown prose unwrapped (one line per paragraph/bullet).
- **A/B discipline**: same pinned nanochat commit + same seed for baseline vs overlay. `--model-tag` is auto-derived in `runs/*.sh` to `d${DEPTH}_${OVERLAY:-baseline}` so runs don't clobber each other.
- **Idea-specific hyperparameters**: passed as env vars (e.g. `Z_LOSS_COEFF=1e-4`), defaulted in the config subclass via `__post_init__` to avoid argparse conflicts with the upstream parser. Documented in each idea's README.
- **Smoke before training**: every new overlay gets a `wrappers/smoke_<idea>.py` that verifies (a) the loss math (b) the monkeypatch substitution. Runs in <5 seconds with no data.

## Glossary

- **overlay**: a model subclass (e.g. `overlay/zloss.py`) that changes the loss/forward of `GPT`.
- **wrapper**: a Python entrypoint (e.g. `wrappers/train_zloss.py`) that monkeypatches `nanochat.gpt.GPT` and `runpy`s the upstream `scripts.base_train`.
- **harness**: the upstream `scripts/base_train.py` training loop. Most overlays don't touch it; a few (online data selection, non-backprop, layer-wise-LR variant B) need a copied version diffed against upstream.
- **`.pth` bootstrap**: a one-line file (`import overlay`) dropped in `.venv/site-packages/` by `runs/setup.sh`. Python's `site.py` reads it on startup, which prepends `../nanochat` to `sys.path`, so `nanochat.*` and `scripts.*` are importable from any `uv run python`.
