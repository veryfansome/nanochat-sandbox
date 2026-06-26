"""
Surface-factoring training launcher (harness change).

Patches nanochat's GPT + the two best-fit dataloaders + evaluate_bpb to their
surface-factoring variants, then runpys `scripts.base_train` UNCHANGED. The
surface streams (space bit + per-word-start cap bits/mask) flow out-of-band via
`overlay.surface_context`: the surface dataloader stashes the current batch's
streams there, and the patched `SurfaceFactoringGPT.forward` — invoked by the
upstream `model(x, y)` — reads them back. (See overlay/surface_context.py for why
the strictly-sequential upstream loop makes this safe.)

PREREQUISITE: the surface-only (spaceless + case-fold, NO force_merges phrases)
tokenizer must already be trained into `$NANOCHAT_BASE_DIR/tokenizer/` so that
`get_tokenizer()` / `get_token_bytes()` resolve to the surface vocab and base-
token byte lengths. Run `wrappers.tok_train_surface` first (or use
`runs/runcpu_surface.sh`, which sequences both).

Env:
  SURFACE_LAMBDA  surface-loss weight λ for TRAINING (default 0.5; eval bpb is λ-free)
  SURFACE_KMAX    cap-slot width (default 1 — surface-only tokens are single words)

Usage (same flags as scripts.base_train):
    uv run python -m wrappers.train_surface_factoring --depth=6 --num-iterations=5000 ...
"""
import os
import runpy
import sys

import nanochat.gpt as gpt_mod
import nanochat.dataloader as dl_mod
import nanochat.loss_eval as le_mod
from nanochat.tokenizer import get_tokenizer

from tools.stack_fm_spaceless import SPACELESS_BODY
from overlay.surface_tokenizer import SurfaceTokenizer
from overlay.surface_dataloader import (
    surface_data_loader_with_state, surface_data_loader, surface_evaluate_bpb,
)
from overlay.surface_factoring import SurfaceFactoringGPT

# --- 1) cap-slot width: surface-only tokens are single words (no phrases), so each
#        token owns at most one word-start → K_max = 1 (the cap is a single bit). ----
KMAX = int(os.environ.get("SURFACE_KMAX", "1"))
os.environ["SURFACE_KMAX"] = str(KMAX)            # SurfaceFactoringGPT.__init__ reads this

# --- 2) build the SurfaceTokenizer over the saved surface encoding -----------
_tok = get_tokenizer()
assert _tok.get_vocab_size() == _tok.enc.n_vocab
SURFACE_TOK = SurfaceTokenizer(_tok.enc, SPACELESS_BODY, kmax=KMAX)
print(f"[surface] tokenizer vocab={SURFACE_TOK.n_vocab:,}, K_max={KMAX}, "
      f"λ={os.environ.get('SURFACE_LAMBDA', '0.5')}")

# --- 2.5) disable base_train's in-training CORE metric + sampling ------------
# Both run the STANDARD scorer / generate on the surface model with base-only inputs
# (no surface streams) → out-of-distribution → a misleading `core_metric` and broken
# samples in WandB. The valid surface CORE is the POST-training base_eval_surface;
# surface-aware generation is still TODO. Default both OFF (the caller can override by
# passing either flag explicitly, e.g. once a surface-aware in-training scorer exists).
if not any(a.startswith("--core-metric-every") for a in sys.argv):
    sys.argv.append("--core-metric-every=-1")
if not any(a.startswith("--sample-every") for a in sys.argv):
    sys.argv.append("--sample-every=-1")
print("[surface] in-training CORE + sampling DISABLED (OOD for the surface model; "
      "post-training base_eval_surface is the valid CORE)")

# --- 3) patch GPT + dataloaders + bpb BEFORE base_train imports them ---------
gpt_mod.GPT = SurfaceFactoringGPT
le_mod.evaluate_bpb = surface_evaluate_bpb

# Persist K_max/λ in the checkpoint meta (a SIBLING of model_config — never inside
# it, or GPTConfig(**) would TypeError) so the eval path can rebuild SurfaceFactoringGPT.
import nanochat.checkpoint_manager as cm_mod  # noqa: E402
_orig_save_checkpoint = cm_mod.save_checkpoint


def _save_with_surface(checkpoint_dir, step, model_data, optimizer_data, meta_data, rank=0):
    meta_data = {**meta_data, "surface_config": {
        "K_max": KMAX, "lambda": float(os.environ.get("SURFACE_LAMBDA", "0.5"))}}
    return _orig_save_checkpoint(checkpoint_dir, step, model_data, optimizer_data, meta_data, rank)


cm_mod.save_checkpoint = _save_with_surface


def _train_loader(tokenizer, B, T, split="train", device="cuda", resume_state_dict=None, **kw):
    # ignore the standard `tokenizer`; use the surface tokenizer captured above
    return surface_data_loader_with_state(SURFACE_TOK, B, T, split, device,
                                          resume_state_dict=resume_state_dict)


def _val_loader(tokenizer, B, T, split="val", device="cuda", **kw):
    return surface_data_loader(SURFACE_TOK, B, T, split, device)


dl_mod.tokenizing_distributed_data_loader_with_state_bos_bestfit = _train_loader
dl_mod.tokenizing_distributed_data_loader_bos_bestfit = _val_loader

# --- 4) raise the dynamo recompile limit -------------------------------------
# nanochat torch.compiles the fused AdamW step (nanochat/optim.py:adamw_step_fused)
# and guards on each parameter's shape. The surface heads/embeddings add several NEW
# AdamW parameter shapes (space + cap embeddings + heads; at K_max=1 the cap and space
# shapes coincide so there are fewer distinct ones — but all are _SHARD-padded anyway),
# so the distinct-shape count blows past dynamo's default recompile limit (8) on the very
# first optimizer.step(). Each shape compiles ONCE — raise the limit so they all fit
# (one-time compile cost, then steady state).
import torch._dynamo  # noqa: E402
for _attr in ("recompile_limit", "cache_size_limit"):
    if hasattr(torch._dynamo.config, _attr):
        setattr(torch._dynamo.config, _attr, 128)

# --- 5) optional seed override -----------------------------------------------
# nanochat hardcodes torch.manual_seed(42) in compute_init (common.py), which per
# its own comment seeds ONLY the model weight init (the dataloader order is
# seed-independent). Override it so SURFACE_SEED gives a fresh init for a
# reproducibility A/B. It's the only global manual_seed call, so substituting the
# value is surgical.
_seed = os.environ.get("SURFACE_SEED")
if _seed is not None:
    import torch
    _orig_manual_seed = torch.manual_seed
    torch.manual_seed = lambda s, _o=_orig_manual_seed, _v=int(_seed): _o(_v)
    print(f"[surface] seed override: model init seeded with SURFACE_SEED={_seed} (was 42)")

# --- 5.5) optional dynamic-λ schedule: tick once per OPTIMIZER step ----------
# Patch the optimizer step() so overlay/surface_lambda advances per real step
# (grad-accumulation-correct). No-op unless SURFACE_LAMBDA_SCHED is set.
if os.environ.get("SURFACE_LAMBDA_SCHED", "none") != "none":
    import nanochat.optim as _optim_mod  # noqa: E402
    from overlay import surface_lambda as _sl  # noqa: E402
    for _cls_name in ("DistMuonAdamW", "MuonAdamW"):
        _cls = getattr(_optim_mod, _cls_name, None)
        if _cls is None:
            continue
        def _ticked_step(self, *a, _orig=_cls.step, **k):
            _sl.tick()
            return _orig(self, *a, **k)
        _cls.step = _ticked_step
    print(f"[surface] dynamic λ: {os.environ['SURFACE_LAMBDA_SCHED']} "
          f"start={os.environ.get('SURFACE_LAMBDA_START', os.environ.get('SURFACE_LAMBDA', '0.5'))} "
          f"end={os.environ.get('SURFACE_LAMBDA_END', '0.1')} "
          f"horizon={os.environ.get('SURFACE_LAMBDA_HORIZON', 'UNSET!')}")

# --- 6) hand off to the upstream training loop, unchanged --------------------
runpy.run_module("scripts.base_train", run_name="__main__")
