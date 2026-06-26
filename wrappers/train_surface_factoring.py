"""
Surface-factoring training launcher (harness change).

Patches nanochat's GPT + the two best-fit dataloaders + evaluate_bpb to their
surface-factoring variants, then runpys `scripts.base_train` UNCHANGED. The
surface streams (space bit + per-word-start cap bits/mask) flow out-of-band via
`overlay.surface_context`: the surface dataloader stashes the current batch's
streams there, and the patched `SurfaceFactoringGPT.forward` — invoked by the
upstream `model(x, y)` — reads them back. (See overlay/surface_context.py for why
the strictly-sequential upstream loop makes this safe.)

PREREQUISITE: the surface (triple = force_merges + spaceless + case-fold)
tokenizer must already be trained into `$NANOCHAT_BASE_DIR/tokenizer/` so that
`get_tokenizer()` / `get_token_bytes()` resolve to the surface vocab and base-
token byte lengths. Run `wrappers.tok_train_surface` first (or use
`runs/runcpu_surface.sh`, which sequences both).

Env:
  SURFACE_LAMBDA  surface-loss weight λ for TRAINING (default 0.5; eval bpb is λ-free)
  SURFACE_KMAX    cap-slot width (default 2 = max forced-phrase word count)

Usage (same flags as scripts.base_train):
    uv run python -m wrappers.train_surface_factoring --depth=6 --num-iterations=5000 ...
"""
import os
import runpy

import nanochat.gpt as gpt_mod
import nanochat.dataloader as dl_mod
import nanochat.loss_eval as le_mod
from nanochat.tokenizer import get_tokenizer

from tools.stack_triple import FOLDED_PAIRS, CFG_TRIPLE
from overlay.surface_tokenizer import SurfaceTokenizer
from overlay.surface_dataloader import (
    surface_data_loader_with_state, surface_data_loader, surface_evaluate_bpb,
)
from overlay.surface_factoring import SurfaceFactoringGPT

# --- 1) cap-slot width: all forced phrases are 2 words and block_leading_space
#        prevents any other multi-word token, so K_max = 2 suffices. -----------
KMAX = int(os.environ.get("SURFACE_KMAX",
                          str(max(2, max(len((a + b).split()) for a, b in FOLDED_PAIRS)))))
os.environ["SURFACE_KMAX"] = str(KMAX)            # SurfaceFactoringGPT.__init__ reads this

# --- 2) build the SurfaceTokenizer over the saved surface encoding -----------
_tok = get_tokenizer()
assert _tok.get_vocab_size() == _tok.enc.n_vocab
SURFACE_TOK = SurfaceTokenizer(_tok.enc, CFG_TRIPLE["spaceless_pattern"], kmax=KMAX)
print(f"[surface] tokenizer vocab={SURFACE_TOK.n_vocab:,}, K_max={KMAX}, "
      f"λ={os.environ.get('SURFACE_LAMBDA', '0.5')}")

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
# and guards on each parameter's shape. The surface heads/embeddings add several
# NEW AdamW parameter shapes (e.g. (2,d),(2K,d),(1,2d),(K,2d) + biases), so the
# distinct-shape count blows past dynamo's default recompile limit (8) on the very
# first optimizer.step(). Each shape only needs to compile ONCE — raise the limit
# so they all fit (one-time compile cost, then steady state).
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

# --- 6) hand off to the upstream training loop, unchanged --------------------
runpy.run_module("scripts.base_train", run_name="__main__")
