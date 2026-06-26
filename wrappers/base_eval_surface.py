"""
Surface-aware base_eval launcher — runs nanochat's `base_eval` with the surface
overlay so the CORE metric (and optionally bpb) score the surface-factored model.

Patches installed BEFORE runpy `scripts.base_eval` (which binds these via `from`
imports, and resolves `build_model`/`evaluate_example` from module globals):
  - `checkpoint_manager.build_model` → `surface_build_model` (rebuild
    SurfaceFactoringGPT from the checkpoint's surface_config / inferred K_max)
  - `core_eval.evaluate_example` → `surface_evaluate_example` (joint-NLL CORE scoring)
  - `loss_eval.evaluate_bpb` + the val dataloader → surface variants (for `--eval bpb`)
  - injects the dual-stream `SurfaceTokenizer` into the scorer

Scope: `--eval core` (default) and `bpb`. **`sample` is stripped** — generation
needs surface-aware autoregressive decode (a separate build); the base fallback
would emit base tokens with no surface bits and decode wrong.

Usage:
    NANOCHAT_BASE_DIR=~/.cache/nanochat-variants/surface_only \
      uv run python -m wrappers.base_eval_surface \
      --model-tag=d6_surface_factoring --eval core --max-per-task=500 --device-batch-size=1
"""
import os
import sys
import runpy

import nanochat.checkpoint_manager as cm_mod
import nanochat.core_eval as ce_mod
import nanochat.loss_eval as le_mod
import nanochat.dataloader as dl_mod
from nanochat.tokenizer import get_tokenizer

from tools.stack_fm_spaceless import SPACELESS_BODY
from overlay.surface_tokenizer import SurfaceTokenizer
from overlay.surface_checkpoint import surface_build_model
from overlay.surface_core_eval import surface_evaluate_example, set_surface_tokenizer
from overlay.surface_dataloader import surface_data_loader, surface_evaluate_bpb

# K_max for the scorer's tokenizer (the checkpoint also carries it for the model load).
# surface-only tokens are single words → K_max = 1 (a checkpoint's surface_config wins).
KMAX = int(os.environ.get("SURFACE_KMAX", "1"))
os.environ["SURFACE_KMAX"] = str(KMAX)

SURFACE_TOK = SurfaceTokenizer(get_tokenizer().enc, SPACELESS_BODY, kmax=KMAX)
set_surface_tokenizer(SURFACE_TOK)
print(f"[surface-eval] vocab={SURFACE_TOK.n_vocab:,}, K_max={KMAX}", file=sys.stderr)

# --- patches ---------------------------------------------------------------- #
cm_mod.build_model = surface_build_model           # K_max-aware load (load_model -> build_model)
ce_mod.evaluate_example = surface_evaluate_example  # surface CORE scoring (via evaluate_task)
le_mod.evaluate_bpb = surface_evaluate_bpb          # surface bpb (for --eval bpb)


def _val_loader(tokenizer, B, T, split="val", device="cuda", **kw):
    return surface_data_loader(SURFACE_TOK, B, T, split, device)


dl_mod.tokenizing_distributed_data_loader_bos_bestfit = _val_loader

# surface params add AdamW shapes during the build/init; raise dynamo's limit
# (same lesson as training) even though eval is no_grad.
import torch._dynamo  # noqa: E402
for _a in ("recompile_limit", "cache_size_limit"):
    if hasattr(torch._dynamo.config, _a):
        setattr(torch._dynamo.config, _a, 128)


# --- arg massage: default to core; strip `sample` (not surface-aware yet) ----- #
def _strip_sample(modes):
    kept = [m for m in modes.split(",") if m.strip() and m.strip() != "sample"]
    return ",".join(kept) or "core"


_argv = sys.argv[1:]
_has_eval = any(a == "--eval" or a.startswith("--eval=") for a in _argv)
if not _has_eval:
    sys.argv += ["--eval", "core"]
else:
    for i, a in enumerate(sys.argv):
        if a == "--eval" and i + 1 < len(sys.argv):
            sys.argv[i + 1] = _strip_sample(sys.argv[i + 1])
        elif a.startswith("--eval="):
            sys.argv[i] = "--eval=" + _strip_sample(a.split("=", 1)[1])

runpy.run_module("scripts.base_eval", run_name="__main__")
