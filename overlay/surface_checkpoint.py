"""
K_max-aware checkpoint load for surface-factoring.

`surface_build_model` is a copy of `nanochat.checkpoint_manager.build_model`
(checkpoint_manager.py:77-115) with ONE change: when the saved meta carries a
`surface_config` (written by the patched `save_checkpoint` in
`wrappers/train_surface_factoring.py`), it sets `SURFACE_KMAX`/`SURFACE_LAMBDA`
from it and builds a `SurfaceFactoringGPT` instead of a vanilla `GPT`, so the
extra surface params strict-load cleanly. (Loading a surface checkpoint through
the vanilla path raises `RuntimeError: Unexpected key(s)` at the strict load.)

The eval launcher installs this as `nanochat.checkpoint_manager.build_model`;
`load_model`/`load_model_from_dir` resolve `build_model` from the module global,
so the patch is picked up.
"""
from __future__ import annotations

import os

import torch

from nanochat.gpt import GPT, GPTConfig
from nanochat.tokenizer import get_tokenizer
from nanochat.checkpoint_manager import (
    load_checkpoint, _patch_missing_config_keys, _patch_missing_keys, log0,
)
from overlay.surface_factoring import SurfaceFactoringGPT


def surface_build_model(checkpoint_dir, step, device, phase):
    """Drop-in for checkpoint_manager.build_model; builds SurfaceFactoringGPT when
    the checkpoint meta has a `surface_config`."""
    assert phase in ["train", "eval"], f"Invalid phase: {phase}"
    model_data, optimizer_data, meta_data = load_checkpoint(checkpoint_dir, step, device, load_optimizer=False)
    if device.type in {"cpu", "mps"}:
        model_data = {k: v.float() if v.dtype == torch.bfloat16 else v for k, v in model_data.items()}
    model_data = {k.removeprefix("_orig_mod."): v for k, v in model_data.items()}
    model_config_kwargs = meta_data["model_config"]
    _patch_missing_config_keys(model_config_kwargs)
    log0(f"Building model with config: {model_config_kwargs}")
    model_config = GPTConfig(**model_config_kwargs)
    _patch_missing_keys(model_data, model_config)

    surface = meta_data.get("surface_config")           # sibling of model_config; None for vanilla ckpts
    if surface is None and any(k.startswith("cap_emb") for k in model_data):
        # Surface model but no surface_config in meta (predates the persistence patch).
        # cap_emb is _SHARD-padded, so K_max is NOT recoverable from its shape — require
        # SURFACE_KMAX in the env.
        kmax_env = os.environ.get("SURFACE_KMAX")
        assert kmax_env is not None, (
            "surface checkpoint lacks surface_config in meta and SURFACE_KMAX is unset; "
            "K_max is unrecoverable (cap_emb is _SHARD-padded). Set SURFACE_KMAX.")
        kmax = int(kmax_env)
        surface = {"K_max": kmax, "lambda": float(os.environ.get("SURFACE_LAMBDA", "0.5"))}
        log0(f"surface_config absent from meta; using SURFACE_KMAX={kmax} from env")
    with torch.device("meta"):
        if surface is not None:
            os.environ["SURFACE_KMAX"] = str(surface["K_max"])
            os.environ.setdefault("SURFACE_LAMBDA", str(surface.get("lambda", 0.5)))
            log0(f"Building SurfaceFactoringGPT (K_max={surface['K_max']}, λ={surface.get('lambda', 0.5)})")
            model = SurfaceFactoringGPT(model_config)
        else:
            model = GPT(model_config)

    model.to_empty(device=device)
    model.init_weights()                                # also inits the surface params + rotary buffers
    model.load_state_dict(model_data, strict=True, assign=True)
    model.eval() if phase == "eval" else model.train()

    tokenizer = get_tokenizer()
    assert tokenizer.get_vocab_size() == model_config_kwargs["vocab_size"], (
        f"Tokenizer vocab {tokenizer.get_vocab_size()} != model vocab {model_config_kwargs['vocab_size']}")
    return model, tokenizer, meta_data
