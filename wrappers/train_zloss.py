"""Train-zloss wrapper.

Monkeypatches `nanochat.gpt.GPT` to the z-loss subclass, then invokes
`scripts.base_train` unmodified via `runpy`. The upstream training script's
`from nanochat.gpt import GPT` resolves to the patched class at import time.

We do NOT patch GPTConfig — the z-loss coefficient is read from the
Z_LOSS_COEFF env var at model construction and held as an instance attribute
on ZLossGPT. That keeps the saved checkpoint's config byte-identical to a
baseline run, so `scripts.base_eval` / `scripts.chat_sft` / `scripts.chat_cli`
can load it as a vanilla GPT without any overlay wrapper. See
`overlay/zloss.py` for the rationale.

Usage (same flags as scripts.base_train, plus env var Z_LOSS_COEFF):

    # single-GPU / CPU
    python -m wrappers.train_zloss --depth=4 --num-iterations=20 ...

    # distributed
    torchrun --nproc_per_node=8 -m wrappers.train_zloss --depth=26 ...
"""
import runpy

import nanochat.gpt as g
from overlay.zloss import ZLossGPT

g.GPT = ZLossGPT

runpy.run_module("scripts.base_train", run_name="__main__")
