"""Train-zloss wrapper.

Monkeypatches `nanochat.gpt.GPT` / `GPTConfig` to the z-loss subclasses, then
invokes `scripts.base_train` unmodified via `runpy`. The upstream training
script's `from nanochat.gpt import GPT, GPTConfig` resolves to the patched
classes at import time.

Usage (same flags as scripts.base_train, plus env var Z_LOSS_COEFF):

    # single-GPU / CPU
    python -m wrappers.train_zloss --depth=4 --num-iterations=20 ...

    # distributed
    torchrun --nproc_per_node=8 -m wrappers.train_zloss --depth=26 ...
"""
import runpy

import nanochat.gpt as g
from overlay.zloss import ZLossGPT, ZLossConfig

g.GPT = ZLossGPT
g.GPTConfig = ZLossConfig

runpy.run_module("scripts.base_train", run_name="__main__")
