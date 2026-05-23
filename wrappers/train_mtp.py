"""Train-MTP wrapper.

Monkeypatches `nanochat.gpt.GPT` to MTPGPT, then invokes scripts.base_train
unmodified via runpy. See ../ideas/mtp/README.md for design rationale and
../overlay/README.md for implementation lessons.

Env vars:
    MTP_HEADS    number of aux heads (default: 3 → predict t+2, t+3, t+4)
    MTP_ALPHA    total aux loss weight (default: 0.3; per-head = α/k)

Usage (same flags as scripts.base_train):

    # single-GPU / CPU
    python -m wrappers.train_mtp --depth=4 --num-iterations=20 ...

    # distributed
    torchrun --nproc_per_node=8 -m wrappers.train_mtp --depth=24 ...
"""
import runpy

import nanochat.gpt as g
from overlay.mtp import MTPGPT

g.GPT = MTPGPT

runpy.run_module("scripts.base_train", run_name="__main__")
