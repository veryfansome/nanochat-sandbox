"""
Drop-in wrapper for `scripts.base_train` (baseline / no-overlay training).

Why this exists: `scripts.base_train` runs mid-training CORE every
`--core-metric-every` steps (default 2000), which calls into
`nanochat.core_eval.batch_sequences_lm`. Non-prefix-stable tokenizers crash
there without the patch in `wrappers/_patches.py`. Routing through this
wrapper triggers `wrappers/__init__.py` which installs the patch first.

Overlay runs (`wrappers.train_<overlay>`) get the same treatment via the
same `__init__.py`; this file is the equivalent for baseline runs.

    python -m wrappers.base_train [args]
    torchrun --standalone --nproc_per_node=N -m wrappers.base_train -- [args]
"""
import runpy
runpy.run_module("scripts.base_train", run_name="__main__")
