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

Optional `SEED` env override (mirrors wrappers/train_surface_factoring.py's SURFACE_SEED):
nanochat's compute_init seeds the model weight INIT (the dataloader order is seed-independent).
It calls BOTH torch.manual_seed(42) AND torch.cuda.manual_seed(42), and init_weights draws from
the CUDA generator — so we patch BOTH or ranks diverge (rank 0 keeps 42, others get the override).
"""
import os
import runpy

import torch

_seed = os.environ.get("SEED")
if _seed is not None:
    _v = int(_seed)
    _om, _ocm = torch.manual_seed, torch.cuda.manual_seed
    torch.manual_seed = lambda s, _o=_om, _x=_v: _o(_x)
    torch.cuda.manual_seed = lambda s, _o=_ocm, _x=_v: _o(_x)
    print(f"[base_train] seed override: model init seeded with SEED={_v} (was 42)")

# FineWeb-Edu corpus override (no-op unless NANOCHAT_DATASET=fineweb)
from overlay.dataset_fineweb import maybe_repoint
maybe_repoint()

runpy.run_module("scripts.base_train", run_name="__main__")
