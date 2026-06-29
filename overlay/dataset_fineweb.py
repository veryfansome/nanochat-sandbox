"""Repoint nanochat.dataset at karpathy's FineWeb-Edu (the pre-ClimbMix corpus) when
NANOCHAT_DATASET=fineweb.

nanochat/dataset.py hardcodes ClimbMix in three module-level constants, and the dataloader
reads whatever .parquet files are present (no on-demand download) treating the LAST sorted
shard as val. This swaps the three constants so both arms of a FineWeb A/B read FineWeb
(training corpus + val-bpb shard). Call maybe_repoint() at the top of every entry wrapper
(train + eval) BEFORE any nanochat data read.

karpathy/fineweb-edu-100b-shuffle: 1823 shards; shard_01822 is the highest index, so once the
train shards (0..N) + shard_01822 are staged, parquets_iter_batched("val") returns shard_01822.

Activate with `NANOCHAT_DATASET=fineweb`. Unset / any other value = no-op (stays on ClimbMix).
"""
import os

_FINEWEB_BASE_URL = "https://huggingface.co/datasets/karpathy/fineweb-edu-100b-shuffle/resolve/main"
_FINEWEB_MAX_SHARD = 1822
_FINEWEB_DATA_SUBDIR = "base_data_fineweb"


def maybe_repoint():
    """Repoint nanochat.dataset → FineWeb-Edu iff NANOCHAT_DATASET=fineweb. Idempotent; returns
    True if it repointed. DATA_DIR is recomputed from the CURRENT NANOCHAT_BASE_DIR, so call this
    after that env var is set (the wrappers run with it set by runpod_ab.sh / speedrun.sh)."""
    if os.environ.get("NANOCHAT_DATASET", "").strip().lower() != "fineweb":
        return False
    import nanochat.dataset as ds
    from nanochat.common import get_base_dir
    ds.BASE_URL = _FINEWEB_BASE_URL
    ds.MAX_SHARD = _FINEWEB_MAX_SHARD
    ds.DATA_DIR = os.path.join(get_base_dir(), _FINEWEB_DATA_SUBDIR)
    os.makedirs(ds.DATA_DIR, exist_ok=True)
    print(f"[dataset_fineweb] nanochat.dataset → FineWeb-Edu "
          f"(BASE_URL=…/fineweb-edu-100b-shuffle, DATA_DIR={ds.DATA_DIR}, MAX_SHARD={ds.MAX_SHARD})",
          flush=True)
    return True
