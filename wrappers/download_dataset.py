"""Dataset download that respects the FineWeb override.

Drop-in for `python -m nanochat.dataset -n N`: applies overlay.dataset_fineweb.maybe_repoint()
(a no-op unless NANOCHAT_DATASET=fineweb) so nanochat.dataset's three constants point at
FineWeb-Edu, then runs nanochat's own CONCURRENT download (a multiprocessing.Pool over
download_single_file). We can't `runpy nanochat.dataset.__main__`: runpy re-executes the module
body, recomputing the constants back to ClimbMix. maybe_repoint is ALSO the Pool initializer, so
each worker (fork inherits the patch; spawn re-applies it) downloads from the right corpus.
download_single_file skips files already present, so re-runs / speedrun.sh's `-n` calls are no-ops.
ClimbMix runs are unaffected (maybe_repoint returns without touching anything).

    python -m wrappers.download_dataset -n 170 -w 8
"""
import argparse
import os
from multiprocessing import Pool

from overlay.dataset_fineweb import maybe_repoint

maybe_repoint()
import nanochat.dataset as ds  # noqa: E402 — import AFTER maybe_repoint so the parent sees the patch


def main():
    ap = argparse.ArgumentParser(description="Download dataset shards (FineWeb-override-aware, concurrent)")
    ap.add_argument("-n", "--num-files", type=int, default=-1, help="train shards to download (-1 = all)")
    ap.add_argument("-w", "--num-workers", type=int, default=8, help="parallel download workers")
    args = ap.parse_args()
    os.makedirs(ds.DATA_DIR, exist_ok=True)
    n = ds.MAX_SHARD if args.num_files == -1 else min(args.num_files, ds.MAX_SHARD)
    ids = list(range(n)) + [ds.MAX_SHARD]                       # train shards + the val shard (always last)
    print(f"Downloading {len(ids)} shards ({args.num_workers} workers) → {ds.DATA_DIR}", flush=True)
    # maybe_repoint as the Pool initializer → each worker (fork OR spawn) re-applies the FineWeb
    # override before download_single_file reads nanochat.dataset's DATA_DIR / BASE_URL.
    with Pool(processes=args.num_workers, initializer=maybe_repoint) as pool:
        results = pool.map(ds.download_single_file, ids)
    ok = sum(1 for r in results if r)
    print(f"Done! Downloaded/present: {ok}/{len(ids)} shards in {ds.DATA_DIR}", flush=True)


if __name__ == "__main__":
    main()
