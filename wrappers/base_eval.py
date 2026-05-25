"""
Drop-in wrapper for `scripts.base_eval`: applies the core_eval prefix-safety
patch (`wrappers/_patches.py`), prints a boundary-crossing summary at end of
eval, and writes a sidecar JSON (`base_eval/lm_boundary_stats.json`) so the
caveat survives the run and surfaces in `tools/compare_runs`. Under torchrun,
summary is aggregated across ranks and emitted once on rank 0.

See `wrappers/_patches.py` for the bug, fix, and semantic caveat.

    python -m wrappers.base_eval [args]
    torchrun --standalone --nproc_per_node=N -m wrappers.base_eval -- [args]
"""
import json
import os
import runpy

import nanochat.common as common  # for patching compute_cleanup
from wrappers._patches import (
    aggregate_boundary_stats,
    reset_boundary_stats,
)

reset_boundary_stats()

# Patched compute_cleanup runs BEFORE the process group is destroyed so the
# all_reduce in aggregate_boundary_stats can still complete. Doing this any
# later (e.g. after runpy returns) would lose the multi-rank counts.
_orig_cleanup = common.compute_cleanup

def _cleanup_with_summary():
    stats, rank = aggregate_boundary_stats()
    if rank == 0 and stats["n_total"] > 0:
        pct = 100.0 * stats["n_boundary_crossing"] / stats["n_total"]

        # stdout: visible during the run
        print(
            f"\n==> batch_sequences_lm: {stats['n_boundary_crossing']}/{stats['n_total']} "
            f"({pct:.1f}%) LM prompts crossed a tokenization boundary (loss window "
            f"expanded left of the prompt/continuation split). Expected 0% for "
            f"prefix-stable tokenizers; non-zero biases cross-tokenizer LM-accuracy "
            f"comparisons (see wrappers/_patches.py)."
        )

        # sidecar JSON: colocated with base_eval/<slug>.csv so runs/{runcpu,speedrun}.sh
        # can copy it into sandbox/results/<MODEL_TAG>/ alongside eval.csv, and
        # tools/compare_runs can surface the bias in its summary table.
        out_dir = os.path.join(common.get_base_dir(), "base_eval")
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "lm_boundary_stats.json"), "w") as f:
            json.dump({
                "schema_version": 1,
                "n_total": stats["n_total"],
                "n_boundary_crossing": stats["n_boundary_crossing"],
                "boundary_crossing_pct": pct,
            }, f, indent=2)
    _orig_cleanup()

common.compute_cleanup = _cleanup_with_summary

runpy.run_module("scripts.base_eval", run_name="__main__")
