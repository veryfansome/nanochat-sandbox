"""Train-STP wrapper.

Monkeypatches `nanochat.gpt.GPT` to the Semantic Tube Prediction subclass,
then invokes `scripts.base_train` unmodified via `runpy`. The upstream
training script's `from nanochat.gpt import GPT` resolves to the patched class
at import time.

We do NOT patch GPTConfig — the STP hyperparameters (STP_COEFF, STP_SPANS,
STP_MAX_SPAN) are read from env at model construction and held as instance
attributes on STPGPT. That keeps the saved checkpoint's config byte-identical
to a baseline run, so `scripts.base_eval` / `scripts.chat_sft` /
`scripts.chat_cli` can load it as a vanilla GPT without any overlay wrapper.
See `overlay/stp.py` for the rationale.

Usage (same flags as scripts.base_train, plus env vars STP_COEFF / STP_SPANS /
STP_MAX_SPAN):

    # single-GPU / CPU
    STP_COEFF=0.02 python -m wrappers.train_stp --depth=4 --num-iterations=20 ...

    # distributed
    STP_COEFF=0.02 torchrun --nproc_per_node=8 -m wrappers.train_stp --depth=26 ...

Diagnostics (opt-in, STP_DIAG=1): the overlay stashes the held-out hidden on
the eval path; here we wrap the wandb run's `.log` so the STP geometric
diagnostics (stp/vel_cos, stp/hidden_var, stp/hidden_rms, stp/eff_rank) ride
base_train's existing `val/bpb` log — same wandb step, no harness edit, and the
diagnostics are computed in eager (outside the compiled forward). Off by default
so the clean A/B run pays nothing. See overlay/stp.py compute_stp_diagnostics.
"""
import os
import runpy

import nanochat.gpt as g
from overlay.stp import STPGPT, enrich_log_data

g.GPT = STPGPT


def _install_stp_diag_wandb_patch() -> None:
    """Wrap wandb.init so the returned run's .log enriches base_train's val/bpb
    entry with the held-out STP diagnostics stashed on the live model. The
    enrichment logic lives in overlay.stp.enrich_log_data (unit-tested by the
    smoke); here we only wire it onto the run object."""
    import wandb

    _real_init = wandb.init

    def _patched_init(*a, **k):
        run = _real_init(*a, **k)
        _real_log = run.log

        def _log(data, *args, **kw):
            enriched = enrich_log_data(data)
            stp = {k: v for k, v in enriched.items() if k.startswith("stp/")}
            if stp:
                step = enriched.get("step", "?")
                print("  STP diag | step " + str(step) + " | "
                      + "  ".join(f"{k.split('/', 1)[1]}={v:.4f}" for k, v in stp.items()), flush=True)
            return _real_log(enriched, *args, **kw)

        try:
            run.log = _log
        except Exception:
            pass  # if the run object forbids it, silently skip diagnostics
        return run

    wandb.init = _patched_init


if os.environ.get("STP_DIAG", "0") == "1":
    _install_stp_diag_wandb_patch()

runpy.run_module("scripts.base_train", run_name="__main__")
