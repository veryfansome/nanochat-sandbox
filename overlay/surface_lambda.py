"""
Optional DYNAMIC SURFACE_LAMBDA schedule for surface-factoring.

The surface (space+cap) loss is an auxiliary objective the model learns fast (it's
low-entropy — the space/cap bits are highly predictable). A *static* λ already self-tapers as
the surface BCE converges, but a schedule can taper sooner / below the residual
plateau to free content capacity earlier — the "favor content over time" idea.

NOT uncertainty weighting: Kendall weighting sets weight ∝ 1/loss, which would
UP-weight the tiny surface loss (wrong direction). This DECAYS λ instead.

The counter ticks once per OPTIMIZER step (wrappers/train_surface_factoring.py patches
the optimizer's step()), so it's grad-accumulation-correct. Default = static (no
schedule), so existing runs/smokes are unaffected.

Env (only active when SURFACE_LAMBDA_SCHED != none):
  SURFACE_LAMBDA_SCHED     none | linear | cosine      (default none)
  SURFACE_LAMBDA_START     λ at step 0                 (default = SURFACE_LAMBDA)
  SURFACE_LAMBDA_END       λ at the horizon (the floor; keep > 0 — the heads must stay
                           accurate for the lossless decode)   (default 0.1)
  SURFACE_LAMBDA_HORIZON   optimizer steps to decay over (REQUIRED for a schedule; set
                           = --num-iterations of the run)
"""
import math
import os

_opt_step = 0


def tick():
    """Advance the schedule by one optimizer step (called from the patched step())."""
    global _opt_step
    _opt_step += 1


def reset():
    global _opt_step
    _opt_step = 0


def current(static_lambda):
    """The λ to use right now. Returns static_lambda unless a schedule is configured."""
    sched = os.environ.get("SURFACE_LAMBDA_SCHED", "none")
    if sched == "none":
        return static_lambda
    start = float(os.environ.get("SURFACE_LAMBDA_START", static_lambda))
    end = float(os.environ.get("SURFACE_LAMBDA_END", "0.1"))
    horizon = int(os.environ.get("SURFACE_LAMBDA_HORIZON", "0"))
    assert horizon > 0, "SURFACE_LAMBDA_HORIZON must be set (=num_iterations) for a λ schedule"
    t = min(1.0, _opt_step / horizon)              # fraction of the decay horizon done
    if sched == "linear":
        return start + (end - start) * t
    if sched == "cosine":
        return end + 0.5 * (start - end) * (1.0 + math.cos(math.pi * t))   # start→end, eased
    raise ValueError(f"unknown SURFACE_LAMBDA_SCHED: {sched}")
