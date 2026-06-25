"""
Out-of-band surface-batch handoff for the harness.

Upstream `scripts/base_train.py` has a fixed `loss = model(x, y)` call and a
fixed `x, y, state = next(loader)` — neither carries the surface streams. Rather
than copy the 630-line training loop, the surface dataloader stashes the current
batch's surface streams HERE, and `SurfaceFactoringGPT.forward`, when called with
the bare `(idx, targets)` signature, reads them back.

This is safe because the upstream loop is strictly sequential per micro-batch:
each `next(loader)` (which sets the context) is immediately followed by the
`model(x, y)` that consumes it — no reordering, no concurrency. `evaluate_bpb`
has the same `next() -> model()` cadence. The streams are stored already SHIFTED
to align with the loader's `inputs = row[:, :-1]` / `targets = row[:, 1:]`:
`*_x` go with `idx`, `*_y` with `targets`.
"""
from __future__ import annotations


class _Ctx:
    current = None  # dict(space_x, space_y, cap_bits_x, cap_bits_y, cap_mask_x, cap_mask_y) | None


def set_surface(batch):
    """Called by the surface dataloader for the batch it is about to yield."""
    _Ctx.current = batch


def get_surface():
    """Called by SurfaceFactoringGPT.forward / surface_evaluate_bpb."""
    return _Ctx.current


def clear_surface():
    _Ctx.current = None
