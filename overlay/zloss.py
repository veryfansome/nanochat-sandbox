"""ZLossGPT — PaLM-style auxiliary z-loss on top of nanochat's GPT.

Design rationale: sandbox/ideas/zloss/README.md.

Adds `z_coeff * mean(logsumexp(logits, dim=-1) ** 2)` to the cross-entropy loss
on the training path. Inference path (targets=None) is unchanged from the base.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from nanochat.gpt import GPT, GPTConfig


def _default_z_coeff() -> float:
    return float(os.environ.get("Z_LOSS_COEFF", "1e-4"))


@dataclass
class ZLossConfig(GPTConfig):
    """GPTConfig + z-loss hyperparameter. Defaulted so upstream construction
    `GPTConfig(sequence_len=..., ...)` keeps working unchanged.
    """
    z_loss_coeff: float = 0.0  # populated by __post_init__ from env if not set

    def __post_init__(self):
        # Allow callers to pass z_loss_coeff=... explicitly; otherwise pull from env.
        if self.z_loss_coeff == 0.0:
            self.z_loss_coeff = _default_z_coeff()


class ZLossGPT(GPT):
    """GPT + auxiliary z-loss term. Forward signature matches the base exactly."""

    def forward(self, idx, targets=None, kv_cache=None, loss_reduction='mean'):
        # Inference / no targets: pass through to base (returns logits).
        if targets is None:
            return super().forward(idx, targets=None, kv_cache=kv_cache, loss_reduction=loss_reduction)

        # Training: get logits via the base forward (softcapped, fp32), then compute
        # CE + z-loss ourselves. Single trunk forward — same cost as baseline.
        logits = super().forward(idx, targets=None, kv_cache=kv_cache)  # (B, T, V)

        ce = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            ignore_index=-1,
            reduction=loss_reduction,
        )

        z_coeff = getattr(self.config, "z_loss_coeff", _default_z_coeff())
        if z_coeff == 0.0:
            return ce

        lse = torch.logsumexp(logits, dim=-1)  # (B, T)
        mask = (targets != -1)
        if mask.any():
            z = (lse[mask] ** 2).mean()
        else:
            z = lse.new_zeros(())
        return ce + z_coeff * z
