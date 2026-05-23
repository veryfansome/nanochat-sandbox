"""MTPGPT — Multi-Token Prediction overlay on nanochat's GPT.

Design rationale: ../ideas/mtp/README.md.

Adds auxiliary next-token-prediction losses for offsets t+2, ..., t+(k+1)
alongside the main t+1 cross-entropy. The trunk + final unembedding are
unchanged; the auxiliary supervision asks the residual-stream representation
at each position to encode enough about the next k+1 tokens (not just the
next 1) to be predictive of all of them through the *same* `lm_head`. At
inference the aux losses are dropped — the model is identical cost.

Implementation choices (per the agreed plan):

- **Shared unembedding with main lm_head.** No extra parameters; the
  "heads" are different prediction targets evaluated against the same
  (B, T, V) logits, not different projection matrices. This is the
  parameter-efficient default per the design doc.
- **Naive only** (for now): k F.cross_entropy calls against shifted target
  slices. A custom autograd fused path would reduce the saved-for-backward
  (B, T, V) tensors from k+1 to 1, but the z-loss work showed the Python
  fused autograd is slower than naive in pure PyTorch (see
  ../overlay/README.md "lessons learned"). Revisit only if memory at scale
  forces it.
- **k = 3 by default** (predict t+2, t+3, t+4). Override with MTP_HEADS.
- **Loss weighting**: total aux contribution = α; per-head weight = α/k,
  so adding heads doesn't increase aux influence on the gradient (just
  spreads it). α defaults to 0.3. Override with MTP_ALPHA.

Hyperparameters are read from env at __init__ and stored as instance
attributes — NOT GPTConfig fields. See ../overlay/README.md "Hyperparam
placement" for why.
"""
from __future__ import annotations

import os

import torch
import torch.nn.functional as F

from nanochat.gpt import GPT


def _mtp_heads_from_env() -> int:
    return int(os.environ.get("MTP_HEADS", "3"))


def _mtp_alpha_from_env() -> float:
    return float(os.environ.get("MTP_ALPHA", "0.3"))


class MTPGPT(GPT):
    """GPT + auxiliary MTP losses at offsets t+2, ..., t+(k+1), shared
    unembedding. Forward signature matches the base exactly.
    """

    def __init__(self, config):
        super().__init__(config)
        self.mtp_heads = _mtp_heads_from_env()   # k extra aux heads
        self.mtp_alpha = _mtp_alpha_from_env()   # total aux weight; per-head = α/k

    def forward(self, idx, targets=None, kv_cache=None, loss_reduction='mean'):
        # Inference / no targets: pass through to base (returns logits).
        if targets is None:
            return super().forward(idx, targets=None, kv_cache=kv_cache, loss_reduction=loss_reduction)

        # Single trunk forward → (B, T, V) softcapped fp32 logits.
        logits = super().forward(idx, targets=None, kv_cache=kv_cache)
        V = logits.size(-1)

        # Main loss: logits[t] vs targets[t] (= idx[t+1] per the loader).
        ce_main = F.cross_entropy(
            logits.view(-1, V),
            targets.view(-1),
            ignore_index=-1,
            reduction=loss_reduction,
        )

        # Skip aux supervision when:
        #   - explicitly disabled (k=0 or α=0), or
        #   - reduction != 'mean' (eval path: `evaluate_bpb` calls with
        #     loss_reduction='none' to get per-token loss for bpb computation;
        #     each aux head produces a different-shaped per-token tensor since
        #     they operate on different-length slices, so summing them isn't
        #     well-defined. Conceptually right too: `val/bpb` should reflect
        #     pure next-token prediction quality, directly comparable to
        #     baseline. Aux MTP losses are a training regularizer only.)
        if self.mtp_heads == 0 or self.mtp_alpha == 0.0 or loss_reduction != 'mean':
            return ce_main

        # Aux losses: for d = 1..k, predict idx[t+d+1] = targets[t+d] from
        # logits at position t. So shift targets by d and clip logits to T-d.
        # d=1 → predict t+2; d=k → predict t+(k+1).
        aux_sum = logits.new_zeros(())
        for d in range(1, self.mtp_heads + 1):
            aux_sum = aux_sum + F.cross_entropy(
                logits[:, :-d, :].reshape(-1, V),
                targets[:, d:].reshape(-1),
                ignore_index=-1,
                reduction=loss_reduction,
            )

        return ce_main + (self.mtp_alpha / self.mtp_heads) * aux_sum
