"""ZLossGPT — PaLM-style auxiliary z-loss on top of nanochat's GPT.

Design rationale: ideas/zloss/README.md.

Adds `z_coeff * mean(logsumexp(logits, dim=-1) ** 2)` to the cross-entropy
loss on the training path. Inference path (targets=None) is unchanged from
the base.

**Two implementations of the loss term:**

- **Naive (default)** — `F.cross_entropy(logits, ...)` + `torch.logsumexp(logits, ...)`.
  PyTorch's C++ CE is highly optimized; on CPU/MPS the saved-for-backward
  memory (~2× baseline) doesn't translate to a throughput penalty worth
  fighting. Measured on M4 d6/5000 iters: ~7% slower than baseline.
- **Fused — Python autograd (opt-in via ZLOSS_FUSED=1)** — a custom
  `torch.autograd.Function` that computes CE + z-term from one `log_softmax`
  and saves only `log_probs` for backward. Numerically correct (bit-exact
  loss, ~2e-8 grad agreement) and saves ~50% backward-time memory, but in
  pure PyTorch it replaces optimized C++ CE with Python ops and ran ~32%
  slower than baseline on M4 (measured against d6_zloss_fused). The literature's
  "fused CE" wins assume a CUDA/Triton kernel (Liger, cut-cross-entropy);
  this Python version exists as a numerical reference and for memory-
  constrained scenarios, not for routine speedup.

**Design note — why z_loss_coeff is NOT a config field.** Putting overlay-
only hyperparameters on a `GPTConfig` subclass pollutes the saved checkpoint
with unknown fields; downstream tools (`scripts.base_eval`, `scripts.chat_sft`,
`scripts.chat_cli`) that load the checkpoint as a vanilla GPT then crash with
`TypeError: GPTConfig.__init__() got an unexpected keyword argument
'z_loss_coeff'`. We read `Z_LOSS_COEFF` from the environment at `__init__`
and store it as a plain Python attribute. The saved checkpoint is config-
identical to a baseline run. Trade-off: the coefficient isn't recorded —
capture it via `MODEL_TAG` (e.g. `d24_zloss_1e4`) and wandb.
"""
from __future__ import annotations

import os

import torch
import torch.nn.functional as F

from nanochat.gpt import GPT


def _z_loss_coeff_from_env() -> float:
    return float(os.environ.get("Z_LOSS_COEFF", "1e-4"))


def _use_fused_from_env() -> bool:
    # Default: naive. The custom autograd Function (`_FusedCEZLoss`) is
    # numerically correct but in pure PyTorch it replaces a highly optimized
    # C++ `F.cross_entropy` with Python ops, paying Python-dispatch + MPS-sync
    # overhead per step. Measured on M4 d6/5000 iters: naive ran at 25.6k
    # tok/sec (~7% under baseline), fused at 18.7k (~32% under baseline).
    # A true memory-AND-throughput "fused CE" requires a CUDA/Triton kernel
    # (Liger, cut-cross-entropy) — out of scope here. Opt in to the Python
    # fused path with ZLOSS_FUSED=1 for memory-constrained scenarios or as
    # a numerical reference.
    return os.environ.get("ZLOSS_FUSED", "0") == "1"


class _FusedCEZLoss(torch.autograd.Function):
    """CE + `z_coeff · mean(logsumexp²)` in one pass, with a single `(N,V)`
    `log_probs` tensor saved for backward.

    The naive two-op formulation (`F.cross_entropy` + `torch.logsumexp`) holds
    *two* `(N,V)` tensors live for backward: the softmax saved internally by
    cross_entropy, plus the original logits required by logsumexp's backward.
    This Function emits both loss terms from one `log_softmax` and writes a
    single analytic backward off the saved `log_probs`, cutting backward-time
    memory ~50% and avoiding the swap behavior observed on M4-class machines.

    Only `reduction='mean'` is supported here; the caller (`ZLossGPT.forward`)
    falls back to the naive path for `'sum'`/`'none'` reductions (used by
    `evaluate_bpb`, which isn't memory-bound).
    """

    @staticmethod
    def forward(ctx, logits: torch.Tensor, targets: torch.Tensor, z_coeff: float, ignore_index: int) -> torch.Tensor:
        V = logits.size(-1)
        logits_flat = logits.reshape(-1, V)   # (N, V)
        targets_flat = targets.reshape(-1)    # (N,)

        log_probs = F.log_softmax(logits_flat, dim=-1)  # (N, V); replaces logits going forward
        mask = (targets_flat != ignore_index)
        n_valid = mask.sum().clamp(min=1)

        # CE = -mean(log_probs[i, targets[i]]) over non-ignored positions
        targets_safe = targets_flat.clamp(min=0)
        nll = -log_probs.gather(1, targets_safe.unsqueeze(1)).squeeze(1)  # (N,)
        ce = (nll * mask).sum() / n_valid

        # Per-row logsumexp from log_softmax: log_softmax = logits - lse  ⇒
        # lse = logits[i, k] - log_probs[i, k] for any k. Pick k=0 (numerically
        # stable since both are well-defined finite values).
        if z_coeff != 0.0:
            lse = logits_flat[:, 0] - log_probs[:, 0]  # (N,)
            z = ((lse * lse) * mask).sum() / n_valid
            loss = ce + z_coeff * z
        else:
            lse = torch.empty(0, dtype=log_probs.dtype, device=log_probs.device)
            loss = ce

        ctx.save_for_backward(log_probs, targets_flat, mask, lse)
        ctx.z_coeff = z_coeff
        ctx.orig_shape = logits.shape
        ctx.n_valid_int = int(n_valid.item())
        return loss

    @staticmethod
    def backward(ctx, grad_loss: torch.Tensor):
        log_probs, targets_flat, mask, lse = ctx.saved_tensors
        z_coeff = ctx.z_coeff
        N = ctx.n_valid_int

        softmax = log_probs.exp()  # (N, V); reuses (N,V) shape

        # ∂CE/∂logits[i,:] = (softmax[i,:] - one_hot(targets[i])) / N  at non-ignored i
        grad_logits = softmax.clone()
        targets_safe = targets_flat.clamp(min=0).unsqueeze(1)
        grad_logits.scatter_add_(1, targets_safe, -torch.ones_like(targets_safe, dtype=grad_logits.dtype))
        grad_logits = grad_logits * mask.unsqueeze(1).to(grad_logits.dtype) / N

        # ∂(z_coeff · mean(lse²))/∂logits[i,:] = z_coeff · 2 · lse[i] · softmax[i,:] / N  at non-ignored i
        if z_coeff != 0.0:
            scale = (2.0 * z_coeff / N) * lse * mask.to(lse.dtype)  # (N,)
            grad_logits = grad_logits + scale.unsqueeze(1) * softmax

        grad_logits = grad_logits * grad_loss
        return grad_logits.reshape(ctx.orig_shape), None, None, None


def _fused_ce_zloss(logits: torch.Tensor, targets: torch.Tensor, z_coeff: float, ignore_index: int = -1) -> torch.Tensor:
    return _FusedCEZLoss.apply(logits, targets, z_coeff, ignore_index)


class ZLossGPT(GPT):
    """GPT + auxiliary z-loss term. `z_coeff` lives as an instance attribute
    (not a config field — see module docstring). Fused autograd by default,
    naive opt-in via `ZLOSS_FUSED=0`.
    """

    def __init__(self, config):
        super().__init__(config)
        self.z_loss_coeff = _z_loss_coeff_from_env()
        self.use_fused = _use_fused_from_env()

    def forward(self, idx, targets=None, kv_cache=None, loss_reduction='mean'):
        # Inference / no targets: defer to base (returns logits).
        if targets is None:
            return super().forward(idx, targets=None, kv_cache=kv_cache, loss_reduction=loss_reduction)

        logits = super().forward(idx, targets=None, kv_cache=kv_cache)  # (B, T, V)

        # Fused path: only handles 'mean'. Eval (loss_reduction='none', used by
        # evaluate_bpb for per-token weighted averaging) falls through to naive.
        if self.use_fused and loss_reduction == 'mean':
            return _fused_ce_zloss(logits, targets, self.z_loss_coeff, ignore_index=-1)

        # Naive path: kept for A/B comparison and as fallback for non-mean reductions.
        ce = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            ignore_index=-1,
            reduction=loss_reduction,
        )
        if self.z_loss_coeff == 0.0:
            return ce
        lse = torch.logsumexp(logits, dim=-1)  # (B, T)
        mask = (targets != -1)
        if mask.any():
            z = (lse[mask] ** 2).mean()
        else:
            z = lse.new_zeros(())
        return ce + self.z_loss_coeff * z
