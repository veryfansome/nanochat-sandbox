"""Smoke test for the z-loss overlay.

Verifies, without training and without data:
  (a) Forward-correctness: ZLossGPT.forward (naive path, the default)
      returns baseline_CE + z_coeff * mean(logsumexp(logits, dim=-1) ** 2)
      computed over non-ignored target positions.
  (b) Config-portability: a checkpoint saved by ZLossGPT loads fine into
      vanilla GPT — i.e. ZLossGPT does NOT add unknown fields to the config.
  (c) Monkeypatch: after `nanochat.gpt.GPT = ZLossGPT`, a fresh
      `from nanochat.gpt import GPT` resolves to ZLossGPT — the mechanism
      `wrappers/train_zloss.py` uses for `scripts.base_train`.
  (d) Fused == naive: the fused autograd path and the naive two-op path
      produce numerically equivalent loss AND gradients on a tiny model.
  (e) Eval-path contract: with a nonzero z_coeff, ZLossGPT called with
      `loss_reduction='none'` (the path nanochat/loss_eval.py uses for
      val/bpb) must return per-token CE that matches vanilla GPT exactly.
      A previous regression silently added the scalar z-loss term on this
      path, inflating val/bpb by a constant offset on every token.

Run:
    uv run python -m wrappers.smoke_zloss
"""
from __future__ import annotations

import os
from dataclasses import asdict

import torch

# Capture the *real* GPT before any patching.
from nanochat.gpt import GPT as RealGPT
from nanochat.gpt import GPTConfig
from overlay.zloss import ZLossGPT


CFG_KWARGS = dict(
    sequence_len=64,
    vocab_size=128,
    n_layer=2,
    n_head=2,
    n_kv_head=2,
    n_embd=64,
    window_pattern="L",  # full context for a tiny T
)


def _env(key: str, value: str | None):
    """Context-manager helper: set/restore an env var around a block."""
    class _C:
        def __enter__(self):
            self.prev = os.environ.get(key)
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
            return self
        def __exit__(self, *a):
            if self.prev is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = self.prev
    return _C()


def check_forward_loss() -> None:
    """Build a baseline and a z-loss model with identical weights; compare losses."""
    device = torch.device("cpu")
    torch.manual_seed(0)

    cfg = GPTConfig(**CFG_KWARGS)

    with _env("Z_LOSS_COEFF", "0.1"):  # exaggerate for numerical clarity
        base = RealGPT(cfg)
        base.init_weights()
        base = base.to(device)

        z = ZLossGPT(cfg).to(device)        # reads Z_LOSS_COEFF here
        z.load_state_dict(base.state_dict())

        assert abs(z.z_loss_coeff - 0.1) < 1e-12, f"z_loss_coeff not read from env (got {z.z_loss_coeff})"
        assert z.use_fused is False, "naive should be the default (fused is opt-in via ZLOSS_FUSED=1)"

        B, T = 2, 16
        idx = torch.randint(0, CFG_KWARGS["vocab_size"], (B, T), device=device)
        targets = torch.randint(0, CFG_KWARGS["vocab_size"], (B, T), device=device)

        with torch.no_grad():
            base.eval()
            z.eval()
            loss_base = base(idx, targets=targets, loss_reduction='mean').item()
            loss_z = z(idx, targets=targets, loss_reduction='mean').item()
            logits = base(idx, targets=None)
            lse = torch.logsumexp(logits, dim=-1)
            expected_z_term = (lse[(targets != -1)] ** 2).mean().item()

        diff = loss_z - loss_base
        expected_diff = z.z_loss_coeff * expected_z_term

        print(f"  baseline_loss        = {loss_base:.6f}")
        print(f"  zloss_loss (naive)   = {loss_z:.6f}")
        print(f"  observed diff        = {diff:.6f}")
        print(f"  z_coeff * E[lse^2]   = {expected_diff:.6f}")
        print(f"  abs error            = {abs(diff - expected_diff):.2e}")

        assert abs(diff - expected_diff) < 1e-5, (
            f"z-loss term mismatch: observed diff={diff:.6e}, expected={expected_diff:.6e}"
        )
        print("[PASS] (a) ZLossGPT.forward = baseline + z_coeff * mean(logsumexp**2).")


def check_config_portability() -> None:
    cfg = GPTConfig(**CFG_KWARGS)
    z = ZLossGPT(cfg)
    base = RealGPT(cfg)
    z_keys = set(asdict(z.config).keys())
    base_keys = set(asdict(base.config).keys())
    extra = z_keys - base_keys
    assert not extra, f"ZLossGPT.config has extra keys vs baseline: {extra} — this will break base_eval / chat_sft"
    print("[PASS] (b) ZLossGPT.config has no extra fields — checkpoint will load with vanilla GPT.")


def check_monkeypatch() -> None:
    import nanochat.gpt as g
    g.GPT = ZLossGPT
    try:
        from nanochat.gpt import GPT as PatchedGPT
        assert PatchedGPT is ZLossGPT, f"GPT did not get patched (got {PatchedGPT})"
        print("[PASS] (c) Monkeypatch: `from nanochat.gpt import GPT` resolves to ZLossGPT.")
    finally:
        g.GPT = RealGPT


def check_fused_equivalence() -> None:
    """Fused and naive paths must produce numerically equivalent loss + gradients."""
    device = torch.device("cpu")
    torch.manual_seed(0)
    cfg = GPTConfig(**CFG_KWARGS)

    with _env("Z_LOSS_COEFF", "0.1"):
        # Build two models with identical weights — one forced naive, one fused.
        m_naive = ZLossGPT(cfg)
        m_naive.init_weights()
        m_naive = m_naive.to(device)
        m_naive.use_fused = False

        m_fused = ZLossGPT(cfg).to(device)
        m_fused.load_state_dict(m_naive.state_dict())
        m_fused.use_fused = True

        B, T = 2, 16
        idx = torch.randint(0, CFG_KWARGS["vocab_size"], (B, T), device=device)
        targets = torch.randint(0, CFG_KWARGS["vocab_size"], (B, T), device=device)
        # sprinkle some ignore_index to exercise the mask
        targets[0, 5] = -1
        targets[1, 10] = -1

        loss_naive = m_naive(idx, targets=targets, loss_reduction='mean')
        loss_naive.backward()
        grads_naive = {n: p.grad.detach().clone() for n, p in m_naive.named_parameters() if p.grad is not None}

        loss_fused = m_fused(idx, targets=targets, loss_reduction='mean')
        loss_fused.backward()
        grads_fused = {n: p.grad.detach().clone() for n, p in m_fused.named_parameters() if p.grad is not None}

        loss_diff = (loss_naive - loss_fused).abs().item()
        max_grad_diff = 0.0
        worst = None
        for n in grads_naive:
            if n not in grads_fused:
                continue
            d = (grads_naive[n] - grads_fused[n]).abs().max().item()
            if d > max_grad_diff:
                max_grad_diff = d
                worst = n

        print(f"  loss (naive)         = {loss_naive.item():.6f}")
        print(f"  loss (fused)         = {loss_fused.item():.6f}")
        print(f"  |Δ loss|             = {loss_diff:.2e}")
        print(f"  max |Δ grad|         = {max_grad_diff:.2e}  (worst param: {worst})")

        assert loss_diff < 1e-5, f"fused/naive loss mismatch: {loss_diff:.2e}"
        assert max_grad_diff < 1e-4, f"fused/naive gradient mismatch: {max_grad_diff:.2e} at {worst}"
        print("[PASS] (d) Fused and naive paths give equivalent loss and gradients.")


def check_eval_path_contract() -> None:
    """With nonzero z_coeff, `loss_reduction='none'` must return per-token CE
    that matches vanilla GPT exactly. This is the path `nanochat/loss_eval.py`
    uses to compute val/bpb. Any contribution from the z-loss term inflates
    the metric and makes overlay runs non-comparable to baseline.

    Regression coverage: a previous version applied `+ z_coeff·z` (a scalar)
    to `ce` regardless of reduction; with reduction='none' the scalar broadcast
    onto every per-token loss, inflating val/bpb by a constant offset.
    """
    device = torch.device("cpu")
    torch.manual_seed(0)
    cfg = GPTConfig(**CFG_KWARGS)

    with _env("Z_LOSS_COEFF", "0.1"):  # exaggerated; bug-was-here would dwarf 1e-4 too
        base = RealGPT(cfg)
        base.init_weights()
        base = base.to(device)

        z = ZLossGPT(cfg).to(device)
        z.load_state_dict(base.state_dict())
        assert z.z_loss_coeff == 0.1 and not z.use_fused

        B, T = 2, 16
        idx = torch.randint(0, CFG_KWARGS["vocab_size"], (B, T), device=device)
        targets = torch.randint(0, CFG_KWARGS["vocab_size"], (B, T), device=device)
        targets[0, 5] = -1  # exercise the mask
        targets[1, 10] = -1

        with torch.no_grad():
            base.eval(); z.eval()
            ce_base_none = base(idx, targets=targets, loss_reduction='none')
            ce_z_none = z(idx, targets=targets, loss_reduction='none')
            ce_base_sum = base(idx, targets=targets, loss_reduction='sum')
            ce_z_sum = z(idx, targets=targets, loss_reduction='sum')

        max_abs_none = (ce_base_none - ce_z_none).abs().max().item()
        abs_sum = (ce_base_sum - ce_z_sum).abs().item()
        print(f"  reduction='none': max |Δ per-token CE| = {max_abs_none:.2e}  (shape: {ce_z_none.shape})")
        print(f"  reduction='sum':  |Δ total|            = {abs_sum:.2e}")
        assert max_abs_none < 1e-6, (
            f"z-loss leaks into eval path (reduction='none'): max per-token diff "
            f"= {max_abs_none:.2e}. The z-loss term must only apply when "
            f"loss_reduction='mean' — see `forward` and overlay/README.md "
            f"'Aux losses are a training-time regularizer'."
        )
        assert abs_sum < 1e-5, f"z-loss leaks into eval path (reduction='sum'): {abs_sum:.2e}"
        print("[PASS] (e) Eval path returns baseline-comparable CE — z-loss term confined to reduction='mean'.")


if __name__ == "__main__":
    print("=== z-loss overlay smoke test ===")
    print("\n[a] forward-correctness (naive default)")
    check_forward_loss()
    print("\n[b] config-portability")
    check_config_portability()
    print("\n[c] monkeypatch mechanism")
    check_monkeypatch()
    print("\n[d] fused == naive (loss + gradients)")
    check_fused_equivalence()
    print("\n[e] eval-path contract (loss_reduction != 'mean' → baseline-comparable)")
    check_eval_path_contract()
    print("\nAll checks passed.")
