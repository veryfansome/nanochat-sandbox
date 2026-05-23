"""Smoke test for the z-loss overlay.

Verifies, without training and without data:
  (a) Forward-correctness: ZLossGPT.forward returns
      baseline_CE + z_coeff * mean(logsumexp(logits, dim=-1) ** 2)
      computed over non-ignored target positions.
  (b) Monkeypatch: after `nanochat.gpt.GPT = ZLossGPT`, a fresh
      `from nanochat.gpt import GPT` resolves to ZLossGPT — i.e. the
      mechanism `wrappers/train_zloss.py` uses for `scripts.base_train` works.

Run:
    uv run python -m wrappers.smoke_zloss
"""
from __future__ import annotations

import torch

# Capture the *real* classes before any patching happens.
from nanochat.gpt import GPT as RealGPT
from nanochat.gpt import GPTConfig as RealGPTConfig
from overlay.zloss import ZLossGPT, ZLossConfig


def check_forward_loss() -> None:
    """Build a baseline and a z-loss model with identical weights; compare losses."""
    device = torch.device("cpu")
    torch.manual_seed(0)

    cfg_kwargs = dict(
        sequence_len=64,
        vocab_size=128,
        n_layer=2,
        n_head=2,
        n_kv_head=2,
        n_embd=64,
        window_pattern="L",  # full context for a tiny T
    )
    base_cfg = RealGPTConfig(**cfg_kwargs)
    z_cfg = ZLossConfig(**cfg_kwargs)
    z_cfg.z_loss_coeff = 0.1  # exaggerate for clarity

    # Build baseline on CPU and initialize.
    base = RealGPT(base_cfg)
    base.init_weights()
    base = base.to(device)

    # Build ZLossGPT with the same weights.
    z = ZLossGPT(z_cfg).to(device)
    z.load_state_dict(base.state_dict())

    B, T = 2, 16
    idx = torch.randint(0, cfg_kwargs["vocab_size"], (B, T), device=device)
    targets = torch.randint(0, cfg_kwargs["vocab_size"], (B, T), device=device)

    with torch.no_grad():
        base.eval()
        z.eval()
        loss_base = base(idx, targets=targets, loss_reduction='mean').item()
        loss_z = z(idx, targets=targets, loss_reduction='mean').item()
        logits = base(idx, targets=None)  # (B, T, V) fp32, softcapped
        lse = torch.logsumexp(logits, dim=-1)  # (B, T)
        expected_z_term = (lse[(targets != -1)] ** 2).mean().item()

    diff = loss_z - loss_base
    expected_diff = z_cfg.z_loss_coeff * expected_z_term

    print(f"  baseline_loss        = {loss_base:.6f}")
    print(f"  zloss_loss           = {loss_z:.6f}")
    print(f"  observed diff        = {diff:.6f}")
    print(f"  z_coeff * E[lse^2]   = {expected_diff:.6f}")
    print(f"  abs error            = {abs(diff - expected_diff):.2e}")

    assert abs(diff - expected_diff) < 1e-5, (
        f"z-loss term mismatch: observed diff={diff:.6e}, expected={expected_diff:.6e}"
    )
    print("[PASS] (a) ZLossGPT.forward = baseline + z_coeff * mean(logsumexp**2).")


def check_monkeypatch() -> None:
    """Verify the patch+import mechanism that `wrappers/train_zloss.py` relies on."""
    import nanochat.gpt as g
    g.GPT = ZLossGPT
    g.GPTConfig = ZLossConfig
    try:
        # This is exactly the line `scripts/base_train.py` runs at import time.
        from nanochat.gpt import GPT as PatchedGPT
        from nanochat.gpt import GPTConfig as PatchedGPTConfig
        assert PatchedGPT is ZLossGPT, f"GPT did not get patched (got {PatchedGPT})"
        assert PatchedGPTConfig is ZLossConfig, f"GPTConfig did not get patched (got {PatchedGPTConfig})"
        print("[PASS] (b) Monkeypatch: `from nanochat.gpt import GPT, GPTConfig` resolves to ZLoss variants.")
    finally:
        # Restore so the smoke is idempotent.
        g.GPT = RealGPT
        g.GPTConfig = RealGPTConfig


if __name__ == "__main__":
    print("=== z-loss overlay smoke test ===")
    print("\n[a] forward-correctness")
    check_forward_loss()
    print("\n[b] monkeypatch mechanism")
    check_monkeypatch()
    print("\nAll checks passed.")
