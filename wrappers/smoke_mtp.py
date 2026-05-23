"""Smoke test for the MTP overlay.

Verifies, without training and without data:
  (a) Forward-correctness: MTPGPT.forward returns
      baseline_CE + (α/k) * Σ_{d=1..k} CE(logits[:, :-d, :], targets[:, d:])
      computed over non-ignored target positions, with logits from a single
      trunk forward shared across the main and all aux heads.
  (b) Config-portability: a checkpoint saved by MTPGPT loads fine into
      vanilla GPT — i.e. MTPGPT does NOT add unknown fields to the config.
  (c) Monkeypatch: after `nanochat.gpt.GPT = MTPGPT`, a fresh
      `from nanochat.gpt import GPT` resolves to MTPGPT — the mechanism
      `wrappers/train_mtp.py` uses for `scripts.base_train`.

Run:
    uv run python -m wrappers.smoke_mtp
"""
from __future__ import annotations

import os
from dataclasses import asdict

import torch
import torch.nn.functional as F

from nanochat.gpt import GPT as RealGPT
from nanochat.gpt import GPTConfig
from overlay.mtp import MTPGPT


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
    """Context manager: set/restore an env var around a block."""
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
    """Build a baseline and an MTP model with identical weights; compare losses."""
    device = torch.device("cpu")
    torch.manual_seed(0)
    cfg = GPTConfig(**CFG_KWARGS)

    # Exaggerated alpha for numerical clarity (defaults are tiny).
    with _env("MTP_HEADS", "3"), _env("MTP_ALPHA", "0.5"):
        base = RealGPT(cfg)
        base.init_weights()
        base = base.to(device)

        m = MTPGPT(cfg).to(device)
        m.load_state_dict(base.state_dict())

        assert m.mtp_heads == 3, f"MTP_HEADS not read from env (got {m.mtp_heads})"
        assert abs(m.mtp_alpha - 0.5) < 1e-12, f"MTP_ALPHA not read from env (got {m.mtp_alpha})"

        B, T = 2, 20  # T > max d (=k=3) so we have meaningful shifted slices
        idx = torch.randint(0, CFG_KWARGS["vocab_size"], (B, T), device=device)
        targets = torch.randint(0, CFG_KWARGS["vocab_size"], (B, T), device=device)
        # Sprinkle some ignore_index positions to exercise the mask
        targets[0, 3] = -1
        targets[1, 7] = -1

        with torch.no_grad():
            base.eval()
            m.eval()
            loss_base = base(idx, targets=targets, loss_reduction='mean').item()
            loss_m = m(idx, targets=targets, loss_reduction='mean').item()

            # Recompute the expected MTP loss from scratch.
            logits = base(idx, targets=None)
            V = logits.size(-1)
            aux_sum = 0.0
            for d in range(1, m.mtp_heads + 1):
                aux_sum += F.cross_entropy(
                    logits[:, :-d, :].reshape(-1, V),
                    targets[:, d:].reshape(-1),
                    ignore_index=-1,
                    reduction='mean',
                ).item()
            expected_loss = loss_base + (m.mtp_alpha / m.mtp_heads) * aux_sum

        diff = loss_m - expected_loss
        print(f"  baseline_loss        = {loss_base:.6f}")
        print(f"  mtp_loss             = {loss_m:.6f}")
        print(f"  expected_loss        = {expected_loss:.6f}")
        print(f"  (α/k) * Σ aux_CE     = {(m.mtp_alpha / m.mtp_heads) * aux_sum:.6f}")
        print(f"  abs error            = {abs(diff):.2e}")

        assert abs(diff) < 1e-5, (
            f"MTP loss mismatch: observed={loss_m:.6e}, expected={expected_loss:.6e}"
        )
        print("[PASS] (a) MTPGPT.forward = baseline_CE + (α/k) * Σ_d CE(logits[:, :-d, :], targets[:, d:]).")


def check_config_portability() -> None:
    cfg = GPTConfig(**CFG_KWARGS)
    m = MTPGPT(cfg)
    base = RealGPT(cfg)
    m_keys = set(asdict(m.config).keys())
    base_keys = set(asdict(base.config).keys())
    extra = m_keys - base_keys
    assert not extra, f"MTPGPT.config has extra keys vs baseline: {extra} — would break base_eval / chat_sft"
    print("[PASS] (b) MTPGPT.config has no extra fields — checkpoint will load with vanilla GPT.")


def check_monkeypatch() -> None:
    import nanochat.gpt as g
    g.GPT = MTPGPT
    try:
        from nanochat.gpt import GPT as PatchedGPT
        assert PatchedGPT is MTPGPT, f"GPT did not get patched (got {PatchedGPT})"
        print("[PASS] (c) Monkeypatch: `from nanochat.gpt import GPT` resolves to MTPGPT.")
    finally:
        g.GPT = RealGPT


if __name__ == "__main__":
    print("=== MTP overlay smoke test ===")
    print("\n[a] forward-correctness")
    check_forward_loss()
    print("\n[b] config-portability")
    check_config_portability()
    print("\n[c] monkeypatch mechanism")
    check_monkeypatch()
    print("\nAll checks passed.")
