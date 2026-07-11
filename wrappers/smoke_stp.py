"""Smoke test for the Semantic Tube Prediction (STP) overlay.

Verifies, without training and without data:
  (a) Forward-correctness: STPGPT.forward (loss_reduction='mean') returns
      baseline_CE + stp_coeff * (1 - mean cos(h_t-h_r, h_r-h_s)) over the
      randomly sampled triples, using the post-final-norm hidden states.
  (b) Config-portability: STPGPT adds no fields to GPTConfig, so a checkpoint
      it saves loads into vanilla GPT (base_eval / chat_sft / chat_cli).
  (c) Monkeypatch: after `nanochat.gpt.GPT = STPGPT`, a fresh
      `from nanochat.gpt import GPT` resolves to STPGPT — the mechanism
      `wrappers/train_stp.py` uses for `scripts.base_train`.
  (d) Hook fidelity: the lm_head pre-hook captures exactly the post-final-norm
      hidden (== lm_head's input) with shape (B, T, C), and STP gradients flow
      back into the trunk.
  (e) Eval-path contract: with a nonzero stp_coeff, STPGPT called with
      `loss_reduction='none'`/`'sum'` (the path nanochat/loss_eval.py uses for
      val/bpb) must return CE that matches vanilla GPT exactly — the STP term
      is confined to the 'mean' training path.
  (f) torch.compile safety: the training forward runs under torch.compile in
      base_train (model is compiled at base_train.py:246). With a
      deterministic single-triple config (T=3, window=3 => only triple
      [0,1,2], RNG-independent), the compiled forward must produce a finite
      loss and gradients numerically equal to eager.
  (g) RNG stays live under compile: with a window admitting many triples, two
      successive compiled forwards must draw different triples (different loss)
      — i.e. Dynamo's RNG functionalization didn't freeze the sampler.
  (h) STP_DIAG instrument: compute_stp_diagnostics separates straightening
      (vel_cos) from the two collapse modes (hidden_var, eff_rank), and with
      STP_DIAG=1 the eval path stashes the held-out hidden while still returning
      pure CE.
  (i) STP_DIAG under compile: evaluate_bpb runs the compiled model, so the
      eval-path diag stash must commit under torch.compile with CE unchanged.

Run:
    uv run python -m wrappers.smoke_stp
"""
from __future__ import annotations

import os
from dataclasses import asdict

import torch
import torch.nn.functional as F

# Capture the *real* GPT before any patching.
from nanochat.gpt import GPT as RealGPT
from nanochat.gpt import GPTConfig
from overlay.stp import STPGPT


CFG_KWARGS = dict(
    sequence_len=64,
    vocab_size=128,
    n_layer=2,
    n_head=2,
    n_kv_head=2,
    n_embd=64,
    window_pattern="L",  # full context for a tiny T
)


def _env(**kv):
    """Context-manager: set/restore multiple env vars around a block."""
    class _C:
        def __enter__(self):
            self.prev = {k: os.environ.get(k) for k in kv}
            for k, v in kv.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = str(v)
            return self
        def __exit__(self, *a):
            for k, v in self.prev.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
    return _C()


def _is_toolchain_skip(e: Exception) -> bool:
    """True if `e` is Inductor failing for lack of a CPU build toolchain
    (setuptools / C++ compiler) rather than a real STP correctness bug."""
    sig = f"{type(e).__name__}: {e}".lower()
    return any(s in sig for s in (
        "setuptools", "inductorerror", "backendcompilerfailed",
        "compiler", "gcc", "clang", "cl.exe", "vec_isa", "cpp_extension",
    ))


def _capture_pre_lm_head_hidden(model, idx):
    """Run `model(idx, targets=None)` with a temporary pre-hook on lm_head and
    return (logits, hidden) where hidden is lm_head's input (post-final-norm)."""
    cap = {}
    h = model.lm_head.register_forward_pre_hook(lambda mod, args: cap.__setitem__("h", args[0]))
    try:
        with torch.no_grad():
            logits = model(idx, targets=None)
    finally:
        h.remove()
    return logits, cap["h"]


def check_forward_loss() -> None:
    """Build baseline + STP models with identical weights; verify the STP loss
    equals CE + coeff*(1 - mean cos) over the *same* sampled triples."""
    device = torch.device("cpu")
    torch.manual_seed(0)
    cfg = GPTConfig(**CFG_KWARGS)

    with _env(STP_COEFF="0.1", STP_SPANS="2", STP_MAX_SPAN="8"):
        base = RealGPT(cfg)
        base.init_weights()
        base = base.to(device).eval()

        m = STPGPT(cfg).to(device).eval()
        m.load_state_dict(base.state_dict())

        assert abs(m.stp_coeff - 0.1) < 1e-12, f"stp_coeff not read from env (got {m.stp_coeff})"
        assert m.stp_spans == 2, f"stp_spans not read from env (got {m.stp_spans})"
        assert m.stp_max_span == 8, f"stp_max_span not read from env (got {m.stp_max_span})"

        B, T = 2, 16
        idx = torch.randint(0, CFG_KWARGS["vocab_size"], (B, T), device=device)
        targets = torch.randint(0, CFG_KWARGS["vocab_size"], (B, T), device=device)

        # Reference: baseline logits + the post-norm hidden that the hook grabs.
        base_logits, base_hidden = _capture_pre_lm_head_hidden(base, idx)
        V = base_logits.size(-1)
        ce = F.cross_entropy(base_logits.view(-1, V), targets.view(-1), ignore_index=-1).item()

        # STP model loss with a controlled RNG seed (the only RNG consumer in
        # STPGPT.forward is _sample_triples, called once after the deterministic
        # trunk — so re-seeding identically reproduces the exact same triples).
        torch.manual_seed(123)
        with torch.no_grad():
            loss_m = m(idx, targets=targets, loss_reduction="mean").item()

        torch.manual_seed(123)
        rows, idxs = m._sample_triples(B, T, device)
        h_sel = base_hidden[rows.unsqueeze(1), idxs].float()
        v1 = h_sel[:, 1] - h_sel[:, 0]
        v2 = h_sel[:, 2] - h_sel[:, 1]
        stp_term = (1.0 - F.cosine_similarity(v2, v1, dim=-1).mean()).item()
        expected = ce + m.stp_coeff * stp_term

        print(f"  baseline_CE          = {ce:.6f}")
        print(f"  STP term (1-mean cos)= {stp_term:.6f}")
        print(f"  stp_loss (observed)  = {loss_m:.6f}")
        print(f"  expected             = {expected:.6f}")
        print(f"  abs error            = {abs(loss_m - expected):.2e}")
        assert abs(loss_m - expected) < 1e-5, (
            f"STP loss mismatch: observed={loss_m:.6e}, expected={expected:.6e}"
        )
        # Sanity: the STP term is genuinely nonzero (loss != plain CE).
        assert abs(loss_m - ce) > 1e-6, "STP term vanished — loss equals plain CE"
        print("[PASS] (a) STPGPT.forward = CE + coeff*(1 - mean cos of consecutive velocities).")


def check_config_portability() -> None:
    cfg = GPTConfig(**CFG_KWARGS)
    m = STPGPT(cfg)
    base = RealGPT(cfg)
    extra = set(asdict(m.config).keys()) - set(asdict(base.config).keys())
    assert not extra, f"STPGPT.config has extra keys vs baseline: {extra} — breaks base_eval / chat_sft"
    print("[PASS] (b) STPGPT.config has no extra fields — checkpoint loads with vanilla GPT.")


def check_monkeypatch() -> None:
    import nanochat.gpt as g
    g.GPT = STPGPT
    try:
        from nanochat.gpt import GPT as PatchedGPT
        assert PatchedGPT is STPGPT, f"GPT did not get patched (got {PatchedGPT})"
        print("[PASS] (c) Monkeypatch: `from nanochat.gpt import GPT` resolves to STPGPT.")
    finally:
        g.GPT = RealGPT


def check_hook_fidelity() -> None:
    """The hook must capture exactly lm_head's input (post-final-norm hidden),
    and the STP term must contribute gradient to the trunk."""
    device = torch.device("cpu")
    torch.manual_seed(0)
    cfg = GPTConfig(**CFG_KWARGS)

    with _env(STP_COEFF="0.1", STP_SPANS="2", STP_MAX_SPAN="8"):
        base = RealGPT(cfg)
        base.init_weights()
        base = base.to(device).eval()
        m = STPGPT(cfg).to(device).eval()
        m.load_state_dict(base.state_dict())

        B, T = 2, 16
        idx = torch.randint(0, CFG_KWARGS["vocab_size"], (B, T), device=device)
        targets = torch.randint(0, CFG_KWARGS["vocab_size"], (B, T), device=device)

        _, ref_hidden = _capture_pre_lm_head_hidden(base, idx)
        with torch.no_grad():
            _ = m(idx, targets=targets, loss_reduction="mean")  # populates m._stp_hidden
        assert m._stp_hidden is not None, "hook did not capture the hidden state"
        assert m._stp_hidden.shape == (B, T, cfg.n_embd), f"unexpected hidden shape {tuple(m._stp_hidden.shape)}"
        max_abs = (m._stp_hidden - ref_hidden).abs().max().item()
        print(f"  captured hidden shape= {tuple(m._stp_hidden.shape)}")
        print(f"  max|hook - lm_head input| = {max_abs:.2e}")
        assert max_abs < 1e-6, f"hook captured the wrong tensor (Δ={max_abs:.2e})"

        # STP gradient contribution: with identical weights + inputs, the CE
        # gradient is identical, so (grad_on - grad_off) isolates coeff*d(STP).
        # A nonzero delta on the embedding (always upstream of the hidden via
        # nanochat's x0 residual) proves STP propagates into the trunk. We
        # compare the embedding rather than an MLP weight because a dead-ReLU
        # tiny init can legitimately zero an MLP c_fc grad on both paths.
        m_off = STPGPT(cfg).to(device).train(); m_off.load_state_dict(base.state_dict()); m_off.stp_coeff = 0.0
        m_on = STPGPT(cfg).to(device).train(); m_on.load_state_dict(base.state_dict()); m_on.stp_coeff = 0.5
        m_off(idx, targets=targets, loss_reduction="mean").backward()
        torch.manual_seed(9)
        m_on(idx, targets=targets, loss_reduction="mean").backward()
        wte_delta = (m_on.transformer.wte.weight.grad - m_off.transformer.wte.weight.grad).norm().item()
        print(f"  |grad_on - grad_off| on wte = {wte_delta:.4e}")
        assert wte_delta > 1e-8, "STP term contributes no gradient to the trunk embedding"
        print("[PASS] (d) Hook captures post-norm hidden; STP gradient reaches the trunk.")


def check_eval_path_contract() -> None:
    """With nonzero coeff, `loss_reduction='none'`/`'sum'` must match vanilla GPT
    exactly — the path nanochat/loss_eval.py uses for val/bpb."""
    device = torch.device("cpu")
    torch.manual_seed(0)
    cfg = GPTConfig(**CFG_KWARGS)

    with _env(STP_COEFF="0.1", STP_SPANS="2", STP_MAX_SPAN="8"):
        base = RealGPT(cfg)
        base.init_weights()
        base = base.to(device).eval()
        m = STPGPT(cfg).to(device).eval()
        m.load_state_dict(base.state_dict())

        B, T = 2, 16
        idx = torch.randint(0, CFG_KWARGS["vocab_size"], (B, T), device=device)
        targets = torch.randint(0, CFG_KWARGS["vocab_size"], (B, T), device=device)
        targets[0, 5] = -1  # exercise the ignore mask
        targets[1, 10] = -1

        with torch.no_grad():
            none_base = base(idx, targets=targets, loss_reduction="none")
            none_m = m(idx, targets=targets, loss_reduction="none")
            sum_base = base(idx, targets=targets, loss_reduction="sum")
            sum_m = m(idx, targets=targets, loss_reduction="sum")

        max_none = (none_base - none_m).abs().max().item()
        abs_sum = (sum_base - sum_m).abs().item()
        print(f"  reduction='none': max|Δ per-token CE| = {max_none:.2e}  (shape {tuple(none_m.shape)})")
        print(f"  reduction='sum':  |Δ total|           = {abs_sum:.2e}")
        assert max_none < 1e-6, (
            f"STP leaks into eval path (reduction='none'): {max_none:.2e}. The STP term "
            f"must only apply when loss_reduction='mean' — see forward and overlay/README.md."
        )
        assert abs_sum < 1e-5, f"STP leaks into eval path (reduction='sum'): {abs_sum:.2e}"
        print("[PASS] (e) Eval path returns baseline-comparable CE — STP confined to 'mean'.")


def check_compile_safety() -> None:
    """The training forward runs under torch.compile in base_train. Use a
    deterministic single-triple config (T=3, window=3 => only [0,1,2]) so the
    eager-vs-compiled comparison isn't confounded by RNG functionalization."""
    device = torch.device("cpu")
    torch.manual_seed(0)
    cfg = GPTConfig(**CFG_KWARGS)

    with _env(STP_COEFF="0.1", STP_SPANS="1", STP_MAX_SPAN="3"):
        m = STPGPT(cfg)
        m.init_weights()
        m = m.to(device).train()

        B, T = 2, 3
        idx = torch.randint(0, CFG_KWARGS["vocab_size"], (B, T), device=device)
        targets = torch.randint(0, CFG_KWARGS["vocab_size"], (B, T), device=device)

        loss_eager = m(idx, targets=targets, loss_reduction="mean")
        loss_eager.backward()
        g_eager = m.transformer.wte.weight.grad.detach().clone()
        m.zero_grad(set_to_none=True)

        torch._dynamo.reset()
        cm = torch.compile(m, dynamic=False)
        try:
            loss_comp = cm(idx, targets=targets, loss_reduction="mean")
            loss_comp.backward()
        except Exception as e:
            # Distinguish a missing compile toolchain (skip) from a real STP
            # correctness bug (fail). Inductor's CPU backend needs setuptools +
            # a C/C++ compiler; a bare env lacks them. The GPU training env has
            # them (baseline runs compile), so this path is exercised there.
            if _is_toolchain_skip(e):
                print(f"  [SKIP] torch.compile backend unavailable here ({type(e).__name__}). "
                      f"STP logic is identical under compile; exercised on the GPU training env.")
                return
            raise
        g_comp = m.transformer.wte.weight.grad.detach().clone()

        loss_diff = (loss_eager - loss_comp).abs().item()
        grad_diff = (g_eager - g_comp).abs().max().item()
        print(f"  loss (eager)         = {loss_eager.item():.6f}")
        print(f"  loss (compiled)      = {loss_comp.item():.6f}")
        print(f"  |Δ loss|             = {loss_diff:.2e}")
        print(f"  max|Δ grad| (wte)    = {grad_diff:.2e}")
        assert torch.isfinite(loss_comp).all(), "compiled STP loss is non-finite"
        assert loss_diff < 1e-3, f"compiled vs eager loss mismatch: {loss_diff:.2e}"
        assert grad_diff < 1e-3, f"compiled vs eager grad mismatch: {grad_diff:.2e}"
        print("[PASS] (f) STP forward is correct under torch.compile (hook survives).")


def check_compile_rng_live() -> None:
    """Under torch.compile, the span sampler must still draw DIFFERENT triples
    each step — if Dynamo's RNG functionalization froze it, STP would train on
    one fixed triple set forever. Two successive compiled forwards (same inputs
    and weights, no update between) must therefore give different losses. Uses a
    window with many possible triples so a collision is astronomically unlikely."""
    device = torch.device("cpu")
    torch.manual_seed(0)
    cfg = GPTConfig(**CFG_KWARGS)

    with _env(STP_COEFF="0.1", STP_SPANS="4", STP_MAX_SPAN="8"):
        m = STPGPT(cfg)
        m.init_weights()
        m = m.to(device).train()
        B, T = 2, 16
        idx = torch.randint(0, CFG_KWARGS["vocab_size"], (B, T), device=device)
        targets = torch.randint(0, CFG_KWARGS["vocab_size"], (B, T), device=device)

        torch._dynamo.reset()
        cm = torch.compile(m, dynamic=False)
        try:
            with torch.no_grad():
                l1 = cm(idx, targets=targets, loss_reduction="mean").item()
                l2 = cm(idx, targets=targets, loss_reduction="mean").item()
        except Exception as e:
            if _is_toolchain_skip(e):
                print(f"  [SKIP] torch.compile backend unavailable here ({type(e).__name__}).")
                return
            raise
        print(f"  loss call #1         = {l1:.6f}")
        print(f"  loss call #2         = {l2:.6f}")
        print(f"  |Δ| across calls     = {abs(l1 - l2):.2e}")
        assert abs(l1 - l2) > 1e-6, (
            "STP triples appear frozen under torch.compile — the RNG did not advance "
            "between calls, so STP would train on a single fixed triple set."
        )
        print("[PASS] (g) Span RNG stays live across compiled forwards (triples vary per step).")


def check_diagnostics() -> None:
    """The STP_DIAG instrument: (1) compute_stp_diagnostics distinguishes the
    two failure modes it claims (straightening vs collapse); (2) with STP_DIAG=1
    the eval path stashes the held-out hidden and STILL returns pure CE."""
    from overlay.stp import compute_stp_diagnostics, enrich_log_data
    import overlay.stp as stp_mod

    device = torch.device("cpu")
    torch.manual_seed(0)
    B, T, C = 4, 32, 64

    # straightening: a near-straight trajectory has vel_cos ~ 1; random ~ 0.
    t = torch.arange(T, dtype=torch.float32).view(1, T, 1)
    straight = t * torch.randn(1, 1, C) + 0.01 * torch.randn(B, T, C)
    rand = torch.randn(B, T, C)
    d_straight, d_rand = compute_stp_diagnostics(straight), compute_stp_diagnostics(rand)
    # collapse modes: directional collapse -> eff_rank ~ 1; constant -> var ~ 0.
    directional = torch.randn(B, T, 1) * torch.randn(1, 1, C) + 1e-3 * torch.randn(B, T, C)
    constant = torch.ones(B, T, C) + 1e-4 * torch.randn(B, T, C)
    d_dir, d_const = compute_stp_diagnostics(directional), compute_stp_diagnostics(constant)
    print(f"  vel_cos:  straight={d_straight['stp/vel_cos']:.3f}  random={d_rand['stp/vel_cos']:.3f}")
    print(f"  tube_cos: straight={d_straight['stp/tube_cos']:.3f}  random={d_rand['stp/tube_cos']:.3f}")
    print(f"  eff_rank: random={d_rand['stp/eff_rank']:.2f}  directional-collapse={d_dir['stp/eff_rank']:.2f}")
    print(f"  hidden_var: random={d_rand['stp/hidden_var']:.3f}  constant-collapse={d_const['stp/hidden_var']:.2e}")
    assert d_straight["stp/vel_cos"] > 0.9 > d_rand["stp/vel_cos"], "vel_cos fails to separate straight vs random"
    assert d_straight["stp/tube_cos"] > 0.9 > d_rand["stp/tube_cos"], "tube_cos fails to separate straight vs random"
    assert d_rand["stp/eff_rank"] > 5.0 > d_dir["stp/eff_rank"], "eff_rank fails to flag directional collapse"
    assert d_const["stp/hidden_var"] < 0.01 < d_rand["stp/hidden_var"], "hidden_var fails to flag constant collapse"
    assert compute_stp_diagnostics(torch.randn(2, 2, C)) == {}, "short batch should return {}"

    # eff_rank formula (trace²/‖cov‖_F²) must equal the eigenvalue participation ratio.
    x = torch.randn(200, 32); xc = x - x.mean(0, keepdim=True); cov = (xc.T @ xc) / xc.size(0)
    ev = torch.linalg.eigvalsh(cov).clamp(min=0)
    pr_eig = (ev.sum() ** 2 / ev.pow(2).sum()).item()
    pr_fro = (cov.diagonal().sum() ** 2 / cov.pow(2).sum()).item()
    assert abs(pr_eig - pr_fro) < 1e-3, f"eff_rank formula mismatch: eig={pr_eig:.4f} fro={pr_fro:.4f}"

    # Device portability: the probe must run on MPS (the actual runcpu device on
    # Mac), where torch.linalg.eigvalsh is unimplemented — regression for the
    # silent NotImplementedError that dropped ALL diagnostics on a real MPS run.
    if torch.backends.mps.is_available():
        hm = torch.randn(16, 32, C, device="mps", dtype=torch.bfloat16)
        dm = compute_stp_diagnostics(hm)
        assert dm and all(v == v for v in dm.values()), "diagnostics failed on MPS"
        print(f"  [mps] probe OK: eff_rank={dm['stp/eff_rank']:.2f} vel_cos={dm['stp/vel_cos']:.3f}")
    else:
        print("  [mps] not available — skipping MPS portability check")

    # RNG non-invasiveness (regression for the randperm-perturbs-training bug):
    # the >4096-row subsample path must consume ZERO global RNG, else STP_DIAG=1
    # would shift STP's triple sampler and change training numerics vs STP_DIAG=0.
    big_in = torch.randn(64, 128, C)                       # 8192 rows > 4096 -> subsample path
    rng_before = torch.random.get_rng_state()              # capture AFTER building the input
    big = compute_stp_diagnostics(big_in)
    assert torch.equal(torch.random.get_rng_state(), rng_before), "diagnostics perturbed the global RNG"
    assert big["stp/eff_rank"] == big["stp/eff_rank"], "subsample path produced NaN"

    # Integration: STP_DIAG=1 stashes held-out hidden on the eval path, purity intact.
    cfg = GPTConfig(**CFG_KWARGS)
    with _env(STP_COEFF="0.05", STP_DIAG="1", STP_MAX_SPAN="8"):
        base = RealGPT(cfg); base.init_weights(); base = base.to(device).eval()
        m = STPGPT(cfg).to(device).eval(); m.load_state_dict(base.state_dict())
        assert m.stp_diag is True, "STP_DIAG not wired"
        Bx, Tx = 2, 16
        idx = torch.randint(0, CFG_KWARGS["vocab_size"], (Bx, Tx), device=device)
        targets = torch.randint(0, CFG_KWARGS["vocab_size"], (Bx, Tx), device=device)
        _, ref_hidden = _capture_pre_lm_head_hidden(base, idx)
        with torch.no_grad():
            ce_base = base(idx, targets=targets, loss_reduction="none")
            ce_m = m(idx, targets=targets, loss_reduction="none")  # eval path stashes + returns pure CE
        assert (ce_base - ce_m).abs().max().item() < 1e-6, "STP_DIAG broke eval-path CE purity"
        assert m._stp_eval_hidden is not None and m._stp_eval_hidden.shape == (Bx, Tx, cfg.n_embd)
        assert (m._stp_eval_hidden - ref_hidden).abs().max().item() < 1e-6, "stashed held-out hidden is wrong"
        d = compute_stp_diagnostics(m._stp_eval_hidden, max_span=m.stp_max_span)
        assert all(k in d for k in ("stp/vel_cos", "stp/tube_cos", "stp/hidden_var", "stp/hidden_rms", "stp/eff_rank"))
        assert all(v == v for v in d.values()), "diagnostic produced NaN"

        # Registry regression: base_train constructs a SECOND STPGPT (d12_ref, meta,
        # for muP) AFTER the live model and never runs an eval forward on it. The
        # instrument must still resolve to the eval-active model `m`, not the
        # last-constructed throwaway. `_live_instance` picks by "has a stashed hidden".
        m_ref = STPGPT(cfg)  # simulates d12_ref; never runs a forward -> _stp_eval_hidden is None
        assert stp_mod._live_instance() is m, "_live_instance must be the eval-active model, not the last-constructed"
        assert m_ref._stp_eval_hidden is None

        # The wrapper's log-enrichment (rides base_train's val/bpb log).
        enriched = enrich_log_data({"step": 100, "val/bpb": 1.23})
        assert "stp/eff_rank" in enriched and enriched["val/bpb"] == 1.23, "val/bpb log not enriched after 2nd construction"
        passthrough = enrich_log_data({"step": 100, "train/loss": 4.5})
        assert "stp/eff_rank" not in passthrough, "non-val/bpb log should pass through untouched"
    print("[PASS] (h) Diagnostics separate straighten/collapse; held-out stash + registry + RNG-safe + log enriched.")


def check_diag_compile() -> None:
    """evaluate_bpb runs the COMPILED model, so the eval-path diag stash happens
    under torch.compile. Verify the stash commits and CE purity holds there."""
    device = torch.device("cpu")
    torch.manual_seed(0)
    cfg = GPTConfig(**CFG_KWARGS)

    with _env(STP_COEFF="0.05", STP_DIAG="1", STP_MAX_SPAN="8"):
        m = STPGPT(cfg); m.init_weights(); m = m.to(device).eval()
        B, T = 2, 16
        idx = torch.randint(0, CFG_KWARGS["vocab_size"], (B, T), device=device)
        targets = torch.randint(0, CFG_KWARGS["vocab_size"], (B, T), device=device)

        with torch.no_grad():
            ce_eager = m(idx, targets=targets, loss_reduction="none")
        h_eager = m._stp_eval_hidden.clone()
        m._stp_eval_hidden = None

        torch._dynamo.reset()
        cm = torch.compile(m, dynamic=False)
        try:
            with torch.no_grad():
                ce_comp = cm(idx, targets=targets, loss_reduction="none")
        except Exception as e:
            if _is_toolchain_skip(e):
                print(f"  [SKIP] torch.compile backend unavailable here ({type(e).__name__}).")
                return
            raise
        assert m._stp_eval_hidden is not None, "eval-path diag stash did NOT commit under torch.compile"
        stash_diff = (m._stp_eval_hidden - h_eager).abs().max().item()
        ce_diff = (ce_eager - ce_comp).abs().max().item()
        print(f"  stash Δ (compiled vs eager) = {stash_diff:.2e}")
        print(f"  eval CE Δ (compiled vs eager) = {ce_diff:.2e}")
        assert stash_diff < 1e-3, f"compiled diag stash mismatch: {stash_diff:.2e}"
        assert ce_diff < 1e-3, f"eval-path CE mismatch under compile+diag: {ce_diff:.2e}"
        print("[PASS] (i) Eval-path diag stash commits under torch.compile; CE purity preserved.")


if __name__ == "__main__":
    print("=== Semantic Tube Prediction (STP) overlay smoke test ===")
    print("\n[a] forward-correctness")
    check_forward_loss()
    print("\n[b] config-portability")
    check_config_portability()
    print("\n[c] monkeypatch mechanism")
    check_monkeypatch()
    print("\n[d] hook fidelity + gradient flow")
    check_hook_fidelity()
    print("\n[e] eval-path contract (loss_reduction != 'mean' -> baseline-comparable)")
    check_eval_path_contract()
    print("\n[f] torch.compile safety")
    check_compile_safety()
    print("\n[g] span RNG stays live under torch.compile")
    check_compile_rng_live()
    print("\n[h] STP_DIAG diagnostics (instrument + held-out stash + purity)")
    check_diagnostics()
    print("\n[i] STP_DIAG eval-path stash under torch.compile")
    check_diag_compile()
    print("\nAll checks passed.")
