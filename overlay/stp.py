"""STPGPT — Semantic Tube Prediction overlay on nanochat's GPT.

Design rationale: ../ideas/semantic-tube/README.md.
Paper: Huang, LeCun, Balestriero 2026, arXiv 2602.22617 (ICML 2026).

Adds a JEPA-style "semantic tube" regularizer on top of the standard
next-token cross-entropy. For random token indices s < r < t along each
sequence, with last-layer hidden states h_s, h_r, h_t, the term is

    L_STP = 1 - cos(h_t - h_r, h_r - h_s)

i.e. it drives consecutive velocity (difference) vectors to be collinear,
confining the hidden-state trajectory to a "tube" around a local geodesic
(the paper's Geodesic Hypothesis: token sequences are locally linear in
representation space). Training loss = CE + coeff * L_STP; the inference and
eval paths are unchanged from the base.

Unlike LLM-JEPA / VL-JEPA, STP needs no paired views — it constrains the
model's own hidden-state trajectory — so it applies to nanochat's plain
next-token stream.

Key implementation points (full rationale in the idea doc):

- **Hidden states via a forward pre-hook on `lm_head`.** nanochat's
  `GPT.forward` returns logits/loss, not hidden states. We capture the input
  to `lm_head` — the post-final-norm hidden `x` (gpt.py:465, the HF
  `hidden_states[-1]` analog) — with a pre-hook that stashes it into
  `self._stp_hidden`. No copy of the upstream forward; trunk changes stay
  transparent. The training forward runs under `torch.compile` (base_train.py
  compiles the model at :246 and trains via `loss = model(x, y)`); the
  set-in-hook / read-in-forward happen within one compiled region, so Dynamo
  connects them (or, worst case, graph-breaks and falls back to eager — still
  correct). Smoke check (f) verifies loss + grads under compile.

- **Last layer only** — the paper's headline `random_span_layer=-1`.
  Intermediate-layer STP would hook a transformer block instead and is left
  as future work (idea doc "Open questions").

- **Local span window.** The paper fine-tunes short single-example sequences;
  nanochat packs many documents into T~2048 rows, so uniform-over-row triples
  would almost all straddle document boundaries and span nonsense. We draw
  triples within a bounded window of width STP_MAX_SPAN to keep them local
  (more likely intra-document) and faithful to the *local* linearity claim.
  Set STP_MAX_SPAN >= sequence_len to recover whole-sequence sampling.
  (Packed nanochat rows have no padding, so every hidden state is a real
  trajectory point — no attention-mask gating is needed, unlike the paper's
  padded fine-tuning batches.)

- **Hyperparameters via env, not GPTConfig fields** (STP_COEFF, STP_SPANS,
  STP_MAX_SPAN), read at __init__, stored as instance attributes — keeps the
  saved checkpoint config-identical to baseline so base_eval / chat_sft /
  chat_cli load it as a vanilla GPT. See ../overlay/README.md "Hyperparam
  placement".

- **Eval-path purity.** On the eval path (loss_reduction != 'mean', used by
  nanochat/loss_eval.py for val/bpb) the STP term is dropped and plain CE is
  returned, so bpb stays baseline-comparable. Smoke check (e) enforces this.
"""
from __future__ import annotations

import os
import weakref

import torch
import torch.nn.functional as F

from nanochat.gpt import GPT


def _stp_coeff_from_env() -> float:
    return float(os.environ.get("STP_COEFF", "0.02"))


def _stp_spans_from_env() -> int:
    return int(os.environ.get("STP_SPANS", "1"))


def _stp_max_span_from_env() -> int:
    return int(os.environ.get("STP_MAX_SPAN", "256"))


def _stp_diag_from_env() -> bool:
    return os.environ.get("STP_DIAG", "0") == "1"


# Registry of live STPGPT instances, so the wandb.init patch in
# wrappers/train_stp.py can reach the training model's stashed held-out hidden
# without a handle to it. NOT "last constructed": base_train builds a SECOND
# model (d12_ref = build_model_meta(12), for muP scaling) after the live one,
# and it is an STPGPT too, so "last constructed" would point at that meta-device
# throwaway (which never runs an eval forward — _stp_eval_hidden stays None) and
# the instrument would log nothing. Instead we track all instances in a WeakSet
# (no leak / no keeping the throwaway alive) and pick the one that actually
# stashed a held-out hidden. Under DDP each rank has its own set; only rank 0 logs.
_INSTANCES: "weakref.WeakSet" = weakref.WeakSet()


def _register_instance(inst) -> None:
    _INSTANCES.add(inst)


def _live_instance():
    """The instance that actually ran an eval forward on data (its
    `_stp_eval_hidden` is set) — the training model, not the meta-device ref."""
    for inst in _INSTANCES:
        if getattr(inst, "_stp_eval_hidden", None) is not None:
            return inst
    return None


@torch.no_grad()
def compute_stp_diagnostics(hidden: torch.Tensor, max_span: int | None = None) -> dict:
    """Cheap geometric diagnostics on a hidden-state batch `(B,T,C)`, for the
    STP_DIAG instrument. Separates "is the trajectory straightening" from "is it
    collapsing" — the failure mode STP has no explicit guard against:

    - `stp/vel_cos`  — mean cos of ADJACENT-position velocities (h_{t+1}-h_t vs
      h_t-h_{t-1}). The most local straightening readout (span 1).
    - `stp/tube_cos` — mean cos of two consecutive length-d velocity segments
      (h_{t+2d}-h_{t+d} vs h_{t+d}-h_t), at span d ≈ `max_span`/3 (the mean
      sub-span of STP's sampled triples). This is the deterministic, equal-span
      analog of STP's ACTUAL objective (loss = 1 - cos over random triples s<r<t
      within the window) at the scale STP operates on — a far better "is STP
      straightening?" readout than the adjacent `vel_cos`. Deterministic (no RNG,
      so it can't perturb training and is reproducible across evals) unlike the
      random-triples loss. Rises toward 1 as long-range trajectories straighten.
    - `stp/hidden_var` — mean per-dim variance of the hidden. → 0 under collapse.
      (Computed post-final-RMSNorm, so ~unit-bounded; only a fall toward 0 is
      informative.)
    - `stp/hidden_rms` — RMS magnitude of the hidden (post-norm ⇒ ~1.0; sanity only).
    - `stp/eff_rank` — participation ratio (Σλ)²/Σλ² of the hidden covariance
      (an effective rank). → 1 under directional collapse; large when
      representations span many dimensions.

    Eager + no_grad; rows are subsampled to bound the covariance cost at scale.
    Returns {} if the batch is too short to be meaningful (e.g. T<3 decode steps).
    """
    h = hidden.detach().float()
    if h.dim() != 3 or h.size(1) < 3:
        return {}
    B, T, C = h.shape
    out = {}

    v = h[:, 1:] - h[:, :-1]                                   # (B,T-1,C)
    cos = F.cosine_similarity(v[:, 1:], v[:, :-1], dim=-1)     # (B,T-2)
    out["stp/vel_cos"] = cos.mean().item()

    # Windowed straightening at STP's operating scale (deterministic, RNG-free).
    span = max_span if max_span is not None else T
    d = max(1, min(span, T - 1) // 3)
    if T > 2 * d:
        va = h[:, d:T - d] - h[:, :T - 2 * d]                  # (B,T-2d,C) span-d velocity at t
        vb = h[:, 2 * d:T] - h[:, d:T - d]                     # (B,T-2d,C) next span-d velocity
        out["stp/tube_cos"] = F.cosine_similarity(vb, va, dim=-1).mean().item()

    flat = h.reshape(-1, C)                                    # (N,C)
    if flat.size(0) > 4096:                                    # bound eig cost at scale
        # Deterministic strided subsample — NOT torch.randperm, which would draw
        # from the global default generator that STP's _sample_triples also uses,
        # perturbing training numerics whenever STP_DIAG=1 (the ON run must stay
        # numerically identical to the OFF run it instruments). Strided is RNG-free
        # and reproducible; evenly spread across the (B,T) rows is representative.
        step = flat.size(0) // 4096
        flat = flat[::step][:4096]
    out["stp/hidden_var"] = flat.var(dim=0, unbiased=False).mean().item()
    out["stp/hidden_rms"] = flat.pow(2).mean().sqrt().item()

    fc = flat - flat.mean(dim=0, keepdim=True)
    cov = (fc.T @ fc) / max(fc.size(0), 1)                     # (C,C), symmetric PSD
    # Participation ratio (Σλ)²/Σλ² WITHOUT eigendecomposition: for symmetric
    # cov, Σλ = trace(cov) and Σλ² = trace(cov²) = ‖cov‖_F². Same value, cheaper,
    # and device-agnostic — torch.linalg.eigvalsh is NOT implemented on MPS
    # (it raised NotImplementedError there, silently disabling the whole probe).
    tr = cov.diagonal().sum()
    fro2 = cov.pow(2).sum()
    out["stp/eff_rank"] = float((tr * tr / fro2.clamp(min=1e-12)).item()) if tr > 0 else 0.0
    return out


_DIAG_WARNED = False


def enrich_log_data(data: dict) -> dict:
    """Merge held-out STP diagnostics into a base_train wandb log dict, keyed on
    the `val/bpb` entry (which fires right after evaluate_bpb, so the live
    model's stashed eval hidden is fresh). Returns `data` unchanged when the
    entry isn't a val/bpb log or no diagnostic hidden is available. Never raises
    — diagnostics must never break real training logging — but prints a one-time
    warning if the probe errors, so a failure is visible rather than silent (an
    earlier version swallowed a NotImplementedError on MPS and logged nothing)."""
    global _DIAG_WARNED
    try:
        if not (isinstance(data, dict) and "val/bpb" in data):
            return data
        inst = _live_instance()
        h = getattr(inst, "_stp_eval_hidden", None) if inst is not None else None
        if h is None:
            return data
        return {**data, **compute_stp_diagnostics(h, max_span=getattr(inst, "stp_max_span", None))}
    except Exception as e:
        if not _DIAG_WARNED:
            _DIAG_WARNED = True
            print(f"[STP_DIAG] diagnostics disabled after error: {type(e).__name__}: {e}", flush=True)
        return data


class STPGPT(GPT):
    """GPT + Semantic Tube Prediction auxiliary loss. Hyperparameters live as
    instance attributes (not config fields). Forward signature matches the base.
    """

    def __init__(self, config):
        super().__init__(config)
        self.stp_coeff = _stp_coeff_from_env()
        self.stp_spans = _stp_spans_from_env()
        self.stp_max_span = _stp_max_span_from_env()
        self.stp_diag = _stp_diag_from_env()
        # Capture the input to lm_head (post-final-norm hidden) each forward.
        self._stp_hidden = None
        # Held-out hidden stashed on the eval path for the STP_DIAG instrument
        # (read by the wandb.init patch in wrappers/train_stp.py). Off by default.
        self._stp_eval_hidden = None
        self.lm_head.register_forward_pre_hook(self._stp_capture_hook)
        _register_instance(self)

    def _stp_capture_hook(self, module, args):
        # forward pre-hook: args[0] is lm_head's input = post-norm hidden (B,T,C)
        self._stp_hidden = args[0]

    def _sample_triples(self, B: int, T: int, device) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (rows, idxs): `rows` (N,) is the row index per triple and
        `idxs` (N,3) holds strictly-increasing token positions s<r<t drawn
        within a window of width min(STP_MAX_SPAN, T). N = B * STP_SPANS.

        Requires min(STP_MAX_SPAN, T) >= 3 (caller guarantees via forward's
        guard). Triples are uniform over distinct positions inside the window.
        """
        W = min(self.stp_max_span, T)
        N = B * self.stp_spans
        # window start s0 in [0, T-W]; three distinct sorted offsets in [0, W)
        s0 = torch.randint(0, T - W + 1, (N, 1), device=device)          # (N,1)
        offs = torch.rand(N, W, device=device).argsort(dim=-1)[:, :3]    # (N,3) distinct in [0,W)
        offs, _ = offs.sort(dim=-1)                                      # s<r<t within window
        idxs = s0 + offs                                                # (N,3) absolute positions
        rows = torch.arange(B, device=device).repeat_interleave(self.stp_spans)  # (N,)
        return rows, idxs

    def _stp_loss(self, hidden: torch.Tensor) -> torch.Tensor:
        """1 - mean cos(h_t - h_r, h_r - h_s) over sampled triples."""
        B, T, _ = hidden.shape
        rows, idxs = self._sample_triples(B, T, hidden.device)
        h_sel = hidden[rows.unsqueeze(1), idxs].float()   # (N,3,C)
        v1 = h_sel[:, 1] - h_sel[:, 0]                    # r - s
        v2 = h_sel[:, 2] - h_sel[:, 1]                    # t - r
        cos = F.cosine_similarity(v2, v1, dim=-1)         # (N,)
        return 1.0 - cos.mean()

    def forward(self, idx, targets=None, kv_cache=None, loss_reduction='mean'):
        # Inference / no targets: pass through to base (returns logits).
        if targets is None:
            return super().forward(idx, targets=None, kv_cache=kv_cache, loss_reduction=loss_reduction)

        logits = super().forward(idx, targets=None, kv_cache=kv_cache)  # (B,T,V); hook stashes hidden
        V = logits.size(-1)

        ce = F.cross_entropy(
            logits.view(-1, V),
            targets.view(-1),
            ignore_index=-1,
            reduction=loss_reduction,
        )

        # Diagnostics (opt-in, STP_DIAG=1): stash the held-out hidden on the eval
        # path (loss_reduction != 'mean' is evaluate_bpb over val data) for the
        # wandb.init patch in wrappers/train_stp.py to read at val/bpb log time.
        # This is a side effect only — it never changes the returned CE, so the
        # eval-path purity contract (smoke (e)) still holds.
        if self.stp_diag and loss_reduction != 'mean':
            self._stp_eval_hidden = self._stp_hidden.detach()

        # Skip STP and return plain CE when:
        #   - disabled (coeff == 0), or
        #   - eval path (loss_reduction != 'mean'): nanochat/loss_eval.py calls
        #     with 'none' to compute per-token val/bpb; adding a scalar STP term
        #     would inflate bpb and break baseline comparability, and
        #     conceptually val/bpb must reflect pure next-token prediction, or
        #   - the window is too short to form a triple (min(max_span, T) < 3).
        T = idx.size(1)
        if self.stp_coeff == 0.0 or loss_reduction != 'mean' or min(self.stp_max_span, T) < 3:
            return ce

        stp = self._stp_loss(self._stp_hidden)
        return ce + self.stp_coeff * stp
