"""SurfaceFactoringGPT — surface-factoring overlay on nanochat's GPT.

Design: ../ideas/surface_factoring/README.md (esp. "Architecture" + "Training
integration"). The tokenizer factors leading SPACE and first-letter CASE out of
token identity into per-token side-channel labels; this overlay predicts them.
Canonical config is surface-only — K_max=1 (one cap bit per single-word token,
conditioned exactly like the space bit); the K>1 multi-slot cap path
(cap_bits[K]/cap_mask[K], per-word-start slots, _cap_compose's slot sum) is a
generalization kept from the deprecated triple (phrase tokens spanning multiple
word-starts) and is no longer the target.

Per position the model is fed `(base_id, space_bit, cap_bits[K], cap_mask[K])`
and predicts the NEXT position's content token (head-1 = the existing lm_head)
and surface label (a space head + a cap head, conditioned on the next base
token). The surface label rides the INPUT too — added to `wte` and to
the per-layer value embeddings — so the trunk is information-equivalent to
baseline (whose `" The"`/`"the"` are distinct tokens).

This is an ARCHITECTURE-subclass overlay (it adds parameters + a copied
`forward`), not a pure-loss one. It therefore overrides the parameter-accounting
methods that assert exact coverage. Hyperparameters via env (read at __init__):
  - SURFACE_KMAX  : cap-slot tensor width (must match the tokenizer; required)
  - SURFACE_LAMBDA: training surface-loss weight λ (default 0.5)

Loss contract (review P3): TRAINING returns `content CE + λ·(space + cap)` BCE
(mean); EVAL (`loss_reduction='none'`, the bpb/CORE path) returns the UNWEIGHTED
joint per-token NLL (content + space + cap, no λ), masked at ignored targets.
"""
from __future__ import annotations

import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from nanochat.gpt import GPT, norm, COMPUTE_DTYPE, has_ve
from overlay.surface_lambda import current as _current_lambda   # optional dynamic-λ schedule


class _CastLinear(nn.Linear):
    """Like nanochat.gpt.Linear (casts the fp32 master weight to the activation
    dtype in forward, replacing autocast — required under torch.compile/inductor,
    which won't promote a bf16 activation × fp32 weight matmul) but ALSO casts the
    bias, since the surface heads use one (nanochat's bias-free Linear drops it)."""
    def forward(self, x):
        b = self.bias.to(dtype=x.dtype) if self.bias is not None else None
        return F.linear(x, self.weight.to(dtype=x.dtype), b)


# nanochat's distributed AdamW (optim.py) reduce_scatters every param with >=1024
# elements along dim 0, asserting shape[0] % world_size == 0; params <1024 elements
# take a safe all_reduce path instead. The surface params have tiny leading dims
# (space=2, cap=2K, heads=1/K) but are >=1024 elements, so on >1 GPU they hit the
# assert. Pad each leading dim up to a FIXED multiple of 8 (covers world_size in
# {1,2,4,8}; fixed — not world_size-derived — so the checkpoint shape is identical
# across GPU counts and loads under base_eval at world_size=1). Padded rows are never
# indexed and padded head outputs are sliced off before the loss => zero gradient =>
# inert (they only weight-decay).
_SHARD = 8
def _pad(n: int) -> int:
    return ((n + _SHARD - 1) // _SHARD) * _SHARD


def _kmax_from_env() -> int:
    k = os.environ.get("SURFACE_KMAX")
    assert k is not None, "SURFACE_KMAX must be set (cap-slot width from the tokenizer)"
    return int(k)


def _lambda_from_env() -> float:
    return float(os.environ.get("SURFACE_LAMBDA", "0.5"))


class SurfaceFactoringGPT(GPT):
    def __init__(self, config):
        super().__init__(config)
        self.surface_kmax = _kmax_from_env()
        self.surface_lambda = _lambda_from_env()
        K, d = self.surface_kmax, config.n_embd
        head_dim = config.n_embd // config.n_head
        kv_dim = config.n_kv_head * head_dim
        # input-side surface embeddings (added to wte; slot-specific cap emb). Leading
        # dims padded to _SHARD for the distributed AdamW reduce_scatter (see _pad);
        # logical widths are 2 (space bit) and 2K (cap slot*2+bit); padded rows unused.
        self.space_emb = nn.Embedding(_pad(2), d)
        self.cap_emb = nn.Embedding(_pad(2 * K), d)      # index = slot*2 + bit (< 2K)
        # surface-conditioned value embeddings, per ve-layer (full info-equivalence)
        ve_layers = [str(i) for i in range(config.n_layer) if has_ve(i, config.n_layer)]
        self.space_value_embeds = nn.ModuleDict({i: nn.Embedding(_pad(2), kv_dim) for i in ve_layers})
        self.cap_value_embeds = nn.ModuleDict({i: nn.Embedding(_pad(2 * K), kv_dim) for i in ve_layers})
        # output heads (token-conditioned: take [h, wte(next_base)]); _CastLinear so the
        # bf16-activation × fp32-weight matmul is dtype-consistent under compile. Output
        # widths padded to _SHARD (logical 1 for space, K for cap); extras sliced off.
        self.space_head = _CastLinear(2 * d, _pad(1))
        self.cap_head = _CastLinear(2 * d, _pad(K))

    # ---- param accounting (must include the new params) -------------------- #
    def _surface_modules(self):
        return [self.space_emb, self.cap_emb, self.space_value_embeds,
                self.cap_value_embeds, self.space_head, self.cap_head]

    def _surface_params(self):
        return [p for m in self._surface_modules() for p in m.parameters()]

    def num_scaling_params(self):
        g = lambda mods: sum(p.numel() for m in mods for p in m.parameters())
        wte = sum(p.numel() for p in self.transformer.wte.parameters())
        value_embeds = sum(p.numel() for p in self.value_embeds.parameters())
        lm_head = sum(p.numel() for p in self.lm_head.parameters())
        tm = sum(p.numel() for p in self.transformer.h.parameters())
        scalars = (self.resid_lambdas.numel() + self.x0_lambdas.numel()
                   + self.smear_gate.weight.numel() + self.smear_lambda.numel()
                   + self.backout_lambda.numel())
        surface = sum(p.numel() for p in self._surface_params())
        total = wte + value_embeds + lm_head + tm + scalars + surface
        assert total == sum(p.numel() for p in self.parameters()), "Parameter count mismatch"
        return dict(wte=wte, value_embeds=value_embeds, lm_head=lm_head,
                    transformer_matrices=tm, scalars=scalars, surface=surface, total=total)

    def estimate_flops(self):
        nparams = sum(p.numel() for p in self.parameters())
        ve = sum(v.weight.numel() for v in self.value_embeds.values())
        # exclude all embedding/scalar params from the 6·matmul term; the surface
        # *embeddings* are lookups, the surface *heads* are matmuls (kept in).
        surf_emb = (self.space_emb.weight.numel() + self.cap_emb.weight.numel()
                    + sum(p.numel() for m in [self.space_value_embeds, self.cap_value_embeds]
                          for p in m.parameters()))
        exclude = (self.transformer.wte.weight.numel() + ve
                   + self.resid_lambdas.numel() + self.x0_lambdas.numel()
                   + self.smear_gate.weight.numel() + self.smear_lambda.numel()
                   + self.backout_lambda.numel() + surf_emb)
        h, q, t = self.config.n_head, self.config.n_embd // self.config.n_head, self.config.sequence_len
        attn = sum(12 * h * q * (t if w[0] < 0 else min(w[0], t)) for w in self.window_sizes)
        return 6 * (nparams - exclude) + attn

    def setup_optimizer(self, unembedding_lr=0.004, embedding_lr=0.2, matrix_lr=0.02,
                        weight_decay=0.0, scalar_lr=0.5):
        # Mirrors GPT.setup_optimizer, adding the surface params: surface
        # embeddings -> embedding-style AdamW, surface heads -> lm_head-style AdamW.
        from nanochat.gpt import get_dist_info, DistMuonAdamW, MuonAdamW
        model_dim = self.config.n_embd
        ddp, rank, local_rank, world_size = get_dist_info()
        matrix_params = list(self.transformer.h.parameters())
        value_embeds_params = list(self.value_embeds.parameters())
        embedding_params = list(self.transformer.wte.parameters())
        lm_head_params = list(self.lm_head.parameters())
        resid_params = [self.resid_lambdas]
        x0_params = [self.x0_lambdas]
        smear_params = [self.smear_gate.weight, self.smear_lambda, self.backout_lambda]
        surf_emb_params = (list(self.space_emb.parameters()) + list(self.cap_emb.parameters())
                           + list(self.space_value_embeds.parameters())
                           + list(self.cap_value_embeds.parameters()))
        surf_head_params = list(self.space_head.parameters()) + list(self.cap_head.parameters())
        assert len(list(self.parameters())) == (
            len(matrix_params) + len(embedding_params) + len(lm_head_params)
            + len(value_embeds_params) + len(resid_params) + len(x0_params)
            + len(smear_params) + len(surf_emb_params) + len(surf_head_params)
        ), "Parameter coverage mismatch (surface params not grouped)"
        s = (model_dim / 768) ** -0.5
        param_groups = [
            dict(kind='adamw', params=lm_head_params, lr=unembedding_lr * s, betas=(0.8, 0.96), eps=1e-10, weight_decay=0.01),
            dict(kind='adamw', params=embedding_params, lr=embedding_lr * s, betas=(0.8, 0.995), eps=1e-10, weight_decay=0.001),
            dict(kind='adamw', params=value_embeds_params, lr=embedding_lr * s * 0.5, betas=(0.8, 0.995), eps=1e-10, weight_decay=0.01),
            dict(kind='adamw', params=resid_params, lr=scalar_lr * 0.01, betas=(0.8, 0.95), eps=1e-10, weight_decay=0.05),
            dict(kind='adamw', params=x0_params, lr=scalar_lr, betas=(0.96, 0.95), eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=smear_params, lr=0.2, betas=(0.8, 0.95), eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=surf_emb_params, lr=embedding_lr * s, betas=(0.8, 0.995), eps=1e-10, weight_decay=0.001),
            dict(kind='adamw', params=surf_head_params, lr=unembedding_lr * s, betas=(0.8, 0.96), eps=1e-10, weight_decay=0.01),
        ]
        for shape in sorted({p.shape for p in matrix_params}):
            gp = [p for p in matrix_params if p.shape == shape]
            param_groups.append(dict(kind='muon', params=gp, lr=matrix_lr,
                                     momentum=0.95, ns_steps=5, beta2=0.9, weight_decay=weight_decay))
        optimizer = (DistMuonAdamW if ddp else MuonAdamW)(param_groups)
        for group in optimizer.param_groups:
            group["initial_lr"] = group["lr"]
        return optimizer

    def init_weights(self):
        super().init_weights()
        surf_embs = [self.space_emb, self.cap_emb, *self.space_value_embeds.values(),
                     *self.cap_value_embeds.values()]
        for emb in surf_embs:
            nn.init.normal_(emb.weight, mean=0.0, std=0.02)
        for head in [self.space_head, self.cap_head]:
            nn.init.normal_(head.weight, mean=0.0, std=0.02)
            nn.init.zeros_(head.bias)
        # Match nanochat: cast the surface embeddings to COMPUTE_DTYPE (like wte +
        # value_embeds, gpt.py) so their adds with the bf16 base embeddings are
        # dtype-consistent. Heads stay fp32 (the _CastLinear casts per-forward).
        if COMPUTE_DTYPE != torch.float16:
            for emb in surf_embs:
                emb.to(dtype=COMPUTE_DTYPE)

    # ---- surface input composition ---------------------------------------- #
    def _cap_compose(self, table, cap_bits, cap_mask):
        """Σ_s mask_s · table[slot s · 2 + bit_s] — slot-specific, injective."""
        K = cap_bits.size(-1)
        slot = torch.arange(K, device=cap_bits.device)
        ce = table(slot * 2 + cap_bits)                 # (B,T,K,dim)
        return (ce * cap_mask.unsqueeze(-1)).sum(dim=2)

    def _compose_ve(self, layer, idx, space_x, cap_bits_x, cap_mask_x):
        """Surface-conditioned value embedding for a ve-layer (full info-
        equivalence): base value embedding + space + slot-specific cap."""
        ve = self.value_embeds[layer](idx)
        ve = ve + self.space_value_embeds[layer](space_x)
        ve = ve + self._cap_compose(self.cap_value_embeds[layer], cap_bits_x, cap_mask_x)
        return ve

    # ---- forward (copied from GPT.forward + surface injections) ------------ #
    def forward(self, idx, targets=None, kv_cache=None, loss_reduction='mean',
                space_x=None, cap_bits_x=None, cap_mask_x=None,
                space_y=None, cap_bits_y=None, cap_mask_y=None,
                return_surface_logits=False):
        # Harness path: upstream base_train calls model(x, y) with no surface
        # kwargs. Pull the current batch's surface streams from the out-of-band
        # context the surface dataloader set (see overlay/surface_context.py).
        if space_x is None and targets is not None and kv_cache is None:
            from overlay.surface_context import get_surface
            sc = get_surface()
            if sc is not None:
                space_x, cap_bits_x, cap_mask_x = sc["space_x"], sc["cap_bits_x"], sc["cap_mask_x"]
                space_y, cap_bits_y, cap_mask_y = sc["space_y"], sc["cap_bits_y"], sc["cap_mask_y"]
        # No surface inputs -> behave exactly like base GPT (generation / no-surface).
        if space_x is None:
            return super().forward(idx, targets, kv_cache, loss_reduction)
        assert kv_cache is None, "surface dual-stream forward is for training/eval (no kv_cache)"
        B, T = idx.size()
        cos_sin = self.cos[:, :T], self.sin[:, :T]

        # Embed: wte + surface (space + slot-specific cap), then norm.
        x = self.transformer.wte(idx)
        x = x + self.space_emb(space_x) + self._cap_compose(self.cap_emb, cap_bits_x, cap_mask_x)
        x = x.to(COMPUTE_DTYPE)
        x = norm(x)
        # Smear (training path: T > 1)
        gate = self.smear_lambda.to(x.dtype) * torch.sigmoid(self.smear_gate(x[:, 1:, :24]))
        x = torch.cat([x[:, :1], x[:, 1:] + gate * x[:, :-1]], dim=1)

        # Trunk with surface-conditioned value embeddings.
        x0 = x
        backout_layer = self.config.n_layer // 2
        x_backout = None
        for i, block in enumerate(self.transformer.h):
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            ve = None
            if str(i) in self.value_embeds:
                ve = self._compose_ve(str(i), idx, space_x, cap_bits_x, cap_mask_x).to(x.dtype)
            x = block(x, ve, cos_sin, self.window_sizes[i], kv_cache)
            if i == backout_layer:
                x_backout = x
        if x_backout is not None:
            x = x - self.backout_lambda.to(x.dtype) * x_backout
        x = norm(x)                                      # x = final hidden state h

        # Content head (= lm_head + softcap), as in base.
        softcap = 15
        logits = self.lm_head(x)[..., :self.config.vocab_size].float()
        logits = softcap * torch.tanh(logits / softcap)
        if targets is None:
            return logits

        # reshape (not view): loader streams are non-contiguous slices (row[:, 1:])
        content = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1),
                                  ignore_index=-1, reduction='none').view(B, T)

        # Surface heads, conditioned on the next base token (teacher-forced = targets).
        # .float() the logits (like the content path) so the BCE runs in fp32 against
        # the fp32 targets — the _CastLinear heads emit in the activation dtype (bf16).
        nxt = self.transformer.wte(targets.clamp_min(0))
        cond = torch.cat([x, nxt], dim=-1)
        # slice off the _SHARD padding to the logical widths (1 for space, K for cap)
        space_logit = self.space_head(cond)[..., 0].float()          # (B,T)
        cap_logit = self.cap_head(cond)[..., :self.surface_kmax].float()   # (B,T,K)
        if return_surface_logits:
            # eval (lossless exact-match): raw content + surface-head logits so the
            # caller can argmax content + threshold space/cap. No targets-y / loss path.
            return logits, space_logit, cap_logit
        space_nll = F.binary_cross_entropy_with_logits(space_logit, space_y.float(),
                                                       reduction='none')
        cap_nll = (F.binary_cross_entropy_with_logits(cap_logit, cap_bits_y.float(),
                                                      reduction='none') * cap_mask_y.float()
                   ).sum(dim=-1)                                     # (B,T)

        valid = (targets != -1)
        if loss_reduction == 'none':
            # EVAL: unweighted joint per-token NLL (no λ), masked at ignored targets.
            joint = content + (space_nll + cap_nll) * valid
            return joint
        # TRAINING: content CE + λ·(space + cap), mean over valid positions.
        v = valid.float()
        denom = v.sum().clamp_min(1.0)
        content_mean = (content * v).sum() / denom
        surface_mean = ((space_nll + cap_nll) * v).sum() / denom
        # λ is static by default; with SURFACE_LAMBDA_SCHED set it decays over training
        # (favor content as the easy surface heads converge). See overlay/surface_lambda.py.
        return content_mean + _current_lambda(self.surface_lambda) * surface_mean
