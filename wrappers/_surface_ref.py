"""
Self-contained REFERENCE implementation of the surface-factoring contract, used
by the smoke tests (wrappers/smoke_surface_factoring*.py). It is NOT the
production overlay — it exists to prove the model-side spec in
ideas/surface_factoring/README.md is implementable and the pieces fit, before
building the real overlay/dataloader/harness and running runcpu.sh.

NOTE: production surface is now surface-only (baseline vocab, K_max=1 — one cap bit
per single-word token, no force_merges). This reference deliberately uses a 2-phrase
(K>1) tokenizer fixture to keep the GENERAL multi-slot cap machinery exercised by the
smokes; it is not the production tokenizer.

Provides:
  - build_tiny_tokenizer(): a tiny spaceless + case-folded + 2-phrase tokenizer
    (rustbpe_force_merges), trained inline on a small corpus — no data files.
  - surface_encode(text)/surface_decode(...): the dual-stream contract
    (base_id, space_bit, cap_bits[K_max], cap_mask[K_max]) per token, lossless.
    Cap slots = the LETTER word-start byte offsets within a token (a fixed
    candidate set derivable from the token's bytes = the mask); the cap VALUES
    are per-occurrence (whether that word-start was titlecase-folded). This is
    the "conservative fixed candidate-offset scheme with masks" the review asked
    for, and resolves the per-occurrence subtlety: a continuation token gets a
    slot too, but its value is ~always 0 (learned from context).
  - MiniSurfaceGPT / MiniBaselineGPT: a tiny causal transformer with the surface
    embedding (wte + space_emb + masked-sum cap_emb) + content/space/cap heads +
    the joint loss, and a baseline twin for the reduces-to-baseline invariant.
"""
from __future__ import annotations

import math

import regex
import torch
import torch.nn as nn
import torch.nn.functional as F

import rustbpe_force_merges
import tiktoken

from nanochat.tokenizer import SPECIAL_TOKENS
from tools.case_factor_compression import fold_case
from tools.stack_fm_spaceless import build_config

# A tiny inline training corpus: prose with caps, single spaces, and the two
# forced phrases (" of the", " in the") repeated enough to merge.
_CORPUS = (
    "The cat sat on the mat. One of the dogs ran in the park. "
    "She was the best of the team. In the morning the sun rose over the hills. "
    "It is the case that the rain in the valley fills the river of the land. "
    "Of The Rings was a book. To do in the city was the plan of the day. "
) * 200

_PHRASES = [(" of", " the"), (" in", " the")]  # baseline-form forced pairs


def build_tiny_tokenizer(vocab_size=320):
    """Train a tiny spaceless + folded-phrase tokenizer inline (no data files)."""
    cfg = build_config(_PHRASES)  # spaceless pattern + leading-space-stripped pairs
    # case-fold the corpus + the phrases (the triple recipe)
    def folded_pairs():
        out, seen = [], set()
        for a, b in cfg["spaceless_forced"]:
            fa, fb = fold_case(a), fold_case(b)
            if (fa, fb) not in seen:
                seen.add((fa, fb)); out.append((fa, fb))
        return out
    forced = folded_pairs()
    docs = [fold_case(d) for d in [_CORPUS[i:i + 4000] for i in range(0, len(_CORPUS), 4000)]]
    tok = rustbpe_force_merges.Tokenizer()
    tok.train_from_iterator(iter(docs), vocab_size - len(SPECIAL_TOKENS),
                            pattern=cfg["spaceless_pattern"], forced_pairs=forced,
                            block_trailing_space=True, block_leading_space=True)
    mr = {bytes(k): v for k, v in tok.get_mergeable_ranks()}
    offset = len(mr)
    special = {nm: offset + i for i, nm in enumerate(SPECIAL_TOKENS)}
    enc = tiktoken.Encoding(name="surface-tiny", pat_str=tok.get_pattern(),
                            mergeable_ranks=mr, special_tokens=special)
    sre = regex.compile(cfg["spaceless_pattern"])
    id2len = [0] * enc.n_vocab
    for tid in range(enc.n_vocab):
        try:
            id2len[tid] = len(enc.decode_single_token_bytes(tid))
        except Exception:
            id2len[tid] = 0
    return enc, sre, id2len


# The dual-stream codec lives in overlay/surface_codec.py (production module);
# re-exported here under the names the smokes import. `_char_word_starts` keeps
# its leading underscore for the existing smoke import.
from overlay.surface_codec import (  # noqa: E402
    char_word_starts as _char_word_starts, compute_kmax, surface_encode, surface_decode,
)


# --------------------------------------------------------------------------- #
# Minimal causal transformer with the surface contract.

class _Block(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.h = h
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.proj = nn.Linear(d, d, bias=False)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
        self.n1 = nn.LayerNorm(d)
        self.n2 = nn.LayerNorm(d)

    def forward(self, x):
        B, T, D = x.shape
        qkv = self.qkv(self.n1(x))
        q, k, v = qkv.split(D, dim=2)
        q = q.view(B, T, self.h, D // self.h).transpose(1, 2)
        k = k.view(B, T, self.h, D // self.h).transpose(1, 2)
        v = v.view(B, T, self.h, D // self.h).transpose(1, 2)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        a = a.transpose(1, 2).reshape(B, T, D)
        x = x + self.proj(a)
        x = x + self.mlp(self.n2(x))
        return x


class MiniBaselineGPT(nn.Module):
    """Plain content-only model — the reduces-to-baseline reference."""
    def __init__(self, vocab, d=64, h=4, n_layer=2, T=64):
        super().__init__()
        self.wte = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(T, d)
        self.blocks = nn.ModuleList([_Block(d, h) for _ in range(n_layer)])
        self.norm = nn.LayerNorm(d)
        self.lm_head = nn.Linear(d, vocab, bias=False)

    def trunk(self, base_x):
        B, T = base_x.shape
        x = self.wte(base_x) + self.pos(torch.arange(T, device=base_x.device))
        for b in self.blocks:
            x = b(x)
        return self.norm(x)

    def forward(self, base_x, base_y):
        h = self.trunk(base_x)
        logits = self.lm_head(h)
        return F.cross_entropy(logits.view(-1, logits.size(-1)), base_y.view(-1))


class MiniSurfaceGPT(nn.Module):
    """Content + surface (space + per-word-start cap) heads, surface input emb."""
    def __init__(self, vocab, K_max, d=64, h=4, n_layer=2, T=64, lam=0.5):
        super().__init__()
        self.K_max = K_max
        self.lam = lam
        self.wte = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(T, d)
        self.space_emb = nn.Embedding(2, d)
        # SLOT-SPECIFIC cap embedding: index = slot*2 + bit. A single shared 0/1
        # embedding would make Σ over slots non-injective ([1,0]==[0,1]).
        self.cap_emb = nn.Embedding(2 * K_max, d)
        self.blocks = nn.ModuleList([_Block(d, h) for _ in range(n_layer)])
        self.norm = nn.LayerNorm(d)
        self.lm_head = nn.Linear(d, vocab, bias=False)
        self.space_head = nn.Linear(2 * d, 1)        # conditioned on next token
        self.cap_head = nn.Linear(2 * d, K_max)      # per-slot, conditioned on next token

    def compose_input(self, base_x, space_x, cap_bits, cap_mask):
        B, T = base_x.shape
        emb = self.wte(base_x) + self.pos(torch.arange(T, device=base_x.device))
        emb = emb + self.space_emb(space_x)
        # masked sum over SLOT-SPECIFIC cap embeddings: Σ_s mask_s * cap_emb[s*2 + bit_s]
        K = cap_bits.size(-1)
        slot = torch.arange(K, device=cap_bits.device)
        idx = slot * 2 + cap_bits                    # (B,T,K) in [0, 2K)
        ce = self.cap_emb(idx)                       # (B,T,K,d)
        emb = emb + (ce * cap_mask.unsqueeze(-1)).sum(dim=2)
        return emb

    def trunk(self, base_x, space_x, cap_bits, cap_mask):
        x = self.compose_input(base_x, space_x, cap_bits, cap_mask)
        for b in self.blocks:
            x = b(x)
        return self.norm(x)

    def forward(self, base_x, space_x, cap_bits, cap_mask,
                base_y, space_y, cap_bits_y, cap_mask_y, return_parts=False):
        h = self.trunk(base_x, space_x, cap_bits, cap_mask)
        content_logits = self.lm_head(h)
        content_ce = F.cross_entropy(content_logits.view(-1, content_logits.size(-1)),
                                     base_y.view(-1))
        # heads conditioned on the (teacher-forced) next base token
        cond = torch.cat([h, self.wte(base_y)], dim=-1)
        space_logits = self.space_head(cond).squeeze(-1)               # (B,T)
        space_bce = F.binary_cross_entropy_with_logits(space_logits, space_y.float())
        cap_logits = self.cap_head(cond)                                # (B,T,K)
        cap_bce_all = F.binary_cross_entropy_with_logits(
            cap_logits, cap_bits_y.float(), reduction="none")
        m = cap_mask_y.float()
        cap_bce = (cap_bce_all * m).sum() / m.sum().clamp_min(1.0)
        loss = content_ce + self.lam * (space_bce + cap_bce)
        if return_parts:
            return loss, dict(content_ce=content_ce, space_bce=space_bce, cap_bce=cap_bce)
        return loss
