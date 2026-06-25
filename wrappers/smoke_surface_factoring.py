"""
Smoke tests for the surface-factoring model contract — rungs 0–3 of the
validation ladder in ideas/surface_factoring/README.md "Training integration".
Validates that the model-side spec is implementable and the pieces fit BEFORE
building the real overlay/dataloader/harness and running runcpu.sh. Self-
contained, no data files, no GPU, runs in seconds.

  Rung 0  encoder property tests (no model): round-trip, byte accounting (the
          bpb denominator invariant), per-occurrence cap-mask correctness.
  Rung 1  unit tests: loss decomposition, mask actually masks, embedding
          composition + zero-degenerate, and the param-accounting asserts on the
          REAL nanochat GPT (naive subclass RAISES, proper subclass PASSES).
  Rung 2  forward/backward on a tiny model: finite loss, grads to all params.
  Rung 3  invariants: reduces-to-baseline (bit-identical content loss with
          surface off), bpb byte total == original bytes, overfit-a-batch.

Run: uv run python -m wrappers.smoke_surface_factoring
"""
import sys
import os

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from _surface_ref import (  # noqa: E402
    build_tiny_tokenizer, compute_kmax, surface_encode, surface_decode,
    MiniSurfaceGPT, MiniBaselineGPT,
)
from nanochat.gpt import GPT, GPTConfig  # noqa: E402

torch.manual_seed(0)

# Shared tiny tokenizer + batch helpers ------------------------------------- #
ENC, SRE, ID2LEN = build_tiny_tokenizer()

BATTERY = [
    "The cat sat on the mat", "one of the dogs ran", "Of The Rings",
    "in the park", "It is the best of the team", "theater bathe the",
    "She was the one of the team", "to do in the city",
    "Łódź in the park", "Ελληνικά and The City",   # unicode multi-byte titlecase
]
KMAX = compute_kmax(ENC, SRE, ID2LEN, BATTERY)


def encode_batch(texts):
    """Pack texts into aligned (base, space, cap_bits, cap_mask) tensors, BOS-
    prepended with a null surface label, padded; returns x/y shifted + a mask."""
    bos = ENC.encode_single_token("<|bos|>")
    rows = []
    for t in texts:
        st = surface_encode(t, ENC, SRE, ID2LEN, KMAX)
        base = [bos] + [s[0] for s in st]
        space = [0] + [s[1] for s in st]
        cb = [[0] * KMAX] + [s[2] for s in st]
        cm = [[0] * KMAX] + [s[3] for s in st]
        rows.append((base, space, cb, cm))
    T = max(len(r[0]) for r in rows)
    def pad(seq, fill, n): return seq + [fill] * (n - len(seq))
    base = torch.tensor([pad(r[0], 0, T) for r in rows])
    space = torch.tensor([pad(r[1], 0, T) for r in rows])
    capb = torch.tensor([pad(r[2], [0] * KMAX, T) for r in rows])
    capm = torch.tensor([pad(r[3], [0] * KMAX, T) for r in rows])
    valid = torch.tensor([pad([1] * len(r[0]), 0, T) for r in rows])
    return base, space, capb, capm, valid


def shift(t):
    return t[:, :-1].contiguous(), t[:, 1:].contiguous()


# --------------------------------------------------------------------------- #
# Rung 0 — encoder property tests (no model)

def check_roundtrip():
    for t in BATTERY:
        st = surface_encode(t, ENC, SRE, ID2LEN, KMAX)
        assert surface_decode(st, ENC) == t, f"round-trip failed: {t!r}"
    print("[PASS] rung0(a) encode->decode round-trip lossless (incl. unicode Łódź / Ελληνικά).")


def check_byte_accounting():
    """Σ base-token bytes + Σ space_bit == original UTF-8 bytes — the invariant
    the bpb byte denominator must use (factored space = 1 byte each)."""
    for t in BATTERY:
        st = surface_encode(t, ENC, SRE, ID2LEN, KMAX)
        base_bytes = sum(ID2LEN[s[0]] for s in st)
        space_bytes = sum(s[1] for s in st)
        assert base_bytes + space_bytes == len(t.encode("utf-8")), (
            f"byte accounting off for {t!r}: {base_bytes}+{space_bytes} != {len(t.encode())}")
    print("[PASS] rung0(b) byte accounting: Σ base_bytes + Σ space_bit == original bytes.")


def check_cap_mask():
    """Per-occurrence cap: the SAME phrase token carries different cap_bits by
    occurrence; a continuation token carries cap=0."""
    def phrase_caps(text):
        for tid, sb, cb, cm in surface_encode(text, ENC, SRE, ID2LEN, KMAX):
            if sum(cm) > 1:  # the multi-word phrase token
                return cb[:2]
        return None
    assert phrase_caps("of the") == [0, 0], "lowercase 'of the' should be [0,0]"
    assert phrase_caps("Of The") == [1, 1], "'Of The' should be [1,1]"
    assert phrase_caps("Of the") == [1, 0], "'Of the' should be [1,0]"
    assert phrase_caps("of The") == [0, 1], "'of The' should be [0,1]"
    # OCCURRENCE-active mask: a single word owns exactly ONE cap slot (its first
    # letter); if it splits across tokens, the continuations own NONE.
    st = surface_encode("establishment", ENC, SRE, ID2LEN, KMAX)
    assert sum(sum(cm) for _, _, _, cm in st) == 1, "one word -> one owned cap slot"
    if len(st) > 1:
        assert any(sum(cm) == 0 for _, _, _, cm in st), "continuation owns no cap slot"
    print("[PASS] rung0(c) per-occurrence cap values {of the,Of The,Of the,of The}->{00,11,10,01}; "
          "occurrence-active mask (1 word = 1 owned slot; continuations own 0).")


def check_cap_injectivity():
    """Slot-specific cap embedding: 'Of the' ([1,0]) and 'of The' ([0,1]) — same
    base token, asymmetric cap — must produce DIFFERENT trunk inputs. A shared
    0/1 cap embedding (masked sum) would collapse them."""
    m = _mini()
    a = encode_batch(["Of the"])
    b = encode_batch(["of The"])
    assert torch.equal(a[0], b[0]), "setup: 'Of the'/'of The' should share base tokens"
    assert not torch.equal(a[2], b[2]), "setup: their cap patterns should differ"
    ea = m.compose_input(a[0], a[1], a[2], a[3])
    eb = m.compose_input(b[0], b[1], b[2], b[3])
    assert not torch.allclose(ea, eb), "'Of the' and 'of The' collapse in the trunk input!"
    print("[PASS] rung1(c2) cap embedding injective: 'Of the' != 'of The' in trunk input.")


# --------------------------------------------------------------------------- #
# Rung 1 — unit tests

def _mini():
    return MiniSurfaceGPT(ENC.n_vocab, KMAX, d=48, h=4, n_layer=2, T=64, lam=0.5)


def check_loss_decomposition():
    m = _mini()
    base, space, capb, capm, _ = encode_batch(BATTERY[:4])
    bx, by = shift(base); sx, sy = shift(space)
    cbx, cby = capb[:, :-1], capb[:, 1:]
    cmx, cmy = capm[:, :-1], capm[:, 1:]
    loss, parts = m(bx, sx, cbx, cmx, by, sy, cby, cmy, return_parts=True)
    recon = parts["content_ce"] + m.lam * (parts["space_bce"] + parts["cap_bce"])
    assert torch.allclose(loss, recon, atol=1e-6), "loss != content + λ(space+cap)"
    for k, v in parts.items():
        assert torch.isfinite(v) and v >= -1e-6, f"{k} not a valid finite loss: {v}"
    print(f"[PASS] rung1(a) loss = content_ce + λ(space_bce + cap_bce) "
          f"({parts['content_ce']:.3f} + {m.lam}·({parts['space_bce']:.3f}+{parts['cap_bce']:.3f})).")


def check_masking():
    """Flipping a MASKED-OUT cap target must not change the cap loss."""
    m = _mini()
    base, space, capb, capm, _ = encode_batch(BATTERY[:4])
    bx, by = shift(base); sx, sy = shift(space)
    cbx, cby = capb[:, :-1], capb[:, 1:]
    cmx, cmy = capm[:, :-1], capm[:, 1:]
    _, p0 = m(bx, sx, cbx, cmx, by, sy, cby, cmy, return_parts=True)
    cby2 = cby.clone()
    cby2[cmy == 0] = 1 - cby2[cmy == 0]  # flip every masked-out slot's target
    assert not torch.equal(cby, cby2), "test setup: no masked-out slots to flip"
    _, p1 = m(bx, sx, cbx, cmx, by, sy, cby2, cmy, return_parts=True)
    assert torch.allclose(p0["cap_bce"], p1["cap_bce"], atol=1e-7), "mask does not mask!"
    print("[PASS] rung1(b) masking: flipping masked-out cap targets leaves cap loss unchanged.")


def check_embedding_composition():
    m = _mini()
    base, space, capb, capm, _ = encode_batch(BATTERY[:2])
    emb = m.compose_input(base, space, capb, capm)
    assert emb.shape == (base.size(0), base.size(1), 48), "bad embedding shape"
    with torch.no_grad():
        m.space_emb.weight.zero_(); m.cap_emb.weight.zero_()
    emb_off = m.compose_input(base, space, capb, capm)
    base_only = m.wte(base) + m.pos(torch.arange(base.size(1)))
    assert torch.allclose(emb_off, base_only, atol=1e-6), "surface-off != wte+pos"
    print("[PASS] rung1(c) input = wte+pos+space_emb+Σ_masked cap_emb; zeros → wte+pos exactly.")


def _cfg():
    return GPTConfig(sequence_len=64, vocab_size=320, n_layer=2, n_head=2,
                     n_kv_head=2, n_embd=64)


class _NaiveSurfaceGPT(GPT):
    """Adds surface params but does NOT update the accounting — should RAISE."""
    def __init__(self, config):
        super().__init__(config)
        self.surface_emb = torch.nn.Embedding(2, config.n_embd)
        self.space_head = torch.nn.Linear(config.n_embd, 1)
        self.cap_head = torch.nn.Linear(config.n_embd, 3)


class _ProperSurfaceGPT(_NaiveSurfaceGPT):
    """Overrides the three accounting paths to include the surface params."""
    def _surface_params(self):
        return (list(self.surface_emb.parameters()) + list(self.space_head.parameters())
                + list(self.cap_head.parameters()))

    def num_scaling_params(self):
        # recompute including a 'surface' group (the parent asserts exact coverage)
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
        exclude = (self.transformer.wte.weight.numel() + ve
                   + self.resid_lambdas.numel() + self.x0_lambdas.numel()
                   + self.smear_gate.weight.numel() + self.smear_lambda.numel()
                   + self.backout_lambda.numel()
                   + self.surface_emb.weight.numel())  # surface_emb is a lookup, not matmul
        h, q, t = self.config.n_head, self.config.n_embd // self.config.n_head, self.config.sequence_len
        attn = sum(12 * h * q * (t if w[0] < 0 else min(w[0], t)) for w in self.window_sizes)
        return 6 * (nparams - exclude) + attn

    def optimizer_param_coverage(self):
        """Mirror setup_optimizer's grouping + the surface group; return whether
        the flattened groups cover ALL params (the gpt.py:386 coverage assert)."""
        groups = (list(self.transformer.h.parameters())
                  + list(self.value_embeds.parameters())
                  + list(self.transformer.wte.parameters())
                  + list(self.lm_head.parameters())
                  + [self.resid_lambdas, self.x0_lambdas]
                  + [self.smear_gate.weight, self.smear_lambda, self.backout_lambda]
                  + self._surface_params())
        return len(list(self.parameters())) == len(groups)


def check_param_accounting():
    cfg = _cfg()
    naive = _NaiveSurfaceGPT(cfg)
    raised = False
    try:
        naive.num_scaling_params()
    except AssertionError:
        raised = True
    assert raised, "naive subclass should trip the Parameter-count-mismatch assert"
    proper = _ProperSurfaceGPT(cfg)
    nsp = proper.num_scaling_params()                # must not raise
    assert nsp["surface"] > 0 and nsp["total"] == sum(p.numel() for p in proper.parameters())
    flops = proper.estimate_flops()                  # must not raise / be finite
    assert flops > 0
    assert proper.optimizer_param_coverage(), "optimizer param groups miss surface params"
    print(f"[PASS] rung1(d) param accounting: naive subclass RAISES; proper subclass passes "
          f"num_scaling_params (surface={nsp['surface']:,}) + estimate_flops + optimizer coverage.")


# --------------------------------------------------------------------------- #
# Rung 2 — forward/backward

def check_forward_backward():
    m = _mini()
    base, space, capb, capm, _ = encode_batch(BATTERY)
    bx, by = shift(base); sx, sy = shift(space)
    loss = m(bx, sx, capb[:, :-1], capm[:, :-1], by, sy, capb[:, 1:], capm[:, 1:])
    assert torch.isfinite(loss), "non-finite loss"
    loss.backward()
    must_have_grad = ["space_emb", "cap_emb", "space_head", "cap_head", "lm_head", "wte"]
    for name in must_have_grad:
        mod = getattr(m, name)
        g = next(mod.parameters()).grad
        assert g is not None and torch.isfinite(g).all() and g.abs().sum() > 0, \
            f"no/zero gradient reached {name} (disconnected)"
    print("[PASS] rung2 forward+backward: finite loss; gradients reach surface_emb/cap_emb/"
          "space_head/cap_head/lm_head/wte.")


# --------------------------------------------------------------------------- #
# Rung 3 — invariants

def check_reduces_to_baseline():
    """With surface_emb/cap_emb=0 and λ=0, the content loss must be BIT-IDENTICAL
    to a baseline twin sharing wte/pos/blocks/norm/lm_head."""
    m = MiniSurfaceGPT(ENC.n_vocab, KMAX, d=48, h=4, n_layer=2, T=64, lam=0.0)
    with torch.no_grad():
        m.space_emb.weight.zero_(); m.cap_emb.weight.zero_()
    b = MiniBaselineGPT(ENC.n_vocab, d=48, h=4, n_layer=2, T=64)
    # share the content submodules
    b.wte.load_state_dict(m.wte.state_dict())
    b.pos.load_state_dict(m.pos.state_dict())
    b.blocks.load_state_dict(m.blocks.state_dict())
    b.norm.load_state_dict(m.norm.state_dict())
    b.lm_head.load_state_dict(m.lm_head.state_dict())
    base, space, capb, capm, _ = encode_batch(BATTERY)
    bx, by = shift(base); sx, sy = shift(space)
    with torch.no_grad():
        _, parts = m(bx, sx, capb[:, :-1], capm[:, :-1], by, sy, capb[:, 1:], capm[:, 1:],
                     return_parts=True)
        base_loss = b(bx, by)
    diff = (parts["content_ce"] - base_loss).abs().item()
    assert diff < 1e-6, f"surface-off content loss not bit-identical to baseline: {diff:.2e}"
    print(f"[PASS] rung3(a) reduces-to-baseline: surface-off content CE == baseline (Δ={diff:.1e}).")


def check_bpb_byte_denominator():
    """A bpb computation over a multi-text 'val' must use Σ base_bytes + Σ space_bit
    as the denominator and recover the original byte count exactly."""
    texts = BATTERY
    total = 0
    for t in texts:
        st = surface_encode(t, ENC, SRE, ID2LEN, KMAX)
        total += sum(ID2LEN[s[0]] for s in st) + sum(s[1] for s in st)
    orig = sum(len(t.encode("utf-8")) for t in texts)
    assert total == orig, f"bpb denominator {total} != original bytes {orig}"
    print(f"[PASS] rung3(b) bpb byte denominator over {len(texts)} texts == original bytes ({orig}).")


def check_overfit_batch():
    m = _mini()
    base, space, capb, capm, _ = encode_batch(BATTERY[:4])
    bx, by = shift(base); sx, sy = shift(space)
    cbx, cby, cmx, cmy = capb[:, :-1], capb[:, 1:], capm[:, :-1], capm[:, 1:]
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    l0 = None
    for step in range(220):
        opt.zero_grad()
        loss, parts = m(bx, sx, cbx, cmx, by, sy, cby, cmy, return_parts=True)
        if l0 is None:
            l0 = loss.item()
        loss.backward(); opt.step()
    # surface-head accuracy on the memorized batch
    with torch.no_grad():
        h = m.trunk(bx, sx, cbx, cmx)
        cond = torch.cat([h, m.wte(by)], dim=-1)
        space_acc = ((m.space_head(cond).squeeze(-1) > 0).long() == sy).float().mean().item()
        cap_pred = (m.cap_head(cond) > 0).long()
        cap_acc = (((cap_pred == cby) & (cmy == 1)).float().sum() / cmy.sum()).item()
    assert loss.item() < 0.15 * l0, f"failed to overfit: {l0:.3f} -> {loss.item():.3f}"
    assert space_acc > 0.95 and cap_acc > 0.95, f"surface heads didn't fit: space {space_acc:.2f} cap {cap_acc:.2f}"
    print(f"[PASS] rung3(c) overfit-a-batch: loss {l0:.2f}->{loss.item():.3f}; "
          f"space acc {space_acc:.2f}, cap acc {cap_acc:.2f}.")


def check_surface_tokenizer():
    """The production SurfaceTokenizer wrapper (overlay/surface_tokenizer.py) must
    agree with the codec: encode == codec, decode round-trips, byte_count == bytes."""
    from overlay.surface_tokenizer import SurfaceTokenizer
    tok = SurfaceTokenizer(ENC, SRE.pattern, kmax=KMAX)
    for t in BATTERY:
        st = tok.encode(t)
        assert st == surface_encode(t, ENC, SRE, ID2LEN, KMAX), f"wrapper encode != codec: {t!r}"
        assert tok.decode(st) == t, f"wrapper round-trip failed: {t!r}"
        assert tok.byte_count(st) == len(t.encode("utf-8")), f"byte_count off: {t!r}"
    assert tok.n_vocab == ENC.n_vocab and tok.kmax == KMAX
    print("[PASS] surface tokenizer wrapper: encode==codec, decode round-trips, byte_count==bytes.")


CHECKS = [
    check_roundtrip, check_byte_accounting, check_cap_mask, check_surface_tokenizer,
    check_loss_decomposition, check_masking, check_embedding_composition,
    check_cap_injectivity, check_param_accounting, check_forward_backward,
    check_reduces_to_baseline, check_bpb_byte_denominator, check_overfit_batch,
]


def main():
    print(f"surface-factoring smoke (rungs 0-3) — tiny tokenizer vocab={ENC.n_vocab}, K_max={KMAX}\n")
    for c in CHECKS:
        c()
    print(f"\nALL {len(CHECKS)} CHECKS PASSED.")


if __name__ == "__main__":
    main()
