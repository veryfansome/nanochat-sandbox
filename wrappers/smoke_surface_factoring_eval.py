"""
Surface-factoring EVAL/GENERATION smoke — rung 4 of the validation ladder in
ideas/surface_factoring/README.md. Validates the paths the CORE/ChatCORE A/B
needs, which the training-only harness change would NOT cover (review finding
P2.5): surface-aware joint-likelihood scoring, and surface-aware generation +
decode. Plumbing only (does the path WORK / produce sane rankings), not quality.

  4(a) joint-likelihood scoring (the CORE-style path): score a sequence by the
       JOINT NLL = content CE + space BCE + masked cap BCE; a model overfit on
       the true sequence must rank it below a corrupted variant.
  4(b) generation + decode: (i) incremental token-by-token decode of an encoded
       sequence round-trips (the generation-side decode path); (ii) an actual
       autoregressive sample (base token -> conditioned space+cap bits) decodes
       to valid round-trippable UTF-8 without crashing.

Run: uv run python -m wrappers.smoke_surface_factoring_eval
"""
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from _surface_ref import (  # noqa: E402
    build_tiny_tokenizer, compute_kmax, surface_encode, surface_decode, MiniSurfaceGPT,
    _char_word_starts,
)

torch.manual_seed(0)
ENC, SRE, ID2LEN = build_tiny_tokenizer()
BOS = ENC.encode_single_token("<|bos|>")

TRAIN = ["one of the dogs ran in the park", "The best of the team",
         "It is the case of the day", "She was in the city", "Łódź in the city"]
KMAX = compute_kmax(ENC, SRE, ID2LEN, TRAIN)


def _gen_mask(tid):
    """Generation-time cap mask for a sampled token = its intrinsic char word-
    starts, clamped to K_max (continuations / byte-split tokens → all-zero)."""
    tb = ENC.decode_single_token_bytes(tid)
    try:
        n = len(_char_word_starts(tb.decode("utf-8")))
    except UnicodeDecodeError:
        n = 0
    return [1 if k < min(n, KMAX) else 0 for k in range(KMAX)]


def encode_seq(text):
    """One sequence -> tensors (1,T) with BOS prepended (null surface label)."""
    st = surface_encode(text, ENC, SRE, ID2LEN, KMAX)
    base = [BOS] + [s[0] for s in st]
    space = [0] + [s[1] for s in st]
    cb = [[0] * KMAX] + [s[2] for s in st]
    cm = [[0] * KMAX] + [s[3] for s in st]
    return (torch.tensor([base]), torch.tensor([space]),
            torch.tensor([cb]), torch.tensor([cm]))


def joint_nll(model, base, space, cb, cm):
    """Sum joint negative log-likelihood (content + surface) over the sequence —
    the quantity a surface-aware CORE/likelihood scorer would compute."""
    bx, by = base[:, :-1], base[:, 1:]
    sx, sy = space[:, :-1], space[:, 1:]
    cbx, cby = cb[:, :-1], cb[:, 1:]
    cmx, cmy = cm[:, :-1], cm[:, 1:]
    with torch.no_grad():
        h = model.trunk(bx, sx, cbx, cmx)
        content = F.cross_entropy(model.lm_head(h).view(-1, ENC.n_vocab), by.reshape(-1),
                                  reduction="sum")
        cond = torch.cat([h, model.wte(by)], dim=-1)
        space_nll = F.binary_cross_entropy_with_logits(
            model.space_head(cond).squeeze(-1), sy.float(), reduction="sum")
        cap_all = F.binary_cross_entropy_with_logits(model.cap_head(cond), cby.float(),
                                                     reduction="none")
        cap_nll = (cap_all * cmy.float()).sum()
    return (content + space_nll + cap_nll).item()


def train_small(steps=400):
    m = MiniSurfaceGPT(ENC.n_vocab, KMAX, d=48, h=4, n_layer=2, T=64, lam=0.5)
    seqs = [encode_seq(t) for t in TRAIN]
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    for _ in range(steps):
        opt.zero_grad()
        loss = 0.0
        for base, space, cb, cm in seqs:
            loss = loss + m(base[:, :-1], space[:, :-1], cb[:, :-1], cm[:, :-1],
                            base[:, 1:], space[:, 1:], cb[:, 1:], cm[:, 1:])
        (loss / len(seqs)).backward()
        opt.step()
    return m


# --------------------------------------------------------------------------- #
# 4(a) joint-likelihood scoring

def check_joint_scoring(model):
    base, space, cb, cm = encode_seq(TRAIN[0])
    nll_true = joint_nll(model, base, space, cb, cm)
    # corrupt the content stream (swap two interior base tokens)
    bad = base.clone()
    i, j = 2, 5
    bad[0, i], bad[0, j] = bad[0, j].clone(), bad[0, i].clone()
    nll_content = joint_nll(model, bad, space, cb, cm)
    # corrupt ONLY the surface stream (flip a space bit) — must also score worse
    badspace = space.clone(); badspace[0, 3] = 1 - badspace[0, 3]
    nll_surface = joint_nll(model, base, badspace, cb, cm)
    assert nll_true < nll_content, f"content corruption not penalized: {nll_true:.2f} !< {nll_content:.2f}"
    assert nll_true < nll_surface, f"surface corruption not penalized: {nll_true:.2f} !< {nll_surface:.2f}"
    print(f"[PASS] rung4(a) joint scoring ranks true seq below corruptions "
          f"(true {nll_true:.2f} < content {nll_content:.2f}, < surface {nll_surface:.2f}).")


# --------------------------------------------------------------------------- #
# 4(b) generation + decode

def check_incremental_decode():
    """The generation-side decoder, applied token-by-token as a stream would
    arrive, must reconstruct the original (no whole-sequence dependency)."""
    for t in TRAIN:
        st = surface_encode(t, ENC, SRE, ID2LEN, KMAX)
        out = ""
        acc = []
        for tok in st:
            acc.append(tok)
            out = surface_decode(acc, ENC)            # decode the growing stream
        assert out == t, f"incremental decode failed: {t!r} -> {out!r}"
    print("[PASS] rung4(b-i) incremental (streaming) decode reconstructs the original.")


def check_autoregressive_generate(model, n_new=12):
    """Sample base token -> conditioned space+cap bits, autoregressively; decode
    the (base, space, cap) stream to valid, round-trippable UTF-8."""
    base = torch.tensor([[BOS]]); space = torch.tensor([[0]])
    cb = torch.zeros(1, 1, KMAX, dtype=torch.long); cm = torch.zeros(1, 1, KMAX, dtype=torch.long)
    torch.manual_seed(1)
    for _ in range(n_new):
        with torch.no_grad():
            h = model.trunk(base, space, cb, cm)[:, -1]              # last position
            nxt = torch.distributions.Categorical(logits=model.lm_head(h)).sample()  # base_{t+1}
            cond = torch.cat([h, model.wte(nxt)], dim=-1)
            sbit = (torch.sigmoid(model.space_head(cond).squeeze(-1)) > 0.5).long()
            mask = torch.tensor([_gen_mask(nxt.item())])     # generation-time mask derivation
            capbit = ((torch.sigmoid(model.cap_head(cond)) > 0.5).long()) * mask
        base = torch.cat([base, nxt[None]], dim=1)
        space = torch.cat([space, sbit[None]], dim=1)
        cb = torch.cat([cb, capbit[:, None]], dim=1)
        cm = torch.cat([cm, mask[:, None]], dim=1)
    # decode the generated stream (skip the BOS at position 0)
    stream = [(base[0, i].item(), space[0, i].item(), cb[0, i].tolist(), cm[0, i].tolist())
              for i in range(1, base.size(1))]
    text = surface_decode(stream, ENC)
    text.encode("utf-8")  # must be valid UTF-8 (would raise otherwise)
    # the generated text must be codec-stable: re-encode -> decode -> same text
    assert surface_decode(surface_encode(text, ENC, SRE, ID2LEN, KMAX), ENC) == text
    print(f"[PASS] rung4(b-ii) autoregressive generate -> decode produced valid UTF-8: {text!r}")


def main():
    print(f"surface-factoring eval smoke (rung 4) — vocab={ENC.n_vocab}, K_max={KMAX}\n")
    model = train_small()
    check_joint_scoring(model)
    check_incremental_decode()
    check_autoregressive_generate(model)
    print("\nALL 3 CHECKS PASSED.")


if __name__ == "__main__":
    main()
