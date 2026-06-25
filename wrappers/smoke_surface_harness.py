"""
Surface HARNESS smoke — validates the data→context→model→bpb path that
`wrappers/train_surface_factoring.py` wires into upstream `base_train`, WITHOUT
real data or a 30-min runcpu. Patches `_document_batches` with synthetic docs.

  1 `_encode_doc` round-trips (the four streams reconstruct the text).
  2 the loader yields aligned base `(inputs, targets)` + sets the surface context
    with correctly-shifted streams; a packed row decodes back to its document
    (packing + shift + context handoff all correct).
  3 the real overlay consumes a loader batch via the context (finite loss, grads).
  4 surface_evaluate_bpb runs and uses the base_bytes+space_bit denominator.

Run: uv run python -m wrappers.smoke_surface_harness
"""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from _surface_ref import build_tiny_tokenizer, compute_kmax  # noqa: E402
from overlay.surface_tokenizer import SurfaceTokenizer  # noqa: E402
from overlay.surface_context import get_surface  # noqa: E402
from nanochat.gpt import GPTConfig  # noqa: E402

torch.manual_seed(0)
ENC, SRE, ID2LEN = build_tiny_tokenizer()
DOCS = ["The cat sat on the mat", "one of the dogs ran in the park",
        "Of The Rings and the team", "She was in the city of light"]
KMAX = compute_kmax(ENC, SRE, ID2LEN, DOCS)
os.environ["SURFACE_KMAX"] = str(KMAX)
TOK = SurfaceTokenizer(ENC, SRE.pattern, kmax=KMAX)

import overlay.surface_dataloader as sdl  # noqa: E402
from overlay.surface_dataloader import surface_evaluate_bpb  # noqa: E402
from overlay.surface_factoring import SurfaceFactoringGPT  # noqa: E402
from overlay.surface_codec import surface_decode  # noqa: E402


def _fake_batches(split, resume_state_dict, tokenizer_batch_size):
    while True:
        yield list(DOCS), (0, 0, 1)


sdl._document_batches = _fake_batches  # no real parquet needed


def _token_bytes():
    tb = torch.tensor([ID2LEN[i] for i in range(ENC.n_vocab)], dtype=torch.long)
    tb[TOK.bos] = 0                                  # special tokens count 0 bytes
    return tb


def check_encode_doc():
    base, space, cb, cm = sdl._encode_doc(TOK, DOCS[1])
    assert base[0] == TOK.bos and space[0] == 0 and cb[0] == [0] * KMAX
    # strip BOS, rebuild the stream, decode -> original
    stream = list(zip(base[1:], space[1:], cb[1:], cm[1:]))
    assert surface_decode(stream, ENC) == DOCS[1]
    print("[PASS] 1 _encode_doc: four BOS-prepended streams round-trip to the text.")


def check_loader_alignment():
    loader = sdl.surface_data_loader_with_state(TOK, B=2, T=48, split="train",
                                                device="cpu", buffer_size=6)
    xi, yt, state = next(loader)
    sc = get_surface()
    B, T = xi.shape
    assert (B, T) == (2, 48)
    for k in ["space_x", "space_y", "cap_bits_x", "cap_bits_y", "cap_mask_x", "cap_mask_y"]:
        assert sc[k] is not None
    assert sc["space_x"].shape == (B, T) and sc["cap_bits_x"].shape == (B, T, KMAX)
    # reconstruct full row 0 streams (inputs=row[:,:-1], targets=row[:,1:])
    full_base = torch.cat([xi[0], yt[0, -1:]]).tolist()
    full_space = torch.cat([sc["space_x"][0], sc["space_y"][0, -1:]]).tolist()
    full_cb = torch.cat([sc["cap_bits_x"][0], sc["cap_bits_y"][0, -1:]]).tolist()
    full_cm = torch.cat([sc["cap_mask_x"][0], sc["cap_mask_y"][0, -1:]]).tolist()
    # first doc = positions 1 .. next BOS (row starts with a BOS at pos 0)
    assert full_base[0] == TOK.bos
    end = next((i for i in range(1, len(full_base)) if full_base[i] == TOK.bos), len(full_base))
    stream = list(zip(full_base[1:end], full_space[1:end], full_cb[1:end], full_cm[1:end]))
    decoded = surface_decode(stream, ENC)
    assert decoded in DOCS, f"first packed doc did not decode to a known doc: {decoded!r}"
    print(f"[PASS] 2 loader: aligned streams + context; first packed doc decodes -> {decoded!r}")


def check_model_consumes_batch():
    loader = sdl.surface_data_loader_with_state(TOK, B=2, T=48, split="train",
                                                device="cpu", buffer_size=6)
    xi, yt, _ = next(loader)                          # sets the surface context
    cfg = GPTConfig(sequence_len=64, vocab_size=ENC.n_vocab, n_layer=2,
                    n_head=2, n_kv_head=2, n_embd=64)
    model = SurfaceFactoringGPT(cfg)
    model.init_weights()
    loss = model(xi, yt)                              # bare (x,y) -> reads context
    assert torch.isfinite(loss) and loss.item() > 0, "model did not produce a finite joint loss via context"
    loss.backward()
    g = next(model.space_head.parameters()).grad
    assert g is not None and g.abs().sum() > 0, "surface head got no gradient through the harness path"
    print(f"[PASS] 3 overlay consumes a loader batch via context: finite joint loss {loss.item():.3f}, grads flow.")


def check_surface_bpb():
    cfg = GPTConfig(sequence_len=64, vocab_size=ENC.n_vocab, n_layer=2,
                    n_head=2, n_kv_head=2, n_embd=64)
    model = SurfaceFactoringGPT(cfg)
    model.init_weights()
    val = sdl.surface_data_loader(TOK, B=2, T=48, split="val", device="cpu", buffer_size=6)
    bpb = surface_evaluate_bpb(model, val, steps=2, token_bytes=_token_bytes())
    assert bpb != float("inf") and bpb > 0, f"surface bpb invalid: {bpb}"
    print(f"[PASS] 4 surface_evaluate_bpb runs (base_bytes+space denominator): bpb={bpb:.3f}.")


CHECKS = [check_encode_doc, check_loader_alignment, check_model_consumes_batch, check_surface_bpb]


def main():
    print(f"surface harness smoke — vocab={ENC.n_vocab}, K_max={KMAX}\n")
    for c in CHECKS:
        c()
    print(f"\nALL {len(CHECKS)} CHECKS PASSED.")


if __name__ == "__main__":
    main()
