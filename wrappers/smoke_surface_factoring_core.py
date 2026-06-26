"""
Surface CORE-scoring smoke — rung 4c. Validates the REAL surface-aware CORE
scorer (overlay/surface_core_eval.surface_evaluate_example) end-to-end against a
tiny SurfaceFactoringGPT overfit on a toy corpus: a multiple-choice item whose
gold continuation appears in the corpus must be picked.

  1 MC scoring: surface_evaluate_example picks the gold continuation (the full
    path — surface-encode each option → joint NLL via the model → mean/argmin).
  2 K_max is inferable from cap_emb.weight (the load fallback for checkpoints
    saved before the surface_config persistence patch).

Run: uv run python -m wrappers.smoke_surface_factoring_core
"""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from _surface_ref import build_tiny_tokenizer, compute_kmax  # noqa: E402
from overlay.surface_tokenizer import SurfaceTokenizer  # noqa: E402
from overlay.surface_core_eval import set_surface_tokenizer, surface_evaluate_example  # noqa: E402
from nanochat.gpt import GPTConfig  # noqa: E402

torch.manual_seed(0)
ENC, SRE, ID2LEN = build_tiny_tokenizer()
# Corpus the gold continuation comes from (repeated so the tiny model overfits it).
CORPUS = ["one of the dogs ran in the park"] * 6 + [
    "the cat sat on the mat", "she was in the city of light",
    "it is the best of the team", "the dogs ran in the park again",
]
KMAX = compute_kmax(ENC, SRE, ID2LEN, CORPUS)
os.environ["SURFACE_KMAX"] = str(KMAX)
TOK = SurfaceTokenizer(ENC, SRE.pattern, kmax=KMAX)
set_surface_tokenizer(TOK)
from overlay.surface_factoring import SurfaceFactoringGPT  # noqa: E402  (after SURFACE_KMAX set)


def _train(steps=450):
    cfg = GPTConfig(sequence_len=64, vocab_size=ENC.n_vocab, n_layer=2,
                    n_head=2, n_kv_head=2, n_embd=64)
    m = SurfaceFactoringGPT(cfg)
    m.init_weights()
    seqs = []
    for t in CORPUS:
        st = TOK.encode(t)
        seqs.append((
            torch.tensor([[TOK.bos] + [s[0] for s in st]]),
            torch.tensor([[0] + [s[1] for s in st]]),
            torch.tensor([[[0] * KMAX] + [s[2] for s in st]]),
            torch.tensor([[[0] * KMAX] + [s[3] for s in st]]),
        ))
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    for _ in range(steps):
        opt.zero_grad()
        loss = 0.0
        for base, sp, cb, cm in seqs:
            loss = loss + m(base[:, :-1], base[:, 1:], loss_reduction='mean',
                            space_x=sp[:, :-1], cap_bits_x=cb[:, :-1], cap_mask_x=cm[:, :-1],
                            space_y=sp[:, 1:], cap_bits_y=cb[:, 1:], cap_mask_y=cm[:, 1:])
        (loss / len(seqs)).backward()
        opt.step()
    return m


def check_core_mc(model):
    """The full surface CORE path picks the gold continuation."""
    item = {"query": "one of the dogs ran in",
            "choices": [" the park", " a house", " to school"], "gold": 0}
    task_meta = {"task_type": "multiple_choice", "num_fewshot": 0, "continuation_delimiter": ""}
    ok = surface_evaluate_example(0, model, ENC, [item], "cpu", task_meta)
    assert ok, "surface CORE scorer did not pick the gold continuation"
    # also confirm a clearly-wrong gold would be marked incorrect (scorer discriminates)
    bad = {**item, "gold": 2}
    assert not surface_evaluate_example(0, model, ENC, [bad], "cpu", task_meta), \
        "scorer marked a wrong gold correct — not discriminating"
    print("[PASS] rung4c CORE: surface_evaluate_example picks the gold MC continuation "
          "(surface-encode → joint NLL → mean/argmin), rejects a wrong gold.")


def check_kmax_inference(model):
    """K_max recoverable from cap_emb.weight (the pre-persistence checkpoint fallback)."""
    inferred = model.state_dict()["cap_emb.weight"].shape[0] // 2
    assert inferred == KMAX, f"K_max inference wrong: {inferred} != {KMAX}"
    print(f"[PASS] K_max inferable from cap_emb.weight (shape[0]//2 = {inferred}) "
          "— surface_build_model fallback for checkpoints without surface_config.")


def main():
    print(f"surface CORE smoke (rung 4c) — vocab={ENC.n_vocab}, K_max={KMAX}\n")
    model = _train()
    check_core_mc(model)
    check_kmax_inference(model)
    print("\nALL 2 CHECKS PASSED.")


if __name__ == "__main__":
    main()
