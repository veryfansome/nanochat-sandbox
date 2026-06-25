"""
Surface-factoring dataloader + bpb eval for the harness.

Mirrors nanochat's BOS-aligned best-fit dataloader, but tokenizes each document
with a `SurfaceTokenizer` (dual stream) and packs ALL FOUR streams in lockstep:
base ids, space bits, cap bits[K], cap mask[K]. It yields the BASE `(inputs,
targets)` exactly like upstream (so the rest of base_train is untouched) and
stashes the four surface streams — already shifted to align with
`inputs=row[:,:-1]` / `targets=row[:,1:]` — in `overlay.surface_context`, where
`SurfaceFactoringGPT.forward` reads them.

`surface_evaluate_bpb` is `nanochat.loss_eval.evaluate_bpb` with the byte
denominator corrected to `token_bytes[base_y] + space_y` (a factored leading
space is 1 original byte; the case fold is byte-preserving).
"""
from __future__ import annotations

import math

import torch
import torch.distributed as dist

from nanochat.dataloader import _document_batches
from overlay.surface_context import set_surface


def _encode_doc(tok, text):
    """text -> four equal-length lists (BOS-prepended with a null surface label)."""
    K = tok.kmax
    stream = tok.encode(text)
    base = [tok.bos] + [s[0] for s in stream]
    space = [0] + [s[1] for s in stream]
    cap_bits = [[0] * K] + [list(s[2]) for s in stream]
    cap_mask = [[0] * K] + [list(s[3]) for s in stream]
    return base, space, cap_bits, cap_mask


def surface_data_loader_with_state(tok, B, T, split, device="cpu",
                                   tokenizer_batch_size=128, resume_state_dict=None,
                                   buffer_size=1000):
    """BOS-aligned best-fit, four streams. Yields (base_inputs, base_targets, state)
    and sets the surface context for each yielded batch."""
    assert split in ["train", "val"]
    K = tok.kmax
    row_capacity = T + 1
    batches = _document_batches(split, resume_state_dict, tokenizer_batch_size)
    doc_buffer = []
    pq_idx, rg_idx, epoch = 0, 0, 1

    def refill():
        nonlocal pq_idx, rg_idx, epoch
        doc_batch, (pq_idx, rg_idx, epoch) = next(batches)
        for text in doc_batch:
            doc_buffer.append(_encode_doc(tok, text))

    # row buffers (CPU); built then shifted into _x/_y views
    base_row = torch.zeros((B, row_capacity), dtype=torch.long)
    space_row = torch.zeros((B, row_capacity), dtype=torch.long)
    capb_row = torch.zeros((B, row_capacity, K), dtype=torch.long)
    capm_row = torch.zeros((B, row_capacity, K), dtype=torch.long)

    def place(row_idx, pos, doc, n):
        base, space, cap_bits, cap_mask = doc
        base_row[row_idx, pos:pos + n] = torch.tensor(base[:n], dtype=torch.long)
        space_row[row_idx, pos:pos + n] = torch.tensor(space[:n], dtype=torch.long)
        capb_row[row_idx, pos:pos + n] = torch.tensor(cap_bits[:n], dtype=torch.long)
        capm_row[row_idx, pos:pos + n] = torch.tensor(cap_mask[:n], dtype=torch.long)

    while True:
        for row_idx in range(B):
            pos = 0
            while pos < row_capacity:
                while len(doc_buffer) < buffer_size:
                    refill()
                remaining = row_capacity - pos
                # largest doc that fits entirely
                best_idx, best_len = -1, 0
                for i, doc in enumerate(doc_buffer):
                    dl = len(doc[0])
                    if dl <= remaining and dl > best_len:
                        best_idx, best_len = i, dl
                if best_idx >= 0:
                    doc = doc_buffer.pop(best_idx)
                    place(row_idx, pos, doc, len(doc[0]))
                    pos += len(doc[0])
                else:
                    # nothing fits: crop the shortest to fill exactly
                    j = min(range(len(doc_buffer)), key=lambda i: len(doc_buffer[i][0]))
                    doc = doc_buffer.pop(j)
                    place(row_idx, pos, doc, remaining)
                    pos += remaining

        base_inputs = base_row[:, :-1].to(device)
        base_targets = base_row[:, 1:].to(device)
        set_surface(dict(
            space_x=space_row[:, :-1].to(device), space_y=space_row[:, 1:].to(device),
            cap_bits_x=capb_row[:, :-1].to(device), cap_bits_y=capb_row[:, 1:].to(device),
            cap_mask_x=capm_row[:, :-1].to(device), cap_mask_y=capm_row[:, 1:].to(device),
        ))
        yield base_inputs, base_targets, {"pq_idx": pq_idx, "rg_idx": rg_idx, "epoch": epoch}


def surface_data_loader(*args, **kwargs):
    """Like above, omitting the state from each yield (for the val loader)."""
    for inputs, targets, _ in surface_data_loader_with_state(*args, **kwargs):
        yield inputs, targets


@torch.no_grad()
def surface_evaluate_bpb(model, batches, steps, token_bytes):
    """nanochat.loss_eval.evaluate_bpb with the surface byte denominator:
    per target token, bytes = token_bytes[base_y] + space_y (factored space)."""
    from overlay.surface_context import get_surface
    device = model.get_device()
    total_nats = torch.tensor(0.0, dtype=torch.float32, device=device)
    total_bytes = torch.tensor(0, dtype=torch.int64, device=device)
    it = iter(batches)
    for _ in range(steps):
        x, y = next(it)
        sc = get_surface()                         # set by the surface val loader
        space_y = sc["space_y"] if sc is not None else torch.zeros_like(y)
        loss2d = model(x, y, loss_reduction='none').reshape(-1)
        y = y.reshape(-1)
        sp = space_y.reshape(-1)
        valid = y >= 0
        y_safe = torch.where(valid, y, torch.zeros_like(y))
        base_bytes = torch.where(valid, token_bytes[y_safe],
                                 torch.zeros_like(y, dtype=token_bytes.dtype))
        # add the factored leading-space byte where the target is a real (counted) token
        num_bytes2d = base_bytes + (sp * (base_bytes > 0)).to(token_bytes.dtype)
        total_nats += (loss2d * (num_bytes2d > 0)).sum()
        total_bytes += num_bytes2d.sum()
    if (dist.is_initialized() and dist.get_world_size() > 1):
        dist.all_reduce(total_nats, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_bytes, op=dist.ReduceOp.SUM)
    total_nats, total_bytes = total_nats.item(), total_bytes.item()
    return float('inf') if total_bytes == 0 else total_nats / (math.log(2) * total_bytes)
