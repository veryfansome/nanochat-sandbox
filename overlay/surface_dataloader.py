"""
Surface-factoring dataloader + bpb eval for the harness.

Mirrors nanochat's BOS-aligned best-fit dataloader, but tokenizes each document
with a `SurfaceTokenizer` (dual stream) and packs ALL FOUR streams in lockstep:
base ids, space bits, cap bits[K], cap mask[K] (K = tok.kmax = 1 for the canonical
surface-only tokenizer — one cap bit per single-word token, like the space bit; the
K-indexed cap arrays generalize the deprecated triple's K_max=3). It yields the BASE
`(inputs, targets)` exactly like upstream (so the rest of base_train is untouched) and
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
import torch.nn.functional as F

from nanochat.dataloader import _document_batches
from nanochat.common import print0
from overlay.surface_context import set_surface

# Last component split from surface_evaluate_bpb (so a logger / wandb hook can read it
# without changing the float return contract). {joint, content, overhead} bits/byte.
LAST_BPB_COMPONENTS: dict = {}


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
def surface_evaluate_bpb(model, batches, steps, token_bytes, return_components=False):
    """nanochat.loss_eval.evaluate_bpb with the surface byte denominator:
    per target token, bytes = token_bytes[base_y] + space_y (factored space).

    ALSO splits the JOINT bpb (= val/bpb, what base_train logs) into its CONTENT stream
    vs the space+cap SURFACE OVERHEAD, so a run logs the two trajectories. The split is
    free of the cross-tokenizer confound (it's within one model): content = CE of the raw
    content logits vs the content targets; overhead = joint - content (the heads' cost of
    predicting the side channels). Surface is a low-entropy task that should converge fast
    and plateau — so the OVERHEAD trajectory flattening while CONTENT keeps descending is
    the signal that λ can be decayed (see overlay/surface_lambda.py); a λ change that lowers
    content while raising overhead is the capacity tradeoff made visible.

    The JOINT return value is unchanged (backward-identical val/bpb). The components are
    also stashed in LAST_BPB_COMPONENTS and returned when return_components=True. Costs one
    extra (content-only) forward per eval batch — eval is infrequent, so this is cheap."""
    from overlay.surface_context import get_surface
    device = model.get_device()
    total_nats = torch.tensor(0.0, dtype=torch.float32, device=device)     # joint
    content_nats = torch.tensor(0.0, dtype=torch.float32, device=device)   # content stream only
    total_bytes = torch.tensor(0, dtype=torch.int64, device=device)
    it = iter(batches)
    for _ in range(steps):
        x, y = next(it)
        sc = get_surface()                         # set by the surface val loader
        space_y = sc["space_y"] if sc is not None else torch.zeros_like(y)
        loss2d = model(x, y, loss_reduction='none').reshape(-1)            # joint per-token NLL
        clogits, _, _ = model(x, y, return_surface_logits=True)            # raw content logits
        content2d = F.cross_entropy(clogits.reshape(-1, clogits.size(-1)), y.reshape(-1),
                                    ignore_index=-1, reduction='none')
        y = y.reshape(-1)
        sp = space_y.reshape(-1)
        valid = y >= 0
        y_safe = torch.where(valid, y, torch.zeros_like(y))
        base_bytes = torch.where(valid, token_bytes[y_safe],
                                 torch.zeros_like(y, dtype=token_bytes.dtype))
        # add the factored leading-space byte where the target is a real (counted) token
        num_bytes2d = base_bytes + (sp * (base_bytes > 0)).to(token_bytes.dtype)
        gate = (num_bytes2d > 0)
        total_nats += (loss2d * gate).sum()
        content_nats += (content2d * gate).sum()
        total_bytes += num_bytes2d.sum()
    if (dist.is_initialized() and dist.get_world_size() > 1):
        dist.all_reduce(total_nats, op=dist.ReduceOp.SUM)
        dist.all_reduce(content_nats, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_bytes, op=dist.ReduceOp.SUM)
    total_nats, content_nats, total_bytes = total_nats.item(), content_nats.item(), total_bytes.item()
    if total_bytes == 0:
        return (float('inf'), float('inf'), float('inf')) if return_components else float('inf')
    denom = math.log(2) * total_bytes
    joint_bpb = total_nats / denom
    content_bpb = content_nats / denom
    overhead_bpb = joint_bpb - content_bpb
    LAST_BPB_COMPONENTS.clear()
    LAST_BPB_COMPONENTS.update(joint=joint_bpb, content=content_bpb, overhead=overhead_bpb)
    print0(f"  [surface bpb] joint {joint_bpb:.5f} = content {content_bpb:.5f} "
           f"+ overhead {overhead_bpb:.5f}")
    return (joint_bpb, content_bpb, overhead_bpb) if return_components else joint_bpb
