"""Shared corpus iteration helpers for tokenizer mining/diagnostic tools.

`parquets_iter_batched` yields one batch per parquet row group (~3M chars
on climbmix). A naive per-batch char cap therefore rounds to the next
row-group boundary — small caps silently process a full row group, large
caps overshoot by up to one. The helpers here trim each batch at the doc
level so `--max-chars` is honored to within one doc (~3K chars typical),
while preserving the batch structure for downstream parallel encoders
(e.g. tiktoken's `encode_ordinary_batch`).
"""

from typing import Iterator


def iter_capped_batches(split: str, max_chars: int) -> Iterator[list[str]]:
    """Yield doc-trimmed batches from `parquets_iter_batched(split)` until
    cumulative char count reaches `max_chars`.

    Each yielded batch is a (possibly truncated) list of docs from the
    underlying batch — the final batch may be shorter if the cap landed
    mid-batch. Empty batches are not yielded.
    """
    from nanochat.dataset import parquets_iter_batched

    total_chars = 0
    for batch in parquets_iter_batched(split=split):
        if total_chars >= max_chars:
            return
        docs: list[str] = []
        for doc in batch:
            docs.append(doc)
            total_chars += len(doc)
            if total_chars >= max_chars:
                break
        if docs:
            yield docs
