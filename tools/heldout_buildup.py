"""Corpus-iteration + vocab utilities for the held-out audit drivers.

Exports `iter_capped_from_path` (char-capped parquet row-group iterator) and
`vocab_str2id` (decoded-string → id map). Imported by
`tools.heldout_fixpoint_cached`, `tools.generalization_check_cached`, and
`tools.token_lineage`.

Historical note: this module also housed a held-out-Pareto **build-up** driver
(single forward pass, reference shifts as keeps accumulate). It was superseded by
`tools.heldout_fixpoint_cached` (greedy fixpoint + corpus-cached trainer, ~6x
faster, generalization baked into selection) and removed in the 2026-06 cleanup.
Only these two utilities — which the fixpoint driver still imports — remain.
"""


def iter_capped_from_path(parquet_path, max_chars):
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(parquet_path)
    total = 0
    for rg in range(pf.num_row_groups):
        if total >= max_chars:
            return
        texts = pf.read_row_group(rg).column("text").to_pylist()
        docs = []
        for d in texts:
            docs.append(d)
            total += len(d)
            if total >= max_chars:
                break
        if docs:
            yield docs
        if total >= max_chars:
            return


def vocab_str2id(tok):
    """{decoded_string: id} for non-special tokens (first id wins)."""
    n = tok.get_vocab_size() - len(tok.get_special_tokens())
    out = {}
    for tid in range(n):
        try:
            s = tok.decode([tid])
        except Exception:
            continue
        out.setdefault(s, tid)
    return out
