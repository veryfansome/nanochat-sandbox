"""
Patches applied to upstream nanochat when running through any sandbox wrapper.

Currently one patch: `patch_core_eval_prefix_safety()`.

## The bug

`nanochat.core_eval.batch_sequences_lm` (LM-style CORE tasks: lambada,
story_cloze, etc.) asserts `tokens_without == tokens_with[:len(tokens_without)]`
— i.e. lengthening a prompt only appends tokens. False for any tokenizer with
context-sensitive merges. Concrete failure: `force_merges` carves out " on the"
as one chunk via `(?=([^a-z]|$))` lookahead, so

    tokenize("The cat sat on")           → [The, ' cat', ' sat', ' on']
    tokenize("The cat sat on the mat")   → [The, ' cat', ' sat', ' on the', ' mat']

Position 3 differs (' on' vs ' on the'); assertion fires; eval crashes.

## The fix

Replace the start index with `find_common_length(tokens, 'left')` — the longest
token-space prefix common to both. For prefix-stable tokenizers this equals
`len(tokens_without)`: no behavioral change. For non-prefix-stable tokenizers,
`common_len < len(tokens_without)`, eval continues. `find_common_length` already
exists in upstream (`core_eval.py:86`) and is used by the other `batch_sequences_*`
variants for the same reason — this patch just brings `batch_sequences_lm` in line.

## The semantic caveat (load-bearing)

When `common_len < len(tokens_without)`, the loss window expands LEFT past the
original prompt/continuation string boundary to include the divergent token.
In the example above, scoring window becomes `[' on the', ' mat']` instead of
`[' mat']` — measuring the model's ability to predict the merged token from a
truncated prompt context. CORE LM accuracy on those examples is no longer
directly comparable to a prefix-stable baseline. The boundary-crossing rate is
tracked in `_BOUNDARY_STATS` so wrappers can report it at end-of-eval and
callers can weight the comparison.

## Application

`wrappers/__init__.py` calls `patch_core_eval_prefix_safety()` after the
sys.path bootstrap, so any `wrappers.*` invocation (`base_train`,
`train_<overlay>`, `base_eval`) gets the patched version automatically —
including mid-training CORE evaluation. Upstream `scripts.base_train` /
`scripts.base_eval` invoked directly do not; route through the wrappers
instead (`runs/runcpu.sh` and `runs/speedrun.sh` already do).
"""

# Mutated by _patched_batch_sequences_lm; readable via get_boundary_stats().
_BOUNDARY_STATS = {"n_total": 0, "n_boundary_crossing": 0}

_PATCHED = False


def _patched_batch_sequences_lm(tokenizer, prompts):
    """Replacement for `nanochat.core_eval.batch_sequences_lm`. See module
    docstring for the bug, fix, and semantic caveat."""
    # Lazy import: sys.path bootstrap (overlay/__init__.py) may not have run
    # when this module is first imported.
    import nanochat.core_eval as ce

    prompt_without, prompt_with = prompts
    # String-level check catches eval-data bugs (typos, mismatched pairs) that
    # the upstream token-level check would have caught; we replace that check
    # below with a looser token-level one, so this is no longer redundant.
    assert prompt_with.startswith(prompt_without), (
        "prompt_without is supposed to be a string prefix of prompt_with:\n"
        f"  without: {prompt_without!r}\n"
        f"  with:    {prompt_with!r}"
    )

    tokens = tokenizer(prompts, prepend=tokenizer.get_bos_token_id())
    tokens_without, tokens_with = tokens

    common_len = ce.find_common_length(tokens, direction="left")
    end_idx = len(tokens_with)
    assert common_len < end_idx, "no continuation tokens to score"

    _BOUNDARY_STATS["n_total"] += 1
    if common_len < len(tokens_without):
        _BOUNDARY_STATS["n_boundary_crossing"] += 1

    return [tokens_with], [common_len], [end_idx]


def patch_core_eval_prefix_safety() -> None:
    """Install `_patched_batch_sequences_lm` into `nanochat.core_eval`. Idempotent."""
    global _PATCHED
    if _PATCHED:
        return
    import nanochat.core_eval as ce
    ce.batch_sequences_lm = _patched_batch_sequences_lm
    _PATCHED = True


def get_boundary_stats() -> dict:
    """Snapshot copy of `{n_total, n_boundary_crossing}` (this rank only)."""
    return dict(_BOUNDARY_STATS)


def reset_boundary_stats() -> None:
    """Zero the counters (for tests / multi-eval-per-process scenarios)."""
    _BOUNDARY_STATS["n_total"] = 0
    _BOUNDARY_STATS["n_boundary_crossing"] = 0


def aggregate_boundary_stats() -> tuple[dict, int]:
    """Sum `_BOUNDARY_STATS` across distributed ranks; return `(stats, rank)`.

    Required because `core_eval.evaluate_task` strides examples across ranks
    (`nanochat/core_eval.py:253`), so each rank's `_BOUNDARY_STATS` reflects
    only its shard. Caller is responsible for guarding any output on `rank == 0`.

    Must run BEFORE the distributed process group is destroyed (i.e. before
    `compute_cleanup` in `scripts.base_eval`). The base_eval wrapper handles
    this by patching `compute_cleanup` to call this function first.

    Single-process / no-dist callers (smokes, runcpu) get the local counters
    back unchanged with `rank = 0`.
    """
    try:
        import torch
        import torch.distributed as dist
    except ImportError:
        return dict(_BOUNDARY_STATS), 0

    if not dist.is_initialized():
        return dict(_BOUNDARY_STATS), 0

    device = (
        f"cuda:{torch.cuda.current_device()}"
        if torch.cuda.is_available()
        else "cpu"
    )
    t = torch.tensor(
        [_BOUNDARY_STATS["n_total"], _BOUNDARY_STATS["n_boundary_crossing"]],
        dtype=torch.long,
        device=device,
    )
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return {"n_total": int(t[0].item()), "n_boundary_crossing": int(t[1].item())}, dist.get_rank()
