# overlay/

Model subclasses that change the loss/forward of `nanochat.gpt.GPT`. One Python file per implemented idea. The matching wrapper in `../wrappers/train_<idea>.py` patches `nanochat.gpt.GPT` to the subclass before invoking upstream `scripts.base_train` via `runpy`.

Pattern reference: [`zloss.py`](zloss.py) is the canonical worked example. The "Adding a new overlay" recipe in [`../SETUP.md`](../SETUP.md) walks through the three files (overlay + wrapper + smoke) for a new idea.

## Implementation lessons

Things we've discovered by actually writing overlays. Each link goes to the war story / fix.

- **Overlay-only hyperparameters go on the model as instance attributes, not as `GPTConfig` fields.** Adding a config field pollutes the saved `meta_*.json` and breaks `scripts.base_eval` / `chat_sft` / `chat_cli` with `TypeError: GPTConfig.__init__() got an unexpected keyword argument '<your field>'`. Read from env at `__init__`, store as `self.X`. The smoke `(b)` check enforces this invariant. See [`../ideas/zloss/README.md`](../ideas/zloss/README.md) "Implementation note" and [`../SETUP.md`](../SETUP.md) "Hyperparameter placement".

- **Don't materialize `(B, T, V)` twice in `forward`.** Anything that uses `logits` *in addition to* `F.cross_entropy` (z-loss, MTP heads, deep-supervision heads) forces autograd to hold two `(B, T, V)` tensors live across the backward pass — the softmax `F.cross_entropy` saves internally + whatever your second op needs. Causes swap pressure and ~7–10% throughput hit at small scale (CPU/MPS); matters at long context / large batch on GPU. The fix is a custom autograd Function that emits both loss components from one `log_softmax` and saves only `log_probs` for backward — but see the next lesson before reaching for it. Worked example: [`zloss.py`](zloss.py)'s `_FusedCEZLoss`. Discussion: [`../ideas/zloss/README.md`](../ideas/zloss/README.md) "Expected cost" and [`../SETUP.md`](../SETUP.md) "Loss-shape gotcha".

- **A Python-level `torch.autograd.Function` is *not* a fused kernel.** Reimplementing CE + aux-loss in a custom autograd Function saves ~50% backward memory but is **slower** than the naive two-op version — PyTorch's C++ `F.cross_entropy` is highly optimized; Python ops + per-op dispatch lose. Measured at d6/M4: naive ~7% under baseline; "fused" Python autograd ~32% under baseline. A real fused-CE throughput win requires a CUDA/Triton kernel (Liger, cut-cross-entropy) — out of scope here. Keep the Python "fused" path only for memory-bound scenarios or as a numerical reference, and default to the naive two-op formulation. See [`../ideas/zloss/README.md`](../ideas/zloss/README.md) "Expected cost".

- **Aux losses are a training-time regularizer; the eval path must return baseline-comparable CE — and the smoke must test it.**

  `nanochat/loss_eval.py` calls `model(x, y, loss_reduction='none')` to get per-token loss for `val/bpb` (per-token CE × token bytes, summed). If your overlay adds anything to the loss tensor on this path — a scalar like z-loss, summed aux losses like MTP — the bpb number is inflated by your overlay's contribution and no longer comparable to baseline. (For MTP-style aux losses on shifted target slices, the shapes don't even broadcast.) Branch on `loss_reduction != 'mean'` and return only the main CE. Worked examples: [`mtp.py`](mtp.py) and [`zloss.py`](zloss.py).

  **Required smoke coverage**: with nonzero overlay coefficient, the overlay's `loss_reduction='none'` and `loss_reduction='sum'` output must match vanilla GPT exactly. See [`../wrappers/smoke_zloss.py`](../wrappers/smoke_zloss.py) check `(e)`. The training-path aux supervision still does its work; it just doesn't pollute the eval metric.

  **Lesson learned the hard way**: zloss carried this bug through its entire d24 trip despite the lesson being documented in this very file before the trip started. Text-only lessons aren't enforcement — the smoke is. Add the eval-path equivalence check to every new overlay's smoke, and back-fill it into existing overlays' smokes.
