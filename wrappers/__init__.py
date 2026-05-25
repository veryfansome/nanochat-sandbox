"""Bootstrap for any `wrappers.*` invocation: sys.path → upstream patches.

`import overlay` prepends `../nanochat` to sys.path so `nanochat.*` and
`scripts.*` are importable. Then `patch_core_eval_prefix_safety()` installs
the prefix-safe `batch_sequences_lm` into `nanochat.core_eval` — required
for non-prefix-stable tokenizers (force_merges, case_marker) and a no-op
otherwise. See `wrappers/_patches.py` for the full rationale.
"""
import overlay  # noqa: F401  triggers sys.path bootstrap; must come first

from wrappers._patches import patch_core_eval_prefix_safety  # noqa: E402
patch_core_eval_prefix_safety()
