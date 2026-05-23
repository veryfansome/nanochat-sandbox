"""Put the sibling nanochat checkout on sys.path so `nanochat.*` and `scripts.*`
are importable without installing nanochat as a package (which would require
touching its pyproject.toml). This runs on first import of the `overlay`
package — every script in `wrappers/` imports `overlay` indirectly, so the path
is always set up before any `from nanochat... import ...` line executes.
"""
import sys
import pathlib

_NANOCHAT_DIR = pathlib.Path(__file__).resolve().parents[2] / "nanochat"
_NANOCHAT_DIR_STR = str(_NANOCHAT_DIR)
if _NANOCHAT_DIR_STR not in sys.path:
    sys.path.insert(0, _NANOCHAT_DIR_STR)
