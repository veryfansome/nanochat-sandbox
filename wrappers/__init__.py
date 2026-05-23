"""Importing `run` first imports `overlay`, which sets up sys.path so that
`nanochat.*` and `scripts.*` are importable from the sibling nanochat checkout.
"""
import overlay  # noqa: F401  triggers the sys.path bootstrap
