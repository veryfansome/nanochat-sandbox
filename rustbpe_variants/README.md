# rustbpe_variants/

Forked tokenizer-training crates. Each variant is a **fully independent Rust crate** that produces a uniquely-named Python module. The pristine upstream lives at [`../rustbpe/`](../rustbpe/) and is never modified.

This mirrors the project's overlay pattern at the Rust layer: pristine upstream stays untouched and `git pull`-able; experimentation lives in parallel crates that the existing pipeline can consume via a thin Python wrapper.

## Layout per variant

```
rustbpe_variants/<name>/
├── Cargo.toml          # package = "rustbpe_<name>"
├── Cargo.lock
├── pyproject.toml      # module-name = "rustbpe_<name>"
├── README.md           # what this variant does + design notes
└── src/
    └── lib.rs          # full fork of pristine src/lib.rs + variant logic
```

Each variant compiles to its own Python module so multiple variants can coexist in one venv. There's no shared base crate at the Rust level — the pristine `src/lib.rs` is the starting point, and each variant copies + diverges. This keeps variants self-contained at the cost of code duplication; when pristine upstream changes, variants are re-ported manually.

## Adding a new variant

1. Copy `../rustbpe/` to `rustbpe_variants/<name>/`:
   ```bash
   cp -R rustbpe rustbpe_variants/<name>
   ```
2. Rename the package in `Cargo.toml`:
   ```toml
   [package]
   name = "rustbpe_<name>"
   [lib]
   name = "rustbpe_<name>"
   ```
3. Rename the module in `pyproject.toml`:
   ```toml
   [project]
   name = "rustbpe_<name>"
   [tool.maturin]
   module-name = "rustbpe_<name>"
   ```
4. Rename the `#[pymodule]` function in `src/lib.rs`:
   ```rust
   #[pymodule]
   fn rustbpe_<name>(m: &Bound<'_, PyModule>) -> PyResult<()> {
       m.add_class::<Tokenizer>()?;
       Ok(())
   }
   ```
5. Add your variant logic in `src/lib.rs`.
6. Build: `VARIANT=<name> bash ../../runs/build_rustbpe.sh`
7. Write a wrapper `wrappers/tok_train_<name>.py` that monkey-aliases `rustbpe`:
   ```python
   import sys, importlib, runpy
   sys.modules['rustbpe'] = importlib.import_module('rustbpe_<name>')
   runpy.run_module('scripts.tok_train', run_name='__main__')
   ```

The wrapper relies on `nanochat/tokenizer.py:160` doing `import rustbpe` at module level; the alias must be installed in `sys.modules` *before* the first `import nanochat.tokenizer`. Mirrors the model-overlay pattern in [`../wrappers/`](../wrappers/).

## Prior art to port (paused, sitting on branches in `veryfansome/nanochat`)

These should be lifted into this directory when re-engaged. See [`../ideas/tokenizer-variants/README.md`](../ideas/tokenizer-variants/README.md) "Prior art" for full design notes.

| Variant | Source branch | Notes |
|---|---|---|
| `seed_tokens` | `origin/seed_tokens` | Morpheme-seeded BPE. Adds `seed_tokens=` kwarg to `train_from_iterator`; constructive merge-chain builder (`ensure_token`, `compute_common_suffixes`, `best_rtl_tail_len`). Seed list in `sandbox/seed_tokens.yaml` (~200 morphemes). |
| `force_merges` | `origin/force_merges_wip` | Forced cross-boundary common-phrase merges. Adds `forced_pairs=`/`blocked_pairs=` kwargs; `apply_forced_merges_at_end` post-training pass. Critical: paired with regex carve-out in the Python wrapper so tiktoken inference stays consistent (see ideas doc). |

Port-forward checklist when lifting a branch variant:
1. `git -C ../nanochat show origin/<branch>:rustbpe/src/lib.rs > rustbpe_variants/<name>/src/lib.rs` (then rename `#[pymodule]`).
2. Lift any branch-side Python-side data (e.g. `sandbox/seed_tokens.yaml`, `FORCED_PAIRS` list) into a parallel sandbox location.
3. Write the matching `wrappers/tok_train_<name>.py`.
4. Run `tools/eval_tokenizer.py` on the resulting tokenizer to confirm structural battery still passes (it should — the round-trip tests are the safety net for regex carve-outs).

## Why this layout (vs. Cargo features or branches)

Considered three options before settling here ([`ideas/tokenizer-variants/README.md`](../ideas/tokenizer-variants/README.md) "Implementation paths"):

- **Cargo features in one crate** — `#[cfg(feature = "...")]` gates. Single source tree but variants can't be installed simultaneously (one build per feature combination); hard to combine variants that touch the same code paths.
- **Git branches per variant** — clean diffs but only one variant installed at a time; switching variants requires checkout + rebuild.
- **Parallel crates (chosen)** — variants coexist, no rebuild to switch, matches the existing `overlay/` pattern. Trade: code duplication and manual port-forward when pristine upstream changes. Acceptable because Karpathy's rustbpe is small and stable (single ~1000-line file, last meaningful change months ago).
