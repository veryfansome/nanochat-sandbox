# rustbpe_variants/

Forked tokenizer-training crates and the Python-only variants built on them. A variant that needs new Rust gets its own **fully independent crate** producing a uniquely-named Python module; variants that only add Python data (block/seed lists) reuse an existing crate instead of forking. The pristine upstream at [`../rustbpe/`](../rustbpe/) is never modified.

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

## Implemented variants

Two live variants, each with its own crate. `force_merges` is the **adopted default** tokenizer (its results are kept); `auto_tune` is the active line of work. Per-variant design notes live in each variant's README; full results in [`../STATUS.md`](../STATUS.md) "Tokenizer variants".

| Variant | Crate | Mechanism | Status |
|---|---|---|---|
| [`force_merges`](force_merges/) | own (`rustbpe_force_merges`) | `forced_pairs=`/`blocked_pairs=` + `apply_forced_merges_at_end` (append) + regex carve-out so cross-boundary phrases survive as one chunk | **adopted default** (Lambda: CORE +5.81%, val/bpb tied) |
| [`auto_tune`](auto_tune/) | own (`rustbpe_auto_tune`) | blocking-only under the **pristine** regex (`blocked_pairs=`); forked from `force_merges` + `load_corpus`/`train_from_cached` so the block-list audit trains many block-lists over one cached corpus | **active** — held-out-Pareto block-list auto-tuning |

Removed (dead research directions, recoverable from git history): `seed_tokens` (within-chunk composition seeds), `blocked_morphemes` (mangled-merge cascade), `manual_merges` (hand-curated blocks + manual merges), and the `space_digits`/`unigram` pre-tokenization/construction probes. Their findings are preserved in [`../STATUS.md`](../STATUS.md).

### Lifting a variant from an upstream branch

`force_merges` was lifted from a branch of [`veryfansome/nanochat`](https://github.com/veryfansome/nanochat) (`origin/force_merges_wip`). To port another branch variant:
1. `git -C ../nanochat show origin/<branch>:rustbpe/src/lib.rs > rustbpe_variants/<name>/src/lib.rs` (then rename `#[pymodule]`).
2. Lift branch-side Python data (e.g. a `FORCED_PAIRS` list) into the parallel `rustbpe_variants/<name>/` location.
3. Write the matching `wrappers/tok_train_<name>.py`.
4. Run `tools/eval_tokenizer.py` to confirm the structural round-trip battery still passes (the safety net for regex carve-outs).

## Why this layout (vs. Cargo features or branches)

Considered three options before settling here ([`ideas/tokenizer-variants/README.md`](../ideas/tokenizer-variants/README.md) "Implementation paths"):

- **Cargo features in one crate** — `#[cfg(feature = "...")]` gates. Single source tree but variants can't be installed simultaneously (one build per feature combination); hard to combine variants that touch the same code paths.
- **Git branches per variant** — clean diffs but only one variant installed at a time; switching variants requires checkout + rebuild.
- **Parallel crates (chosen)** — variants coexist, no rebuild to switch, matches the existing `overlay/` pattern. Trade: code duplication and manual port-forward when pristine upstream changes. Acceptable because Karpathy's rustbpe is small and stable (single ~1000-line file, last meaningful change months ago).
