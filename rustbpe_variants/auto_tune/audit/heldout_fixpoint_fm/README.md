# heldout_fixpoint_fm — firing-33 fm-context block-list audit

The audit run that produced the **126 blocks** shipped as [`force_merges_combined`](../../../force_merges_combined/README.md). This doc is how to **resume or extend** the run: the exact driver invocation, the per-round judging methodology (the "goal language"), and the tool inventory. For the *result* (validation, firing characterization, A/B plan) see the variant README; for the shared eject-cooldown + transactional-net-commit machinery (and the earlier, different r194 run) see the [parent audit README](../README.md).

## State

`kept=126`, a **hand-stopped point — NOT a driver fixpoint.** The audit status is still `running`: the driver writes `status="fixpoint"` only on a round with zero gate-passing keeps *and* an empty cooldown bench (`tools/heldout_fixpoint_cached.py`), and that was never reached. At round 256, 1,344 candidates still pass the per-candidate gate, but the board is topped by hard semantic vetoes (`ob+by` +3/−15.1 shatters hobby/lobby; `s+hip` +2/−5.6 shatters `-ship`), so the orchestrated loop was **manually halted** at 126 — a judgment that the remaining board (vetoes + small net-negative substitutes) wasn't worth continuing, not a proof it's empty. Two routes could still extend it: small free-on-compression gate-passers on the round-256 board were never net-commit-tested (resume + judge them — "Resume / extend"), and a banked compression surplus is unspent (variant README "Open directions"). Files here:

- `checkpoint.json` — live audit state (kept set, reference metrics, candidate pool, cooldown jails). **Authoritative.**
- `ledger.md` — human round-by-round log: gate definition + per-round commit table + running tally.
- `lineage.jsonl` — ordered commit record (provenance).
- `.gitignore` — keeps per-run backups/scratch (`backup_*/`, `*.prerun.*`, `propose.md`, `loo_words.md`, `inspect_r*.md`) out of git; the committed record is the three files above + this README.

## Run config — what makes this the "fm-context firing-33" run

- **fm-context** (`--fm-context`): trains/evaluates under force_merges' carve-out `SPLIT_PATTERN` + 122 `FORCED_PAIRS` + `block_trailing_space`/`block_leading_space` predicates + base `BLOCKED_PAIRS=[]`, at **vocab 32890** — so the mined blocks are tuned to the tokenizer they ship in. This is the key difference from the r194 run, which mined under the **pristine** regex.
- **firing-percentile-33 mining** (`--mine-mode none --mine-firing-percentile 33`): candidates are the formation routes of every fireable merge token at/below the 33rd percentile of the firing distribution — a pure firing cut, the held-out gate is the sole arbiter. This is **not** the suffix/prefix mangled detector (that is `--mode suffix/prefix/both`, the road not taken — see the variant README "Open directions" #2).
- **gate**: held-out Pareto — covΔ ≥ 0, compΔ ≤ +3, deadΔ ≤ +1, ≥1 strict improvement. **Net-commit** transaction: trial-commit → settle leave-one-out ejections → gate the *net* delta (net-negative trials vetoed + jailed). Eject-cooldown base 4. (Shared machinery — see parent README.)

## Driver command

Canonical per-round invocation — single line (run in background; the eval is the slow part). Same flags for `--propose` and `--commit`:

```
uv run --directory /Users/fanzhu/PyCharmProjects/sandbox python -m tools.heldout_fixpoint_cached --propose --fm-context --gate-comp-tolerance 3 --gate-dead-tolerance 1 --cooldown-base 4 --net-commit --net-retries 99 --mine-mode none --mine-firing-percentile 33 --eval-batch 200 --propose-top-n 20 --state-dir rustbpe_variants/auto_tune/audit/heldout_fixpoint_fm
```

Always pass `--directory /Users/fanzhu/PyCharmProjects/sandbox` — uv resolves the project from cwd, and a drifted cwd picks the wrong venv (`ModuleNotFoundError: No module named 'tools'`). Pass flags literally (zsh does not word-split a `$VAR` holding flags, so a flags-in-a-variable becomes one bad argument).

## Per-round judging methodology (the "goal language")

This run is a **Claude-orchestrated loop**, not an auto-committing script. The driver does the machinery (mine → eval pool → gate → rank keeps by net-firing-coverage); **a human/Claude picks which keep to commit each round**, because a pure firing cut proposes blocking rare-but-legitimate tokens (proper nouns, acronyms, productive morphemes) that the statistics cannot tell apart from mangled junk.

**Each round:**
1. `--propose` (background): evaluates the pool, ranks the top-20 gate-passing keeps by net-firing-coverage, and prints each with the surface token it prevents + the whole-words it adds/loses, **realness-annotated** `fire[sStandalone·dict·flags]` (written to `propose.md`).
2. Read the rows. **Judge the actual words by hand — not the headline Δ.** Apply the criteria below; pick the best non-vetoed keep.
3. `--commit '["L","R"]'`: net-commits it (a net-negative pick is vetoed + jailed → pick the next). Repeat.

**Commit criteria (priority order):**
1. **Net firing whole-words added** — top priority; a whole-word token earns its slot only if it's actually used (`word_realness` standalone usage proves it).
2. **Total whole-words added** — below firing-cov but **above compression**; do *not* deprioritize a real-but-rare word (the sample is small; firing is a tiebreak *among* real words, not a gate on whether a real word counts). Weigh option value / cascade potential (a block may reshape the merge tree so a later block surfaces a bigger word family).
3. **Compression** — discounted (corpus-dependent, noisy).
4. **Dead reduction** — weak tiebreaker.

**Read each word through the realness lens** (`tools/word_realness.py`): DICT word + real standalone usage = coverage; high firing but ~0 standalone (`ART?`) = reroute fragment (discount); born under many candidates (`AMBIENT`) = slot-filler (discount).

**Hard semantic veto:** never commit a block whose blocked surface token is a legitimate token — proper noun, acronym, real word, or productive morpheme (e.g. `ob+by` shatters hobby/lobby; `s+hip` shatters the `-ship` morpheme). Scan to the best non-vetoed candidate. This veto is the entire reason the loop is orchestrated.

**The net-commit gate still gates the pick.** The semantic judgment is a filter *on top of* the statistical gate, never a bypass. **Never** run the bare autonomous loop (`--resume`, with no `--propose`/`--commit`) to harvest — it auto-commits on the statistical gate alone and will commit exactly the junk hand-review rejects (it once committed `ob+by` + ART fragments). Commit only via `--commit` (one judged pick) or `tools/harvest.py --pairs` (a hand-vetted list).

**Stop condition (target vs. where this run actually stopped):** the driver's *true* fixpoint is a round with **zero gate-passing keeps AND an empty cooldown bench** (it then writes `status="fixpoint"`). This run was **hand-stopped at kept=126 before reaching that** — the gate-passing board stayed large (1,344 at round 256) but topped by semantic vetoes / net-negative substitutes, so the loop was halted by judgment (see "State"). To reach a true fixpoint, resume and judge each round's board down to zero committable keeps; to go further on coverage past the free wins, spend compression budget (Resume / extend, below).

## Tools

| tool | role in this run |
|---|---|
| `tools/heldout_fixpoint_cached.py` | the driver. `--propose` (eval pool + rank keeps), `--commit '["L","R"]'` (net-commit a pick), `--pool-override '[["L","R"]]'` (score an explicit pool — one-invocation commit when paired with `--commit`), `--inspect-round N` (re-judge a past round, read-only), `--loo-words` (retention check — surface a keep whose unique provides degraded to artifacts), `--bulk-pairs` (grind a large free-win set, coverage-safe), `--rerun N` (roll back to keeps 1..N-1), `--no-mine` (skip the per-commit re-mine) |
| `tools/harvest.py` | free-win executor (not a judge). `--suggest` (rank the board's free wins as exact (L,R) tuples), `--pairs '[[L,R],…]'` (commit a hand-vetted list in order, net-commit re-validated each) |
| `tools/word_realness.py` | realness lens. `python -m tools.word_realness " airs" " insure" [--corpus val|holdout --contexts N]` — standalone counts + dict status + `ART?`/`AMBIENT` flags |
| `tools/generalization_check_cached.py --fm-context` | fresh-shard validation of the kept set on never-selected shards |
| `tools/blocked_pairs_report.py` | verify blocks are live in a baked tokenizer (0 no-op / 0 latent) + per-operand firing deltas |
| `tools/find_mangled_morphemes.py --mode suffix/prefix/both` | the morpheme-decomposition miner (Open direction #2 — **not** used for these 126; firing-percentile was) |
| `tools/compare_tokenizers.py` | three-way baseline / force_merges / combined comparison — per-shard compression, whole-word firing coverage (born/died classified real vs artifact), morpheme-atom firing. The variant's *evaluation* method; reproduces the figures in the variant README. `uv run python -m tools.compare_tokenizers` |
| `tools/heldout_buildup.py` | helpers `iter_capped_from_path`, `vocab_str2id` (used by `compare_tokenizers` + the driver) |

## Resume / extend

- **Continue the free-coverage harvest** (if a fresh `--propose` still finds net-positive keeps): run the driver command, judge per the methodology above, `--commit`.
- **Open direction #1 — spend compression for more coverage:** raise `--gate-comp-tolerance` (the kept set banked ~−891/shard on fresh shards) to admit coverage-positive-but-compression-costly blocks; sweep the tolerance to map the coverage↔compression frontier and pick a point.
- **Open direction #2 — complementary morpheme mine:** add `--mine-mode suffix/prefix/both` (or run `find_mangled_morphemes --mode both` standalone), which targets cleaner suffix/prefix firing — a different axis. Test whether it adds morpheme cleanliness while *holding* whole-word coverage (a layer on top of the 126, not a replacement).
- After any batch commit, run `--loo-words` to confirm no kept block regressed to artifacts under later additions.
