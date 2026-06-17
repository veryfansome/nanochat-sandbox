# auto_tune

A tokenizer overlay that forbids a curated set of mangled stem+suffix merges during BPE training.

Pristine pretokenization `SPLIT_PATTERN` (no carve-out), blocking only — no forced merges, no vocab bump. `BLOCKED_PAIRS` in [`pairs.py`](pairs.py) lists `(L, R)` pairs the trainer must not merge (via the `rustbpe_auto_tune` crate's `blocked_pairs=` kwarg — a fork of `rustbpe_force_merges` that adds `load_corpus`/`train_from_cached` for the block-list audit); the freed merge slots get reallocated by BPE to alternative merges. Blocking `(L, R)` stops the token `L+R` from forming, so words route through the cleaner `[stem][suffix]` split instead.

## Layout

```
auto_tune/
├── README.md            # this file (CLAUDE.md → README.md): the overlay + methodology
├── Cargo.toml + pyproject.toml + src/lib.rs   # the rustbpe_auto_tune crate (force_merges fork + corpus caching)
├── pairs.py             # BLOCKED_PAIRS (the only thing the trainer reads; currently empty — see file comment)
└── audit/               # the block-list audit: live log + checked-in resume state
    ├── README.md        # live audit log — round plan, status, results, exploration directions (CLAUDE.md → README.md)
    └── checkpoint.json + ledger.md + lineage.jsonl  # CANONICAL converged auto-mine run (comptol6, promoted to top level); earlier passes removed 2026-06-14 — lessons in audit/README.md §Superseded passes
```

`pairs.py` holds **`BLOCKED_PAIRS`** — *the only thing the trainer reads*; it is currently **empty** (the downstream-tested r194 set is env-baked, pending promotion of the held-out-vetted fixpoint winners — see the file's comment). The audit's candidate pool is no longer a static list: the driver mines it adaptively from the reference vocab each round (`--mine`, the default). The old static `CANDIDATES*` queues (and the generators that produced them) were retired in the 2026-06 cleanup — git history preserves them. **Live audit status, round plan, and per-round results: [`audit/README.md`](audit/README.md).**

## Tuning objective

We want to **improve whole-word coverage, improve compression, and reduce semantically-weak mangled tokens** (tokens split at non-morphemic boundaries). In strict priority order (the commit rule below is lexicographic over them):

1. **whole-word coverage** = *total* whole-word vocab tokens (space-prefixed dictionary words, base + inflected) — higher is better. Both a base form (` government`) and an inflected form (` governments`) are real words worth a dedicated token, so total is the coverage signal; we are *not* trying to shed inflected tokens. (Track the base/inflected split too — base is the direct-dict-hit subset — but classify on total.)
2. **compression** (total tokens on the held-out corpus) — lower is better.
3. **mangled tokens remaining** — lower is better.

There is no fixed exchange rate between these yet. The lexicographic rule never pays a higher-priority cost for a lower-priority gain; pairs that improve coverage *but* cost compression are FLAGGED (not auto-kept) for a human cheap-wins-first / expensive-later call. Log each kept pair's per-metric contribution so a sense of "good rate" emerges from the data.

## Tuning methodology

The methodology itself is under active tuning — refine it as evidence accumulates. The core discipline: **every entry in `BLOCKED_PAIRS` must have a proven net-positive effect; nothing that doesn't pull its weight stays in.**

### The driver: held-out-Pareto fixpoint
`tools/heldout_fixpoint_cached.py` rebuilds `BLOCKED_PAIRS` from empty over an adaptively-mined candidate pool (`--mine`, the default: each round re-mines the reference vocab for mangled + low-firing tokens), training each candidate on top of the growing kept set and measuring it on **10 held-out shards** (never trained on). It uses the `rustbpe_auto_tune` crate's `load_corpus`/`train_from_cached` — the training corpus is read+tokenized once and reused for every candidate (~30s/candidate vs ~174s), which is what makes the maximally-thorough scheme below affordable.

- **Gate (held-out Pareto):** KEEP a candidate iff, **averaged over the 10 held-out shards, whole-word coverage ↑ AND compression ≤ 0 AND dead ≤ 0**. Generalization is baked into selection — no separate after-the-fact check. Mangled is *logged* (on the tune shard, for keeps) but **not gated**: it's a heuristic that can be gamed by shoving tokens into its blind spots, and held-out shards don't catch that.
- **Greedy fixpoint:** each round trains the whole remaining pool against the current reference, commits the single best keep (max Δcov, then min Δcomp, then min Δdead), and re-checks *everything* against the new reference — so unlock-combos (a pair that fails alone but passes once another block reshapes the tree) are caught inline. Converges when a full round finds no keep; a final **leave-one-out** pass flags any kept pair made redundant by a later keep. Not globally optimal (the gate is non-monotonic, so no greedy method is) — a complete greedy fixpoint.
- **Why build-up / retrain per candidate:** a candidate redundant with an already-kept pair shows ~0 gain on top of `kept`, so it drops naturally. Training is deterministic, so one retrain gives a real delta (`train_from_cached` is bit-identical to a full `train_from_iterator`).

*Predecessor:* the earlier **single-shard lexicographic** build-up (`tools/audit_blocked_pairs.py`, removed in the 2026-06 cleanup; rounds 1–3 + a flag-revisit; coverage > compression > mangled on one shard) produced the promoted 96-pair list — but single-shard compression turned out to overfit (held-out mean went the wrong way; see `audit/README.md` End-game), which is why the held-out-Pareto redesign replaced it. That history + per-round results live in [`audit/README.md`](audit/README.md).

### Candidate generation (how to find pairs worth testing)
Most room to improve here — no settled "best" generator yet.
- **Mangled tokens** in the existing vocab. Long agglutinated suffixes are good for compression: if a mangled token is rare-ish and contains one, blocking it tends to cost little compression and can yield cleaner splits that raise base coverage — sometimes net-positive on compression too.
- **Dead tokens** — they don't fire (often superseded by a whole-word token), so blocking them is nearly free and frees a slot.
- `tools/find_mangled_morphemes.py` emits candidates (`--exclude-blocks-py` skips already-blocked ones). One lead source, not an oracle — its heuristic is lossy and opinionated; cross-check every candidate.
- `tools.token_contexts <token>` (heuristic) for a token's vocab impact and how to block it; `tools.word_tokens <word>` for how a word currently splits.
- The blocked-pairs report's sections (blocked-R diff, cleanup candidates, latent routes) are additional per-round signal.

### Multi-pair follow-up: co-requisite & cascade sets
Single-pair-in-isolation misses pairs that pay off only **together**, so a DROP means "not useful alone," not "never useful." The predecessor's `--mode groups` re-tested **sets** (`CANDIDATE_GROUPS*`) jointly (the 2 passes below; both that driver and the static group queues were retired in the 2026-06 cleanup, but the finding stands); future combination tests would instead seed the fixpoint with the proven reference and feed the sets as candidates (see `audit/README.md` direction 1). Two flavors:
- **Co-requisite (multi-route):** a surface form reachable via >1 merge route — every route must be blocked or the token still forms (`astically` = `ast+ically` *or* `astic+ally`).
- **Cascade-enabling:** blocking one pair reshapes the tree so a cheaper block suffices elsewhere (`b+ility` lets `ubility` fall to `ub+ility`).
- **What we've seen so far (small sample — 2 families, ~32 sets, 0 keeps):** in `-ically` and `-ation`, the enumerated routes to a surface form were *not* co-equal — one carried the delta and blocking the siblings added nothing, so the set behaved like its best single route. **Too little data to generalize** (other families, and cascade-enabling sets, are untested). A lead worth trying: prioritize sets where **no single route shows the full delta** (candidate co-equal routes). Details + the open question in [`audit/README.md`](audit/README.md).

### Lessons so far
- **So far the keeps have come from single-pair audits of new candidate sources.** The two groups passes (re-grouping DROPs) produced 0 keeps; a new suffix family (`-ation`) added 10. But that's only 2 groups passes on co-requisite sets — *not* a verdict that re-grouping can't pay off (open question — see direction (1) in `audit/README.md`).
- **Most hand-curated single blocks don't pull their weight** (≈¾ DROP so far). Keeps are a minority that earn a clean compression gain or coverage at zero compression cost.
- **In the families seen so far (`-ically`, `-ation`), one split was the productive route** (`X+ation`) and its siblings (`Xa+tion`) were inert; a few productive-route blocks were *harmful* (they shatter whole-word tokens) and the rule dropped them.
- **"Inert vs the current reference" ≠ "useless."** A no-op block can become live in combination or as the reference grows, so a DROP is reference-specific, not permanent. (There's also no cheap *static* shortcut to pre-screen DROPs — a static re-encode doesn't predict the retrain verdict because it omits slot reallocation; see `audit/README.md` directions (1)+(2).)

### Running the audit
In-process + serial; the corpus is cached once (~140s), then ~30s/candidate. State (`checkpoint.json` + `ledger.md` + `lineage.jsonl`) is checked in and rewritten every eval batch (`--eval-batch`, default 30), so a crash resumes mid-round and `commit`+`pull` resumes on another machine. The canonical (converged) run is promoted to the **top level of `audit/`**; to resume/rerun it pass `--state-dir rustbpe_variants/auto_tune/audit` (its invocation: `--mine --gate-comp-tolerance 6 --gate-dead-tolerance 1 --cooldown-base 4 --net-commit`; see `audit/README.md`).
```
uv run python -m tools.heldout_fixpoint_cached            # full run (adaptive --mine pool, ~16–30h)
uv run python -m tools.heldout_fixpoint_cached --resume   # continue after any interruption
```
Why in-process + cached, not parallel subprocesses: a single BPE train already saturates all cores (rayon), so running many trains at once doesn't add throughput — and the dominant cost was re-reading the corpus (~85% of a train), not the merge loop. Caching the corpus once and running merge loops back-to-back gives the ~6× speedup; concurrency would only oversubscribe.

After convergence: promote the fixpoint winners into `pairs.py`'s `BLOCKED_PAIRS`, distill the result into `audit/README.md`, then `uv run python -m wrappers.tok_train_auto_tune` to bake the canonical `~/.cache/nanochat-variants/auto_tune/tokenizer`. The ultimate test is a speedrun A/B (tokenizer metrics are proxies).

Per-candidate training is in-memory (no subprocess/disk per candidate); only KEPT tokenizers are retained, under `--work-base/kept_tokenizers/` (gitignored, re-derivable) for post-hoc `tools.token_lineage`. For one-off manual trains, the wrapper still honors `AUTO_TUNE_BLOCKED_PAIRS_JSON` (a JSON `[L,R]` list overriding `pairs.py`).

## One-off inspection commands

For verifying/inspecting a specific `BLOCKED_PAIRS` list (e.g. after promoting survivors) — the build-up audit above is the driver for *choosing* pairs.

1. Smoke (after changing wrapper code): `uv run python -m wrappers.smoke_tok_train_auto_tune`
2. Train the canonical tokenizer + full report (eval_tokenizer → pass_metrics → dump_vocab + blocked-pairs sections):
   ```
   uv run python -m wrappers.tok_train_auto_tune && \
   uv run python -m tools.blocked_pairs_report \
       --variant ~/.cache/nanochat-variants/auto_tune/tokenizer \
       --pairs-module rustbpe_variants.auto_tune.pairs
   ```
   See STATUS.md "Evaluating a (re)trained tokenizer variant" for reading the output. The dump lands at `/tmp/auto_tune_vocab.txt`; the cached baseline dump at `/tmp/baseline_vocab.txt`.
3. Inspect a token's contexts: `uv run python -m tools.token_contexts --tokenizer ~/.cache/nanochat-variants/auto_tune/tokenizer <token>`
4. Inspect a word's split: `uv run python -m tools.word_tokens --tokenizer ~/.cache/nanochat-variants/auto_tune/tokenizer <word>`
