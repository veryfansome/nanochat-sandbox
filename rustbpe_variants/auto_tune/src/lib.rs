use std::cmp::Ordering;
use std::collections::HashMap as StdHashMap;

use dary_heap::OctonaryHeap;
use fancy_regex::Regex;
use pyo3::prelude::*;

use ahash::{AHashMap, AHashSet};
use compact_str::CompactString;
use rayon::prelude::*;

// Default GPT-4 style regex pattern for splitting text
const GPT4_PATTERN: &str = r"'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+";

type Pair = (u32, u32);

/// A Byte Pair Encoding tokenizer that matches the GPT-4 style implementation
#[pyclass]
pub struct Tokenizer {
    /// Maps pairs of token IDs to their merged token ID
    pub merges: StdHashMap<Pair, u32>,
    /// The regex pattern used for text splitting
    pub pattern: String,
    /// Compiled regex for efficiency
    compiled_pattern: Regex,
    /// Materialized corpus (unique words + counts) cached by `load_corpus`, so
    /// `train_from_cached` can run the merge loop repeatedly while only the
    /// `blocked_pairs` vary — without re-reading/re-tokenizing the corpus (the
    /// dominant cost). None until `load_corpus`. Cloned per train (the merge loop
    /// consumes/mutates its inputs), so the cache stays pristine across trains.
    cached_words: Option<Vec<Word>>,
    cached_counts: Option<Vec<i32>>,
    /// Held-out shards staged by `add_holdout_shard` (name, unique chunks, counts),
    /// consumed by `finalize_holdout` into the global-dedup form below.
    holdout_raw: Vec<(String, Vec<Word>, Vec<i32>)>,
    /// Held-out corpus as GLOBAL unique chunks (deduped across all shards) so each
    /// candidate encodes every distinct chunk once, plus per-shard sparse counts
    /// `(global_chunk_index, count)`. Lets `evaluate_many` compute per-shard
    /// compression + dead in Rust without re-encoding repeats. None until finalize.
    holdout_chunks: Option<Vec<Word>>,
    holdout_shard_counts: Option<Vec<Vec<(u32, i32)>>>,
    holdout_names: Option<Vec<String>>,
    /// Lowercased dictionary wordlist for the in-Rust whole-word coverage scan.
    wordlist: Option<AHashSet<String>>,
}

// ------------------------ internal helpers ------------------------

#[derive(Clone, Debug)]
struct Word {
    ids: Vec<u32>,
}

impl Word {
    #[inline]
    fn new(ids: Vec<u32>) -> Self {
        Self { ids }
    }

    #[inline]
    fn pairs<'a>(&'a self) -> impl Iterator<Item = Pair> + 'a {
        self.ids.windows(2).map(|w| (w[0], w[1]))
    }

    /// Merge all non-overlapping occurrences of pair -> new_id.
    /// Returns a small Vec of local pair-count deltas for THIS word only:
    ///   -1 for removed pairs, +1 for newly created pairs.
    ///
    /// NOTE: this version deliberately avoids a HashMap in the hot loop.
    fn merge_pair(&mut self, pair: Pair, new_id: u32) -> Vec<(Pair, i32)> {
        let (a, b) = pair;
        let n = self.ids.len();
        if n < 2 {
            return Vec::new();
        }

        let mut out: Vec<u32> = Vec::with_capacity(n);
        let mut deltas: Vec<(Pair, i32)> = Vec::with_capacity(6);

        let mut i = 0;
        while i < n {
            if i + 1 < n && self.ids[i] == a && self.ids[i + 1] == b {
                let left = out.last().copied();
                let right = if i + 2 < n { Some(self.ids[i + 2]) } else { None };

                // remove old pairs
                if let Some(x) = left {
                    deltas.push(((x, a), -1));
                    deltas.push(((x, new_id), 1));
                }
                deltas.push(((a, b), -1));
                if let Some(y) = right {
                    deltas.push(((b, y), -1));
                    deltas.push(((new_id, y), 1));
                }

                // write merged token
                out.push(new_id);
                i += 2; // skip 'a' and 'b'
            } else {
                out.push(self.ids[i]);
                i += 1;
            }
        }

        self.ids = out;
        deltas
    }
}

#[derive(Debug, Eq)]
struct MergeJob {
    pair: Pair,
    count: u64,
    /// set of word indices where this pair may occur and needs processing
    pos: AHashSet<usize>,
}

impl PartialEq for MergeJob {
    fn eq(&self, other: &Self) -> bool {
        self.count == other.count && self.pair == other.pair
    }
}

impl PartialOrd for MergeJob {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for MergeJob {
    fn cmp(&self, other: &Self) -> Ordering {
        // Max-heap by count; tie-break to ascending pair order (deterministic)
        if self.count != other.count {
            self.count.cmp(&other.count)
        } else {
            // ascending order on the pair when counts tie
            other.pair.cmp(&self.pair)
        }
    }
}

#[inline]
fn count_pairs_parallel(
    words: &[Word],
    counts: &[i32],
) -> (AHashMap<Pair, i32>, AHashMap<Pair, AHashSet<usize>>) {
    words
        .par_iter()
        .enumerate()
        .fold(
            || {
                (
                    AHashMap::<Pair, i32>::new(),
                    AHashMap::<Pair, AHashSet<usize>>::new(),
                )
            },
            |(mut local_pc, mut local_wtu), (i, w)| {
                if w.ids.len() >= 2 && counts[i] != 0 {
                    for (a, b) in w.pairs() {
                        *local_pc.entry((a, b)).or_default() += counts[i];
                        local_wtu.entry((a, b)).or_default().insert(i);
                    }
                }
                (local_pc, local_wtu)
            },
        )
        .reduce(
            || {
                (
                    AHashMap::<Pair, i32>::new(),
                    AHashMap::<Pair, AHashSet<usize>>::new(),
                )
            },
            |(mut acc_pc, mut acc_wtu), (pc, wtu)| {
                for (k, v) in pc {
                    *acc_pc.entry(k).or_default() += v;
                }
                for (k, s) in wtu {
                    acc_wtu.entry(k).or_default().extend(s);
                }
                (acc_pc, acc_wtu)
            },
        )
}

#[derive(Clone, Debug)]
struct BlockedPairSpec {
    left_bytes: Vec<u8>,
    right_bytes: Vec<u8>,
    left_str: String,
    right_str: String,
}

#[derive(Clone, Debug)]
struct ForcedMergeSpec {
    concat_bytes: Vec<u8>,
    left_bytes: Vec<u8>,
    right_bytes: Vec<u8>,
    // Keep original strings for logs / debugging
    left_str: String,
    right_str: String,
}

type BytePairLookup = AHashMap<Vec<u8>, AHashMap<Vec<u8>, usize>>;
type ByteLookup = AHashMap<Vec<u8>, usize>;

trait PairSpecBytes {
    fn pair_bytes(&self) -> (&[u8], &[u8]);
}

impl PairSpecBytes for BlockedPairSpec {
    fn pair_bytes(&self) -> (&[u8], &[u8]) {
        (&self.left_bytes, &self.right_bytes)
    }
}

impl PairSpecBytes for ForcedMergeSpec {
    fn pair_bytes(&self) -> (&[u8], &[u8]) {
        (&self.left_bytes, &self.right_bytes)
    }
}

fn insert_pair_lookup(
    lookup: &mut BytePairLookup,
    left: &[u8],
    right: &[u8],
    spec_idx: usize,
) {
    lookup
        .entry(left.to_vec())
        .or_default()
        // Preserve the old linear-scan behavior for duplicate specs: the
        // earliest matching entry is the one used for diagnostics.
        .entry(right.to_vec())
        .or_insert(spec_idx);
}

#[inline]
fn find_pair_lookup(lookup: &BytePairLookup, left: &[u8], right: &[u8]) -> Option<usize> {
    lookup
        .get(left)
        .and_then(|right_lookup| right_lookup.get(right))
        .copied()
}

fn build_pair_lookup<T: PairSpecBytes>(specs: &[T]) -> BytePairLookup {
    let mut lookup = AHashMap::with_capacity(specs.len());
    for (idx, spec) in specs.iter().enumerate() {
        let (left, right) = spec.pair_bytes();
        insert_pair_lookup(&mut lookup, left, right, idx);
    }
    lookup
}

fn build_forced_concat_lookup(specs: &[ForcedMergeSpec]) -> ByteLookup {
    let mut lookup = AHashMap::with_capacity(specs.len());
    for (idx, spec) in specs.iter().enumerate() {
        lookup.entry(spec.concat_bytes.clone()).or_insert(idx);
    }
    lookup
}

#[inline]
fn increment_chunk_count(counts: &mut AHashMap<CompactString, i32>, piece: &str) {
    if let Some(count) = counts.get_mut(piece) {
        *count += 1;
    } else {
        counts.insert(CompactString::from(piece), 1);
    }
}

/// Apply forced merges *after* normal training.
/// - `vocab_size` is the final vocab size limit
/// - `merges_done` is the number of merges that have already allocated token IDs
fn apply_forced_merges_at_end(
    blocked_pair_lookup: &BytePairLookup,
    forced_specs: &[ForcedMergeSpec],
    merges: &mut StdHashMap<Pair, u32>,
    vocab_size: u32,
    merges_done: &mut u32,
) {
    if forced_specs.is_empty() {
        return;
    }

    let num_merges_capacity = vocab_size - 256;

    // Highest token id that currently exists
    let max_token_id = if *merges_done == 0 {
        255
    } else {
        256 + *merges_done - 1
    };

    let (mut token_bytes, mut bytes_to_id) = build_token_bytes_and_map(merges, max_token_id);

    'outer_forced: for spec in forced_specs {
        let l_str = &spec.left_str;
        let r_str = &spec.right_str;

        // 1) If this pair is explicitly blocked, do not create a forced merge for it.
        if find_pair_lookup(blocked_pair_lookup, &spec.left_bytes, &spec.right_bytes).is_some() {
            log::debug!(
                "Skipping forced merge {:?}+{:?} because it is in blocked_pairs",
                spec.left_str,
                spec.right_str
            );
            continue 'outer_forced;
        }

        // 2) Both operands must exist as tokens.
        let left_id = match bytes_to_id.get(&spec.left_bytes) {
            Some(&id) => id,
            None => {
                log::warn!(
                    "Skipping forced merge {:?}+{:?}: left token not present in vocab",
                    l_str,
                    r_str
                );
                continue;
            }
        };

        let right_id = match bytes_to_id.get(&spec.right_bytes) {
            Some(&id) => id,
            None => {
                log::warn!(
                    "Skipping forced merge {:?}+{:?}: right token not present in vocab",
                    l_str,
                    r_str
                );
                continue;
            }
        };

        let pair = (left_id, right_id);

        // If a merge for this pair already exists, nothing to do.
        if merges.contains_key(&pair) {
            log::debug!(
                "Forced merge {:?}+{:?} already present as pair {:?}",
                l_str,
                r_str,
                pair
            );
            continue;
        }

        // 3) Compute concatenated bytes: left || right.
        let mut concat = spec.left_bytes.clone();
        concat.extend_from_slice(&spec.right_bytes);

        // 4) If a token already has these bytes, reuse its id (no new capacity).
        let new_id = if let Some(&existing_id) = bytes_to_id.get(&concat) {
            existing_id
        } else {
            // 4) Otherwise, allocate a new token id, respecting capacity.
            if *merges_done >= num_merges_capacity {
                log::warn!(
                    "Skipping forced merge {:?}+{:?}: no remaining merge slots ({}/{})",
                    l_str,
                    r_str,
                    merges_done,
                    num_merges_capacity
                );
                break; // no more capacity for further forced merges
            }

            let id = 256 + *merges_done;
            if token_bytes.len() <= id as usize {
                token_bytes.resize(id as usize + 1, Vec::new());
            }
            token_bytes[id as usize] = concat.clone();
            bytes_to_id.insert(concat, id);
            *merges_done += 1;
            id
        };

        merges.insert(pair, new_id);
        log::info!(
            "Applied forced merge at end: {:?}+{:?} -> {} (pair={:?})",
            l_str,
            r_str,
            new_id,
            pair
        );
    }
}

/// Reconstruct token_id -> bytes and bytes -> token_id from the merges we’ve
/// learned so far, up to `max_token_id`.
fn build_token_bytes_and_map(
    merges: &StdHashMap<Pair, u32>,
    max_token_id: u32,
) -> (Vec<Vec<u8>>, AHashMap<Vec<u8>, u32>) {
    // Allocate [0..=max_token_id] as empty byte vectors
    let mut token_bytes: Vec<Vec<u8>> = (0..=max_token_id).map(|_| Vec::new()).collect();

    // Base vocabulary: bytes 0..=255
    for i in 0..256u32 {
        token_bytes[i as usize] = vec![i as u8];
    }

    // Rebuild merged tokens by increasing token id
    let mut sorted_merges: Vec<_> = merges.iter().collect();
    sorted_merges.sort_by_key(|&(_, &tid)| tid);

    for (&(left, right), &new_id) in sorted_merges {
        // Safety: in BPE, left/right are always < new_id, and we sorted by new_id
        let mut merged = token_bytes[left as usize].clone();
        merged.extend_from_slice(&token_bytes[right as usize]);
        token_bytes[new_id as usize] = merged;
    }

    // Build reverse map bytes -> id
    let mut bytes_to_id: AHashMap<Vec<u8>, u32> = AHashMap::with_capacity(token_bytes.len());
    for (id, bytes) in token_bytes.iter().enumerate() {
        if !bytes.is_empty() {
            bytes_to_id.insert(bytes.clone(), id as u32);
        }
    }

    (token_bytes, bytes_to_id)
}

/// Encode one pre-split chunk (bytes as u32 ids) with the learned `merges`,
/// applying the lowest-rank (= lowest new_id) mergeable adjacent pair repeatedly.
/// This is the per-chunk core of `Tokenizer::encode` without the regex split —
/// the chunk is already a regex piece. Matches tiktoken's canonical BPE encode of
/// the same chunk with the same ranks, so corpus token counts are identical.
fn encode_word(merges: &StdHashMap<Pair, u32>, bytes: &[u32]) -> Vec<u32> {
    let mut ids: Vec<u32> = bytes.to_vec();
    while ids.len() >= 2 {
        let mut best: Option<(usize, u32)> = None; // (index, new_id/rank)
        for i in 0..ids.len() - 1 {
            if let Some(&new_id) = merges.get(&(ids[i], ids[i + 1])) {
                if best.map_or(true, |(_, b)| new_id < b) {
                    best = Some((i, new_id));
                }
            }
        }
        match best {
            Some((idx, new_id)) => {
                ids[idx] = new_id;
                ids.remove(idx + 1);
            }
            None => break,
        }
    }
    ids
}

/// Reconstruct token_id -> bytes for every token in [0, 256 + merges.len()).
/// Lighter than `build_token_bytes_and_map` (no reverse map); used per candidate
/// in `evaluate_many` for the coverage scan.
fn build_token_bytes(merges: &StdHashMap<Pair, u32>) -> Vec<Vec<u8>> {
    let n = 256 + merges.len();
    let mut tb: Vec<Vec<u8>> = vec![Vec::new(); n];
    for i in 0..256usize {
        tb[i] = vec![i as u8];
    }
    let mut sorted: Vec<(&Pair, &u32)> = merges.iter().collect();
    sorted.sort_by_key(|&(_, &id)| id);
    for (&(l, r), &id) in sorted {
        let mut b = tb[l as usize].clone();
        let right = tb[r as usize].clone();
        b.extend_from_slice(&right);
        tb[id as usize] = b;
    }
    tb
}

/// Mirror of `tools.blocked_pairs_report._is_word`'s inflection branch: `text`
/// (already lowercased, and already known NOT to be a direct dict hit) counts as a
/// whole word if a light inflectional strip of it is in `words`
/// (governments->government, runs->run, cities->city). Suffix bytes are ASCII, so
/// the byte slicing stays on char boundaries.
fn is_inflected_word(text: &str, words: &AHashSet<String>) -> bool {
    for suf in ["s", "es", "ed", "ing", "d"] {
        if text.len() > suf.len() + 1 && text.ends_with(suf) && words.contains(&text[..text.len() - suf.len()]) {
            return true;
        }
    }
    if text.len() > 4 && text.ends_with("ies") {
        let mut stem = text[..text.len() - 3].to_string();
        stem.push('y');
        if words.contains(&stem) {
            return true;
        }
    }
    false
}

// ------------------------ END helpers ------------------------

impl Tokenizer {

    /// Read a corpus from a Python string-iterator, split it with the pattern,
    /// and count unique chunks — returning the materialized (words, counts).
    /// This is the read/tokenize/count pass shared by `train_from_iterator`
    /// (one-shot) and `load_corpus` (cache for repeated `train_from_cached`).
    /// It is the dominant cost of a train (~85% of wall time on a real corpus),
    /// and is independent of `blocked_pairs` — hence worth caching. Sets
    /// `self.pattern` / `self.compiled_pattern` as a side effect.
    fn ingest_corpus(
        &mut self,
        py: pyo3::Python<'_>,
        iterator: &pyo3::Bound<'_, pyo3::PyAny>,
        pattern: Option<String>,
        buffer_size: usize,
    ) -> PyResult<(Vec<Word>, Vec<i32>)> {
        // Use provided pattern or default to GPT-4 pattern
        let pattern_str = pattern.unwrap_or_else(|| GPT4_PATTERN.to_string());

        // Update the stored pattern and compile it
        self.pattern = pattern_str.clone();
        self.compiled_pattern = Regex::new(&pattern_str)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("Invalid regex pattern: {}", e)))?;

        // Prepare a true Python iterator object
        let py_iter: pyo3::Py<pyo3::PyAny> = unsafe {
            pyo3::Py::from_owned_ptr_or_err(py, pyo3::ffi::PyObject_GetIter(iterator.as_ptr()))?
        };

        // Global chunk counts
        let mut counts: AHashMap<CompactString, i32> = AHashMap::new();

        // Temporary buffer we refill under the GIL
        let mut buf: Vec<String> = Vec::with_capacity(buffer_size);

        log::info!("Processing sequences from iterator (buffer_size: {})", buffer_size);
        let mut total_sequences = 0u64;

        // Helper: refill `buf` with up to `buffer_size` strings from the Python iterator.
        // Returns Ok(true) if the iterator is exhausted, Ok(false) otherwise.
        let refill = |buf: &mut Vec<String>| -> PyResult<bool> {
            pyo3::Python::with_gil(|py| {
                buf.clear();
                let it = py_iter.bind(py);
                loop {
                    if buf.len() >= buffer_size {
                        return Ok(false);
                    }
                    // next(it)
                    let next_obj = unsafe {
                        pyo3::Bound::from_owned_ptr_or_opt(py, pyo3::ffi::PyIter_Next(it.as_ptr()))
                    };
                    match next_obj {
                        Some(obj) => {
                            let s: String = obj.extract()?;
                            buf.push(s);
                        }
                        None => {
                            if pyo3::PyErr::occurred(py) {
                                return Err(pyo3::PyErr::fetch(py));
                            } else {
                                return Ok(true); // exhausted
                            }
                        }
                    }
                }
            })
        };

        // Stream ingestion loop: refill under GIL, process without GIL (parallel)
        loop {
            let exhausted = refill(&mut buf)?;
            if buf.is_empty() && exhausted {
                break;
            }

            total_sequences += buf.len() as u64;

            let pattern = self.compiled_pattern.clone();
            let local: AHashMap<CompactString, i32> = py.allow_threads(|| {
                buf.par_iter()
                    .fold(
                        || AHashMap::<CompactString, i32>::new(),
                        |mut m: AHashMap<CompactString, i32>, s| {
                            for mat in pattern.find_iter(s) {
                                let piece = mat.expect("regex match failed").as_str();
                                increment_chunk_count(&mut m, piece);
                            }
                            m
                        },
                    )
                    .reduce(
                        || AHashMap::<CompactString, i32>::new(),
                        |mut a, b| {
                            for (k, v) in b {
                                *a.entry(k).or_default() += v;
                            }
                            a
                        },
                    )
            });

            // Merge local into global (single-threaded)
            for (k, v) in local {
                *counts.entry(k).or_default() += v;
            }

            if exhausted {
                break;
            }
        }
        log::info!("Processed {} sequences total, {} unique", total_sequences, counts.len());

        // Materialize words & counts
        let mut words = Vec::with_capacity(counts.len());
        let mut cvec = Vec::with_capacity(counts.len());
        for (chunk, c) in counts.into_iter() {
            words.push(Word::new(chunk.as_bytes().iter().map(|&b| b as u32).collect()));
            cvec.push(c);
        }
        Ok((words, cvec))
    }

    /// Core incremental BPE training given unique words and their counts.
    /// `words`: one entry per unique chunk (Vec<u32> of token-ids/bytes).
    /// `counts`: same length as `words`, count per chunk.
    fn train_core_incremental(
        &mut self,
        mut words: Vec<Word>,
        counts: Vec<i32>,
        vocab_size: u32,
        blocked_pairs: Option<Vec<(String, String)>>,
        forced_pairs: Option<Vec<(String, String)>>,
        block_trailing_space: bool,
        block_leading_space: bool,
    ) {
        assert!(vocab_size >= 256, "vocab_size must be at least 256");

        let blocked_specs: Vec<BlockedPairSpec> = blocked_pairs
            .unwrap_or_default()
            .into_iter()
            .map(|(l, r)| BlockedPairSpec {
                left_bytes: l.as_bytes().to_vec(),
                right_bytes: r.as_bytes().to_vec(),
                left_str: l,
                right_str: r,
            })
            .collect();

        let forced_specs: Vec<ForcedMergeSpec> = forced_pairs
            .unwrap_or_default()
            .into_iter()
            .map(|(l, r)| {
                let left_bytes = l.as_bytes().to_vec();
                let right_bytes = r.as_bytes().to_vec();
                let mut concat_bytes = left_bytes.clone();
                concat_bytes.extend_from_slice(&right_bytes);
                ForcedMergeSpec {
                    concat_bytes,
                    left_bytes,
                    right_bytes,
                    left_str: l,
                    right_str: r,
                }
            })
            .collect();

        let blocked_pair_lookup = build_pair_lookup(&blocked_specs);
        let forced_pair_lookup = build_pair_lookup(&forced_specs);
        let forced_concat_lookup = build_forced_concat_lookup(&forced_specs);

        let total_capacity = vocab_size - 256;
        let reserved_merges = forced_specs.len() as u32;
        if reserved_merges > total_capacity {
            log::warn!(
                "There are {} forced-merge pairs but only {} available merge slots (vocab_size = {}). \
                 Some forced merges will be skipped.",
                reserved_merges, total_capacity, vocab_size
            );
        }
        // Phase 1 target: leave room for `reserved_merges` forced specs. If fewer than
        // `reserved_merges` actually fire (operands missing, or the concat is already a token),
        // Phase 2 backfills the unused slots with normal BPE merges so the final vocab is
        // exactly `vocab_size` regardless of how many forced specs succeed. Fixes a bug
        // where vocab silently underfilled when forced specs were skipped.
        let phase1_target = total_capacity.saturating_sub(reserved_merges);
        log::info!(
            "Starting BPE training: total capacity = {} merges ({} normal phase-1 + {} reserved \
             for forced merges; unused reservations will be backfilled with normal merges), \
             blocking {} pairs from merging",
            total_capacity, phase1_target, reserved_merges.min(total_capacity), blocked_specs.len(),
        );
        self.merges.clear();

        // --- Track token bytes incrementally so we can recognize forced pairs ---
        let mut token_bytes: Vec<Vec<u8>> =
            (0..256_u32).map(|i| vec![i as u8]).collect();

        // Pairs that we are not allowed to merge in the normal loop
        let mut banned_pairs: AHashSet<Pair> = AHashSet::new();

        // ---- Initial pair_counts and where_to_update (parallel) ----
        log::info!("Computing initial pair counts from {} unique sequences", words.len());
        let (mut pair_counts, mut where_to_update) = count_pairs_parallel(&words, &counts);

        // ---- Build heap ----
        log::info!("Building heap with {} unique pairs", pair_counts.len());
        let mut heap = OctonaryHeap::with_capacity(pair_counts.len());
        for (pair, pos) in where_to_update.drain() {
            let c = *pair_counts.get(&pair).unwrap_or(&0);
            if c > 0 {
                heap.push(MergeJob {
                    pair,
                    count: c as u64,
                    pos,
                });
            }
        }

        // ---- Merge loop ----
        // Wrapped in a phase-loop so we can re-enter after `apply_forced_merges_at_end`
        // to backfill any reservation slots that went unused.
        log::info!("Starting merge loop (phase 1, target = {} merges)", phase1_target);
        let mut merges_done = 0u32;
        let mut last_log_percent = 0u32;
        let mut phase: u8 = 1;
        let mut current_target = phase1_target;
        let mut merged_bytes_scratch: Vec<u8> = Vec::new();

        'phase_loop: loop {
        'merge_loop: while merges_done < current_target {
            let Some(mut top) = heap.pop() else { break; };

            // Lazy refresh
            let current = *pair_counts.get(&top.pair).unwrap_or(&0);
            if top.count != current as u64 {
                top.count = current as u64;
                if top.count > 0 {
                    heap.push(top);
                }
                continue;
            }
            if top.count == 0 {
                break;
            }

            // On phase-2 re-entry, the heap may pop a pair that was already merged
            // by apply_forced_merges_at_end (the heap doesn't know about those).
            // Skip — re-inserting would clobber the forced merge rule with a new id.
            if self.merges.contains_key(&top.pair) {
                continue 'merge_loop;
            }

            // Skip pairs we already decided are banned
            if banned_pairs.contains(&top.pair) {
                continue 'merge_loop;
            }

            // --- Suppress natural merges for forced pairs ---
            if !forced_specs.is_empty() || !blocked_specs.is_empty() || block_trailing_space || block_leading_space {
                let (left, right) = top.pair;
                let left_bytes = &token_bytes[left as usize];
                let right_bytes = &token_bytes[right as usize];

                // 0) Trailing-space rule (block_trailing_space): forbid any merge whose
                //    right operand ends in a space and whose left operand is not all
                //    whitespace. Kills "X " junk tokens the carve-out's internal spaces
                //    would otherwise spawn, while leaving leading-space tokens (" word"),
                //    internal-space forced phrases (" of the"), and pure whitespace runs
                //    untouched. A predicate, so it's complete + needs no enumerated list.
                if block_trailing_space
                    && right_bytes.last() == Some(&b' ')
                    && !left_bytes.iter().all(|&b| b == b' ')
                {
                    banned_pairs.insert(top.pair);
                    continue 'merge_loop;
                }

                // 0b) Leading-space rule (block_leading_space): forbid any merge whose
                //     right operand STARTS with a space and whose left operand is not all
                //     whitespace — i.e. a multi-word bigram attaching a new word. Multi-word
                //     phrases come ONLY from the injected forced list; inside a carved chunk
                //     the natural bigram (' It'+' is') would otherwise form first and
                //     intercept the forced merge ('. It is'), so banning them all makes the
                //     forced phrases the sole multi-word tokens. Pure-whitespace runs survive.
                if block_leading_space
                    && right_bytes.first() == Some(&b' ')
                    && !left_bytes.iter().all(|&b| b == b' ')
                {
                    banned_pairs.insert(top.pair);
                    continue 'merge_loop;
                }

                // 1) Explicitly blocked pairs: NEVER allowed to merge.
                if let Some(spec_idx) =
                    find_pair_lookup(&blocked_pair_lookup, left_bytes, right_bytes)
                {
                    let spec = &blocked_specs[spec_idx];
                    log::debug!(
                        "Blocking merge of explicitly blocked pair {:?}+{:?}",
                        spec.left_str,
                        spec.right_str
                    );
                    banned_pairs.insert(top.pair);
                    continue 'merge_loop;
                }

                // 2) Forced pairs: suppress natural merges of the forced pair itself,
                //    and any other merge that would create the same concat as a forced pair.
                if !forced_specs.is_empty() {
                    // 2a) Block exact forced pair
                    if let Some(spec_idx) =
                        find_pair_lookup(&forced_pair_lookup, left_bytes, right_bytes)
                    {
                        let spec = &forced_specs[spec_idx];
                        log::debug!(
                            "Suppressing natural merge of forced pair {:?}+{:?} during normal training",
                            spec.left_str,
                            spec.right_str
                        );
                        banned_pairs.insert(top.pair);
                        continue 'merge_loop;
                    }

                    // 2b) Block any merge whose concat == forced concat (different decomposition)
                    merged_bytes_scratch.clear();
                    merged_bytes_scratch.extend_from_slice(left_bytes);
                    merged_bytes_scratch.extend_from_slice(right_bytes);

                    if let Some(spec_idx) =
                        forced_concat_lookup.get(merged_bytes_scratch.as_slice())
                    {
                        let spec = &forced_specs[*spec_idx];
                        log::debug!(
                            "Suppressing natural merge producing concat of forced pair {:?}+{:?}",
                            spec.left_str,
                            spec.right_str
                        );
                        banned_pairs.insert(top.pair);
                        continue 'merge_loop;
                    }
                }
            }

            // Record merge
            let new_id = 256 + merges_done;
            self.merges.insert(top.pair, new_id);

            // Update token_bytes for the new token so we can recognize future forced pairs
            let (left, right) = top.pair;
            let mut merged_bytes = token_bytes[left as usize].clone();
            merged_bytes.extend_from_slice(&token_bytes[right as usize]);
            if token_bytes.len() <= new_id as usize {
                token_bytes.resize(new_id as usize + 1, Vec::new());
            }
            token_bytes[new_id as usize] = merged_bytes;

            // Merge this pair in all words where it occurs
            let mut local_pos_updates: AHashMap<Pair, AHashSet<usize>> = AHashMap::new();
            for &word_idx in &top.pos {
                // Apply merge to this word and collect pair-count deltas
                let changes = words[word_idx].merge_pair(top.pair, new_id);
                // Update global pair counts based on this word's count
                for (pair, delta) in changes {
                    let delta_total = delta * counts[word_idx];
                    if delta_total != 0 {
                        *pair_counts.entry(pair).or_default() += delta_total;
                        if delta > 0 {
                            local_pos_updates.entry(pair).or_default().insert(word_idx);
                        }
                    }
                }
            }

            // Add the updated pair counts back to the heap
            for (pair, pos) in local_pos_updates {
                let cnt = *pair_counts.get(&pair).unwrap_or(&0);
                if cnt > 0 {
                    heap.push(MergeJob {
                        pair,
                        count: cnt as u64,
                        pos,
                    });
                }
            }

            merges_done += 1;

            // Log progress every 1% (relative to current phase target; avoids div-by-zero
            // when current_target == 0, which can happen if all capacity is reserved).
            if current_target > 0 {
                let current_percent = (merges_done * 100) / current_target;
                if current_percent > last_log_percent {
                    log::info!(
                        "Phase {} progress: {}% ({}/{} merges) - Last merge: {:?} -> {} (frequency: {})",
                        phase, current_percent, merges_done, current_target, top.pair, new_id, top.count
                    );
                    last_log_percent = current_percent;
                }
            }
        }
        // --- End of inner 'merge_loop ---

        match phase {
            1 => {
                log::info!("Finished phase-1: {} normal merges completed", merges_done);
                let before_forced = merges_done;
                apply_forced_merges_at_end(
                    &blocked_pair_lookup,
                    &forced_specs,
                    &mut self.merges,
                    vocab_size,
                    &mut merges_done,
                );
                let applied = merges_done - before_forced;
                let unused_reservation = reserved_merges.saturating_sub(applied);
                log::info!(
                    "Applied {}/{} forced merges; {} reservation slots unused",
                    applied, reserved_merges, unused_reservation,
                );
                if merges_done < total_capacity {
                    // Phase 2: backfill any unused reservation slots with normal BPE merges
                    // so vocab is exactly `vocab_size`. The heap and pair_counts state is
                    // still alive; apply_forced_merges_at_end only touched self.merges and
                    // merges_done. The "skip if already merged" check at the top of the
                    // loop handles heap entries for pairs that were force-merged above.
                    phase = 2;
                    current_target = total_capacity;
                    last_log_percent = 0;
                    log::info!(
                        "Starting phase-2 backfill ({} additional normal merges to reach \
                         total capacity = {})",
                        current_target - merges_done, total_capacity,
                    );
                    continue 'phase_loop;
                }
                break 'phase_loop;
            }
            _ => {
                log::info!("Finished phase-2 backfill: {} total merges completed", merges_done);
                break 'phase_loop;
            }
        }
        } // end 'phase_loop
    }
}

/// Public methods for the Tokenizer class that will be exposed to Python.
#[pymethods]
impl Tokenizer {
    /// Create a new Tokenizer
    #[new]
    pub fn new() -> Self {
        Self {
            merges: StdHashMap::new(),
            pattern: String::new(),
            compiled_pattern: Regex::new("").expect("Empty regex should be valid"),
            cached_words: None,
            cached_counts: None,
            holdout_raw: Vec::new(),
            holdout_chunks: None,
            holdout_shard_counts: None,
            holdout_names: None,
            wordlist: None,
        }
    }

    /// Train from a streaming iterator (parallel ingestion).
    /// We refill a Rust Vec<String> buffer under the GIL, then release the GIL
    /// to do the heavy splitting and counting **in parallel** with rayon.
    #[pyo3(signature = (iterator, vocab_size, buffer_size=8192, pattern=None, blocked_pairs=None, forced_pairs=None, block_trailing_space=false, block_leading_space=false))]
    #[pyo3(text_signature = "(self, iterator, vocab_size, buffer_size=8192, pattern=None, blocked_pairs=None, forced_pairs=None, block_trailing_space=False, block_leading_space=False)")]
    pub fn train_from_iterator(
        &mut self,
        py: pyo3::Python<'_>,
        iterator: &pyo3::Bound<'_, pyo3::PyAny>,
        vocab_size: u32,
        buffer_size: usize,
        pattern: Option<String>,
        blocked_pairs: Option<Vec<(String, String)>>,
        forced_pairs: Option<Vec<(String, String)>>,
        block_trailing_space: bool,
        block_leading_space: bool,
    ) -> PyResult<()> {
        let (words, cvec) = self.ingest_corpus(py, iterator, pattern, buffer_size)?;
        self.train_core_incremental(words, cvec, vocab_size, blocked_pairs, forced_pairs, block_trailing_space, block_leading_space);
        Ok(())
    }

    /// Read + split + count a corpus once and CACHE the result (unique words +
    /// counts) on the tokenizer, so `train_from_cached` can run the merge loop
    /// repeatedly while only `blocked_pairs` vary — skipping the dominant
    /// read/tokenize cost on every subsequent train. Overwrites any prior cache.
    #[pyo3(signature = (iterator, buffer_size=8192, pattern=None))]
    #[pyo3(text_signature = "(self, iterator, buffer_size=8192, pattern=None)")]
    pub fn load_corpus(
        &mut self,
        py: pyo3::Python<'_>,
        iterator: &pyo3::Bound<'_, pyo3::PyAny>,
        buffer_size: usize,
        pattern: Option<String>,
    ) -> PyResult<()> {
        let (words, cvec) = self.ingest_corpus(py, iterator, pattern, buffer_size)?;
        log::info!("Cached corpus: {} unique words", words.len());
        self.cached_words = Some(words);
        self.cached_counts = Some(cvec);
        Ok(())
    }

    /// Train the merge loop against the corpus cached by `load_corpus`, with the
    /// given `blocked_pairs`/`forced_pairs`. Clones the cache (the merge loop
    /// consumes its inputs) so it stays reusable. Bit-identical to a
    /// `train_from_iterator` call with the same corpus + args. Errors if no
    /// corpus has been cached.
    #[pyo3(signature = (vocab_size, blocked_pairs=None, forced_pairs=None, block_trailing_space=false, block_leading_space=false))]
    #[pyo3(text_signature = "(self, vocab_size, blocked_pairs=None, forced_pairs=None, block_trailing_space=False, block_leading_space=False)")]
    pub fn train_from_cached(
        &mut self,
        vocab_size: u32,
        blocked_pairs: Option<Vec<(String, String)>>,
        forced_pairs: Option<Vec<(String, String)>>,
        block_trailing_space: bool,
        block_leading_space: bool,
    ) -> PyResult<()> {
        let words = self.cached_words.clone().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("no cached corpus; call load_corpus() first")
        })?;
        let counts = self.cached_counts.clone().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("no cached corpus; call load_corpus() first")
        })?;
        self.train_core_incremental(words, counts, vocab_size, blocked_pairs, forced_pairs, block_trailing_space, block_leading_space);
        Ok(())
    }

    /// Cache the lowercased dictionary wordlist for the in-Rust coverage scan.
    pub fn set_wordlist(&mut self, words: Vec<String>) {
        self.wordlist = Some(words.into_iter().map(|w| w.to_lowercase()).collect());
    }

    /// Drop any staged/finalized held-out corpus.
    pub fn clear_holdout(&mut self) {
        self.holdout_raw.clear();
        self.holdout_chunks = None;
        self.holdout_shard_counts = None;
        self.holdout_names = None;
    }

    /// Stage one held-out shard: read + split + count it (same path/pattern as the
    /// training corpus), keeping its unique chunks + counts for `finalize_holdout`.
    #[pyo3(signature = (name, iterator, buffer_size=8192, pattern=None))]
    #[pyo3(text_signature = "(self, name, iterator, buffer_size=8192, pattern=None)")]
    pub fn add_holdout_shard(
        &mut self,
        py: pyo3::Python<'_>,
        name: String,
        iterator: &pyo3::Bound<'_, pyo3::PyAny>,
        buffer_size: usize,
        pattern: Option<String>,
    ) -> PyResult<()> {
        let (words, counts) = self.ingest_corpus(py, iterator, pattern, buffer_size)?;
        self.holdout_raw.push((name, words, counts));
        Ok(())
    }

    /// Collapse the staged shards into the global-dedup form: one shared list of
    /// distinct chunks across all shards + per-shard `(chunk_index, count)` lists,
    /// so each candidate encodes every distinct chunk exactly once.
    pub fn finalize_holdout(&mut self) {
        let mut index: AHashMap<Vec<u32>, u32> = AHashMap::new();
        let mut chunks: Vec<Word> = Vec::new();
        let mut names: Vec<String> = Vec::new();
        let mut shard_counts: Vec<Vec<(u32, i32)>> = Vec::new();
        for (name, words, counts) in self.holdout_raw.drain(..) {
            names.push(name);
            let mut sc: Vec<(u32, i32)> = Vec::with_capacity(words.len());
            for (w, c) in words.into_iter().zip(counts.into_iter()) {
                let idx = match index.get(&w.ids) {
                    Some(&i) => i,
                    None => {
                        let i = chunks.len() as u32;
                        index.insert(w.ids.clone(), i);
                        chunks.push(w);
                        i
                    }
                };
                sc.push((idx, c));
            }
            shard_counts.push(sc);
        }
        self.holdout_chunks = Some(chunks);
        self.holdout_shard_counts = Some(shard_counts);
        self.holdout_names = Some(names);
    }

    /// Held-out shard names, in the order `evaluate_many` returns per-shard metrics.
    pub fn get_holdout_names(&self) -> Vec<String> {
        self.holdout_names.clone().unwrap_or_default()
    }

    /// FUSED train + measure for MANY candidates in PARALLEL. For each candidate:
    /// train the merge loop (cached training corpus), then measure against the
    /// finalized held-out corpus + wordlist — all in Rust, GIL released, rayon
    /// across candidates (`num_threads` caps the pool). Returns one
    /// `(coverage_base, coverage_inflected, [(tokens, dead) per shard])` per
    /// candidate (per-shard order = `get_holdout_names`). Bit-identical to the
    /// Python `build_tok` + `measure` path, so callers can drop the per-candidate
    /// tiktoken round-trip entirely. Requires `load_corpus`, `finalize_holdout`,
    /// and `set_wordlist` first.
    #[pyo3(signature = (vocab_size, blocked_pairs_batch, num_threads=None, forced_pairs=None, block_trailing_space=false, block_leading_space=false))]
    #[pyo3(text_signature = "(self, vocab_size, blocked_pairs_batch, num_threads=None, forced_pairs=None, block_trailing_space=False, block_leading_space=False)")]
    pub fn evaluate_many(
        &self,
        py: pyo3::Python<'_>,
        vocab_size: u32,
        blocked_pairs_batch: Vec<Vec<(String, String)>>,
        num_threads: Option<usize>,
        forced_pairs: Option<Vec<(String, String)>>,
        block_trailing_space: bool,
        block_leading_space: bool,
    ) -> PyResult<Vec<(usize, usize, Vec<(u64, usize)>)>> {
        let words = self.cached_words.as_ref().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("no cached corpus; call load_corpus() first")
        })?;
        let counts = self.cached_counts.as_ref().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("no cached corpus; call load_corpus() first")
        })?;
        let chunks = self.holdout_chunks.as_ref().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("no held-out; call finalize_holdout() first")
        })?;
        let shard_counts = self.holdout_shard_counts.as_ref().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("no held-out; call finalize_holdout() first")
        })?;
        let wordlist = self.wordlist.as_ref().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("no wordlist; call set_wordlist() first")
        })?;

        let run = || {
            blocked_pairs_batch
                .par_iter()
                .map(|blocked| {
                    // --- train ---
                    let mut t = Tokenizer::new();
                    t.train_core_incremental(
                        words.clone(),
                        counts.clone(),
                        vocab_size,
                        Some(blocked.clone()),
                        forced_pairs.clone(),
                        block_trailing_space,
                        block_leading_space,
                    );
                    let n_merge = 256 + t.merges.len();

                    // --- coverage (whole-word vocab tokens) ---
                    let token_bytes = build_token_bytes(&t.merges);
                    let mut base = 0usize;
                    let mut infl = 0usize;
                    for b in token_bytes.iter().take(n_merge) {
                        if let Ok(s) = std::str::from_utf8(b) {
                            if let Some(rest) = s.strip_prefix(' ') {
                                let w = rest.to_lowercase();
                                if wordlist.contains(&w) {
                                    base += 1;
                                } else if is_inflected_word(&w, wordlist) {
                                    infl += 1;
                                }
                            }
                        }
                    }

                    // --- compression + dead per shard (encode each chunk once) ---
                    let encoded: Vec<Vec<u32>> =
                        chunks.iter().map(|c| encode_word(&t.merges, &c.ids)).collect();
                    let per_shard: Vec<(u64, usize)> = shard_counts
                        .iter()
                        .map(|sc| {
                            let mut fired = vec![false; n_merge];
                            let mut tokens: u64 = 0;
                            for &(cidx, cnt) in sc {
                                let ids = &encoded[cidx as usize];
                                tokens += ids.len() as u64 * cnt as u64;
                                for &id in ids {
                                    fired[id as usize] = true;
                                }
                            }
                            let dead = fired.iter().filter(|&&f| !f).count();
                            (tokens, dead)
                        })
                        .collect();

                    (base, infl, per_shard)
                })
                .collect::<Vec<_>>()
        };
        let results = py.allow_threads(|| match num_threads {
            Some(p) => rayon::ThreadPoolBuilder::new()
                .num_threads(p)
                .build()
                .expect("failed to build rayon pool")
                .install(run),
            None => run(),
        });
        Ok(results)
    }

    /// Return the regex pattern
    pub fn get_pattern(&self) -> String {
        self.pattern.clone()
    }

    /// Return the mergeable ranks (token bytes -> token id / rank)
    pub fn get_mergeable_ranks(&self) -> Vec<(Vec<u8>, u32)> {
        let mut mergeable_ranks = Vec::new();

        // Build vocabulary incrementally from low to high token IDs
        let mut token_bytes: Vec<Vec<u8>> = (0..256_u32).map(|i| vec![i as u8]).collect();

        for (i, bytes) in token_bytes.iter().enumerate() {
            mergeable_ranks.push((bytes.clone(), i as u32));
        }

        // Sort merges by token id (so we can reconstruct bytes progressively)
        let mut sorted_merges: Vec<_> = self.merges.iter().collect();
        sorted_merges.sort_by_key(|&(_, &token_id)| token_id);

        for (&pair, &merged_id) in sorted_merges {
            let (left, right) = pair;
            let mut merged_bytes = token_bytes[left as usize].clone();
            merged_bytes.extend(&token_bytes[right as usize]);

            if token_bytes.len() <= merged_id as usize {
                token_bytes.resize(merged_id as usize + 1, Vec::new());
            }
            token_bytes[merged_id as usize] = merged_bytes.clone();

            mergeable_ranks.push((merged_bytes, merged_id));
        }

        mergeable_ranks
    }

    /// Encode a string into token IDs
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut all_ids = Vec::new();

        // Split text using the regex pattern
        for m in self.compiled_pattern.find_iter(text) {
            let chunk = m.expect("regex match failed").as_str();

            // Convert chunk to bytes then to u32 IDs
            let mut ids: Vec<u32> = chunk.bytes().map(|b| b as u32).collect();

            // Apply merges iteratively
            while ids.len() >= 2 {
                // Find the best pair to merge
                let mut best_pair: Option<(usize, Pair, u32)> = None;

                for i in 0..ids.len() - 1 {
                    let pair: Pair = (ids[i], ids[i + 1]);
                    if let Some(&new_id) = self.merges.get(&pair) {
                        if best_pair.is_none() || new_id < best_pair.unwrap().2 {
                            best_pair = Some((i, pair, new_id));
                        }
                    }
                }

                // If we found a pair to merge, apply it
                if let Some((idx, _pair, new_id)) = best_pair {
                    ids[idx] = new_id;
                    ids.remove(idx + 1);
                } else {
                    // No more merges possible
                    break;
                }
            }

            all_ids.extend(ids);
        }

        all_ids
    }
}

#[pymodule]
fn rustbpe_auto_tune(m: &Bound<'_, PyModule>) -> PyResult<()> {
    pyo3_log::init(); // forwards Rust `log` to Python's `logging`
    m.add_class::<Tokenizer>()?;
    Ok(())
}
