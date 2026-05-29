"""Post-hoc token lineage + influence over the held-out build-up artifacts.

Reads the retained keep-tokenizers (`<work-base>/kept_tokenizers/<pass>/keep_NNNN`,
where `keep_0000` is the empty reference and `keep_K` is the tokenizer after the
K-th KEPT pair) plus the pass's `checkpoint.json` / `lineage.jsonl`, to answer two
questions about any token WITHOUT retraining:

  LINEAGE   — how the token is built (its BPE merge tree in the latest tokenizer
              that contains it) and which keep introduced / removed it (born/died
              across the chain).
  INFLUENCE — how the token's firing mass shifts as blocked pairs accumulate: its
              firing count on a corpus at every retained keep, with each Δ attributed
              to the pair that keep added. This is the "what does blocking reroute"
              signal — a token can gain mass (words now route through it) or lose it
              (it was the agglutinated route that got blocked).

  uv run python -m tools.token_lineage --token ' running'      # full trace
  uv run python -m tools.token_lineage --token ation --no-trajectory  # skip corpus encodes
  uv run python -m tools.token_lineage                          # overview from lineage.jsonl

Operates on whatever keeps are present, so it's safe to run mid-build-up.
"""
import argparse
import glob
import json
import os
from pathlib import Path

from tools.heldout_buildup import iter_capped_from_path, vocab_str2id
from tools.pass_metrics import fire_counts


def merge_parents(token_bytes, ranks):
    """The two child tokens whose merge formed `token_bytes` — i.e. replay BPE over
    the byte sequence using `ranks` (bytes->rank); the LAST merge is the parents.
    Returns (left_bytes, right_bytes) or None for a single-byte token."""
    parts = [bytes([b]) for b in token_bytes]
    last = None
    while len(parts) > 1:
        best_rank, best_i = None, -1
        for i in range(len(parts) - 1):
            r = ranks.get(parts[i] + parts[i + 1])
            if r is not None and (best_rank is None or r < best_rank):
                best_rank, best_i = r, i
        if best_i < 0:
            break  # no further mergeable pair (shouldn't happen for a real token)
        last = (parts[best_i], parts[best_i + 1])
        parts[best_i:best_i + 2] = [parts[best_i] + parts[best_i + 1]]
    return last


def print_merge_tree(token_bytes, ranks, indent=0, prefix=""):
    """Recursively print the BPE merge tree of a token (children that merged to form it)."""
    try:
        s = token_bytes.decode("utf-8")
        label = repr(s)
    except UnicodeDecodeError:
        label = token_bytes.hex()
    r = ranks.get(token_bytes)
    rank_s = f"  (rank {r})" if r is not None else ""
    print(f"{'  ' * indent}{prefix}{label}{rank_s}")
    kids = merge_parents(token_bytes, ranks)
    if kids:
        print_merge_tree(kids[0], ranks, indent + 1, "├─ ")
        print_merge_tree(kids[1], ranks, indent + 1, "└─ ")


def load_keeps(retain_dir):
    """[(N, dir)] for keep_NNNN dirs present, sorted by N (0 = empty reference)."""
    out = []
    for d in sorted(glob.glob(str(Path(retain_dir) / "keep_*"))):
        try:
            out.append((int(Path(d).name.split("_")[1]), d))
        except (IndexError, ValueError):
            continue
    return sorted(out)


def main():
    home = os.path.expanduser("~")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--token", help="Token string to trace (include the leading space for word-initial, e.g. ' running')")
    p.add_argument("--state-dir", default="rustbpe_variants/auto_tune/audit/heldout_pass")
    p.add_argument("--retain-dir", default=None,
                   help="kept_tokenizers/<pass> dir (default: derived from --work-base + state-dir name)")
    p.add_argument("--work-base", default=f"{home}/.cache/nanochat-variants/auto_tune_heldout")
    p.add_argument("--shard", default=f"{home}/.cache/nanochat/base_data_climbmix/shard_06542.parquet",
                   help="Corpus for the influence trajectory (default: tune shard)")
    p.add_argument("--max-chars", type=int, default=10_000_000)
    p.add_argument("--no-trajectory", action="store_true", help="Skip the per-keep corpus encodes")
    p.add_argument("--top", type=int, default=20, help="Overview: keeps to show")
    args = p.parse_args()

    state_dir = Path(args.state_dir)
    retain_dir = Path(args.retain_dir) if args.retain_dir else Path(args.work_base) / "kept_tokenizers" / state_dir.name
    ckpt = json.loads((state_dir / "checkpoint.json").read_text()) if (state_dir / "checkpoint.json").exists() else {}
    kept = [tuple(p) for p in ckpt.get("kept", [])]  # kept[k] is the pair added AT keep k+1

    # ---- overview mode (no token): summarize the keep chain from lineage.jsonl ----
    if not args.token:
        lp = state_dir / "lineage.jsonl"
        if not lp.exists():
            raise SystemExit(f"no lineage at {lp} (run the build-up first)")
        recs = [json.loads(l) for l in lp.read_text().splitlines() if l.strip()]
        print(f"[lineage] {len(recs)} keeps in {lp}  (showing first {args.top})\n")
        print(f"{'keep':>4}  {'pair':<22} {'born':>5} {'died':>5}  top gainer / top loser")
        for r in recs[:args.top]:
            L, R = r["pair"]
            g = r["top_gainers"][0] if r["top_gainers"] else ["", 0]
            ls = r["top_losers"][0] if r["top_losers"] else ["", 0]
            print(f"{r['keep']:>4}  {repr(L + '+' + R):<22} {r['n_born']:>5} {r['n_died']:>5}  "
                  f"{g[0]!r} {g[1]:+}  /  {ls[0]!r} {ls[1]:+}")
        print(f"\nTrace one token: uv run python -m tools.token_lineage --token '<tok>'")
        return

    # ---- token mode ----
    from nanochat.tokenizer import RustBPETokenizer
    tok_str = args.token
    keeps = load_keeps(retain_dir)
    if not keeps:
        raise SystemExit(f"no retained keep-tokenizers under {retain_dir} "
                         "(retention is serial-only and post-dates the flag-revisit; "
                         "the held-out pass retains them)")
    print(f"== token {tok_str!r} ==  ({len(keeps)} retained keeps under {retain_dir.name})\n")

    # Single pass over the keep chain: collect vocab presence/id (LINEAGE) and, unless
    # skipped, firing count on the corpus (INFLUENCE) — so each keep tokenizer loads once.
    do_traj = not args.no_trajectory
    corpus = list(iter_capped_from_path(args.shard, args.max_chars)) if do_traj else None
    present = []   # (N, id) where token is in vocab
    traj = []      # (N, fires|None)
    latest_tok = None
    for n, d in keeps:
        t = RustBPETokenizer.from_directory(d)
        tid = vocab_str2id(t).get(tok_str)
        if tid is not None:
            present.append((n, tid))
            latest_tok = t
        if do_traj:
            traj.append((n, fire_counts(t, corpus).get(tid, 0) if tid is not None else None))

    # LINEAGE: first appearance (+ the pair that introduced it) and the merge tree.
    if present:
        first_n = present[0][0]
        pair_at = f" (added by blocking {kept[first_n-1][0]}+{kept[first_n-1][1]})" if 0 < first_n <= len(kept) else " (in the empty-reference vocab)" if first_n == 0 else ""
        print(f"LINEAGE: present in keeps {present[0][0]}..{present[-1][0]}; first appears at keep {first_n}{pair_at}")
        ranks = latest_tok.enc._mergeable_ranks
        tb = tok_str.encode("utf-8")
        if tb in ranks:
            print("  merge tree (latest tokenizer that contains it):")
            print_merge_tree(tb, ranks)
    else:
        print(f"LINEAGE: {tok_str!r} not present in any retained keep vocab.")

    # INFLUENCE: firing trajectory across keeps.
    if not do_traj:
        return
    print(f"\nINFLUENCE: firing count on {Path(args.shard).stem} ({args.max_chars:,} chars), per keep:")
    print(f"  {'keep':>4}  {'pair added':<22} {'fires':>10} {'Δ':>10}")
    prev = None
    for n, fires in traj:
        pair = f"{kept[n-1][0]}+{kept[n-1][1]}" if 0 < n <= len(kept) else "(empty reference)" if n == 0 else ""
        if fires is None:
            print(f"  {n:>4}  {pair:<22} {'absent':>10} {'':>10}")
            prev = None
        else:
            delta = "" if prev is None else f"{fires - prev:+}"
            mark = "  <<" if prev is not None and abs(fires - prev) > 0 else ""
            print(f"  {n:>4}  {pair:<22} {fires:>10,} {delta:>10}{mark}")
            prev = fires


if __name__ == "__main__":
    main()
