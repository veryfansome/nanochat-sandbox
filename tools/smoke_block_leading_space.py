"""Smoke test: the block_leading_space crate predicate.

block_leading_space bans any base merge whose RIGHT operand starts with a space and
whose LEFT operand isn't all-whitespace — i.e. natural multi-word bigrams (' It'+' is',
'.'+' T') that would otherwise intercept the injected forced phrases. It's the mirror of
block_trailing_space. This trains a tiny synthetic corpus (whole-line chunks, so
multi-word merges CAN form) with the predicate off vs on, reconstructs each token's
formation operands, and asserts: off → some space-leading-R merges form; on → none do
(while pure-whitespace runs, L all-space, are unaffected). Fast, no external data.

    uv run python -m tools.smoke_block_leading_space
"""
import rustbpe_auto_tune


def _train(block_leading_space):
    # multi-word phrases (so space-leading-R bigrams CAN form) + very frequent
    # all-whitespace runs (all-whitespace L, which the predicate must NOT block).
    lines = ([" It is a cat", " It is a dog", " The cat is here", " It is fun now"]
             + ["        "] * 4)  # 8-space indentation runs, frequent enough to make the vocab
    text = "\n".join(lines * 500)
    tr = rustbpe_auto_tune.Tokenizer()
    tr.load_corpus(iter([text]), pattern=r"[^\n]+")  # each line is one chunk -> bigrams can form
    tr.train_from_cached(500, block_leading_space=block_leading_space)
    return {bytes(k): v for k, v in tr.get_mergeable_ranks()}


def _operands(ranks, tb):
    rank = ranks[tb]
    parts = [bytes([x]) for x in tb]
    while len(parts) > 1:
        bi = br = None
        for i in range(len(parts) - 1):
            r = ranks.get(parts[i] + parts[i + 1])
            if r is not None and r < rank and (br is None or r < br):
                bi, br = i, r
        if bi is None:
            break
        parts = parts[:bi] + [parts[bi] + parts[bi + 1]] + parts[bi + 2:]
    return tuple(parts) if len(parts) == 2 else None


def _violations(ranks):
    """tokens whose formation R starts with a space and L is not all-whitespace."""
    out = []
    for b in ranks:
        if len(b) < 2:
            continue
        op = _operands(ranks, b)
        if op and op[1][:1] == b" " and not all(c == 0x20 for c in op[0]):
            out.append(b)
    return out


def test_predicate():
    off = _train(False)
    on = _train(True)
    v_off, v_on = _violations(off), _violations(on)
    assert v_off, "test corpus should produce space-leading-R merges with the predicate OFF"
    assert not v_on, ("block_leading_space must ban every space-leading-R merge, got: "
                      + repr([b.decode("latin1") for b in v_on][:8]))
    # whitespace runs (L all-space) must survive — the '  indented' lines should still let
    # multi-space tokens form even with the predicate on.
    ws_on = [b for b in on if len(b) >= 2 and all(c == 0x20 for c in b)]
    assert ws_on, "pure-whitespace runs (all-space L) should NOT be blocked by block_leading_space"
    print(f"  [ok] block_leading_space: OFF→{len(v_off)} space-leading-R merges, ON→0; "
          f"whitespace runs preserved ({len(ws_on)})")


if __name__ == "__main__":
    print("smoke: block_leading_space predicate")
    test_predicate()
    print("PASS")
