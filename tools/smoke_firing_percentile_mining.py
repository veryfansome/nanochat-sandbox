"""Smoke test: firing-percentile mining.

Verifies the repurposed dead-orphan path — `find_firing_percentile_candidates`
(a raw firing-percentile cut over fireable merges, specials + base bytes excluded)
plus its `mine_candidates` / driver-CLI wiring. Runs in <5s with no data: a tiny
synthetic vocab + firing dict, so the percentile/exclusion logic is checked
deterministically without loading a tokenizer or corpus.

    uv run python -m tools.smoke_firing_percentile_mining
"""
import inspect
import subprocess
import sys


def _synthetic():
    """256 base bytes (all firing 0) + 5 high-firing leaf merges + 4 target tokens
    with varied firing, each formed from the leaves. Returns (b2i, i2b, fire)."""
    i2b = {i: bytes([i]) for i in range(256)}
    b2i = {bytes([i]): i for i in range(256)}

    def add(s, tid):
        bs = s.encode()
        i2b[tid] = bs
        b2i[bs] = tid

    add("at", 256); add("ed", 257); add("ing", 258); add("es", 259); add("or", 260)
    add("ated", 261); add("ating", 262); add("ates", 263); add("ator", 264)
    # leaves fire a lot (excluded from low-pct cuts); targets span the firing range
    fire = {256: 500, 257: 500, 258: 500, 259: 500, 260: 500,
            261: 100, 262: 3, 263: 8, 264: 50}
    return b2i, i2b, fire


def test_function():
    from tools.find_mangled_morphemes import find_firing_percentile_candidates as fpc
    b2i, i2b, fire = _synthetic()
    c0 = fpc(fire, b2i, i2b, 0)
    c10 = fpc(fire, b2i, i2b, 10)
    c25 = fpc(fire, b2i, i2b, 25)
    c50 = fpc(fire, b2i, i2b, 50)

    assert c0 == set(), f"percentile 0 should disable (empty), got {c0}"
    # 'ating' (firing 3) is selected at pct 10; 'ated' (firing 100) is not
    assert ("at", "ing") in c10, f"ating route missing at pct 10: {c10}"
    assert ("at", "ed") not in c10, f"ated should NOT be selected at pct 10: {c10}"
    # higher percentile is a strict superset (monotone)
    assert c10 <= c25 <= c50, f"not monotone: {c10} <= {c25} <= {c50}"
    # 'ates'/'ator' enter by pct 25
    assert {("at", "es"), ("at", "or")} <= c25, f"ates/ator missing at pct 25: {c25}"
    # BYTE EXCLUSION: were the 256 firing-0 base bytes counted, the median firing
    # would be 0 and pct 50 would select nothing that fires — so 'ated' (firing 100)
    # landing in c50 proves bytes are excluded from the percentile distribution.
    assert ("at", "ed") in c50, f"ated missing at pct 50 — byte exclusion broken? {c50}"
    assert all(isinstance(L, str) and isinstance(R, str) for L, R in c50)
    print("  [ok] find_firing_percentile_candidates: pct-0 empty, low-firing cut, monotone, byte-excluded")


def test_mine_candidates_signature():
    from tools.find_mangled_morphemes import mine_candidates
    params = inspect.signature(mine_candidates).parameters
    assert "firing_percentile" in params, "mine_candidates missing firing_percentile"
    assert "firing_max_chars" in params, "mine_candidates missing firing_max_chars"
    assert "dead_orphans" not in params, "dead_orphans should be repurposed away"
    assert "words" not in params, "words (dead-orphan-only) should be gone"
    print("  [ok] mine_candidates signature: firing_percentile/firing_max_chars in, dead_orphans/words out")


def test_cli():
    out = subprocess.run(
        [sys.executable, "-m", "tools.heldout_fixpoint_cached", "--help"],
        capture_output=True, text=True, timeout=120)
    h = out.stdout + out.stderr
    assert "--mine-firing-percentile" in h, "CLI missing --mine-firing-percentile"
    assert "--mine-firing-max-chars" in h, "CLI missing --mine-firing-max-chars"
    assert "--mine-dead-orphans" not in h, "CLI still exposes the old --mine-dead-orphans"
    print("  [ok] driver CLI: --mine-firing-percentile/--mine-firing-max-chars wired; --mine-dead-orphans gone")


if __name__ == "__main__":
    print("smoke: firing-percentile mining")
    test_function()
    test_mine_candidates_signature()
    test_cli()
    print("PASS")
