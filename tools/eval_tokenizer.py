"""
Offline tokenizer eval harness.

Runs a battery of metrics that **filter** tokenizer candidates before paying
GPU A/B cost. Standalone — only imports `nanochat.tokenizer` + `nanochat.dataset`
(no training-loop deps).

Tier 1 (default, seconds):
  - compression       — bytes/token across a probe corpus battery
  - vocab inspection  — length distribution, longest tokens, single-byte
                        fraction, digit-token analysis, special tokens
  - structural        — round-trip on a curated edge-case battery
  - task probes       — token counts on capability-relevant strings
                        (arithmetic, code idioms, URLs, multilingual)

Tier 2 (--full, minutes):
  - coverage curve    — top-N tokens' cumulative share of a held-out corpus;
                        dead-token count

What this does NOT do: predict downstream CORE / SFT / RL gain. These are
FILTERS. See ideas/tokenizer-variants/README.md "What offline metrics will
NOT do" before drawing speedrun conclusions from these numbers.

Usage:
    # Default: cached tokenizer vs GPT-2 / GPT-4 baselines, Tier 1 only
    uv run python -m tools.eval_tokenizer

    # Compare multiple custom tokenizers + the baselines
    uv run python -m tools.eval_tokenizer \\
        --tokenizers ours,gpt2,gpt4,/path/to/tokenizer_v2

    # Tier 1 + Tier 2 (coverage curve; tokenizes a ClimbMix val chunk)
    uv run python -m tools.eval_tokenizer --full

    # Dump full results as JSON for downstream tools
    uv run python -m tools.eval_tokenizer --json out.json

    # Also write a Markdown report (e.g. to paste into run notes)
    uv run python -m tools.eval_tokenizer --markdown out.md
"""

import argparse
import json
import os
import sys
import time
from collections import Counter
from dataclasses import dataclass, field, asdict

from nanochat.tokenizer import RustBPETokenizer, HuggingFaceTokenizer, get_tokenizer

# ANSI colors (match style of scripts/tok_eval.py)
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
GRAY = "\033[90m"
BOLD = "\033[1m"
RESET = "\033[0m"

# ---------------------------------------------------------------------------
# Probe corpora
#
# Kept inline (not files) so the harness is self-contained and reproducible
# without external data. Expand by adding to the dict.

PROBE_CORPORA = {
    # Standard English prose
    "english": (
        "The quick brown fox jumps over the lazy dog. "
        "She sells seashells by the seashore. "
        "Pack my box with five dozen liquor jugs. "
        "The five boxing wizards jump quickly. "
        "Sphinx of black quartz, judge my vow."
    ),
    # Modern English with contractions, punctuation, mixed case
    "modern_prose": (
        "I'm not sure if you're going to believe this, but it's already "
        "happened — we've shipped the v2 model and it's working better "
        "than we'd hoped. Won't you join us for the release party? "
        "It'll be on the 14th at 7:30pm."
    ),
    # Python code with realistic structure
    "code_python": (
        "import os\n"
        "from typing import Optional\n\n"
        "def normalize(x: float, lo: float = 0.0, hi: float = 1.0) -> float:\n"
        "    if hi - lo < 1e-9:\n"
        "        return lo\n"
        "    return (x - lo) / (hi - lo)\n\n"
        "class Cache(dict):\n"
        "    def __init__(self, maxsize: int = 1024):\n"
        "        super().__init__()\n"
        "        self.maxsize = maxsize\n"
    ),
    # JS / TS code
    "code_javascript": (
        "const fetchUser = async (id) => {\n"
        "  const res = await fetch(`/api/users/${id}`);\n"
        "  if (!res.ok) throw new Error(`HTTP ${res.status}`);\n"
        "  return res.json();\n"
        "};\n\n"
        "export default fetchUser;\n"
    ),
    # Math: LaTeX + ASCII formulas
    "math_latex": (
        "The sum of cubes identity: \\sum_{k=1}^{n} k^3 = "
        "\\left(\\frac{n(n+1)}{2}\\right)^2. "
        "For n=10 this gives 55^2 = 3025."
    ),
    # Arithmetic prose
    "arithmetic": (
        "123 + 456 = 579. 1024 * 2048 = 2097152. "
        "The year 2026 minus 1969 equals 57. "
        "If x = 3.14159 and y = 2.71828, then x*y is approximately 8.5397."
    ),
    # Scientific dense vocabulary
    "science": (
        "Photosynthesis converts CO2 and H2O into glucose (C6H12O6) and O2 "
        "via chlorophyll-mediated light absorption at ~680 nm and ~700 nm "
        "(PSII / PSI). The Calvin-Benson cycle then fixes carbon in the stroma."
    ),
    # Multilingual probes — short representative strings
    "multi_spanish": (
        "El rápido zorro marrón salta sobre el perro perezoso. "
        "Mañana voy a la playa con mis amigos."
    ),
    "multi_french": (
        "Le vif renard brun saute par-dessus le chien paresseux. "
        "Comment ça va aujourd'hui?"
    ),
    "multi_german": (
        "Der schnelle braune Fuchs springt über den faulen Hund. "
        "Wie geht es Ihnen heute, Herr Müller?"
    ),
    "multi_chinese": "敏捷的棕色狐狸跳过了懒狗。今天天气真好,我们去公园散步吧。",
    "multi_japanese": "素早い茶色の狐は怠惰な犬を飛び越える。今日は良い天気ですね。",
    "multi_korean": "빠른 갈색 여우가 게으른 개를 뛰어 넘는다. 오늘 날씨가 좋네요.",
    "multi_arabic": "الثعلب البني السريع يقفز فوق الكلب الكسول. كيف حالك اليوم؟",
    "multi_hindi": "तेज़ भूरी लोमड़ी आलसी कुत्ते के ऊपर कूदती है। आज मौसम कैसा है?",
    "multi_russian": (
        "Быстрая коричневая лиса перепрыгивает через ленивую собаку. "
        "Как у вас дела сегодня?"
    ),
    # Real-world structured strings — URLs, paths, identifiers
    "structured": (
        "Visit https://github.com/karpathy/nanochat or email "
        "alice.smith@example.com for details. The config is at "
        "/usr/local/etc/nanochat/config.json (PID 12345, user_id=abc-def-123)."
    ),
    # JSON
    "json": (
        '{"name": "Claude", "version": "4.7", "model_id": "claude-opus-4-7", '
        '"context": 200000, "capabilities": ["text", "vision", "tool_use"]}'
    ),
    # Whitespace-heavy
    "whitespace": "  leading  spaces\n\n\nmultiple\n\nnewlines\n\tand\ttabs\t\there",
    # Emoji / pictographs
    "emoji": "Status update: 🚀 shipped! 💯 tests passing ✅ Time for ☕ and 🎉",
}

# ---------------------------------------------------------------------------
# Structural battery — encode/decode round-trips. Tokenizer is BROKEN if any
# of these fail. (Distinct from probes because we report PASS/FAIL, not counts.)

STRUCTURAL_TESTS = [
    ("empty",                ""),
    ("single_char",          "a"),
    ("single_unicode",       "你"),
    ("ascii_printable",      "".join(chr(i) for i in range(32, 127))),
    ("whitespace_mix",       " \t\n\r\v\f   "),
    ("contractions",         "I'm you're it's won't they've we'd I'll he's"),
    ("digits_long",          "1234567890" * 5),
    ("digits_spaced",        "1 2 3 4 5 6 7 8 9 10 11 12"),
    ("digits_mixed",         "v1.2.3-rc4 build 99999 on 2026-05-24T14:30:00Z"),
    ("urls_emails",          "https://x.com/a/b?c=d#e foo.bar@example.co.uk"),
    ("unicode_cjk",          "你好世界 こんにちは 안녕하세요"),
    ("unicode_emoji",        "🚀🎉💯🧠🤖🌍 mixed with text"),
    ("unicode_combining",    "café naïve résumé Zoë"),
    ("unicode_rtl",          "العربية עברית"),
    ("zwj_emoji",            "👨‍👩‍👧‍👦 👩🏽‍💻 🏳️‍🌈"),
    ("code_python_indent",   "def f():\n    if True:\n        return 1\n"),
    ("code_tab_indent",      "def f():\n\tif True:\n\t\treturn 1\n"),
    ("long_repeat",          "ababab" * 50),
    ("punctuation_dense",    "!!!??? ...,,, ;;:: (([])) {{}} <<>> /\\|"),
]

# ---------------------------------------------------------------------------
# Task probes — token-counts that directly cap "what's expressible per token".
# Lower = better; ratios across tokenizers are what matters.

TASK_PROBES = {
    "arith_2digit":   "12 + 34 = 46",
    "arith_3digit":   "123 + 456 = 579",
    "arith_4digit":   "1234 + 5678 = 6912",
    "arith_5digit":   "12345 + 67890 = 80235",
    "arith_long":     "1234567890 + 9876543210 = 11111111100",
    "decimal_pi":     "3.14159265358979323846",
    "year_range":     "1990 1991 1992 ... 2024 2025 2026",
    "phone_numbers":  "+1 (415) 555-0123, +44 20 7946 0958, +81 3-1234-5678",
    "dates_iso":      "2026-05-24 2025-12-31 2024-02-29",
    "uuid":           "550e8400-e29b-41d4-a716-446655440000",
    "code_def":       "def foo(x, y):\n    return x + y",
    "code_import":    "import os\nimport sys\nfrom typing import Optional",
    "code_class":     "class Cache(dict):\n    def __init__(self):\n        super().__init__()",
    "code_main":      'if __name__ == "__main__":\n    main()',
    "code_oneliner":  "result = [x*2 for x in range(100) if x % 3 == 0]",
    "url_github":     "https://github.com/karpathy/nanochat/blob/main/README.md",
    "named_entity":   "San Francisco, California, United States of America",
    "chemical":       "C6H12O6 + 6 O2 -> 6 CO2 + 6 H2O",
    "currency":       "$1,234.56 €999.99 £42.00 ¥10000",
}

# ---------------------------------------------------------------------------
# Tokenizer loading

def load_tokenizer(spec: str):
    """
    Load a tokenizer by spec:
      'ours'       — $NANOCHAT_BASE_DIR/tokenizer (the cached one)
      'gpt2'       — GPT-2 base via tiktoken
      'gpt4'       — cl100k_base via tiktoken
      'gpt4o'      — o200k_base via tiktoken
      path         — load from a directory; `~` and `~user` are expanded
                     (bash doesn't expand `~` after a comma in
                     `--tokenizers a,~/path/b`, so we expand here).
                     Dispatches by file present:
                       tokenizer.json → HuggingFaceTokenizer (HF format,
                                        e.g. unigram variant)
                       tokenizer.pkl  → RustBPETokenizer (BPE format,
                                        scripts.tok_train / rustbpe-variants)
    """
    if spec == "ours":
        return get_tokenizer()
    if spec == "gpt2":
        return RustBPETokenizer.from_pretrained("gpt2")
    if spec == "gpt4":
        return RustBPETokenizer.from_pretrained("cl100k_base")
    if spec == "gpt4o":
        return RustBPETokenizer.from_pretrained("o200k_base")
    spec = os.path.expanduser(spec)
    if os.path.isdir(spec):
        if os.path.exists(os.path.join(spec, "tokenizer.json")):
            return HuggingFaceTokenizer.from_directory(spec)
        return RustBPETokenizer.from_directory(spec)
    raise ValueError(f"Cannot resolve tokenizer spec: {spec!r}")


# ---------------------------------------------------------------------------
# Metrics

@dataclass
class TokenizerReport:
    name: str
    spec: str
    vocab_size: int
    num_special: int
    compression: dict           # corpus_name -> {bytes, tokens, ratio}
    vocab: dict                 # vocab inspection stats
    structural: dict            # test_name -> {ok, n_tokens, sample}
    task_probes: dict           # probe_name -> {text, tokens, bytes}
    coverage: dict | None = None  # Tier 2; None if not run


def compute_compression(tok) -> dict:
    """Per-corpus bytes/token + raw counts."""
    out = {}
    for name, text in PROBE_CORPORA.items():
        ids = tok.encode(text)
        byts = len(text.encode("utf-8"))
        out[name] = {
            "bytes": byts,
            "tokens": len(ids),
            "bytes_per_token": byts / len(ids) if ids else 0.0,
        }
    return out


# The GPT-2 / HF tokenizers ByteLevel pre-tokenizer maps each of the 256 input
# bytes to a unique printable Unicode char (byte 0x20 (space) → "Ġ", byte 0xC4
# → "Ä", etc.). To recover real represented bytes from `id_to_token()`'s
# output, we reverse this mapping char-by-char.
#
# IMPORTANT: do NOT use `pre_tokenizers.ByteLevel.alphabet()` to build the
# reverse map by `enumerate()`. That method returns the 256 chars in
# alphabetized order, NOT in byte-value order — so `enumerate` would map
# each char to its list-position, not to the byte it actually represents.
# The correct mapping is computed via the canonical GPT-2 algorithm below.
_BYTELEVEL_CHAR_TO_BYTE: dict | None = None


def _build_bytelevel_reverse() -> dict:
    """
    Compute the GPT-2 / HF ByteLevel byte → char mapping, then invert it.

    Reference: https://github.com/openai/gpt-2/blob/master/src/encoder.py
    Bytes in printable ranges (33–126, 161–172, 174–255) map to themselves;
    remaining bytes (0–32, 127–160, 173) map to chars 256+ (in order).
    """
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = list(bs)
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    # bs[i] is the byte, cs[i] is the codepoint it maps to.
    return {chr(c): b for b, c in zip(bs, cs)}


def _is_bytelevel_hf(tok) -> bool:
    """True iff `tok` is a HuggingFaceTokenizer using a ByteLevel decoder."""
    inner = getattr(tok, "tokenizer", None)
    if inner is None:
        return False
    try:
        decoder = inner.decoder
    except AttributeError:
        return False
    return decoder is not None and type(decoder).__name__ == "ByteLevel"


def _safe_token_bytes(tok, tid):
    """
    Return raw bytes for a token id (the actual text bytes the token
    represents on encode), or None if the id is a vocab gap (e.g. cl100k_base
    reserves id 100256 between regular and special tokens).

    Dispatch by tokenizer family:
      - tiktoken (RustBPETokenizer): `decode_single_token_bytes` (exact).
      - HuggingFaceTokenizer with ByteLevel decoder: reverse-map `id_to_token`'s
        ByteLevel-encoded form back to bytes via `_BYTELEVEL_CHAR_TO_BYTE`.
        Without this step, `id_to_token().encode("utf-8")` returns the internal
        glyph bytes (e.g. b"\\xc4\\xa0the" for the " the" token), not the
        represented text bytes (b" the") — overcounts byte lengths and breaks
        single-byte / digit-only detection.
      - HuggingFaceTokenizer without ByteLevel (e.g. plain WordPiece): the
        id_to_token form already IS the represented text → UTF-8 encode it.
    """
    enc = getattr(tok, "enc", None)
    if enc is not None:
        try:
            return enc.decode_single_token_bytes(tid)
        except (KeyError, Exception):
            return None
    # HuggingFaceTokenizer path.
    try:
        s = tok.id_to_token(tid)
    except Exception:
        return None
    if not s:
        return None
    if _is_bytelevel_hf(tok):
        global _BYTELEVEL_CHAR_TO_BYTE
        if _BYTELEVEL_CHAR_TO_BYTE is None:
            _BYTELEVEL_CHAR_TO_BYTE = _build_bytelevel_reverse()
        try:
            return bytes(_BYTELEVEL_CHAR_TO_BYTE[c] for c in s)
        except KeyError:
            # Token contains chars outside the ByteLevel alphabet — almost
            # certainly a special token (e.g. "<|bos|>", "<|unk|>") whose id
            # is in get_special_tokens() and gets filtered downstream by
            # `if s in special_set`. Return the literal UTF-8 so the caller
            # can still display/compare it; the resulting byte count is
            # excluded from the length distribution via the special-token
            # filter in compute_vocab_stats.
            return s.encode("utf-8")
    # Non-ByteLevel HF (WordPiece, plain BPE without ByteLevel pre-tok, etc.):
    # id_to_token() returns the represented text directly.
    return s.encode("utf-8")


def compute_vocab_stats(tok) -> dict:
    """
    Walk the whole vocab once. Cheap (32K-200K decodes).

    Returns:
      length_hist: token-byte-length histogram
      mean / median / max bytes-per-token
      single_byte_fraction
      digit_token_count: tokens whose decoded form is *all digits*
      digit_token_max_len: longest all-digit token
      longest_tokens: top-15 by byte length (with a snippet)
      special_tokens: list of special-token strings
      vocab_gaps: count of reserved / non-decodable ids (e.g. tiktoken gaps)
    """
    V = tok.get_vocab_size()
    special_set = set(tok.get_special_tokens())

    lengths = []
    digit_tokens = []
    all_lens = []
    vocab_gaps = 0

    for tid in range(V):
        b = _safe_token_bytes(tok, tid)
        if b is None:
            vocab_gaps += 1
            continue
        # Decode for display + digit check; replace unrenderable bytes
        s = b.decode("utf-8", errors="replace")
        if s in special_set:
            continue
        nbytes = len(b)
        lengths.append(nbytes)
        all_lens.append((nbytes, tid, s))
        # Only count cleanly-decodable digit-only tokens
        try:
            clean = b.decode("utf-8")
            if clean and clean.strip("0123456789") == "":
                digit_tokens.append(clean)
        except UnicodeDecodeError:
            pass

    lengths_sorted = sorted(lengths)
    n = len(lengths_sorted)
    median = lengths_sorted[n // 2] if n else 0
    mean = sum(lengths_sorted) / n if n else 0.0
    mx = lengths_sorted[-1] if n else 0
    single_byte = sum(1 for x in lengths if x == 1)

    # Length histogram (bytes 1..max)
    hist = Counter(lengths)

    # Top-15 longest tokens (truncate display to 40 chars)
    all_lens.sort(key=lambda t: -t[0])
    longest = []
    for b, tid, s in all_lens[:15]:
        disp = repr(s)[1:-1]  # strip outer quotes from repr
        if len(disp) > 40:
            disp = disp[:37] + "..."
        longest.append({"id": tid, "bytes": b, "repr": disp})

    # Digit token stats: how many, longest, distribution by length
    digit_by_len = Counter(len(s) for s in digit_tokens)

    return {
        "vocab_size": V,
        "num_non_special": n,
        "num_special": len(special_set),
        "vocab_gaps": vocab_gaps,
        "bytes_per_token_mean": mean,
        "bytes_per_token_median": median,
        "bytes_per_token_max": mx,
        "single_byte_fraction": single_byte / n if n else 0.0,
        "length_histogram": dict(sorted(hist.items())),
        "longest_tokens": longest,
        "special_tokens": sorted(special_set),
        "digit_token_count": len(digit_tokens),
        "digit_token_by_len": dict(sorted(digit_by_len.items())),
        "digit_token_max_len": max((len(s) for s in digit_tokens), default=0),
    }


def compute_structural(tok) -> dict:
    """Encode/decode round-trip the structural battery. PASS/FAIL per test."""
    out = {}
    for name, text in STRUCTURAL_TESTS:
        try:
            ids = tok.encode(text)
            decoded = tok.decode(ids)
            ok = (decoded == text)
            out[name] = {
                "ok": ok,
                "input_bytes": len(text.encode("utf-8")),
                "n_tokens": len(ids),
                "decoded_sample": None if ok else decoded[:80],
            }
        except Exception as e:
            out[name] = {"ok": False, "error": f"{type(e).__name__}: {e}"}
    return out


def compute_task_probes(tok) -> dict:
    """Token counts for capability-relevant probe strings."""
    out = {}
    for name, text in TASK_PROBES.items():
        ids = tok.encode(text)
        out[name] = {
            "text": text,
            "bytes": len(text.encode("utf-8")),
            "tokens": len(ids),
        }
    return out


def compute_coverage(tok, max_chars: int) -> dict:
    """
    Tier 2: tokenize a held-out corpus chunk, compute token-frequency
    statistics and cumulative coverage at top-K. Uses ClimbMix val (last
    parquet shard) via nanochat.dataset.
    """
    from nanochat.dataset import parquets_iter_batched

    counts = Counter()
    total_tokens = 0
    total_chars = 0
    t0 = time.time()
    for batch in parquets_iter_batched(split="val"):
        ids = tok.encode(batch)  # batched encode
        for row in ids:
            counts.update(row)
            total_tokens += len(row)
        total_chars += sum(len(d) for d in batch)
        if total_chars >= max_chars:
            break
    elapsed = time.time() - t0

    V = tok.get_vocab_size()
    seen_vocab = len(counts)
    dead = V - seen_vocab

    # Cumulative coverage at top-K
    sorted_counts = sorted(counts.values(), reverse=True)
    cumsum = 0
    coverage_at = {}
    Ks = [100, 500, 1000, 5000, 10000, 32000, V]
    Ks = sorted(set(k for k in Ks if k <= V))
    k_iter = iter(Ks)
    next_k = next(k_iter, None)
    for i, c in enumerate(sorted_counts, start=1):
        cumsum += c
        while next_k is not None and i >= next_k:
            coverage_at[next_k] = cumsum / total_tokens
            next_k = next(k_iter, None)
    while next_k is not None:
        # Vocab smaller than total_tokens; fill remaining Ks with full coverage
        coverage_at[next_k] = 1.0
        next_k = next(k_iter, None)

    return {
        "total_chars_tokenized": total_chars,
        "total_tokens": total_tokens,
        "elapsed_sec": elapsed,
        "vocab_size": V,
        "tokens_seen": seen_vocab,
        "tokens_dead": dead,
        "dead_fraction": dead / V,
        "coverage_at_topk": coverage_at,
    }


def run_one(tok, name: str, spec: str, full: bool, coverage_chars: int) -> TokenizerReport:
    return TokenizerReport(
        name=name,
        spec=spec,
        vocab_size=tok.get_vocab_size(),
        num_special=len(tok.get_special_tokens()),
        compression=compute_compression(tok),
        vocab=compute_vocab_stats(tok),
        structural=compute_structural(tok),
        task_probes=compute_task_probes(tok),
        coverage=compute_coverage(tok, coverage_chars) if full else None,
    )


# ---------------------------------------------------------------------------
# Reporting — CLI tables

def _fmt_pct(x, digits=1):
    return f"{100*x:.{digits}f}%"


def _color_better(val, ref_val, lower_is_better=True):
    """Color a value GREEN if better than ref, RED if worse, GRAY if equal."""
    if val == ref_val:
        return GRAY
    if lower_is_better:
        return GREEN if val < ref_val else RED
    return GREEN if val > ref_val else RED


def print_header(reports):
    print(f"\n{BOLD}=== Tokenizer eval ==={RESET}")
    print(f"Reference (= column 1): {reports[0].name}  [{reports[0].spec}]")
    print(f"Vocab sizes:")
    for r in reports:
        special_note = f" (+{r.num_special} special)" if r.num_special else ""
        print(f"  {r.name:<20} {r.vocab_size:>7}{special_note}")


def print_compression(reports):
    print(f"\n{BOLD}-- Compression (bytes / token; higher = better) --{RESET}")
    ref = reports[0]
    # Header
    cols = "  ".join(f"{r.name:<14}" for r in reports)
    print(f"{'corpus':<18}  {'bytes':>6}  {cols}")
    print("-" * (26 + 16 * len(reports)))
    for corpus in PROBE_CORPORA:
        ref_bpt = ref.compression[corpus]["bytes_per_token"]
        ref_bytes = ref.compression[corpus]["bytes"]
        cells = []
        for r in reports:
            bpt = r.compression[corpus]["bytes_per_token"]
            tok_count = r.compression[corpus]["tokens"]
            color = "" if r is ref else _color_better(bpt, ref_bpt, lower_is_better=False)
            cell = f"{color}{bpt:5.2f} ({tok_count:>4}t){RESET}"
            # Pad to 14 chars accounting for ANSI escapes (just pad raw width)
            cells.append(cell)
        print(f"{corpus:<18}  {ref_bytes:>6}  " + "  ".join(f"{c:<14}" for c in cells))


def print_vocab(reports):
    print(f"\n{BOLD}-- Vocab inspection --{RESET}")
    rows = [
        ("vocab_size",            lambda r: f"{r.vocab['vocab_size']}"),
        ("non-special tokens",    lambda r: f"{r.vocab['num_non_special']}"),
        ("vocab gaps",            lambda r: f"{r.vocab['vocab_gaps']}"),
        ("bytes/tok mean",        lambda r: f"{r.vocab['bytes_per_token_mean']:.2f}"),
        ("bytes/tok median",      lambda r: f"{r.vocab['bytes_per_token_median']}"),
        ("bytes/tok max",         lambda r: f"{r.vocab['bytes_per_token_max']}"),
        ("single-byte fraction",  lambda r: _fmt_pct(r.vocab['single_byte_fraction'])),
        ("digit-only tokens",     lambda r: f"{r.vocab['digit_token_count']} (max len {r.vocab['digit_token_max_len']})"),
    ]
    name_col = max(len(label) for label, _ in rows)
    header = f"{'metric':<{name_col}}  " + "  ".join(f"{r.name:<18}" for r in reports)
    print(header)
    print("-" * len(header))
    for label, fn in rows:
        cells = "  ".join(f"{fn(r):<18}" for r in reports)
        print(f"{label:<{name_col}}  {cells}")

    # Digit-token breakdown by length — directly relevant to Karpathy's
    # \p{N}{1,2} tuning question (ideas/tokenizer-variants/README.md).
    print(f"\n  digit-token count by length:")
    print(f"  {'length':<8} " + "  ".join(f"{r.name:<10}" for r in reports))
    all_lens = sorted(set().union(*(set(r.vocab["digit_token_by_len"].keys()) for r in reports)))
    for L in all_lens:
        cells = "  ".join(f"{r.vocab['digit_token_by_len'].get(L, 0):<10}" for r in reports)
        print(f"  {L:<8} {cells}")


def print_structural(reports):
    print(f"\n{BOLD}-- Structural round-trip battery --{RESET}")
    header = f"{'test':<22}  " + "  ".join(f"{r.name:<14}" for r in reports)
    print(header)
    print("-" * len(header))
    for name, _text in STRUCTURAL_TESTS:
        cells = []
        for r in reports:
            d = r.structural[name]
            if d.get("ok"):
                cell = f"{GREEN}PASS{RESET} {d['n_tokens']:>3}t"
            elif "error" in d:
                cell = f"{RED}ERROR{RESET}"
            else:
                cell = f"{RED}FAIL{RESET} {d['n_tokens']:>3}t"
            cells.append(cell)
        print(f"{name:<22}  " + "  ".join(f"{c:<14}" for c in cells))

    # Summarize any FAILs in detail
    for r in reports:
        fails = [(n, d) for n, d in r.structural.items()
                 if not d.get("ok") and "error" not in d]
        if fails:
            print(f"\n  {r.name} failures:")
            for n, d in fails[:5]:
                print(f"    {n}: decoded → {d.get('decoded_sample')!r}")


def print_task_probes(reports):
    print(f"\n{BOLD}-- Task probes (token counts; lower = better) --{RESET}")
    ref = reports[0]
    header = f"{'probe':<18}  {'bytes':>5}  " + "  ".join(f"{r.name:<14}" for r in reports)
    print(header)
    print("-" * len(header))
    for name, text in TASK_PROBES.items():
        b = len(text.encode("utf-8"))
        ref_t = ref.task_probes[name]["tokens"]
        cells = []
        for r in reports:
            t = r.task_probes[name]["tokens"]
            color = "" if r is ref else _color_better(t, ref_t, lower_is_better=True)
            cells.append(f"{color}{t:>3}{RESET}")
        print(f"{name:<18}  {b:>5}  " + "  ".join(f"{c:<14}" for c in cells))


def print_longest(reports):
    print(f"\n{BOLD}-- Longest tokens (top-10 per tokenizer) --{RESET}")
    for r in reports:
        print(f"\n  {r.name}:")
        for entry in r.vocab["longest_tokens"][:10]:
            print(f"    [{entry['id']:>6}] {entry['bytes']:>3}B  {entry['repr']}")


def print_coverage(reports):
    if not any(r.coverage for r in reports):
        return
    print(f"\n{BOLD}-- Coverage curve (Tier 2; ClimbMix val) --{RESET}")
    for r in reports:
        if not r.coverage:
            continue
        c = r.coverage
        print(f"\n  {r.name}:")
        print(f"    tokenized {c['total_chars_tokenized']:,} chars → {c['total_tokens']:,} tokens "
              f"in {c['elapsed_sec']:.1f}s")
        print(f"    dead tokens: {c['tokens_dead']:,} / {c['vocab_size']:,} ({_fmt_pct(c['dead_fraction'])})")
        print(f"    cumulative coverage:")
        for k, frac in c["coverage_at_topk"].items():
            print(f"      top-{k:>6}: {_fmt_pct(frac, digits=2)}")


# ---------------------------------------------------------------------------
# Markdown report — for pasting into run notes / GitHub

def write_markdown(reports, path):
    lines = []
    lines.append(f"# Tokenizer eval — {len(reports)} tokenizers")
    lines.append("")
    lines.append(f"Reference: **{reports[0].name}** ({reports[0].spec})")
    lines.append("")

    # Vocab summary
    lines.append("## Vocab sizes")
    lines.append("")
    lines.append("| tokenizer | spec | vocab | special |")
    lines.append("|---|---|---:|---:|")
    for r in reports:
        lines.append(f"| {r.name} | `{r.spec}` | {r.vocab_size} | {r.num_special} |")
    lines.append("")

    # Compression
    lines.append("## Compression (bytes / token; higher = better)")
    lines.append("")
    header = "| corpus | bytes |" + "".join(f" {r.name} |" for r in reports)
    sep = "|---|---:|" + "---:|" * len(reports)
    lines.append(header)
    lines.append(sep)
    for corpus in PROBE_CORPORA:
        cells = []
        for r in reports:
            d = r.compression[corpus]
            cells.append(f" {d['bytes_per_token']:.2f} ({d['tokens']}t) |")
        lines.append(f"| {corpus} | {reports[0].compression[corpus]['bytes']} |" + "".join(cells))
    lines.append("")

    # Vocab inspection
    lines.append("## Vocab inspection")
    lines.append("")
    rows = [
        ("bytes/tok mean",        lambda r: f"{r.vocab['bytes_per_token_mean']:.2f}"),
        ("bytes/tok median",      lambda r: f"{r.vocab['bytes_per_token_median']}"),
        ("bytes/tok max",         lambda r: f"{r.vocab['bytes_per_token_max']}"),
        ("single-byte fraction",  lambda r: _fmt_pct(r.vocab['single_byte_fraction'])),
        ("digit-only tokens",     lambda r: f"{r.vocab['digit_token_count']} (max len {r.vocab['digit_token_max_len']})"),
    ]
    lines.append("| metric |" + "".join(f" {r.name} |" for r in reports))
    lines.append("|---|" + "---|" * len(reports))
    for label, fn in rows:
        lines.append(f"| {label} |" + "".join(f" {fn(r)} |" for r in reports))
    lines.append("")

    # Structural
    lines.append("## Structural round-trip")
    lines.append("")
    lines.append("| test |" + "".join(f" {r.name} |" for r in reports))
    lines.append("|---|" + "---|" * len(reports))
    for name, _text in STRUCTURAL_TESTS:
        cells = []
        for r in reports:
            d = r.structural[name]
            if d.get("ok"):
                cells.append(f" PASS ({d['n_tokens']}t) |")
            elif "error" in d:
                cells.append(" ERROR |")
            else:
                cells.append(f" **FAIL** ({d['n_tokens']}t) |")
        lines.append(f"| {name} |" + "".join(cells))
    lines.append("")

    # Task probes
    lines.append("## Task probes (token count; lower = better)")
    lines.append("")
    lines.append("| probe | bytes |" + "".join(f" {r.name} |" for r in reports))
    lines.append("|---|---:|" + "---:|" * len(reports))
    for name, text in TASK_PROBES.items():
        b = len(text.encode("utf-8"))
        cells = "".join(f" {r.task_probes[name]['tokens']} |" for r in reports)
        lines.append(f"| {name} | {b} |{cells}")
    lines.append("")

    # Coverage (if present)
    if any(r.coverage for r in reports):
        lines.append("## Coverage curve (Tier 2)")
        lines.append("")
        for r in reports:
            if not r.coverage:
                continue
            c = r.coverage
            lines.append(f"### {r.name}")
            lines.append(f"- tokenized {c['total_chars_tokenized']:,} chars → {c['total_tokens']:,} tokens")
            lines.append(f"- dead tokens: {c['tokens_dead']:,} / {c['vocab_size']:,} ({_fmt_pct(c['dead_fraction'])})")
            lines.append(f"- cumulative coverage:")
            for k, frac in c["coverage_at_topk"].items():
                lines.append(f"  - top-{k}: {_fmt_pct(frac, digits=2)}")
            lines.append("")

    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"\nWrote Markdown report to {path}")


# ---------------------------------------------------------------------------
# Main

def parse_args():
    p = argparse.ArgumentParser(
        description="Offline tokenizer eval harness "
                    "(see ideas/tokenizer-variants/README.md for the workflow).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--tokenizers",
        default="ours,gpt2,gpt4",
        help="Comma-separated list of tokenizers (built-ins: ours, gpt2, gpt4, gpt4o; "
             "or absolute dir paths). Default: ours,gpt2,gpt4. "
             "First listed is the reference for delta colors.",
    )
    p.add_argument(
        "--full",
        action="store_true",
        help="Include Tier 2 (coverage curve over ClimbMix val; takes ~1-2 min).",
    )
    p.add_argument(
        "--coverage-chars",
        type=int,
        default=20_000_000,
        help="Max chars to tokenize for coverage curve (default: 20M ~ 1-2 min).",
    )
    p.add_argument("--json", metavar="OUT", help="Write full results as JSON.")
    p.add_argument("--markdown", metavar="OUT", help="Write a Markdown report.")
    p.add_argument(
        "--no-longest",
        action="store_true",
        help="Skip the 'longest tokens' section in CLI output.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    specs = [s.strip() for s in args.tokenizers.split(",") if s.strip()]
    if not specs:
        print("error: no tokenizers given", file=sys.stderr)
        sys.exit(2)

    reports = []
    for spec in specs:
        # Display name: basename for paths, spec verbatim otherwise
        name = os.path.basename(spec.rstrip("/")) if os.path.isdir(spec) else spec
        print(f"Loading tokenizer: {name}  ({spec})")
        tok = load_tokenizer(spec)
        reports.append(run_one(tok, name, spec, args.full, args.coverage_chars))

    print_header(reports)
    print_compression(reports)
    print_vocab(reports)
    print_structural(reports)
    print_task_probes(reports)
    if not args.no_longest:
        print_longest(reports)
    if args.full:
        print_coverage(reports)

    if args.json:
        payload = [asdict(r) for r in reports]
        with open(args.json, "w") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print(f"\nWrote JSON results to {args.json}")

    if args.markdown:
        write_markdown(reports, args.markdown)


if __name__ == "__main__":
    main()
