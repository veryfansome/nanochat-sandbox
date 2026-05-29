"""
MANUAL_PAIRS — hand-curated merges, applied FIRST, for the manual_merges tokenizer.

Hand-written counterpart to seed_tokens (same rank-insertion engine). For each
`(L, R)`, the concat `S = L + R` is inserted into `mergeable_ranks` at its
natural-firing rank `max(id_L, id_R) + 1` (clamped to 256). For byte-bigram
pairs (single-char operands), that clamp lands them at **rank 256 — the first
merges**, so they win the merge race wherever their operands are adjacent.
That's the intent here: the merges are done *first* (highest priority), unlike
`force_merges`, whose `apply_forced_merges_at_end` appends at the lowest
priority (right only when you must force a pair that otherwise can't be
mergeable, paired with a regex carve-out).

Properties:
  - **Pairs are hand-written** here, not mined.
  - **Near-standard pre-tokenization regex** (no carve-out), via pristine
    `rustbpe`. One deliberate tweak (in `wrappers/tok_train_manual_merges.py`):
    the opening-delimiter chars `" ( , “` are removed from the leading-char
    word-gluing clause, so junk `delimiter+word` tokens (`"The`, `(x`, `,y`,
    `“The`) don't form — they were verified to be near-pure artifacts of quoted /
    parenthetical / list text (the `-`/`.`/`/` glued tokens, which *do* form
    units like `-based`/`.com`/`/api`, are kept). Everything else is the stock
    regex, so every pair must still be one the pre-tokenizer keeps *within a
    single chunk* — a within-word / sub-word composition. For a pair with
    multi-char operands (e.g. `("educ","ation")`), "first" means right after its
    operands (`max(id_L,id_R)+1`), not literally rank 256, since the operands
    must form before the merge can fire. Cross-boundary phrase pairs (` of`,
    ` the`) cannot fire here — those belong in `force_merges`.
  - If `S` already exists naturally, it is PROMOTED to its insertion rank (moved
    to the front), not skipped — the point is priority.

Fill `MANUAL_PAIRS` below.
"""

MANUAL_PAIRS: list[tuple[str, str]] = [
    # (L, R) pairs to force-merge — within-chunk / sub-word only.
    # e.g. ("educ", "ation"), ("un", "happy"), ...
]

# BLOCKED_PAIRS — pairs that must NOT be merged during BPE training. Same
# mechanism as blocked_morphemes / force_merges (rustbpe_force_merges'
# `blocked_pairs=`): the trainer drops any merge candidate (L, R) listed here.
# Use to forbid specific (non-morphemic) merges while the MANUAL_PAIRS seeds
# above force the desired ones. Empty = no blocking (pristine-equivalent base).
BLOCKED_PAIRS: list[tuple[str, str]] = [
    # (L, R) pairs to forbid, e.g. ("at", "ion").
    #
    # ── CURATION METHODOLOGY (how to decide what belongs here) ──────────────
    #
    # GOAL: forbid merges that form *mangled* tokens — fragments that split a
    # word at a non-morphemic boundary — so the model sees clean [stem][suffix]
    # atoms instead. Blocking (L, R) stops the token S = L+R from forming; BPE
    # then routes those words through the cleaner split.
    #
    # The hard part is telling a mangled fragment from a real word. For each
    # candidate, look at how the NO-SPACE token S = L+R actually fires on the
    # corpus — run `uv run python -m tools.token_contexts <S>` (words it fires in
    # + bare-word %; do NOT eyeball it — we got red/bed/led wrong by guessing).
    # 3 cases:
    #
    #   1. S is a MANGLED stem+suffix fragment  → BLOCK. It splits real words at
    #      a non-morphemic point ('ulated'=ulat+ed, 'ception'=cept+ion). Forcing
    #      [stem][suffix] is the whole point. This is the bulk of the list.
    #
    #   2. S is a complete CLEAN WORD that fires AS itself  → DON'T block
    #      (the "river" rule). Blocking shatters a good atom. Witnesses we
    #      removed: species (spec+ies), question (quest+ion), thing (th+ing),
    #      string (str+ing). Also leave genuine free morphemes: need, feed, fed.
    #
    #   3. S spells a real word but the NO-SPACE token is a PARASITE  → BLOCK
    #      anyway (the "king" rule). It steals a stem-final consonant from OTHER
    #      words rather than being the word itself: 'king' fires as working→
    #      wor+king / parking→par+king (never the word "king", which is safe as
    #      the space-prefixed ' king'); 'bed' fires as adsorbed→adsor+bed.
    #
    # WHY the word survives a block: blocks hit the NO-SPACE token; the real word
    # almost always lives as a separate space-prefixed ' word' token the block
    # never touches, so harm is bounded to mid-word occurrences.
    #
    # QUICK TELL — bare-word %: for a single-syllable Xsuffix, what fraction of
    # the no-space token's firings ARE the bare word? bare≈0 → parasite, block
    # ('bed'); bare high → real word, leave ('need'); middling → MIXED, read the
    # word list ('red': bare ~11%, but its firings are spread thin across stem-
    # stealers, -rred doublers, names and the color word with no dominant pattern
    # — so low-stakes either way; a judgment call, not a clear block or leave).
    #
    # FRAGMENTS / DEAD TOKENS — lean BLOCK, cheaply: if S is a sub-fragment of a
    # monomorphemic word (oment≠moment, liament≠parliament, ament≠lament) there
    # is no clean morpheme to protect, so blocking just reshuffles junk and
    # reclaims a vocab slot. Same for trained-but-dead tokens (fire ~0×): a block
    # reclaims the slot, not blocking lets them re-form. Weak positive, no harm.
    #
    # MULTI-PATH ("hard to kill"): a token forms via several binary splits
    # (oment = o+ment OR om+ent OR ome+nt …). Blocking one leaves the rest; it
    # re-forms via the next-cheapest. To definitively kill S, block ALL n−1 of
    # its binary splits (why some entries list ('o','ment'),('om','ent'),
    # ('ome','nt'),('omen','t') together). Chase LIVE mangles this way; for dead
    # ones it isn't worth it. Chase by firing count, not by mangledness.
    # ────────────────────────────────────────────────────────────────────────

    ("ag", "ation"),
    ("aga", "tion"),
    ("am", "ation"),
    ("ama", "tion"),
    ("amin", "ation"),
    ("an", "ation"),
    ("ana", "tion"),
    ("apor", "ation"),
    ("ar", "ation"),
    ("ara", "tion"),
    ("atur", "ation"),
    ("celer", "ation"),
    ("centr", "ation"),
    ("edi", "ation"),
    ("edia", "tion"),
    ("ener", "ation"),
    ("ens", "ation"),
    ("ent", "ation"),
    ("er", "ation"),
    ("ert", "ation"),
    ("erv", "ation"),
    ("estr", "ation"),
    ("flamm", "ation"),
    ("form", "ation"),
    ("i", "ation"),
    ("ibr", "ation"),
    ("id", "ation"),
    ("ida", "tion"),
    ("ig", "ation"),
    ("igr", "ation"),
    ("il", "ation"),
    ("ilit", "ation"),
    ("ill", "ation"),
    ("illa", "tion"),
    ("iltr", "ation"),
    ("im", "ation"),
    ("ima", "tion"),
    ("imin", "ation"),
    ("imul", "ation"),
    ("in", "ation"),
    ("in", "ations"),
    ("ip", "ation"),
    ("ipit", "ation"),
    ("ir", "ation"),
    ("istr", "ation"),
    ("it", "ation"),
    ("iv", "ation"), # *ivate -ion, not -ation
    ("iva", "tion"),
    ("l", "ation"),
    ("la", "tion"),
    ("lar", "ation"),
    ("lim", "ation"),
    ("ment", "ation"),
    ("min", "ation"),
    ("nov", "ation"),
    ("oc", "ation"),
    ("ol", "ation"),
    ("ola", "tion"),
    ("or", "ation"),
    ("ot", "ation"),
    ("ota", "tion"),
    ("par", "ation"),
    ("pir", "ation"),
    ("pl", "ications"),
    ("plement", "ation"),
    ("port", "ation"),
    ("pret", "ation"),
    ("r", "ation"),  # *rate -ion, not -ation
    ("re", "ation"),
    ("rea", "tion"),
    ("reat", "ion"),
    ("reci", "ation"),
    ("redit", "ation"),
    ("reg", "ation"),
    ("rig", "ation"),
    ("u", "ation"),
    ("ua", "tion"),
    ("ul", "ation"),
    ("ul", "ations"),
    ("umin", "ation"),
    ("unci", "ation"),
    ("ur", "ation"),
    ("ut", "ation"),
    ("v", "ation"),
    ("va", "tion"),
]

# Slot reservation is disabled (see tok_train_manual_merges.py): natural BPE
# trains to this full vocab size, and seeds are added on top. So the FINAL
# mergeable vocab = RECOMMENDED_VOCAB_SIZE + (count of genuinely-new seeds);
# promoted-existing seeds don't change the count. `tok_train_manual_merges.py`
# reads this.
RECOMMENDED_VOCAB_SIZE = 32768
