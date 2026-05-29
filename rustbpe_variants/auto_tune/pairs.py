"""
auto_tune — block list for the auto_tune tokenizer overlay.

Pairs (L, R) that must NOT merge during BPE training (rustbpe_force_merges'
`blocked_pairs=`): the trainer drops any such merge candidate, so words route
through the cleaner [stem][suffix] split. Blocking only — no forced merges, no
vocab bump.
"""

# Round 3 single-pair queue: the find_mangled_morphemes-mined candidates
# (../blocked_morphemes/mined/), deduped against everything already audited and
# ordered by firing count. Auto-generated; regenerate with
# `uv run python -m tools.gen_mined_candidates`. Audited (highest-firing first,
# staged) via --candidates-attr CANDIDATES_MINED --state-dir audit/mined_pass1.
from rustbpe_variants.auto_tune.mined_candidates import CANDIDATES_MINED  # noqa: F401

# Flag-revisit queue: FLAGs from the single-pair passes (coverage up but compression
# worse → deferred). The held-out generalization check showed compression was eval-
# shard noise while coverage/mangled generalize, so these are re-tested on top of the
# promoted keeps. Auto-generated; regenerate with `tools.gen_flag_candidates`.
from rustbpe_variants.auto_tune.flag_candidates import CANDIDATES_FLAGS  # noqa: F401

# Pass-2 held-out-Pareto build-up queue: every coverage-gain pair (coverage-driven
# KEEPs + all FLAGs), re-evaluated from empty under the held-out Pareto rule (see
# tools.heldout_buildup). Auto-generated; regenerate with tools.gen_pass2_candidates.
from rustbpe_variants.auto_tune.pass2_candidates import CANDIDATES_PASS2  # noqa: F401

# Proven block list — what actually trains. A candidate graduates here only after a
# build-up retrain proves it earns its slot (see audit/README.md). 96 pairs:
#   Round 1 (68 candidates, build-up from empty): 6
#   Round 2 (86 new -ation-family from ../manual_merges superset): 10
#   Round 3 (602 find_mangled_morphemes-mined candidates): 80
# Validated on 10 held-out shards: coverage (+48 words) and mangled (−~87, 10/10
# shards) GENERALIZE; per-shard compression "gains" were eval-shard noise (held-out
# cost ~+0.0024%, the accepted morphology tradeoff). See audit/README.md.
BLOCKED_PAIRS: list[tuple[str, str]] = [
    # Round 1 (-ically / -ness / -ability)
    ("em", "ically"),
    ("it", "ness"),
    ("ograph", "ically"),
    ("r", "ically"),
    ("ustain", "ability"),
    ("ware", "ness"),
    # Round 2 (-ation family)
    ("ent", "ation"),
    ("er", "ation"),
    ("ert", "ation"),
    ("l", "ation"),
    ("lim", "ation"),
    ("ment", "ation"),
    ("nov", "ation"),
    ("or", "ation"),
    ("pl", "ications"),
    ("rig", "ation"),
    # Round 3 (mined, fires-ordered)
    ("ay", "ing"),
    ("u", "ber"),
    ("un", "ning"),
    ("i", "king"),
    ("il", "ting"),
    ("ay", "ed"),
    ("hys", "ical"),
    ("ot", "ting"),
    ("ck", "ers"),
    ("ite", "ly"),
    ("ix", "ing"),
    ("ool", "ing"),
    ("or", "ious"),
    ("du", "ced"),
    ("aps", "es"),
    ("ffic", "ial"),
    ("oad", "ing"),
    ("oot", "ing"),
    ("ffect", "ive"),
    ("ar", "king"),
    ("el", "ting"),
    ("orn", "ed"),
    ("ough", "ly"),
    ("ar", "ning"),
    ("atter", "ing"),
    ("ond", "ing"),
    ("ig", "ers"),
    ("or", "ning"),
    ("ri", "ages"),
    ("ature", "d"),
    ("erv", "ous"),
    ("ict", "ure"),
    ("ient", "ed"),
    ("oler", "ance"),
    ("overn", "ment"),
    ("ren", "ches"),
    ("ib", "ration"),
    ("orm", "ally"),
    ("row", "ning"),
    ("tain", "ment"),
    ("urn", "ed"),
    ("aw", "ning"),
    ("icular", "ly"),
    ("ire", "ment"),
    ("ull", "ing"),
    ("um", "per"),
    ("umin", "ous"),
    ("cess", "ion"),
    ("i", "ators"),
    ("if", "ter"),
    ("il", "ant"),
    ("il", "ian"),
    ("osp", "or"),
    ("uck", "ily"),
    ("ut", "ational"),
    ("amm", "ing"),
    ("ific", "ant"),
    ("ire", "ments"),
    ("ress", "or"),
    ("ruct", "or"),
    ("cept", "or"),
    ("ig", "ure"),
    ("ns", "ure"),
    ("ould", "er"),
    ("ud", "ding"),
    ("ateg", "ory"),
    ("ct", "ure"),
    ("il", "age"),
    ("im", "eters"),
    ("ipt", "ical"),
    ("alle", "led"),
    ("ist", "ication"),
    ("l", "ished"),
    ("lish", "ing"),
    ("orn", "ings"),
    ("ort", "ment"),
    ("ress", "ure"),
    ("uc", "ent"),
    ("ul", "ses"),
    ("ustain", "able"),
]

# Audit candidate queue — the prior hand-curated list, now under retest ("nothing
# is golden"). NOT active (not passed to the trainer); tools/audit_blocked_pairs.py
# walks these one at a time, training kept+candidate and keeping only net-positive
# ones. Survivors get promoted into BLOCKED_PAIRS above.
# Pass-1 verdicts (KEEP/FLAG/DROP) for the full original list are recorded in
# audit/ledger.md; the 6 KEEPs were moved up. What remains below is the 62
# DROP+FLAG queue, pending the multi-pair follow-up (DROPs) and FLAG revisit.
CANDIDATES: list[tuple[str, str]] = [
    ("ail", "ability"),
    ("ard", "less"),
    ("arsighted", "ness"),
    ("ast", "ically"),
    ("astic", "ally"),
    ("at", "ibility"),
    ("at", "ically"),
    ("atic", "ally"),
    ("b", "ability"),
    ("b", "ility"),
    ("e", "less"),
    ("e", "ness"),
    ("el", "ess"),
    ("emic", "ally"),
    ("en", "ess"),
    ("ene", "ss"),
    ("ent", "iful"),
    ("enti", "ful"),
    ("er", "ed"),
    ("et", "ically"),
    ("ethe", "less"),
    ("etic", "ally"),
    ("hemat", "ically"),
    ("hematic", "ally"),
    ("het", "ically"),
    ("hetic", "ally"),
    ("i", "ability"),
    ("ick", "ness"),
    ("ighted", "ness"),
    ("ique", "ness"),
    ("iquen", "ess"),
    ("ist", "ically"),
    ("istic", "ally"),
    ("istical", "ly"),
    ("izz", "iness"),
    ("le", "ness"),
    ("ob", "ility"),
    ("obil", "ity"),
    ("ographic", "ally"),
    ("ographical", "ly"),
    ("olog", "ically"),
    ("ological", "ly"),
    ("one", "liness"),
    ("onel", "iness"),
    ("oubted", "ly"),
    ("p", "less"),
    ("pl", "ess"),
    ("ple", "ss"),
    ("ples", "s"),
    ("reat", "ion"),
    ("ric", "ally"),
    ("rical", "ly"),
    ("t", "ically"),
    ("the", "less"),
    ("tic", "ally"),
    ("tical", "ly"),
    ("ub", "ility"),
    ("uccess", "ful"),
    ("ur", "ability"),
    ("urab", "ility"),
    ("us", "iness"),
    ("yp", "ically"),
]

# Round 2 single-pair queue — the NEW pairs from ../manual_merges/pairs.py
# (a 154-pair superset of round 1's 68); these 86 are the superset minus the 68
# already audited, i.e. the whole -ation/-tion/-ations/-ications family that
# round 1 didn't cover. Tested on top of the 6 round-1 keeps via:
#   --mode single --candidates-attr CANDIDATES_MANUAL --state-dir audit/manual_pass1
CANDIDATES_MANUAL: list[tuple[str, str]] = [
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
    ("iv", "ation"),
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
    ("opul", "ation"),
    ("or", "ation"),
    ("ot", "ation"),
    ("ota", "tion"),
    ("par", "ation"),
    ("pir", "ation"),
    ("pl", "ications"),
    ("plement", "ation"),
    ("port", "ation"),
    ("pret", "ation"),
    ("r", "ation"),
    ("ratul", "ations"),
    ("re", "ation"),
    ("rea", "tion"),
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

# Multi-pair follow-up queue (pass 2): co-requisite route-sets. A surface form
# reachable via more than one merge route needs EVERY route blocked or the token
# still forms — so single-pair pass 1 sees each route as a no-op (~40 of 53 DROPs
# had all-zero deltas). Here each group is the full route-set for one surface form
# (pairs whose L+R concatenate to the same string), tested/committed JOINTLY by
# tools/audit_blocked_pairs.py --mode groups (unit = the set, same lexicographic
# rule). Built from pass-1's multi-route DROP/FLAG families; groups whose primary
# route is already in BLOCKED_PAIRS list only the remaining (secondary) routes.
CANDIDATE_GROUPS: list[list[tuple[str, str]]] = [
    [("ast", "ically"), ("astic", "ally")],                       # astically
    [("at", "ically"), ("atic", "ally")],                         # atically
    [("e", "less"), ("el", "ess")],                               # eless
    [("e", "ness"), ("en", "ess"), ("ene", "ss")],                # eness
    [("ent", "iful"), ("enti", "ful")],                           # entiful
    [("et", "ically"), ("etic", "ally")],                         # etically
    [("hemat", "ically"), ("hematic", "ally")],                   # hematically
    [("het", "ically"), ("hetic", "ally")],                       # hetically
    [("ique", "ness"), ("iquen", "ess")],                         # iqueness
    [("ist", "ically"), ("istic", "ally"), ("istical", "ly")],    # istically
    [("ob", "ility"), ("obil", "ity")],                           # obility
    [("olog", "ically"), ("ological", "ly")],                     # ologically
    [("one", "liness"), ("onel", "iness")],                       # oneliness
    [("p", "less"), ("pl", "ess"), ("ple", "ss"), ("ples", "s")], # pless
    [("t", "ically"), ("tic", "ally"), ("tical", "ly")],          # tically
    [("ur", "ability"), ("urab", "ility")],                       # urability
    [("ographic", "ally"), ("ographical", "ly")],                 # ographically (ograph+ically already KEPT)
    [("ric", "ally"), ("rical", "ly")],                           # rically (r+ically already KEPT)
]

# Round 2 multi-pair follow-up: -ation-family co-requisite route-sets, grouped by
# surface form over the manual_merges superset, excluding routes already in
# BLOCKED_PAIRS. `reation` includes `reat+ion` from round 1 (cross-round route
# closure). Run on top of the 16 keeps via:
#   --mode groups --groups-attr CANDIDATE_GROUPS_MANUAL --state-dir audit/manual_groups
# Predicted ≈0 keeps: single-pass showed every `Xa+tion` route inert while
# `X+ation` carried the delta (routes not co-equal) — testing to confirm.
CANDIDATE_GROUPS_MANUAL: list[list[tuple[str, str]]] = [
    [("ag", "ation"), ("aga", "tion")],                   # agation
    [("am", "ation"), ("ama", "tion")],                   # amation
    [("an", "ation"), ("ana", "tion")],                   # anation
    [("ar", "ation"), ("ara", "tion")],                   # aration
    [("edi", "ation"), ("edia", "tion")],                 # ediation
    [("id", "ation"), ("ida", "tion")],                   # idation
    [("ill", "ation"), ("illa", "tion")],                 # illation
    [("im", "ation"), ("ima", "tion")],                   # imation
    [("iv", "ation"), ("iva", "tion")],                   # ivation
    [("ol", "ation"), ("ola", "tion")],                   # olation
    [("ot", "ation"), ("ota", "tion")],                   # otation
    [("re", "ation"), ("rea", "tion"), ("reat", "ion")],  # reation (3 routes; reat+ion from round 1)
    [("u", "ation"), ("ua", "tion")],                     # uation
    [("v", "ation"), ("va", "tion")],                     # vation
]

# Blocking only removes merge candidates; freed slots get reallocated by BPE.
RECOMMENDED_VOCAB_SIZE = 32768
