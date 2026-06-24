"""force_merges + the 126 fm-context mined blocks  (the 'combined' tokenizer).

Reuses force_merges' carve-out SPLIT_PATTERN + 122 forced phrases + RECOMMENDED_VOCAB_SIZE
(32890) and the same rustbpe_force_merges crate; the ONLY difference vs force_merges is the
enumerated BLOCKED_PAIRS below. Those 126 are the HAND-CURATED output of the held-out-Pareto
firing-33 block-list audit (auto_tune/audit/heldout_fixpoint_fm, 2026-06-23): 126 judged commits
over 255 rounds. NOT a driver fixpoint — the loop was manually stopped at round 256 (status still
'running'; 1,344 candidates pass the per-candidate gate, but the board is topped by hard semantic
vetoes like ob+by/s+hip). Treat as a validated A/B candidate, not a proven optimum; it can be
extended by resuming the loop or by spending the banked compression surplus (see the variant
README + the audit README). Fresh-shard validation (20 never-selected shards) of the 126 ON TOP of
force_merges: Δcomp -891.6/shard (77%% of the -1152.6 gate tally; 20/20 shards improved),
Δdead -41.3 (83%%; 20/20), coverage Δ+52 (vocab-exact).

This is the in-context re-derivation of the old force_merges+r194 "combined" (which was
strongly sub-additive over force_merges — STATUS.md). Bake: wrappers.tok_train_force_merges_combined.
"""
from rustbpe_variants.force_merges.pairs import (  # noqa: F401  (re-exported for the wrapper)
    SPLIT_PATTERN,
    FORCED_PAIRS,
    FORCED_PAIRS_EXPR,
    RECOMMENDED_VOCAB_SIZE,
)

BLOCKED_PAIRS = [
    (' inv', 'ol'), (' inc', 're'), ('f', 'ter'), (' Man', 'ufact'),
    (' pro', 'du'), ('ah', 'u'), (' contin', 'u'), (' imm', 'edi'),
    ('....', '....'), (' diff', 'ere'), (' tra', 'um'), ('I', 'H'),
    (' ag', 'ricult'), (' antib', 'iot'), (' Ind', 'ivid'), (' Pro', 'ble'),
    (' Te', 'ac'), (' ass', 'um'), (' P', 'upp'), (' D', 'ise'),
    (' F', 'if'), (' V', 'ac'), (' acc', 'ur'), (' m', 'ov'),
    ('·', '·'), (' inc', 'lud'), (' exper', 'im'), ('ot', 'ted'),
    ('ot', 'ropic'), ('av', 'ig'), ('u', 'ild'), (' w', 'r'),
    (' cit', 'iz'), (' est', 'ab'), ('th', 'rop'), (' inst', 'it'),
    ('S', 'F'), ('om', 'ile'), (' ex', 'ce'), ('op', 'art'),
    (' rep', 'res'), (' t', 'iss'), (' reg', 'ul'), (' vol', 'un'),
    (' encou', 'rag'), (' cont', 'roll'), ('ar', 'as'), (' prov', 'id'),
    (',', '00'), ('on', 'el'), (' b', 'reat'), ('el', 'ry'),
    ('ther', 'net'), ('ress', 'ions'), ('ign', 'ing'), ('n', 'ate'),
    ('ang', 'ar'), (' ', '\uf0b7'), (' Ad', 'vis'), (' An', 'alog'),
    (' An', 'im'), (' >', '>'), (' A', 'AA'), (' Air', 'bus'),
    ('n', 'iv'), ('por', 'ary'), ('x', 'im'), ('ec', 'ause'),
    (' desc', 'rib'), (' ele', 'ph'), ('auc', 'oma'), (' colle', 'ag'),
    (' cl', 'us'), (' d', 'imens'), (' mat', 'hemat'), (' sh', 'ap'),
    (' c', 'ateg'), (' diss', 'oci'), ('du', 'ate'), ('nov', 'ation'),
    (' ch', 'ann'), (' influ', 'en'), ('y', 'pe'), ('ak', 'u'),
    ('e', 'ah'), (' es', 'oph'), ('N', 'OT'), (' se', 'arc'),
    (' dis', 'pl'), (' pl', 'aus'), ('ric', 'ulum'), (' out', 'bre'),
    (' port', 'ra'), (' sub', 'sequ'), (' Int', 'ellig'), (' bi', 'op'),
    (' c', 'igaret'), (' pers', 'pect'), ('an', 'ish'), (' const', 'ra'),
    ('ic', 'ose'), ('n', 'om'), (' rep', 'ut'), ('ns', 'ure'),
    ('iber', 'ty'), ('ol', 'ver'), (' exper', 'ien'), (' rein', 'for'),
    ('ab', 'it'), ('ab', 'ul'), ('cc', 'ess'), ('meric', 'an'),
    (' rest', 'ric'), ('-', 'hyd'), ('under', 'st'), (' p', 'estic'),
    ('.', 'in'), ('r', 'ateg'), ('ros', 'so'), (' micro', 'cont'),
    ('ay', 'enne'), ('yth', 'ons'), (' earth', 'qu'), ('-b', 'ut'),
    ('S', 'ar'), ('ur', 'is'),
]
