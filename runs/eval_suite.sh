#!/bin/bash
# Permanent post-run eval suite — the surface-specific probes that bpb/CORE miss. Run on the
# persisted checkpoints after a paired A/B (both arms done). ALWAYS includes the boolq
# passage-swap counterfactual: the boolq CORE score is ambiguous without it (a yes/no prior
# vs actually reading the passage are indistinguishable from the answer distribution alone).
#
#   SURF_CKPT=.../d24_surf_..._s8  BASE_CKPT=.../d24_baseline_..._s8 \
#   SURF_TOK=.../surface_only_*/tokenizer  BASE_TOK=.../baseline_*/tokenizer \
#   NANOCHAT_DATASET=fineweb  OUT=/workspace/.../eval_suite  bash runs/eval_suite.sh
set -uo pipefail
. "$HOME/.local/bin/env" 2>/dev/null || true
export PATH="$HOME/.local/bin:$PATH"
: "${SURF_CKPT:?set SURF_CKPT}" "${BASE_CKPT:?set BASE_CKPT}" "${SURF_TOK:?set SURF_TOK}" "${BASE_TOK:?set BASE_TOK}"
OUT="${OUT:-./eval_suite_out}"; mkdir -p "$OUT"
DS="${NANOCHAT_DATASET:-climbmix}"
PAIR="--surf-ckpt $SURF_CKPT --base-ckpt $BASE_CKPT --surf-tok $SURF_TOK --base-tok $BASE_TOK"

run() {  # run <out-name> <wrapper-module> <args...>
    echo "=== $1 ==="
    NANOCHAT_DATASET="$DS" uv run python -m "wrappers.$2" ${3:-} > "$OUT/$1.txt" 2>&1 \
        && echo "$1 OK" || echo "$1 FAILED (see $OUT/$1.txt)"
}

# cross-tokenizer surface-vs-baseline probes (per RENDERED byte)
run robustness       surface_robustness "$PAIR"
run longtail         longtail_nll       "$PAIR"
run cloze_climbmix   rare_entity_cloze  "$PAIR --source climbmix"
run cloze_fineweb    rare_entity_cloze  "$PAIR --source fineweb"
run fineweb_holdout  fineweb_holdout    "$PAIR"
# boolq passage-swap counterfactual — per model (does it read the passage or apply a prior?)
run boolq_swap_surf  boolq_swap         "--ckpt $SURF_CKPT --tok $SURF_TOK"
run boolq_swap_base  boolq_swap         "--ckpt $BASE_CKPT --tok $BASE_TOK"

touch "$OUT/.evals_done"
echo "ALL EVALS DONE → $OUT"
