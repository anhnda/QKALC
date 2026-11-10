#!/usr/bin/env bash
# GPTQ-based experiments (T3: is the QK correction still complementary to a stronger base?). Usage:
#   MODEL_PATH=meta-llama/Llama-3.2-1B BITS=3 bash scripts/gptq/run_gptq_ablation.sh
#   ARMS=gptq,gptq_clcq,gptq_qk BITS=4 bash scripts/gptq/run_gptq_ablation.sh
#   GPTQ_ARGS="--no-gptq-act-order" bash scripts/gptq/run_gptq_ablation.sh     # GPTQ variant
#
# GPTQ here: fp32, damp 0.01, blocksize 128, act-order on, dynamic groups, true-sequential
# ([q,k,v] -> QK correction -> [o] -> [gate,up] -> [down]); same asymmetric min-max g128 lattice as RTN/AWQ.
# For any sweep / config choice keep the WikiText-2 test split untouched:
#   EXTRA_ARGS="--diag-split validation" ...   (PPL is still reported on test; do not tune on it)
source "$(dirname "${BASH_SOURCE[0]}")/../common.sh"

SLUG=$(basename "$MODEL_PATH")
GPTQ_ARGS=${GPTQ_ARGS:-}
OUT=${OUT:-results/gptq/${SLUG}_w${BITS}g${GS}}
mkdir -p "$OUT"

B=(--base gptq $GPTQ_ARGS)
QK=(--qk-correct full)

want_arm fp              && run_arm "$OUT" fp              --base none
want_arm gptq            && run_arm "$OUT" gptq            "${B[@]}"
want_arm gptq_clcq       && run_arm "$OUT" gptq_clcq       "${B[@]}" --clc-targets q
want_arm gptq_clc        && run_arm "$OUT" gptq_clc        "${B[@]}" --clc-targets all
want_arm gptq_qk         && run_arm "$OUT" gptq_qk         "${B[@]}" "${QK[@]}" --qk-schedule Q
want_arm gptq_qkq        && run_arm "$OUT" gptq_qkq        "${B[@]}" "${QK[@]}" --qk-schedule QKQ
want_arm gptq_clcnq_qk   && run_arm "$OUT" gptq_clcnq_qk   "${B[@]}" --clc-targets all_but_q "${QK[@]}" --qk-schedule Q
want_arm gptq_clcnq_qkq  && run_arm "$OUT" gptq_clcnq_qkq  "${B[@]}" --clc-targets all_but_q "${QK[@]}" --qk-schedule QKQ
# ablations specific to the GPTQ base
want_arm gptq_qk_noacc   && run_arm "$OUT" gptq_qk_noacc   "${B[@]}" "${QK[@]}" --qk-accept none
want_arm gptq_qk_f02     && run_arm "$OUT" gptq_qk_f02     "${B[@]}" "${QK[@]}" --qk-max-flip-percent 0.02

"$PYTHON_BIN" summarize.py --results-dir "$OUT"
