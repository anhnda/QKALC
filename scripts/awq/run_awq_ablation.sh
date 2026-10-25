#!/usr/bin/env bash
# AWQ-based experiments (run after the RTN study). Usage:
#   MODEL_PATH=meta-llama/Meta-Llama-3-8B BITS=3 bash scripts/awq/run_awq_ablation.sh
# AWQ here is the CLC repo's variant: per-Linear input scales searched on a salience^alpha grid,
# fake-quantized weights (scales not folded into the preceding norm).
source "$(dirname "${BASH_SOURCE[0]}")/../common.sh"

SLUG=$(basename "$MODEL_PATH")
OUT=${OUT:-results/awq/${SLUG}_w${BITS}g${GS}}
mkdir -p "$OUT"

want_arm awq             && run_arm "$OUT" awq             --base awq
want_arm awq_clcq        && run_arm "$OUT" awq_clcq        --base awq --clc-targets q
want_arm awq_clc         && run_arm "$OUT" awq_clc         --base awq --clc-targets all
want_arm awq_qk          && run_arm "$OUT" awq_qk          --base awq --qk-correct full
want_arm awq_clc_qk      && run_arm "$OUT" awq_clc_qk      --base awq --clc-targets all --qk-correct full
want_arm awq_clcnq_qk    && run_arm "$OUT" awq_clcnq_qk    --base awq --clc-targets all_but_q --qk-correct full
want_arm awq_qkq         && run_arm "$OUT" awq_qkq         --base awq --qk-correct full --qk-schedule QKQ
want_arm awq_clcnq_qkq   && run_arm "$OUT" awq_clcnq_qkq   --base awq --clc-targets all_but_q --qk-correct full --qk-schedule QKQ
want_arm awq_qk_mq      && run_arm "$OUT" awq_qk_mq       --base awq --qk-correct meanquery --qk-accept none
want_arm awq_qk_noacc    && run_arm "$OUT" awq_qk_noacc    --base awq --qk-correct full --qk-accept none
want_arm awq_qk_k1       && run_arm "$OUT" awq_qk_k1       --base awq --qk-correct full --qk-strata 1

"$PYTHON_BIN" summarize.py --results-dir "$OUT"
