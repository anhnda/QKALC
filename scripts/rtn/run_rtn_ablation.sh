#!/usr/bin/env bash
# RTN-based experiments (step 1-3 of the plan). Usage:
#   MODEL_PATH=meta-llama/Meta-Llama-3-8B BITS=3 bash scripts/rtn/run_rtn_ablation.sh
#   ARMS=rtn,rtn_clcq,rtn_qk BITS=4 bash scripts/rtn/run_rtn_ablation.sh
source "$(dirname "${BASH_SOURCE[0]}")/../common.sh"

SLUG=$(basename "$MODEL_PATH")
OUT=${OUT:-results/rtn/${SLUG}_w${BITS}g${GS}}
mkdir -p "$OUT"

# --- reference
want_arm fp              && run_arm "$OUT" fp              --base none

# --- step 1: diagnostics of the starting points (diag_before = go/no-go numbers)
want_arm rtn             && run_arm "$OUT" rtn             --base rtn
want_arm rtn_clcq        && run_arm "$OUT" rtn_clcq        --base rtn --clc-targets q
want_arm rtn_clc         && run_arm "$OUT" rtn_clc         --base rtn --clc-targets all

# --- step 2: main method (Q-only, full query, exact acceptance, 3 entropy strata)
want_arm rtn_qk          && run_arm "$OUT" rtn_qk          --base rtn --qk-correct full
want_arm rtn_clc_qk      && run_arm "$OUT" rtn_clc_qk      --base rtn --clc-targets all --qk-correct full
want_arm rtn_clcnq_qk    && run_arm "$OUT" rtn_clcnq_qk    --base rtn --clc-targets all_but_q --qk-correct full

# --- K-pass (Q fixed) and alternating schedules
want_arm rtn_k           && run_arm "$OUT" rtn_k           --base rtn --qk-correct full --qk-schedule K
want_arm rtn_qkq         && run_arm "$OUT" rtn_qkq         --base rtn --qk-correct full --qk-schedule QKQ
want_arm rtn_kq          && run_arm "$OUT" rtn_kq          --base rtn --qk-correct full --qk-schedule KQ
want_arm rtn_clcnq_qkq   && run_arm "$OUT" rtn_clcnq_qkq   --base rtn --clc-targets all_but_q --qk-correct full --qk-schedule QKQ

# --- step 3: ablations of the QK correction (on plain RTN)
want_arm rtn_qk_mq       && run_arm "$OUT" rtn_qk_mq       --base rtn --qk-correct meanquery --qk-accept none
want_arm rtn_qk_mq_acc   && run_arm "$OUT" rtn_qk_mq_acc   --base rtn --qk-correct meanquery --qk-accept exact
want_arm rtn_qk_noacc    && run_arm "$OUT" rtn_qk_noacc    --base rtn --qk-correct full --qk-accept none
want_arm rtn_qk_k1       && run_arm "$OUT" rtn_qk_k1       --base rtn --qk-correct full --qk-strata 1 --qk-audit-every 5
want_arm rtn_qk_k3audit  && run_arm "$OUT" rtn_qk_k3audit  --base rtn --qk-correct full --qk-strata 3 --qk-audit-every 5
want_arm rtn_qk_acc64    && run_arm "$OUT" rtn_qk_acc64    --base rtn --qk-correct full --qk-accept-seqs 2  --qk-accept-queries-per-seq 32
want_arm rtn_qk_acc1024  && run_arm "$OUT" rtn_qk_acc1024  --base rtn --qk-correct full --qk-accept-seqs 32 --qk-accept-queries-per-seq 32
want_arm rtn_qk_f02      && run_arm "$OUT" rtn_qk_f02      --base rtn --qk-correct full --qk-max-flip-percent 0.02
want_arm rtn_qk_f10      && run_arm "$OUT" rtn_qk_f10      --base rtn --qk-correct full --qk-max-flip-percent 0.10

"$PYTHON_BIN" summarize.py --results-dir "$OUT"
