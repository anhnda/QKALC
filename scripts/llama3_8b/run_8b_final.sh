#!/usr/bin/env bash
# Everything still needed on Llama-3-8B before moving to another model family.
#
#   STEP=1 bash scripts/llama3_8b/run_8b_final.sh   # main table, seed 42, + PPL at 8K context, + missing arm
#   STEP=2 bash scripts/llama3_8b/run_8b_final.sh   # seeds 1, 2 (W3; set SEED_BITS="3 4" for W4 too)
#   STEP=3 bash scripts/llama3_8b/run_8b_final.sh   # AWQ W3/W4, seed 42, + 8K
#   STEP=4 bash scripts/llama3_8b/run_8b_final.sh   # lm-eval zero-shot on the key arms (W3)
#   STEP=5 bash scripts/llama3_8b/run_8b_final.sh   # aggregate seeds (mean +- std)
#   STEP=all ...                                    # 1 -> 5
# Priority if time is short: 1 > 2 > 3 > 4.  Everything goes to results/llama3_8b_final/.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

export MODEL_PATH=${MODEL_PATH:-meta-llama/Meta-Llama-3-8B}
export N_CALIB=${N_CALIB:-128}
STEP=${STEP:-1}
BASE_OUT=${BASE_OUT:-results/llama3_8b_final}
CTX=${CTX:-"--ppl-extra-lengths 8192"}
SEED_BITS=${SEED_BITS:-"3"}
LM_TASKS=${LM_TASKS:-"arc_easy arc_challenge hellaswag piqa winogrande"}
R8=scripts/llama3_8b/run_8b.sh
USER_EXTRA=${EXTRA_ARGS:-}

MAIN_RTN=fp,rtn,rtn_clcq,rtn_qk,rtn_qkq,rtn_clc,rtn_clcnq_qk,rtn_clcnq_qkq
SEED_RTN=rtn_clcq,rtn_qk,rtn_clc,rtn_clcnq_qk,rtn_clcnq_qkq          # rtn / fp are deterministic
MAIN_AWQ=awq,awq_clcq,awq_qk,awq_clc,awq_clcnq_qk,awq_clcnq_qkq
LM_RTN=rtn,rtn_clcq,rtn_qk,rtn_clc,rtn_clcnq_qk,rtn_clcnq_qkq

step1() {
  SEED=42 ROOT_OUT=$BASE_OUT/s42 ARMS=$MAIN_RTN BITS_LIST="3 4" EXTRA_ARGS="$USER_EXTRA $CTX" PHASE=rtn bash $R8
}
step2() {
  for S in 1 2; do
    SEED=$S ROOT_OUT=$BASE_OUT/s$S ARMS=$SEED_RTN BITS_LIST="$SEED_BITS" EXTRA_ARGS="$USER_EXTRA $CTX" PHASE=rtn bash $R8
  done
}
step3() {
  SEED=42 ROOT_OUT=$BASE_OUT/s42 ARMS=$MAIN_AWQ BITS_LIST="3 4" EXTRA_ARGS="$USER_EXTRA $CTX" PHASE=awq bash $R8
}
step4() {
  local LMB="--lm-eval-batch-size ${LM_BS:-16}"
  SEED=42 ROOT_OUT=$BASE_OUT/lmeval ARMS=$LM_RTN BITS_LIST="3" LM_EVAL_TASKS="$LM_TASKS" EXTRA_ARGS="$USER_EXTRA $LMB" PHASE=rtn bash $R8
  # FP reference for lm-eval
  SEED=42 ROOT_OUT=$BASE_OUT/lmeval ARMS=fp BITS_LIST="3" LM_EVAL_TASKS="$LM_TASKS" EXTRA_ARGS="$USER_EXTRA $LMB" PHASE=rtn bash $R8
}
step5() {
  for B in $SEED_BITS; do
    echo "== RTN W${B}: mean +- std over seeds =="
    python aggregate_seeds.py $BASE_OUT/s42/rtn_w${B}g128 $BASE_OUT/s1/rtn_w${B}g128 $BASE_OUT/s2/rtn_w${B}g128 \
      --csv $BASE_OUT/seeds_rtn_w${B}.csv
  done
}

case "$STEP" in
  1) step1 ;;
  2) step2 ;;
  3) step3 ;;
  4) step4 ;;
  5) step5 ;;
  all) step1; step2; step3; step4; step5 ;;
  *) echo "STEP must be 1..5 or all"; exit 1 ;;
esac
