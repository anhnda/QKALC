#!/usr/bin/env bash
# K-pass after the diag-H fix (both moments on the same query subset).
#   MODEL_PATH=meta-llama/Llama-3.2-1B BITS=3 N_CALIB=128 bash scripts/rtn/run_kpass_sweep2.sh
#
# Check first: k_sp of rtn_k_h32 / rtn_k_hall must now be POSITIVE (was -0.30 with the old H),
# and k_acc should go up. Then compare quality/time with rtn_qkq, rtn_qkqkq_kb64, rtn_qkq_kb64_r1.
source "$(dirname "${BASH_SOURCE[0]}")/../common.sh"

SLUG=$(basename "$MODEL_PATH")
OUT=${OUT:-results/rtn/${SLUG}_w${BITS}g${GS}}
mkdir -p "$OUT"
EXTRA_ARGS="${EXTRA_ARGS:-} --diag-seqs ${DIAG_SEQS:-8}"
QK=(--base rtn --qk-correct full)

# predictor check: K only, subset H (32 q/seq) vs exact H (all queries)
want_arm rtn_k_h32            && run_arm "$OUT" rtn_k_h32            "${QK[@]}" --qk-schedule K
want_arm rtn_k_hall           && run_arm "$OUT" rtn_k_hall           "${QK[@]}" --qk-schedule K --qk-k-hsub-queries 0
# schedules with the fixed H
want_arm rtn_qkq_h32          && run_arm "$OUT" rtn_qkq_h32          "${QK[@]}" --qk-schedule QKQ
want_arm rtn_qkqkq_h32        && run_arm "$OUT" rtn_qkqkq_h32        "${QK[@]}" --qk-schedule QKQKQ
# full pipeline with CLC on the other linears
want_arm rtn_clcnq_qkqkq_h32  && run_arm "$OUT" rtn_clcnq_qkqkq_h32  "${QK[@]}" --clc-targets all_but_q --qk-schedule QKQKQ

"$PYTHON_BIN" summarize.py --results-dir "$OUT" \
  --runs rtn,rtn_qk,rtn_k_kb64,rtn_qkq,rtn_qkq_kb64_r1,rtn_qkqkq_kb64,rtn_clcnq_qkq,rtn_k_h32,rtn_k_hall,rtn_qkq_h32,rtn_qkqkq_h32,rtn_clcnq_qkqkq_h32
