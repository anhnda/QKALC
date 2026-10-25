#!/usr/bin/env bash
# K-pass batch-size / refresh sweep (RTN). Writes into the same folder as run_rtn_ablation.sh so the
# new arms are compared directly with the existing rtn, rtn_qk, rtn_k, rtn_qkq, rtn_clcnq_qkq runs.
#
#   MODEL_PATH=meta-llama/Llama-3.2-1B BITS=3 N_CALIB=128 bash scripts/rtn/run_kpass_sweep.sh
#   ARMS=rtn_k_kb64,rtn_qkq_kb64 BITS=3 bash scripts/rtn/run_kpass_sweep.sh
#
# What to look at in the final table:
#   k_sp   median per-layer Spearman(pred, exact dJ) of K passes  (was negative with big batches)
#   k_acc  accepted fraction of proposed K batches                 (higher = fewer wasted rounds)
#   qk_time_s, kl_after, ppl_wiki   vs rtn_qkq (old K batches: init 64, max 512, row cap 4)
source "$(dirname "${BASH_SOURCE[0]}")/../common.sh"

SLUG=$(basename "$MODEL_PATH")
OUT=${OUT:-results/rtn/${SLUG}_w${BITS}g${GS}}
mkdir -p "$OUT"
EXTRA_ARGS="${EXTRA_ARGS:-} --diag-seqs ${DIAG_SEQS:-8}"

QK=(--base rtn --qk-correct full)

# K-only: is the predictor fixed by small batches?  (compare with rtn_k)
want_arm rtn_k_kb64        && run_arm "$OUT" rtn_k_kb64        "${QK[@]}" --qk-schedule K
# QKQ with the new K defaults (batch 16 -> 64, row cap 1, refresh 5)
want_arm rtn_qkq_kb64      && run_arm "$OUT" rtn_qkq_kb64      "${QK[@]}" --qk-schedule QKQ
# smaller K batches
want_arm rtn_qkq_kb16      && run_arm "$OUT" rtn_qkq_kb16      "${QK[@]}" --qk-schedule QKQ \
                                --qk-k-batch-init 4 --qk-k-batch-max 16
# intermediate K batches
want_arm rtn_qkq_kb256     && run_arm "$OUT" rtn_qkq_kb256     "${QK[@]}" --qk-schedule QKQ \
                                --qk-k-batch-init 32 --qk-k-batch-max 256 --qk-k-row-cap 2
# fresher K gradient (exact refresh every round)
want_arm rtn_qkq_kb64_r1   && run_arm "$OUT" rtn_qkq_kb64_r1   "${QK[@]}" --qk-schedule QKQ --qk-k-refresh-every 1
# longer alternation
want_arm rtn_qkqkq_kb64    && run_arm "$OUT" rtn_qkqkq_kb64    "${QK[@]}" --qk-schedule QKQKQ
# best-so-far pipeline with CLC on the other linears
want_arm rtn_clcnq_qkq_kb64 && run_arm "$OUT" rtn_clcnq_qkq_kb64 "${QK[@]}" --clc-targets all_but_q --qk-schedule QKQ

"$PYTHON_BIN" summarize.py --results-dir "$OUT" \
  --runs rtn,rtn_qk,rtn_k,rtn_qkq,rtn_clcnq_qk,rtn_clcnq_qkq,rtn_k_kb64,rtn_qkq_kb64,rtn_qkq_kb16,rtn_qkq_kb256,rtn_qkq_kb64_r1,rtn_qkqkq_kb64,rtn_clcnq_qkq_kb64
