#!/usr/bin/env bash
# Llama-3-8B experiments, in phases.
#
#   PHASE=smoke bash scripts/llama3_8b/run_8b.sh                  # layers 0-1, no PPL: API / OOM / k_sp check
#   PHASE=rtn   BITS_LIST="3 4" bash scripts/llama3_8b/run_8b.sh  # RTN arms
#   PHASE=awq   BITS_LIST="3 4" bash scripts/llama3_8b/run_8b.sh  # AWQ arms
#   PHASE=all   bash scripts/llama3_8b/run_8b.sh
#   ARMS=rtn,rtn_qkq PHASE=rtn bash scripts/llama3_8b/run_8b.sh   # subset of arms
#
# Memory: ~16 GB model + ~2 GB calib hidden states + ~5 GB correction stats (+ transient).
# If the GPU has < 40 GB:  EXTRA_ARGS="--inps-device cpu --qk-opt-seqs 32"
# Fixed settings from the 1B study: --qk-strata 1, K batches 16->64 row cap 1, K diag-H on 32 q/seq.
export MODEL_PATH=${MODEL_PATH:-meta-llama/Meta-Llama-3-8B}
export N_CALIB=${N_CALIB:-128}
PHASE=${PHASE:-smoke}
BITS_LIST=${BITS_LIST:-"3 4"}
source "$(dirname "${BASH_SOURCE[0]}")/../common.sh"

ROOT_OUT=${ROOT_OUT:-results/llama3_8b}
EXTRA_ARGS="${EXTRA_ARGS:-} --diag-seqs ${DIAG_SEQS:-8} --qk-strata 1"
QK=(--qk-correct full)

summ() { "$PYTHON_BIN" summarize.py --results-dir "$1" --csv "$1/summary.csv" || true; }

if [[ "$PHASE" == "smoke" ]]; then
  OUT="$ROOT_OUT/smoke"; mkdir -p "$OUT"
  BITS=3
  run_arm "$OUT" smoke_rtn_qkq --base rtn "${QK[@]}" --qk-schedule QKQ --qk-layers 0-1 \
          --n-calib 32 --qk-opt-seqs 16 --qk-accept-seqs 8 --diag-seqs 2 --no-eval-ppl
  summ "$OUT"
  echo ">>> check: no error, KL_full decreases on layers 0-1, k_sp > 0 in the table above"
  exit 0
fi

if [[ "$PHASE" == "rtn" || "$PHASE" == "all" ]]; then
  if want_arm fp && [[ ! -f "$ROOT_OUT/fp/fp/summary.json" ]]; then
    mkdir -p "$ROOT_OUT/fp"; run_arm "$ROOT_OUT/fp" fp --base none
  fi
  for BITS in $BITS_LIST; do
    OUT="$ROOT_OUT/rtn_w${BITS}g${GS}"; mkdir -p "$OUT"
    want_arm rtn                 && run_arm "$OUT" rtn                 --base rtn
    want_arm rtn_clcq            && run_arm "$OUT" rtn_clcq            --base rtn --clc-targets q
    want_arm rtn_qk              && run_arm "$OUT" rtn_qk              --base rtn "${QK[@]}" --qk-schedule Q
    want_arm rtn_qkq             && run_arm "$OUT" rtn_qkq             --base rtn "${QK[@]}" --qk-schedule QKQ
    want_arm rtn_clc             && run_arm "$OUT" rtn_clc             --base rtn --clc-targets all
    want_arm rtn_clcnq_qkq       && run_arm "$OUT" rtn_clcnq_qkq       --base rtn --clc-targets all_but_q "${QK[@]}" --qk-schedule QKQ
    want_arm rtn_clcnq_qkqkq     && run_arm "$OUT" rtn_clcnq_qkqkq     --base rtn --clc-targets all_but_q "${QK[@]}" --qk-schedule QKQKQ
    summ "$OUT"
  done
fi

if [[ "$PHASE" == "awq" || "$PHASE" == "all" ]]; then
  for BITS in $BITS_LIST; do
    OUT="$ROOT_OUT/awq_w${BITS}g${GS}"; mkdir -p "$OUT"
    want_arm awq                 && run_arm "$OUT" awq                 --base awq
    want_arm awq_clcq            && run_arm "$OUT" awq_clcq            --base awq --clc-targets q
    want_arm awq_qk              && run_arm "$OUT" awq_qk              --base awq "${QK[@]}" --qk-schedule Q
    want_arm awq_qkq             && run_arm "$OUT" awq_qkq             --base awq "${QK[@]}" --qk-schedule QKQ
    want_arm awq_clc             && run_arm "$OUT" awq_clc             --base awq --clc-targets all
    want_arm awq_clcnq_qkq       && run_arm "$OUT" awq_clcnq_qkq       --base awq --clc-targets all_but_q "${QK[@]}" --qk-schedule QKQ
    summ "$OUT"
  done
fi
