#!/usr/bin/env bash
# Shared settings for the experiment wrappers. Override any variable from the environment.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN=${PYTHON_BIN:-python}
MODEL_PATH=${MODEL_PATH:-meta-llama/Meta-Llama-3-8B}
MODELS_ROOT=${MODELS_ROOT:-/models}
BITS=${BITS:-3}
GS=${GS:-128}
SEED=${SEED:-42}
N_CALIB=${N_CALIB:-128}
SEQLEN=${SEQLEN:-2048}
CALIB_DATASET=${CALIB_DATASET:-c4}
FLIP=${FLIP:-0.05}              # same per-row flip budget for CLC and QK (fair comparison)
OPT_SEQS=${OPT_SEQS:-64}
ACC_SEQS=${ACC_SEQS:-16}
DIAG=${DIAG:-1}
EVAL_C4=${EVAL_C4:-0}
LM_EVAL_TASKS=${LM_EVAL_TASKS:-}  # e.g. "arc_easy arc_challenge hellaswag piqa winogrande"
EXTRA_ARGS=${EXTRA_ARGS:-}

common_args() {
  local a=(--model-path "$MODEL_PATH" --models-root "$MODELS_ROOT" --seed "$SEED"
           --bits "$BITS" --group-size "$GS"
           --calib-dataset "$CALIB_DATASET" --n-calib "$N_CALIB" --calib-seqlen "$SEQLEN"
           --clc-max-flip-percent "$FLIP" --qk-max-flip-percent "$FLIP"
           --qk-opt-seqs "$OPT_SEQS" --qk-accept-seqs "$ACC_SEQS")
  [[ "$DIAG" == "1" ]] && a+=(--diag)
  [[ "$EVAL_C4" == "1" ]] && a+=(--eval-c4)
  [[ -n "$LM_EVAL_TASKS" ]] && a+=(--lm-eval-tasks $LM_EVAL_TASKS)
  printf '%s\n' "${a[@]}"
}

run_arm() {
  # run_arm <output_dir> <run_name> [extra args...]
  local out=$1 name=$2; shift 2
  mapfile -t base < <(common_args)
  echo "=================== $name ==================="
  # shellcheck disable=SC2086
  "$PYTHON_BIN" run_qk.py "${base[@]}" --output-dir "$out" --run-name "$name" "$@" $EXTRA_ARGS
}

want_arm() {
  # ARMS="rtn,rtn_qk" restricts which arms run; empty = all
  [[ -z "${ARMS:-}" ]] && return 0
  [[ ",$ARMS," == *",$1,"* ]]
}
