#!/usr/bin/env bash
set -euo pipefail

# One-click ablation eval runner for R3.
# Run this script under nohup if you want it to keep running after logout.

NPROC="${NPROC:-8}"
EVAL_MODE="${EVAL_MODE:-r3}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-checkpoints/stage5_full_last/step_30000}"
MODEL_NAME="${MODEL_NAME:-models/Qwen3-VL-8B-Instruct}"
VAL_JSONL="${VAL_JSONL:-data/unified/val.jsonl}"
IMAGE_ROOT="${IMAGE_ROOT:-data/raw}"
INDEX_DIR="${INDEX_DIR:-indices}"
BATCH_SIZE="${BATCH_SIZE:-64}"
MAX_EVAL_SAMPLES="${MAX_EVAL_SAMPLES:-2048}"
CORR_LEVELS="${CORR_LEVELS:-0,0.2,0.4,0.6,0.8,1}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-64}"
ANSWER_ONLY="${ANSWER_ONLY:-1}"
LOAD_LORA="${LOAD_LORA:-1}"
DATASET_PREFIXES="${DATASET_PREFIXES:-screenqa,chartqa,infovqa}"
DATASET="${DATASET:-}"
OUT_DIR="${OUT_DIR:-results/ablations}"
LOG_DIR="${LOG_DIR:-logs/ablations}"
SLEEP_BETWEEN="${SLEEP_BETWEEN:-2}"
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-0}"

mkdir -p "$OUT_DIR" "$LOG_DIR"

BASE_ARGS=(
  --eval_mode "$EVAL_MODE"
  --checkpoint_dir "$CHECKPOINT_DIR"
  --model_name "$MODEL_NAME"
  --val_jsonl "$VAL_JSONL"
  --image_root "$IMAGE_ROOT"
  --index_dir "$INDEX_DIR"
  --batch_size "$BATCH_SIZE"
  --max_eval_samples "$MAX_EVAL_SAMPLES"
  --corruption_levels "$CORR_LEVELS"
  --max_new_tokens "$MAX_NEW_TOKENS"
)

if [[ "$ANSWER_ONLY" == "1" ]]; then
  BASE_ARGS+=(--answer_only)
fi
if [[ "$LOAD_LORA" == "1" ]]; then
  BASE_ARGS+=(--load_lora_adapter)
fi
if [[ -n "$DATASET" ]]; then
  BASE_ARGS+=(--dataset "$DATASET")
else
  BASE_ARGS+=(--dataset_prefixes "$DATASET_PREFIXES")
fi

ABLATIONS=(
  "full|"
  "no_text_retrieval|--disable_text_retrieval"
  "no_image_retrieval|--disable_image_retrieval"
  "no_retrieval|--disable_text_retrieval --disable_image_retrieval"
  "no_prefix|--disable_prefix"
  "no_memory|--disable_memory"
  "no_visual_memory|--disable_visual_memory"
  "no_latent_tokens|--disable_latent_tokens"
  "no_gate|--disable_gate"
  "no_pseudotext|--no_pseudo_text --corrupt_text_target question"
)

for entry in "${ABLATIONS[@]}"; do
  name="${entry%%|*}"
  flags="${entry#*|}"
  out_json="${OUT_DIR}/${name}.json"
  log="${LOG_DIR}/${name}.log"

  extra_args=()
  if [[ -n "$flags" ]]; then
    read -r -a extra_args <<< "$flags"
  fi

  echo "[$(date +'%F %T')] Running ${name} -> ${out_json}"
  if ! torchrun --nproc_per_node="$NPROC" scripts/eval_r3.py \
    "${BASE_ARGS[@]}" \
    "${extra_args[@]}" \
    --out_json "$out_json" \
    > "$log" 2>&1; then
    echo "Ablation ${name} failed. See ${log}" >&2
    if [[ "$CONTINUE_ON_ERROR" != "1" ]]; then
      exit 1
    fi
  fi
  sleep "$SLEEP_BETWEEN"
done

echo "All ablation runs completed."
