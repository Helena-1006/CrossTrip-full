#!/usr/bin/env bash
set -euo pipefail

# Re-train Yelp with trial_001 params and save full checkpoints,
# then evaluate test split with model_best/model_best_f1/model_best_pairs.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

DATASET_NAME="Yelp"
SEED="2050"
DEVICE="cuda:0"

SAVE_PATH="../../Yelp/model_save_new"
RUN_NAME="yelp_trial001_fullckpt"

ORI_DATA="../../Yelp/home.txt"
DST_DATA="../../Yelp/oot.txt"
TRANS_DATA="../../Yelp/travel.txt"
ORI_DATA_ENRICHED="../../Yelp/extendData/enriched_home.txt"
DST_DATA_ENRICHED="../../Yelp/extendData/enriched_oot.txt"
SPLIT_PATH="/root/autodl-tmp/MyCrossCity/Yelp/spottrip_baseline_split.pkl"

LOG_PATH="/root/autodl-tmp/MyCrossCity/code/new_citypref_llm/optuna_runs/yelp_fullcombo_tune_v1/logs"

TRAIN_COMMON_ARGS=(
  --mode train
  --dataset_name "$DATASET_NAME"
  --ori_data "$ORI_DATA"
  --dst_data "$DST_DATA"
  --trans_data "$TRANS_DATA"
  --use_enriched_data
  --ori_data_enriched "$ORI_DATA_ENRICHED"
  --dst_data_enriched "$DST_DATA_ENRICHED"
  --data_split_path "$SPLIT_PATH"
  --split_strategy legacy
  --split_singleton_to_train 1
  --save_path "$SAVE_PATH"
  --name "$RUN_NAME"
  --seed "$SEED"
  --device "$DEVICE"
  --log
  --log_path "$LOG_PATH"
  --semantic_backend qwen
  --qwen_strict
  --qwen_train_soft_prompt 1
  --mamba_strict
  --save_trainable_only 0
  --save_optimizer_state 0
  --save_dual_best 1
  --early_stop_metric full_combo
  --lr 0.00012007600802363917
  --lambda_pair 0.9036724214705179
  --pair_max_future 4
  --lambda_transition 0.5540582351353283
  --transition_logit_scale 0.9825550032246688
  --beam_size 5
  --beam_len_penalty 0.559720660610013
  --combo_beta 2.2634933830964377
  --dropout 0.2374616517436221
  --temperature 0.10745187659658728
  --seq_num_layers 2
  --mamba_d_state 16
  --mamba_expand 2
)

TEST_COMMON_ARGS=(
  --mode test
  --dataset_name "$DATASET_NAME"
  --ori_data "$ORI_DATA"
  --dst_data "$DST_DATA"
  --trans_data "$TRANS_DATA"
  --use_enriched_data
  --ori_data_enriched "$ORI_DATA_ENRICHED"
  --dst_data_enriched "$DST_DATA_ENRICHED"
  --data_split_path "$SPLIT_PATH"
  --split_strategy legacy
  --split_singleton_to_train 1
  --save_path "$SAVE_PATH"
  --name "$RUN_NAME"
  --seed "$SEED"
  --device "$DEVICE"
  --log
  --log_path "$LOG_PATH"
  --semantic_backend qwen
  --qwen_strict
  --qwen_train_soft_prompt 1
  --mamba_strict
)

echo "[STEP] Start retraining $RUN_NAME with full checkpoints"
python main.py "${TRAIN_COMMON_ARGS[@]}"

RUN_DIR="$SAVE_PATH/$RUN_NAME"
if [[ ! -d "$RUN_DIR" ]]; then
  echo "[ERROR] Run directory not found: $RUN_DIR"
  exit 1
fi

declare -a CKPTS=("model_best.xhr" "model_best_f1.xhr" "model_best_pairs.xhr")

for ckpt in "${CKPTS[@]}"; do
  if [[ ! -f "$RUN_DIR/$ckpt" ]]; then
    echo "[WARN] Missing checkpoint: $RUN_DIR/$ckpt"
    continue
  fi

  echo "[STEP] Test with checkpoint: $ckpt"
  python main.py "${TEST_COMMON_ARGS[@]}" --ckpt_name "$ckpt"
done

echo "[DONE] Retrain and test flow finished for $RUN_NAME"