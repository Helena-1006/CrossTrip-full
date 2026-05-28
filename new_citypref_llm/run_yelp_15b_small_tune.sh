#!/usr/bin/env bash
set -euo pipefail

# Small-scale retuning for Yelp with Qwen 1.5B.
# - Keeps combo_beta fixed by setting min=max.
# - Objective can be full_pairs_f1 or full_f1.
# - Uses stable setting qwen_train_soft_prompt=0 by default.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

STUDY_NAME="${STUDY_NAME:-yelp_15b_small_tune_v1}"
OUT_DIR="${OUT_DIR:-/root/autodl-tmp/MyCrossCity/code/new_citypref_llm/optuna_runs/${STUDY_NAME}}"
OBJECTIVE_METRIC="${OBJECTIVE_METRIC:-full_pairs_f1}"   # full_pairs_f1 | full_f1
EARLY_STOP_METRIC="${EARLY_STOP_METRIC:-full_pairs_f1}" # full_pairs_f1 | full_f1 | full_combo
N_TRIALS="${N_TRIALS:-12}"
SEED="${SEED:-2050}"

# Keep fixed to your preferred beta so cross-trial objective remains comparable.
FIXED_COMBO_BETA="${FIXED_COMBO_BETA:-2.2634933830964377}"

# Stable default for 1.5B.
QWEN_TRAIN_SOFT_PROMPT="${QWEN_TRAIN_SOFT_PROMPT:-0}"

mkdir -p "$OUT_DIR"

echo "[INFO] STUDY_NAME=$STUDY_NAME"
echo "[INFO] OUT_DIR=$OUT_DIR"
echo "[INFO] OBJECTIVE_METRIC=$OBJECTIVE_METRIC"
echo "[INFO] EARLY_STOP_METRIC=$EARLY_STOP_METRIC"
echo "[INFO] N_TRIALS=$N_TRIALS"
echo "[INFO] FIXED_COMBO_BETA=$FIXED_COMBO_BETA"
echo "[INFO] QWEN_TRAIN_SOFT_PROMPT=$QWEN_TRAIN_SOFT_PROMPT"

python optuna_tune.py \
  --study_name "$STUDY_NAME" \
  --output_dir "$OUT_DIR" \
  --dataset_name Yelp \
  --objective_metric "$OBJECTIVE_METRIC" \
  --early_stop_metric "$EARLY_STOP_METRIC" \
  --data_split_path /root/autodl-tmp/MyCrossCity/Yelp/spottrip_baseline_split.pkl \
  --split_strategy legacy \
  --split_singleton_to_train 1 \
  --save_path ../../Yelp/model_save_new \
  --n_trials "$N_TRIALS" \
  --seed "$SEED" \
  --use_enriched_data 1 \
  --save_trainable_only 1 \
  --save_optimizer_state 0 \
  --save_dual_best 1 \
  --copy_best_artifacts 1 \
  --semantic_backend qwen \
  --qwen_strict 1 \
  --qwen_train_soft_prompt "$QWEN_TRAIN_SOFT_PROMPT" \
  --llm_model_name Qwen/Qwen2.5-1.5B-Instruct \
  --llm_fallback_names Qwen/Qwen2.5-1.5B-Instruct \
  --llm_dtype bfloat16 \
  --mamba_strict 1 \
  --device cuda:0 \
  --combo_beta_min "$FIXED_COMBO_BETA" \
  --combo_beta_max "$FIXED_COMBO_BETA" \
  --lr_min 8e-5 \
  --lr_max 5e-4 \
  --lambda_pair_min 0.05 \
  --lambda_pair_max 0.6 \
  --lambda_transition_min 0.05 \
  --lambda_transition_max 0.6 \
  --transition_logit_scale_min 0.1 \
  --transition_logit_scale_max 0.8 \
  --beam_size_choices 3,5,7 \
  --beam_len_penalty_min 0.0 \
  --beam_len_penalty_max 0.8 \
  --dropout_min 0.08 \
  --dropout_max 0.3 \
  --temperature_min 0.07 \
  --temperature_max 0.12 \
  --pair_max_future_choices 4,8 \
  --seq_num_layers_choices 2 \
  --mamba_d_state_choices 16,32 \
  --mamba_expand_choices 2
