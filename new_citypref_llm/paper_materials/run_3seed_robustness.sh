#!/usr/bin/env bash
set -euo pipefail

# Run from: /root/autodl-tmp/MyCrossCity/code/new_citypref_llm
# Example:
#   cd /root/autodl-tmp/MyCrossCity/code/new_citypref_llm
#   bash paper_materials/run_3seed_robustness.sh

PY_BIN="/root/miniconda3/envs/SPOT-Trip/bin/python"
SEEDS=(2024 3407 4096)

run_yelp() {
  local s="$1"
  "$PY_BIN" ./main.py \
    --mode train --log \
    --name "yelp_soft_pipeline_v1_final_s${s}" \
    --dataset_name Yelp \
    --data_split_path /root/autodl-tmp/MyCrossCity/Yelp/spottrip_baseline_split.pkl \
    --save_path ../../Yelp/model_save_new \
    --ori_data ../../Yelp/home.txt \
    --dst_data ../../Yelp/oot.txt \
    --trans_data ../../Yelp/travel.txt \
    --ori_data_enriched ../../Yelp/extendData/enriched_home.txt \
    --dst_data_enriched ../../Yelp/extendData/enriched_oot.txt \
    --device cuda:0 --seed "$s" \
    --semantic_backend qwen \
    --llm_model_name Qwen/Qwen2.5-1.5B-Instruct \
    --llm_fallback_names Qwen/Qwen2.5-1.5B-Instruct \
    --llm_dtype bfloat16 \
    --use_mamba_backbone 1 \
    --seq_num_layers 3 \
    --mamba_d_state 32 \
    --mamba_d_conv 4 \
    --mamba_expand 2 \
    --dropout 0.19870688874069625 \
    --temperature 0.09073104164491952 \
    --enable_pairwise_loss 1 \
    --lambda_pair 0.14146035888136166 \
    --pair_max_future 6 \
    --lambda_transition 0.05573173796754207 \
    --transition_logit_scale 0.05749702283258237 \
    --use_beam_search 1 \
    --beam_size 3 \
    --beam_len_penalty 0.5933113553841403 \
    --use_no_repeat_mask 1 \
    --pref_factor_k 4 \
    --lambda_decouple 0.1 \
    --lambda_semantic 0.1 \
    --eta_fixed -1.0 \
    --decode_constraint_mode soft \
    --soft_constraint_scale 0.0011742263783192195 \
    --soft_constraint_dist_emb_dim 16 \
    --enforce_start_end_constraints 1 \
    --epoch 30 \
    --stop_epoch 8 \
    --lr 0.00030668492537806405 \
    --l2 1e-05 \
    --lr_dc 0.3 \
    --lr_dc_step 8 \
    --save_trainable_only 0 \
    --save_optimizer_state 0 \
    --run_final_test_after_train \
    --early_stop_metric full_pairs_f1 \
    --combo_beta 3.974611656262081 \
    --use_f1_floor_filter 1 \
    --f1_floor_margin 0.002 \
    --save_dual_best 1 \
    --log_path /root/autodl-tmp/MyCrossCity/code/new_citypref_llm/pipeline_runs/yelp_soft_pipeline_v1/final_train/logs_seed3 \
    --best_save --use_enriched_data --qwen_strict --mamba_strict
}

run_fsq() {
  local s="$1"
  "$PY_BIN" ./main.py \
    --mode train --log \
    --name "fsq_soft_pipeline_v1_final_s${s}" \
    --dataset_name Foursquare \
    --data_split_path /root/autodl-tmp/MyCrossCity/code/new_citypref_llm/data_split_new.pkl \
    --save_path ../../Foursquare/model_save_new \
    --ori_data ../../Foursquare/home.txt \
    --dst_data ../../Foursquare/oot.txt \
    --trans_data ../../Foursquare/travel.txt \
    --ori_data_enriched ../../Foursquare/extendData/enriched_home.txt \
    --dst_data_enriched ../../Foursquare/extendData/enriched_oot.txt \
    --device cuda:0 --seed "$s" \
    --semantic_backend qwen \
    --llm_model_name Qwen/Qwen2.5-1.5B-Instruct \
    --llm_fallback_names Qwen/Qwen2.5-0.5B-Instruct \
    --llm_dtype bfloat16 \
    --use_mamba_backbone 1 \
    --seq_num_layers 3 \
    --mamba_d_state 32 \
    --mamba_d_conv 4 \
    --mamba_expand 2 \
    --dropout 0.16266454424451948 \
    --temperature 0.09317100529492414 \
    --enable_pairwise_loss 1 \
    --lambda_pair 0.18934240398207877 \
    --pair_max_future 4 \
    --lambda_transition 0.052888310885315946 \
    --transition_logit_scale 0.2885730879561912 \
    --use_beam_search 1 \
    --beam_size 3 \
    --beam_len_penalty 0.4539723779761229 \
    --use_no_repeat_mask 1 \
    --pref_factor_k 4 \
    --lambda_decouple 0.1 \
    --lambda_semantic 0.1 \
    --eta_fixed -1.0 \
    --decode_constraint_mode soft \
    --soft_constraint_scale 0.07462942141466841 \
    --soft_constraint_dist_emb_dim 32 \
    --enforce_start_end_constraints 1 \
    --epoch 30 \
    --stop_epoch 8 \
    --lr 0.000309272394422365 \
    --l2 1e-05 \
    --lr_dc 0.3 \
    --lr_dc_step 8 \
    --save_trainable_only 0 \
    --save_optimizer_state 0 \
    --run_final_test_after_train \
    --early_stop_metric full_pairs_f1 \
    --combo_beta 4.0 \
    --use_f1_floor_filter 1 \
    --f1_floor_margin 0.002 \
    --save_dual_best 1 \
    --log_path /root/autodl-tmp/MyCrossCity/code/new_citypref_llm/pipeline_runs/fsq_soft_pipeline_v1/final_train/logs_seed3 \
    --best_save --use_enriched_data --qwen_strict --mamba_strict
}

main() {
  for s in "${SEEDS[@]}"; do
    echo "[RUN] Yelp seed=${s}"
    run_yelp "$s"
  done

  for s in "${SEEDS[@]}"; do
    echo "[RUN] Foursquare seed=${s}"
    run_fsq "$s"
  done

  echo "[DONE] 3-seed runs submitted for Yelp and Foursquare."
}

main "$@"
