# Yelp Soft Constraint Pipeline Report

- Start Time: 2026-04-03 23:17:09
- End Time: 2026-04-04 04:03:55
- Status: ok

## Stage Summary

| Stage | Status | Log |
|---|---|---|
| tuning | ok | /root/autodl-tmp/MyCrossCity/code/new_citypref_llm/pipeline_runs/yelp_soft_pipeline_v1/tuning/stage_tuning.log |
| final_train | ok | /root/autodl-tmp/MyCrossCity/code/new_citypref_llm/pipeline_runs/yelp_soft_pipeline_v1/final_train/stage_final_train.log |
| ablation_hparam | ok | /root/autodl-tmp/MyCrossCity/code/new_citypref_llm/pipeline_runs/yelp_soft_pipeline_v1/ablation/stage_ablation.log |

## Commands

### tuning

```bash
/root/miniconda3/envs/SPOT-Trip/bin/python ./optuna_tune.py --study_name yelp_soft_pipeline_v1_tune --output_dir /root/autodl-tmp/MyCrossCity/code/new_citypref_llm/pipeline_runs/yelp_soft_pipeline_v1/tuning --n_trials 12 --timeout 0 --objective_metric full_combo --seed 2050 --main_script ./main.py --dataset_name Yelp --data_split_path /root/autodl-tmp/MyCrossCity/Yelp/spottrip_baseline_split.pkl --save_path ../../Yelp/model_save_new --ori_data ../../Yelp/home.txt --dst_data ../../Yelp/oot.txt --trans_data ../../Yelp/travel.txt --ori_data_enriched ../../Yelp/extendData/enriched_home.txt --dst_data_enriched ../../Yelp/extendData/enriched_oot.txt --device cuda:0 --epoch 30 --stop_epoch 8 --use_enriched_data 1 --semantic_backend qwen --qwen_strict 1 --llm_model_name Qwen/Qwen2.5-1.5B-Instruct --llm_fallback_names Qwen/Qwen2.5-1.5B-Instruct --llm_dtype bfloat16 --use_mamba_backbone 1 --mamba_strict 1 --mamba_d_conv 4 --seq_num_layers_choices 2,3 --mamba_d_state_choices 32 --mamba_expand_choices 2 --pair_max_future_choices 4,6 --beam_size_choices 3 --lr_min 0.0003 --lr_max 0.0006 --lambda_pair_min 0.03 --lambda_pair_max 0.2 --lambda_transition_min 0.05 --lambda_transition_max 0.2 --transition_logit_scale_min 0.05 --transition_logit_scale_max 0.3 --beam_len_penalty_min 0.25 --beam_len_penalty_max 0.6 --dropout_min 0.08 --dropout_max 0.2 --temperature_min 0.06 --temperature_max 0.12 --combo_beta_min 1.5 --combo_beta_max 4.0 --enable_pairwise_loss 1 --save_trainable_only 1 --save_optimizer_state 0 --save_dual_best 1 --use_f1_floor_filter 1 --f1_floor_margin 0.002 --use_no_repeat_mask 1 --early_stop_metric full_combo --decode_constraint_mode soft --enforce_start_end_constraints 1 --soft_constraint_scale_min 0.0 --soft_constraint_scale_max 0.08 --soft_constraint_dist_emb_dim_choices 16,32
```

### final_train

```bash
/root/miniconda3/envs/SPOT-Trip/bin/python ./main.py --mode train --log --name yelp_soft_pipeline_v1_final_s2050 --dataset_name Yelp --data_split_path /root/autodl-tmp/MyCrossCity/Yelp/spottrip_baseline_split.pkl --save_path ../../Yelp/model_save_new --ori_data ../../Yelp/home.txt --dst_data ../../Yelp/oot.txt --trans_data ../../Yelp/travel.txt --ori_data_enriched ../../Yelp/extendData/enriched_home.txt --dst_data_enriched ../../Yelp/extendData/enriched_oot.txt --device cuda:0 --seed 2050 --semantic_backend qwen --llm_model_name Qwen/Qwen2.5-1.5B-Instruct --llm_fallback_names Qwen/Qwen2.5-1.5B-Instruct --llm_dtype bfloat16 --use_mamba_backbone 1 --seq_num_layers 3 --mamba_d_state 32 --mamba_d_conv 4 --mamba_expand 2 --dropout 0.19870688874069625 --temperature 0.09073104164491952 --enable_pairwise_loss 1 --lambda_pair 0.14146035888136166 --pair_max_future 6 --lambda_transition 0.05573173796754207 --transition_logit_scale 0.05749702283258237 --use_beam_search 1 --beam_size 3 --beam_len_penalty 0.5933113553841403 --use_no_repeat_mask 1 --pref_factor_k 4 --lambda_decouple 0.1 --lambda_semantic 0.1 --eta_fixed -1.0 --decode_constraint_mode soft --soft_constraint_scale 0.0011742263783192195 --soft_constraint_dist_emb_dim 16 --enforce_start_end_constraints 1 --epoch 30 --stop_epoch 8 --lr 0.00030668492537806405 --l2 1e-05 --lr_dc 0.3 --lr_dc_step 8 --save_trainable_only 0 --save_optimizer_state 0 --run_final_test_after_train --early_stop_metric full_pairs_f1 --combo_beta 3.974611656262081 --use_f1_floor_filter 1 --f1_floor_margin 0.002 --save_dual_best 1 --log_path /root/autodl-tmp/MyCrossCity/code/new_citypref_llm/pipeline_runs/yelp_soft_pipeline_v1/final_train/logs --best_save --use_enriched_data --qwen_strict --mamba_strict
```

### ablation_hparam

```bash
/root/miniconda3/envs/SPOT-Trip/bin/python ./batch_ablation_hparam.py --dataset_name Yelp --run_mode train --seed 2050 --device cuda:0 --output_dir /root/autodl-tmp/MyCrossCity/code/new_citypref_llm/pipeline_runs/yelp_soft_pipeline_v1/ablation/batch_output --exp_prefix yelp_soft_pipeline_v1_ablation --log_path /root/autodl-tmp/MyCrossCity/code/new_citypref_llm/pipeline_runs/yelp_soft_pipeline_v1/final_train/logs --data_split_path /root/autodl-tmp/MyCrossCity/Yelp/spottrip_baseline_split.pkl --save_path ../../Yelp/model_save_new --ori_data ../../Yelp/home.txt --dst_data ../../Yelp/oot.txt --trans_data ../../Yelp/travel.txt --ori_data_enriched ../../Yelp/extendData/enriched_home.txt --dst_data_enriched ../../Yelp/extendData/enriched_oot.txt --semantic_backend qwen --llm_model_name Qwen/Qwen2.5-1.5B-Instruct --llm_fallback_names Qwen/Qwen2.5-1.5B-Instruct --llm_dtype bfloat16 --qwen_strict 1 --use_enriched_data 1 --use_mamba_backbone 1 --mamba_strict 1 --seq_num_layers 3 --mamba_d_state 32 --mamba_d_conv 4 --mamba_expand 2 --dropout 0.19870688874069625 --temperature 0.09073104164491952 --enable_pairwise_loss 1 --lambda_pair 0.14146035888136166 --pair_max_future 6 --lambda_transition 0.05573173796754207 --transition_logit_scale 0.05749702283258237 --use_beam_search 1 --beam_size 3 --beam_len_penalty 0.5933113553841403 --use_no_repeat_mask 1 --pref_factor_k 4 --lambda_decouple 0.1 --lambda_semantic 0.1 --eta_fixed -1.0 --decode_constraint_mode soft --soft_constraint_scale 0.0011742263783192195 --soft_constraint_dist_emb_dim 16 --enforce_start_end_constraints 1 --epoch 30 --stop_epoch 8 --lr 0.00030668492537806405 --l2 1e-05 --lr_dc 0.3 --lr_dc_step 8 --write_incremental_csv 1 --save_trainable_only 1 --early_stop_metric full_pairs_f1 --combo_beta 3.974611656262081 --use_f1_floor_filter 1 --f1_floor_margin 0.002 --save_dual_best 1 --append_full_reference 0 --hparam_seq_num_layers 1,2,3 --hparam_transition_strength 0.5,1.0,1.5 --hparam_eta_fixed 0.2,0.5,0.8
```

## Key Artifacts

- tuning_dir: /root/autodl-tmp/MyCrossCity/code/new_citypref_llm/pipeline_runs/yelp_soft_pipeline_v1/tuning
- best_params_json: /root/autodl-tmp/MyCrossCity/code/new_citypref_llm/pipeline_runs/yelp_soft_pipeline_v1/tuning/yelp_soft_pipeline_v1_tune_best_params.json
- best_trial_log: /root/autodl-tmp/MyCrossCity/code/new_citypref_llm/pipeline_runs/yelp_soft_pipeline_v1/tuning/logs/04-04-01:04 yelp_soft_pipeline_v1_tune_trial_011(2050).log
- best_trial_model_dir: /root/autodl-tmp/MyCrossCity/Yelp/model_save_new/yelp_soft_pipeline_v1_tune_trial_011
- final_stage_log: /root/autodl-tmp/MyCrossCity/code/new_citypref_llm/pipeline_runs/yelp_soft_pipeline_v1/final_train/stage_final_train.log
- final_model_log: /root/autodl-tmp/MyCrossCity/code/new_citypref_llm/pipeline_runs/yelp_soft_pipeline_v1/final_train/logs/04-04-01:15 yelp_soft_pipeline_v1_final_s2050(2050).log
- final_model_dir: /root/autodl-tmp/MyCrossCity/Yelp/model_save_new/yelp_soft_pipeline_v1_final_s2050
- ablation_output_dir: /root/autodl-tmp/MyCrossCity/code/new_citypref_llm/pipeline_runs/yelp_soft_pipeline_v1/ablation/batch_output
- ablation_summary_latest_csv: /root/autodl-tmp/MyCrossCity/code/new_citypref_llm/pipeline_runs/yelp_soft_pipeline_v1/ablation/batch_output/summary_yelp_soft_pipeline_v1_ablation_latest.csv

## Notes

- All stages enforce best-save policy (equivalent to --best_save).
- Tuning stage keeps trainable-only checkpoints for storage efficiency.
- Final retrain stage saves full best checkpoint for deployment/evaluation.
