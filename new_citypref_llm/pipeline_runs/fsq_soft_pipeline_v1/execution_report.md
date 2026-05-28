# Yelp Soft Constraint Pipeline Report

- Start Time: 2026-04-04 14:16:43
- End Time: 2026-04-04 18:14:04
- Status: ok

## Stage Summary

| Stage | Status | Log |
|---|---|---|
| tuning | ok | /root/autodl-tmp/MyCrossCity/code/new_citypref_llm/pipeline_runs/fsq_soft_pipeline_v1/tuning/stage_tuning.log |
| final_train | ok | /root/autodl-tmp/MyCrossCity/code/new_citypref_llm/pipeline_runs/fsq_soft_pipeline_v1/final_train/stage_final_train.log |
| ablation_hparam | ok | /root/autodl-tmp/MyCrossCity/code/new_citypref_llm/pipeline_runs/fsq_soft_pipeline_v1/ablation/stage_ablation.log |

## Commands

### tuning

```bash
/root/miniconda3/envs/SPOT-Trip/bin/python ./optuna_tune.py --study_name fsq_soft_pipeline_v1_tune --output_dir /root/autodl-tmp/MyCrossCity/code/new_citypref_llm/pipeline_runs/fsq_soft_pipeline_v1/tuning --n_trials 12 --timeout 0 --objective_metric combo --seed 2050 --main_script ./main.py --dataset_name Foursquare --data_split_path /root/autodl-tmp/MyCrossCity/code/new_citypref_llm/data_split_new.pkl --save_path ../../Foursquare/model_save_new --ori_data ../../Foursquare/home.txt --dst_data ../../Foursquare/oot.txt --trans_data ../../Foursquare/travel.txt --ori_data_enriched ../../Foursquare/extendData/enriched_home.txt --dst_data_enriched ../../Foursquare/extendData/enriched_oot.txt --device cuda:0 --epoch 30 --stop_epoch 8 --use_enriched_data 1 --semantic_backend qwen --qwen_strict 1 --llm_model_name Qwen/Qwen2.5-1.5B-Instruct --llm_fallback_names Qwen/Qwen2.5-0.5B-Instruct --llm_dtype bfloat16 --use_mamba_backbone 1 --mamba_strict 1 --mamba_d_conv 4 --seq_num_layers_choices 2,3 --mamba_d_state_choices 32 --mamba_expand_choices 2 --pair_max_future_choices 4,6 --beam_size_choices 3 --lr_min 0.0003 --lr_max 0.0006 --lambda_pair_min 0.03 --lambda_pair_max 0.2 --lambda_transition_min 0.05 --lambda_transition_max 0.2 --transition_logit_scale_min 0.05 --transition_logit_scale_max 0.3 --beam_len_penalty_min 0.25 --beam_len_penalty_max 0.6 --dropout_min 0.08 --dropout_max 0.2 --temperature_min 0.06 --temperature_max 0.12 --combo_beta_min 4.0 --combo_beta_max 4.0 --enable_pairwise_loss 1 --save_trainable_only 1 --save_optimizer_state 0 --save_dual_best 1 --use_f1_floor_filter 1 --f1_floor_margin 0.002 --use_no_repeat_mask 1 --early_stop_metric combo --decode_constraint_mode soft --enforce_start_end_constraints 1 --soft_constraint_scale_min 0.0 --soft_constraint_scale_max 0.08 --soft_constraint_dist_emb_dim_choices 16,32
```

### final_train

```bash
/root/miniconda3/envs/SPOT-Trip/bin/python ./main.py --mode train --log --name fsq_soft_pipeline_v1_final_s2050 --dataset_name Foursquare --data_split_path /root/autodl-tmp/MyCrossCity/code/new_citypref_llm/data_split_new.pkl --save_path ../../Foursquare/model_save_new --ori_data ../../Foursquare/home.txt --dst_data ../../Foursquare/oot.txt --trans_data ../../Foursquare/travel.txt --ori_data_enriched ../../Foursquare/extendData/enriched_home.txt --dst_data_enriched ../../Foursquare/extendData/enriched_oot.txt --device cuda:0 --seed 2050 --semantic_backend qwen --llm_model_name Qwen/Qwen2.5-1.5B-Instruct --llm_fallback_names Qwen/Qwen2.5-0.5B-Instruct --llm_dtype bfloat16 --use_mamba_backbone 1 --seq_num_layers 3 --mamba_d_state 32 --mamba_d_conv 4 --mamba_expand 2 --dropout 0.16266454424451948 --temperature 0.09317100529492414 --enable_pairwise_loss 1 --lambda_pair 0.18934240398207877 --pair_max_future 4 --lambda_transition 0.052888310885315946 --transition_logit_scale 0.2885730879561912 --use_beam_search 1 --beam_size 3 --beam_len_penalty 0.4539723779761229 --use_no_repeat_mask 1 --pref_factor_k 4 --lambda_decouple 0.1 --lambda_semantic 0.1 --eta_fixed -1.0 --decode_constraint_mode soft --soft_constraint_scale 0.07462942141466841 --soft_constraint_dist_emb_dim 32 --enforce_start_end_constraints 1 --epoch 30 --stop_epoch 8 --lr 0.000309272394422365 --l2 1e-05 --lr_dc 0.3 --lr_dc_step 8 --save_trainable_only 0 --save_optimizer_state 0 --run_final_test_after_train --early_stop_metric full_pairs_f1 --combo_beta 4.0 --use_f1_floor_filter 1 --f1_floor_margin 0.002 --save_dual_best 1 --log_path /root/autodl-tmp/MyCrossCity/code/new_citypref_llm/pipeline_runs/fsq_soft_pipeline_v1/final_train/logs --best_save --use_enriched_data --qwen_strict --mamba_strict
```

### ablation_hparam

```bash
/root/miniconda3/envs/SPOT-Trip/bin/python ./batch_ablation_hparam.py --dataset_name Foursquare --run_mode train --seed 2050 --device cuda:0 --output_dir /root/autodl-tmp/MyCrossCity/code/new_citypref_llm/pipeline_runs/fsq_soft_pipeline_v1/ablation/batch_output --exp_prefix fsq_soft_pipeline_v1_ablation --log_path /root/autodl-tmp/MyCrossCity/code/new_citypref_llm/pipeline_runs/fsq_soft_pipeline_v1/final_train/logs --data_split_path /root/autodl-tmp/MyCrossCity/code/new_citypref_llm/data_split_new.pkl --save_path ../../Foursquare/model_save_new --ori_data ../../Foursquare/home.txt --dst_data ../../Foursquare/oot.txt --trans_data ../../Foursquare/travel.txt --ori_data_enriched ../../Foursquare/extendData/enriched_home.txt --dst_data_enriched ../../Foursquare/extendData/enriched_oot.txt --semantic_backend qwen --llm_model_name Qwen/Qwen2.5-1.5B-Instruct --llm_fallback_names Qwen/Qwen2.5-0.5B-Instruct --llm_dtype bfloat16 --qwen_strict 1 --use_enriched_data 1 --use_mamba_backbone 1 --mamba_strict 1 --seq_num_layers 3 --mamba_d_state 32 --mamba_d_conv 4 --mamba_expand 2 --dropout 0.16266454424451948 --temperature 0.09317100529492414 --enable_pairwise_loss 1 --lambda_pair 0.18934240398207877 --pair_max_future 4 --lambda_transition 0.052888310885315946 --transition_logit_scale 0.2885730879561912 --use_beam_search 1 --beam_size 3 --beam_len_penalty 0.4539723779761229 --use_no_repeat_mask 1 --pref_factor_k 4 --lambda_decouple 0.1 --lambda_semantic 0.1 --eta_fixed -1.0 --decode_constraint_mode soft --soft_constraint_scale 0.07462942141466841 --soft_constraint_dist_emb_dim 32 --enforce_start_end_constraints 1 --epoch 30 --stop_epoch 8 --lr 0.000309272394422365 --l2 1e-05 --lr_dc 0.3 --lr_dc_step 8 --write_incremental_csv 1 --save_trainable_only 1 --early_stop_metric full_pairs_f1 --combo_beta 4.0 --use_f1_floor_filter 1 --f1_floor_margin 0.002 --save_dual_best 1 --append_full_reference 0 --hparam_seq_num_layers 1,2,3 --hparam_transition_strength 0.5,1.0,1.5 --hparam_eta_fixed 0.2,0.5,0.8
```

## Key Artifacts

- tuning_dir: /root/autodl-tmp/MyCrossCity/code/new_citypref_llm/pipeline_runs/fsq_soft_pipeline_v1/tuning
- best_params_json: /root/autodl-tmp/MyCrossCity/code/new_citypref_llm/pipeline_runs/fsq_soft_pipeline_v1/tuning/fsq_soft_pipeline_v1_tune_best_params.json
- best_trial_log: /root/autodl-tmp/MyCrossCity/code/new_citypref_llm/pipeline_runs/fsq_soft_pipeline_v1/tuning/logs/04-04-14:30 fsq_soft_pipeline_v1_tune_trial_002(2050).log
- best_trial_model_dir: /root/autodl-tmp/MyCrossCity/Foursquare/model_save_new/fsq_soft_pipeline_v1_tune_trial_002
- final_stage_log: /root/autodl-tmp/MyCrossCity/code/new_citypref_llm/pipeline_runs/fsq_soft_pipeline_v1/final_train/stage_final_train.log
- final_model_log: /root/autodl-tmp/MyCrossCity/code/new_citypref_llm/pipeline_runs/fsq_soft_pipeline_v1/final_train/logs/04-04-15:54 fsq_soft_pipeline_v1_final_s2050(2050).log
- final_model_dir: /root/autodl-tmp/MyCrossCity/Foursquare/model_save_new/fsq_soft_pipeline_v1_final_s2050
- ablation_output_dir: /root/autodl-tmp/MyCrossCity/code/new_citypref_llm/pipeline_runs/fsq_soft_pipeline_v1/ablation/batch_output
- ablation_summary_latest_csv: /root/autodl-tmp/MyCrossCity/code/new_citypref_llm/pipeline_runs/fsq_soft_pipeline_v1/ablation/batch_output/summary_fsq_soft_pipeline_v1_ablation_latest.csv

## Notes

- All stages enforce best-save policy (equivalent to --best_save).
- Tuning stage keeps trainable-only checkpoints for storage efficiency.
- Final retrain stage saves full best checkpoint for deployment/evaluation.
