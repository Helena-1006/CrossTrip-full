# Both Datasets Pipeline Report

- 开始时间: 2026-04-16 14:53:44
- 结束时间: 2026-04-16 23:38:10
- 整体状态: ok

## Yelp — 阶段汇总

| 阶段 | 状态 | 日志 |
|---|---|---|
| tuning | ok | /root/autodl-tmp/MyCrossCity/code/new_citypref_llm/pipeline_runs/yelp_both_pipeline_v1/tuning/stage_tuning.log |
| final_train | ok | /root/autodl-tmp/MyCrossCity/code/new_citypref_llm/pipeline_runs/yelp_both_pipeline_v1/final_train/stage_final_train.log |
| ablation_hparam | ok | /root/autodl-tmp/MyCrossCity/code/new_citypref_llm/pipeline_runs/yelp_both_pipeline_v1/ablation/stage_ablation.log |

## Yelp — 命令记录

### tuning

```bash
/root/miniconda3/envs/SPOT-Trip/bin/python ./optuna_tune.py --study_name yelp_both_pipeline_v1_tune --output_dir /root/autodl-tmp/MyCrossCity/code/new_citypref_llm/pipeline_runs/yelp_both_pipeline_v1/tuning --n_trials 12 --timeout 0 --objective_metric full_combo --objective_combo_beta_fixed 4.0 --seed 2050 --main_script ./main.py --dataset_name Yelp --data_split_path /root/autodl-tmp/MyCrossCity/Yelp/spottrip_baseline_split.pkl --save_path ../../Yelp/model_save_new --ori_data ../../Yelp/home.txt --dst_data ../../Yelp/oot.txt --trans_data ../../Yelp/travel.txt --ori_data_enriched ../../Yelp/extendData/enriched_home.txt --dst_data_enriched ../../Yelp/extendData/enriched_oot.txt --device cuda:0 --epoch 30 --stop_epoch 8 --use_enriched_data 1 --semantic_backend qwen --qwen_strict 1 --llm_model_name Qwen/Qwen2.5-1.5B-Instruct --llm_fallback_names Qwen/Qwen2.5-1.5B-Instruct --llm_dtype bfloat16 --use_mamba_backbone 1 --mamba_strict 1 --mamba_d_conv 4 --seq_num_layers_choices 2,3 --mamba_d_state_choices 32 --mamba_expand_choices 2 --pair_max_future_choices 4,6 --beam_size_choices 3 --lr_min 0.0003 --lr_max 0.0005 --lambda_pair_min 0.03 --lambda_pair_max 0.2 --lambda_transition_min 0.05 --lambda_transition_max 0.2 --transition_logit_scale_min 0.05 --transition_logit_scale_max 0.3 --beam_len_penalty_min 0.25 --beam_len_penalty_max 0.6 --dropout_min 0.08 --dropout_max 0.2 --temperature_min 0.06 --temperature_max 0.12 --combo_beta_min 1.5 --combo_beta_max 4.0 --enable_pairwise_loss 1 --save_trainable_only 1 --save_optimizer_state 0 --save_dual_best 1 --use_f1_floor_filter 0 --f1_floor_margin 0.002 --use_no_repeat_mask 1 --early_stop_metric full_combo --decode_constraint_mode soft --enforce_start_end_constraints 1 --soft_constraint_scale_min 0.0 --soft_constraint_scale_max 0.08 --soft_constraint_dist_emb_dim_choices 16,32
```

### final_train

```bash
/root/miniconda3/envs/SPOT-Trip/bin/python ./main.py --mode train --log --name yelp_both_pipeline_v1_final_s2050 --dataset_name Yelp --data_split_path /root/autodl-tmp/MyCrossCity/Yelp/spottrip_baseline_split.pkl --save_path ../../Yelp/model_save_new --ori_data ../../Yelp/home.txt --dst_data ../../Yelp/oot.txt --trans_data ../../Yelp/travel.txt --ori_data_enriched ../../Yelp/extendData/enriched_home.txt --dst_data_enriched ../../Yelp/extendData/enriched_oot.txt --device cuda:0 --seed 2050 --semantic_backend qwen --llm_model_name Qwen/Qwen2.5-1.5B-Instruct --llm_fallback_names Qwen/Qwen2.5-1.5B-Instruct --llm_dtype bfloat16 --use_mamba_backbone 1 --seq_num_layers 2 --mamba_d_state 32 --mamba_d_conv 4 --mamba_expand 2 --dropout 0.08703438600353917 --temperature 0.07863357855049277 --enable_pairwise_loss 1 --lambda_pair 0.16659825439743392 --pair_max_future 6 --lambda_transition 0.08260948751514435 --transition_logit_scale 0.2907728698865922 --use_beam_search 1 --beam_size 3 --beam_len_penalty 0.46331447067470943 --use_no_repeat_mask 1 --pref_factor_k 4 --lambda_decouple 0.1 --lambda_semantic 0.1 --eta_fixed -1.0 --decode_constraint_mode soft --soft_constraint_scale 0.07903668540158991 --soft_constraint_dist_emb_dim 32 --enforce_start_end_constraints 1 --epoch 30 --stop_epoch 8 --lr 0.0004673570731306835 --l2 1e-05 --lr_dc 0.3 --lr_dc_step 8 --save_trainable_only 0 --save_optimizer_state 0 --run_final_test_after_train --early_stop_metric full_pairs_f1 --combo_beta 2.5602566338108823 --use_f1_floor_filter 0 --f1_floor_margin 0.002 --save_dual_best 1 --log_path /root/autodl-tmp/MyCrossCity/code/new_citypref_llm/pipeline_runs/yelp_both_pipeline_v1/final_train/logs --best_save --use_enriched_data --qwen_strict --mamba_strict
```

### ablation_hparam

```bash
/root/miniconda3/envs/SPOT-Trip/bin/python ./batch_ablation_hparam.py --dataset_name Yelp --run_mode train --seed 2050 --device cuda:0 --output_dir /root/autodl-tmp/MyCrossCity/code/new_citypref_llm/pipeline_runs/yelp_both_pipeline_v1/ablation/batch_output --exp_prefix yelp_both_pipeline_v1_ablation --log_path /root/autodl-tmp/MyCrossCity/code/new_citypref_llm/pipeline_runs/yelp_both_pipeline_v1/final_train/logs --data_split_path /root/autodl-tmp/MyCrossCity/Yelp/spottrip_baseline_split.pkl --save_path ../../Yelp/model_save_new --ori_data ../../Yelp/home.txt --dst_data ../../Yelp/oot.txt --trans_data ../../Yelp/travel.txt --ori_data_enriched ../../Yelp/extendData/enriched_home.txt --dst_data_enriched ../../Yelp/extendData/enriched_oot.txt --semantic_backend qwen --llm_model_name Qwen/Qwen2.5-1.5B-Instruct --llm_fallback_names Qwen/Qwen2.5-1.5B-Instruct --llm_dtype bfloat16 --qwen_strict 1 --use_enriched_data 1 --use_mamba_backbone 1 --mamba_strict 1 --seq_num_layers 2 --mamba_d_state 32 --mamba_d_conv 4 --mamba_expand 2 --dropout 0.08703438600353917 --temperature 0.07863357855049277 --enable_pairwise_loss 1 --lambda_pair 0.16659825439743392 --pair_max_future 6 --lambda_transition 0.08260948751514435 --transition_logit_scale 0.2907728698865922 --use_beam_search 1 --beam_size 3 --beam_len_penalty 0.46331447067470943 --use_no_repeat_mask 1 --pref_factor_k 4 --lambda_decouple 0.1 --lambda_semantic 0.1 --eta_fixed -1.0 --decode_constraint_mode soft --soft_constraint_scale 0.07903668540158991 --soft_constraint_dist_emb_dim 32 --enforce_start_end_constraints 1 --epoch 30 --stop_epoch 8 --lr 0.0004673570731306835 --l2 1e-05 --lr_dc 0.3 --lr_dc_step 8 --write_incremental_csv 1 --save_trainable_only 1 --early_stop_metric full_pairs_f1 --combo_beta 2.5602566338108823 --use_f1_floor_filter 0 --f1_floor_margin 0.002 --save_dual_best 1 --append_full_reference 0 --hparam_seq_num_layers 1,2,3 --hparam_lambda_pair 0.25,0.5,1.0,1.5,2.0 --hparam_transition_strength 1.0 --hparam_eta_fixed 0.0,0.2,0.4,0.6,0.8
```

## Yelp — 产出物

- tuning_dir: /root/autodl-tmp/MyCrossCity/code/new_citypref_llm/pipeline_runs/yelp_both_pipeline_v1/tuning
- best_params_json: /root/autodl-tmp/MyCrossCity/code/new_citypref_llm/pipeline_runs/yelp_both_pipeline_v1/tuning/yelp_both_pipeline_v1_tune_best_params.json
- best_trial_log: /root/autodl-tmp/MyCrossCity/code/new_citypref_llm/pipeline_runs/yelp_both_pipeline_v1/tuning/logs/04-16-15:57 yelp_both_pipeline_v1_tune_trial_006(2050).log
- best_trial_model_dir: /root/autodl-tmp/MyCrossCity/Yelp/model_save_new/yelp_both_pipeline_v1_tune_trial_006
- final_stage_log: /root/autodl-tmp/MyCrossCity/code/new_citypref_llm/pipeline_runs/yelp_both_pipeline_v1/final_train/stage_final_train.log
- final_model_log: /root/autodl-tmp/MyCrossCity/code/new_citypref_llm/pipeline_runs/yelp_both_pipeline_v1/final_train/logs/04-16-17:01 yelp_both_pipeline_v1_final_s2050(2050).log
- final_model_dir: /root/autodl-tmp/MyCrossCity/Yelp/model_save_new/yelp_both_pipeline_v1_final_s2050
- ablation_output_dir: /root/autodl-tmp/MyCrossCity/code/new_citypref_llm/pipeline_runs/yelp_both_pipeline_v1/ablation/batch_output
- ablation_summary_latest_csv: /root/autodl-tmp/MyCrossCity/code/new_citypref_llm/pipeline_runs/yelp_both_pipeline_v1/ablation/batch_output/summary_yelp_both_pipeline_v1_ablation_latest.csv

## Foursquare — 阶段汇总

| 阶段 | 状态 | 日志 |
|---|---|---|
| tuning | ok | /root/autodl-tmp/MyCrossCity/code/new_citypref_llm/pipeline_runs/fsq_both_pipeline_v1/tuning/stage_tuning.log |
| final_train | ok | /root/autodl-tmp/MyCrossCity/code/new_citypref_llm/pipeline_runs/fsq_both_pipeline_v1/final_train/stage_final_train.log |
| ablation_hparam | ok | /root/autodl-tmp/MyCrossCity/code/new_citypref_llm/pipeline_runs/fsq_both_pipeline_v1/ablation/stage_ablation.log |

## Foursquare — 命令记录

### tuning

```bash
/root/miniconda3/envs/SPOT-Trip/bin/python ./optuna_tune.py --study_name fsq_both_pipeline_v1_tune --output_dir /root/autodl-tmp/MyCrossCity/code/new_citypref_llm/pipeline_runs/fsq_both_pipeline_v1/tuning --n_trials 12 --timeout 0 --objective_metric full_combo --objective_combo_beta_fixed 4.0 --seed 2050 --main_script ./main.py --dataset_name Foursquare --data_split_path ../../Foursquare/data_split_new.pkl --save_path ../../Foursquare/model_save_new --ori_data ../../Foursquare/home.txt --dst_data ../../Foursquare/oot.txt --trans_data ../../Foursquare/travel.txt --ori_data_enriched ../../Foursquare/extendData/enriched_home.txt --dst_data_enriched ../../Foursquare/extendData/enriched_oot.txt --device cuda:0 --epoch 30 --stop_epoch 8 --use_enriched_data 1 --semantic_backend qwen --qwen_strict 1 --llm_model_name Qwen/Qwen2.5-1.5B-Instruct --llm_fallback_names Qwen/Qwen2.5-1.5B-Instruct --llm_dtype bfloat16 --use_mamba_backbone 1 --mamba_strict 1 --mamba_d_conv 4 --seq_num_layers_choices 2,3 --mamba_d_state_choices 32 --mamba_expand_choices 2 --pair_max_future_choices 4,6 --beam_size_choices 3 --lr_min 0.0003 --lr_max 0.0005 --lambda_pair_min 0.03 --lambda_pair_max 0.2 --lambda_transition_min 0.05 --lambda_transition_max 0.2 --transition_logit_scale_min 0.05 --transition_logit_scale_max 0.3 --beam_len_penalty_min 0.25 --beam_len_penalty_max 0.6 --dropout_min 0.08 --dropout_max 0.2 --temperature_min 0.06 --temperature_max 0.12 --combo_beta_min 1.5 --combo_beta_max 4.0 --enable_pairwise_loss 1 --save_trainable_only 1 --save_optimizer_state 0 --save_dual_best 1 --use_f1_floor_filter 0 --f1_floor_margin 0.002 --use_no_repeat_mask 1 --early_stop_metric full_combo --decode_constraint_mode soft --enforce_start_end_constraints 1 --soft_constraint_scale_min 0.0 --soft_constraint_scale_max 0.08 --soft_constraint_dist_emb_dim_choices 16,32
```

### final_train

```bash
/root/miniconda3/envs/SPOT-Trip/bin/python ./main.py --mode train --log --name fsq_both_pipeline_v1_final_s2050 --dataset_name Foursquare --data_split_path ../../Foursquare/data_split_new.pkl --save_path ../../Foursquare/model_save_new --ori_data ../../Foursquare/home.txt --dst_data ../../Foursquare/oot.txt --trans_data ../../Foursquare/travel.txt --ori_data_enriched ../../Foursquare/extendData/enriched_home.txt --dst_data_enriched ../../Foursquare/extendData/enriched_oot.txt --device cuda:0 --seed 2050 --semantic_backend qwen --llm_model_name Qwen/Qwen2.5-1.5B-Instruct --llm_fallback_names Qwen/Qwen2.5-1.5B-Instruct --llm_dtype bfloat16 --use_mamba_backbone 1 --seq_num_layers 3 --mamba_d_state 32 --mamba_d_conv 4 --mamba_expand 2 --dropout 0.13030822625040137 --temperature 0.06413239744570416 --enable_pairwise_loss 1 --lambda_pair 0.049009977603333685 --pair_max_future 6 --lambda_transition 0.1374167334183384 --transition_logit_scale 0.2222178005094156 --use_beam_search 1 --beam_size 3 --beam_len_penalty 0.44349753088705746 --use_no_repeat_mask 1 --pref_factor_k 4 --lambda_decouple 0.1 --lambda_semantic 0.1 --eta_fixed -1.0 --decode_constraint_mode soft --soft_constraint_scale 0.07498466069744883 --soft_constraint_dist_emb_dim 32 --enforce_start_end_constraints 1 --epoch 30 --stop_epoch 8 --lr 0.0004562487002247504 --l2 1e-05 --lr_dc 0.3 --lr_dc_step 8 --save_trainable_only 0 --save_optimizer_state 0 --run_final_test_after_train --early_stop_metric full_pairs_f1 --combo_beta 2.0984134923431585 --use_f1_floor_filter 0 --f1_floor_margin 0.002 --save_dual_best 1 --log_path /root/autodl-tmp/MyCrossCity/code/new_citypref_llm/pipeline_runs/fsq_both_pipeline_v1/final_train/logs --best_save --use_enriched_data --qwen_strict --mamba_strict
```

### ablation_hparam

```bash
/root/miniconda3/envs/SPOT-Trip/bin/python ./batch_ablation_hparam.py --dataset_name Foursquare --run_mode train --seed 2050 --device cuda:0 --output_dir /root/autodl-tmp/MyCrossCity/code/new_citypref_llm/pipeline_runs/fsq_both_pipeline_v1/ablation/batch_output --exp_prefix fsq_both_pipeline_v1_ablation --log_path /root/autodl-tmp/MyCrossCity/code/new_citypref_llm/pipeline_runs/fsq_both_pipeline_v1/final_train/logs --data_split_path ../../Foursquare/data_split_new.pkl --save_path ../../Foursquare/model_save_new --ori_data ../../Foursquare/home.txt --dst_data ../../Foursquare/oot.txt --trans_data ../../Foursquare/travel.txt --ori_data_enriched ../../Foursquare/extendData/enriched_home.txt --dst_data_enriched ../../Foursquare/extendData/enriched_oot.txt --semantic_backend qwen --llm_model_name Qwen/Qwen2.5-1.5B-Instruct --llm_fallback_names Qwen/Qwen2.5-1.5B-Instruct --llm_dtype bfloat16 --qwen_strict 1 --use_enriched_data 1 --use_mamba_backbone 1 --mamba_strict 1 --seq_num_layers 3 --mamba_d_state 32 --mamba_d_conv 4 --mamba_expand 2 --dropout 0.13030822625040137 --temperature 0.06413239744570416 --enable_pairwise_loss 1 --lambda_pair 0.049009977603333685 --pair_max_future 6 --lambda_transition 0.1374167334183384 --transition_logit_scale 0.2222178005094156 --use_beam_search 1 --beam_size 3 --beam_len_penalty 0.44349753088705746 --use_no_repeat_mask 1 --pref_factor_k 4 --lambda_decouple 0.1 --lambda_semantic 0.1 --eta_fixed -1.0 --decode_constraint_mode soft --soft_constraint_scale 0.07498466069744883 --soft_constraint_dist_emb_dim 32 --enforce_start_end_constraints 1 --epoch 30 --stop_epoch 8 --lr 0.0004562487002247504 --l2 1e-05 --lr_dc 0.3 --lr_dc_step 8 --write_incremental_csv 1 --save_trainable_only 1 --early_stop_metric full_pairs_f1 --combo_beta 2.0984134923431585 --use_f1_floor_filter 0 --f1_floor_margin 0.002 --save_dual_best 1 --append_full_reference 0 --hparam_seq_num_layers 1,2,3 --hparam_lambda_pair 0.25,0.5,1.0,1.5,2.0 --hparam_transition_strength 1.0 --hparam_eta_fixed 0.0,0.2,0.4,0.6,0.8
```

## Foursquare — 产出物

- tuning_dir: /root/autodl-tmp/MyCrossCity/code/new_citypref_llm/pipeline_runs/fsq_both_pipeline_v1/tuning
- best_params_json: /root/autodl-tmp/MyCrossCity/code/new_citypref_llm/pipeline_runs/fsq_both_pipeline_v1/tuning/fsq_both_pipeline_v1_tune_best_params.json
- best_trial_log: /root/autodl-tmp/MyCrossCity/code/new_citypref_llm/pipeline_runs/fsq_both_pipeline_v1/tuning/logs/04-16-19:56 fsq_both_pipeline_v1_tune_trial_002(2050).log
- best_trial_model_dir: /root/autodl-tmp/MyCrossCity/Foursquare/model_save_new/fsq_both_pipeline_v1_tune_trial_002
- final_stage_log: /root/autodl-tmp/MyCrossCity/code/new_citypref_llm/pipeline_runs/fsq_both_pipeline_v1/final_train/stage_final_train.log
- final_model_log: /root/autodl-tmp/MyCrossCity/code/new_citypref_llm/pipeline_runs/fsq_both_pipeline_v1/final_train/logs/04-16-21:23 fsq_both_pipeline_v1_final_s2050(2050).log
- final_model_dir: /root/autodl-tmp/MyCrossCity/Foursquare/model_save_new/fsq_both_pipeline_v1_final_s2050
- ablation_output_dir: /root/autodl-tmp/MyCrossCity/code/new_citypref_llm/pipeline_runs/fsq_both_pipeline_v1/ablation/batch_output
- ablation_summary_latest_csv: /root/autodl-tmp/MyCrossCity/code/new_citypref_llm/pipeline_runs/fsq_both_pipeline_v1/ablation/batch_output/summary_fsq_both_pipeline_v1_ablation_latest.csv

## 说明

- 两个数据集顺序运行，一个失败不影响另一个继续执行。
- 调优阶段仅保存可训练参数（存储高效）；最终训练阶段保存完整 checkpoint。
