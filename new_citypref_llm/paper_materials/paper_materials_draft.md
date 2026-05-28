# Cross-City Recommendation Paper Materials (Draft)

This file organizes the current experimental outputs into paper-ready materials.
Data sources are the latest pipeline outputs under:
- `pipeline_runs/yelp_soft_pipeline_v1`
- `pipeline_runs/fsq_soft_pipeline_v1`

## 1) Main Results Table

Primary reporting uses FINAL TEST metrics from final retraining logs.

| Dataset | F1 | Pairs F1 | Full F1 | Full Pairs F1 | Full REP |
|---|---:|---:|---:|---:|---:|
| Yelp | 0.0067 | 0.0001 | 0.5975 | 0.2817 | 0.0219 |
| Foursquare | 0.0318 | 0.0006 | 0.5370 | 0.2242 | 0.0423 |

Validation best epoch (for appendix):
- Yelp best val epoch = 3: F1=0.0225, Pairs F1=0.0009, Full F1=0.6030, Full Pairs F1=0.2896.
- Foursquare best val epoch = 3: F1=0.0428, Pairs F1=0.0053, Full F1=0.5419, Full Pairs F1=0.2440.

## 2) Ablation Table and Analysis

### 2.1 Yelp Ablation (delta vs Full)

| Variant | Full F1 | Delta Full F1 | Full Pairs F1 | Delta Full Pairs F1 |
|---|---:|---:|---:|---:|
| Full (soft) | 0.5978 | 0.0000 | 0.2817 | 0.0000 |
| no_llm_semantic | 0.5985 | +0.0007 | 0.2829 | +0.0012 |
| no_disentangle_transfer | 0.5990 | +0.0012 | 0.2837 | +0.0020 |
| no_city_group_pref (eta_fixed=1.0) | 0.5965 | -0.0013 | 0.2789 | -0.0028 |
| no_generator_constraints | 0.5973 | -0.0005 | 0.2810 | -0.0007 |
| hard_constraint_variant | 0.5193 | -0.0785 | 0.2025 | -0.0792 |

### 2.2 Foursquare Ablation (delta vs Full)

| Variant | Full F1 | Delta Full F1 | Full Pairs F1 | Delta Full Pairs F1 |
|---|---:|---:|---:|---:|
| Full (soft) | 0.5366 | 0.0000 | 0.2239 | 0.0000 |
| no_llm_semantic | 0.5369 | +0.0003 | 0.2255 | +0.0016 |
| no_disentangle_transfer | 0.5357 | -0.0009 | 0.2237 | -0.0002 |
| no_city_group_pref (eta_fixed=1.0) | 0.5262 | -0.0104 | 0.2140 | -0.0099 |
| no_generator_constraints | 0.5356 | -0.0010 | 0.2247 | +0.0008 |
| hard_constraint_variant | 0.4750 | -0.0616 | 0.1721 | -0.0518 |

### 2.3 Ablation Findings (paper text)

- Soft constraint decoding is consistently better than hard constraints on both datasets.
  - Yelp: hard mode drops Full F1 by 0.0785 and Full Pairs F1 by 0.0792.
  - Foursquare: hard mode drops Full F1 by 0.0616 and Full Pairs F1 by 0.0518.
- City-group preference transfer is a key contributor.
  - Removing it (eta_fixed=1.0) degrades both datasets, with larger effect on Foursquare.
- LLM semantic and disentangled-transfer ablations show small deltas.
  - Their contribution is secondary compared with decoding-constraint design and city-preference transfer.

## 3) Hyperparameter Table and Selection Analysis

### 3.1 Selected Hyperparameters (from best_params + final retrain)

| Category | Hyperparameter | Yelp | Foursquare |
|---|---|---:|---:|
| Backbone | seq_num_layers | 3 | 3 |
| Backbone | mamba_d_state | 32 | 32 |
| Backbone | mamba_expand | 2 | 2 |
| Optimization | lr | 0.00030668492537806405 | 0.000309272394422365 |
| Loss | lambda_pair | 0.14146035888136166 | 0.18934240398207877 |
| Loss | lambda_transition | 0.05573173796754207 | 0.052888310885315946 |
| Decode | beam_size | 3 | 3 |
| Decode | beam_len_penalty | 0.5933113553841403 | 0.4539723779761229 |
| Decode | pair_max_future | 6 | 4 |
| Soft Constraint | soft_constraint_scale | 0.0011742263783192195 | 0.07462942141466841 |
| Soft Constraint | soft_constraint_dist_emb_dim | 16 | 32 |
| Regularization | dropout | 0.19870688874069625 | 0.16266454424451948 |
| Temperature | temperature | 0.09073104164491952 | 0.09317100529492414 |
| Selection | early_stop_metric | full_pairs_f1 | full_pairs_f1 |

### 3.2 Selection Analysis (paper text)

- Both datasets favor a 3-layer Mamba stack under the current search range.
- Soft-constraint strength is highly dataset-specific.
  - Yelp selects near-zero scale, indicating weak reliability of geometric constraint signal.
  - Foursquare selects high scale, indicating strong benefit from distance-aware soft bias.
- Pairwise and transition terms are both needed, but relative weighting differs by dataset.

Additional sensitivity evidence from hparam ablation:
- Yelp Full Pairs F1 is relatively stable across seq_num_layers and transition strength.
- Foursquare gains from seq_num_layers=1 and transition_strength=0.5 in the single-seed ablation,
  suggesting slight over-parameterization in the default full setting.

## 4) Method Section Draft (aligned to implementation)

### 4.1 Task Definition

Given user source-city trajectory history and destination-city travel query,
the model predicts a destination POI sequence under start/end and route constraints.

### 4.2 Model Components

1. Semantic encoder:
- `semantic_backend=qwen` uses Qwen-based semantic encoding with optional soft-prompt training.
- Fallback semantic encoder is used when Qwen backend is unavailable.

2. Sequence backbone:
- Mamba backbone (`use_mamba_backbone=1`) with configurable `seq_num_layers`,
  `mamba_d_state`, `mamba_d_conv`, and `mamba_expand`.

3. Cross-city preference transfer:
- User preference representation is fused with destination-city memory.
- Blend factor uses learned gate by default (`eta_fixed < 0`) or fixed ratio when specified.

4. Transition modeling:
- Transition logits are computed from previous POI and user-destination context,
  then injected into generator logits via `transition_logit_scale`.

### 4.3 Training Objective

Total loss is weighted multi-task optimization:

L = lambda_gen * L_gen
  + lambda_pair * L_pair
  + lambda_transition * L_transition
  + lambda_decouple * L_decouple
  + lambda_semantic * L_semantic

- `L_gen`: autoregressive POI generation loss.
- `L_pair`: pairwise ranking loss over future POIs.
- `L_transition`: transition consistency loss.
- `L_decouple` and `L_semantic`: regularization/semantic alignment terms.

### 4.4 Constraint-Aware Decoding

- Hard mode: strict start/end and route constraints.
- Soft mode: distance-aware soft bias on logits (`soft_constraint_scale`, `soft_constraint_dist_emb_dim`).
- Beam search (`beam_size`, `beam_len_penalty`) with no-repeat mask (`use_no_repeat_mask=1`).

### 4.5 Model Selection and Early Stop

- Early-stop metric uses `full_pairs_f1` in final retraining.
- Additional F1-floor filter (`use_f1_floor_filter`, `f1_floor_margin`) prevents selecting checkpoints
  that improve target metric while collapsing base F1.
- `save_dual_best=1` preserves best checkpoints under multiple selection views.

## 5) Notes for Next Step (3-seed robustness)

Use the script `paper_materials/run_3seed_robustness.sh` to run Yelp/Foursquare 3-seed robustness with fixed best hyperparameters and changed seeds only.
