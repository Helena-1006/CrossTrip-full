#!/usr/bin/env bash
set -euo pipefail

# Decode-parameter tuning without changing model structure:
# 1) Grid search on VALID for three checkpoints.
# 2) Pick best by Full_Pairs_F1, tie-break by Full_F1.
# 3) Run TEST once with the selected configuration.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# IMPORTANT:
# The current yelp_trial001_fullckpt checkpoints were trained with Qwen2.5-0.5B.
# Full checkpoints require identical model architecture at load time.
# Override by env only if your checkpoint was trained with another LLM.
LLM_MODEL_NAME="${LLM_MODEL_NAME:-Qwen/Qwen2.5-0.5B-Instruct}"
LLM_FALLBACK_NAMES="${LLM_FALLBACK_NAMES:-Qwen/Qwen2.5-0.5B-Instruct,Qwen/Qwen2.5-1.5B-Instruct}"

BASE_ARGS=(
  --dataset_name Yelp
  --ori_data ../../Yelp/home.txt
  --dst_data ../../Yelp/oot.txt
  --trans_data ../../Yelp/travel.txt
  --use_enriched_data
  --ori_data_enriched ../../Yelp/extendData/enriched_home.txt
  --dst_data_enriched ../../Yelp/extendData/enriched_oot.txt
  --data_split_path /root/autodl-tmp/MyCrossCity/Yelp/spottrip_baseline_split.pkl
  --split_strategy legacy
  --split_singleton_to_train 1
  --save_path ../../Yelp/model_save_new
  --name yelp_trial001_fullckpt
  --seed 2050
  --device cuda:0
  --semantic_backend qwen
  --llm_model_name "$LLM_MODEL_NAME"
  --llm_fallback_names "$LLM_FALLBACK_NAMES"
  --qwen_strict
  --qwen_train_soft_prompt 1
  --mamba_strict
)

CKPTS=(model_best.xhr model_best_f1.xhr model_best_pairs.xhr)
BEAM_SIZES=(3 5 7)
LEN_PENALTIES=(0.10 0.30 0.56)
TRANS_SCALES=(0.30 0.60 0.98)

ROOT_LOG_DIR="/root/autodl-tmp/MyCrossCity/code/new_citypref_llm/optuna_runs/yelp_fullcombo_tune_v1/logs"
VALID_LOG_DIR="$ROOT_LOG_DIR/valid_grid"
TEST_LOG_DIR="$ROOT_LOG_DIR/test_best"

mkdir -p "$VALID_LOG_DIR" "$TEST_LOG_DIR"

echo "[INFO] LLM_MODEL_NAME=$LLM_MODEL_NAME"
echo "[INFO] LLM_FALLBACK_NAMES=$LLM_FALLBACK_NAMES"

echo "[STEP] VALID grid search starts"
for CKPT in "${CKPTS[@]}"; do
  for BS in "${BEAM_SIZES[@]}"; do
    for LP in "${LEN_PENALTIES[@]}"; do
      for TS in "${TRANS_SCALES[@]}"; do
        echo "[VALID] ckpt=$CKPT beam_size=$BS beam_len_penalty=$LP transition_logit_scale=$TS"
        python validate.py \
          --split valid \
          --ckpt_name "$CKPT" \
          --beam_size "$BS" \
          --beam_len_penalty "$LP" \
          --transition_logit_scale "$TS" \
          --log \
          --log_path "$VALID_LOG_DIR" \
          "${BASE_ARGS[@]}"
      done
    done
  done
done

BEST_ENV_FILE="/tmp/yelp_valid_best.env"

echo "[STEP] Parse VALID logs and pick best config"
python - << 'PY'
import glob
import os
import re

log_dir = "/root/autodl-tmp/MyCrossCity/code/new_citypref_llm/optuna_runs/yelp_fullcombo_tune_v1/logs/valid_grid"
files = sorted(glob.glob(os.path.join(log_dir, "*.log")))

pat_ns = re.compile(
    r"ckpt_name='([^']+)'.*beam_size=(\d+).*beam_len_penalty=([0-9.]+).*transition_logit_scale=([0-9.]+)"
)
pat_valid = re.compile(r"\[VALID\].*Full_F1:\s*([0-9.]+)\s*\|\s*Full_Pairs_F1:\s*([0-9.]+)")

best = None
for f in files:
    ckpt = None
    bs = None
    lp = None
    ts = None
    full_f1 = None
    full_pairs = None

    with open(f, "r", encoding="utf-8") as fh:
        for line in fh:
            if ckpt is None:
                m = pat_ns.search(line)
                if m:
                    ckpt = m.group(1)
                    bs = int(m.group(2))
                    lp = float(m.group(3))
                    ts = float(m.group(4))

            m2 = pat_valid.search(line)
            if m2:
                full_f1 = float(m2.group(1))
                full_pairs = float(m2.group(2))

    if ckpt is None or full_f1 is None or full_pairs is None:
        continue

    item = (full_pairs, full_f1, ckpt, bs, lp, ts, f)
    if (best is None) or (item[0] > best[0]) or (item[0] == best[0] and item[1] > best[1]):
        best = item

if best is None:
    raise SystemExit("No valid records found in VALID logs.")

print("BEST_FULL_PAIRS_F1", best[0])
print("BEST_FULL_F1", best[1])
print("BEST_CKPT", best[2])
print("BEST_BEAM_SIZE", best[3])
print("BEST_LEN_PENALTY", best[4])
print("BEST_TRANSITION_SCALE", best[5])
print("BEST_LOG", best[6])

with open("/tmp/yelp_valid_best.env", "w", encoding="utf-8") as w:
    w.write(f"CKPT={best[2]}\n")
    w.write(f"BS={best[3]}\n")
    w.write(f"LP={best[4]}\n")
    w.write(f"TS={best[5]}\n")
PY

if [[ ! -f "$BEST_ENV_FILE" ]]; then
  echo "[ERROR] Best env file not found: $BEST_ENV_FILE"
  exit 1
fi

source "$BEST_ENV_FILE"

echo "[STEP] Run TEST with selected config"
echo "[BEST] ckpt=$CKPT beam_size=$BS beam_len_penalty=$LP transition_logit_scale=$TS"
python main.py \
  --mode test \
  --ckpt_name "$CKPT" \
  --beam_size "$BS" \
  --beam_len_penalty "$LP" \
  --transition_logit_scale "$TS" \
  --log \
  --log_path "$TEST_LOG_DIR" \
  "${BASE_ARGS[@]}"

LATEST_TEST_LOG="$(ls -t "$TEST_LOG_DIR"/*.log | head -n 1)"
echo "[DONE] FINAL_TEST_LOG=$LATEST_TEST_LOG"
grep -E "\[TEST\]" "$LATEST_TEST_LOG" || true
