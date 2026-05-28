"""
一键双数据集实验脚本：对 Yelp 和 Foursquare 数据集顺序执行完整的三阶段流程：
  Stage A: 超参调优 (Optuna)
  Stage B: 最终模型训练 + 测试
  Stage C: 消融实验 + 超参敏感性实验

用法示例：
  # 跑全部两个数据集
  python run_both_datasets_pipeline.py

  # 只跑 Yelp
  python run_both_datasets_pipeline.py --datasets yelp

  # 只跑 Foursquare
  python run_both_datasets_pipeline.py --datasets foursquare

  # 跳过调优阶段，直接使用 base_* 参数进行训练
  python run_both_datasets_pipeline.py --skip_tuning 1

服务器路径示例（按需覆盖默认值）：
  python run_both_datasets_pipeline.py \\
    --yelp_data_split_path /root/autodl-tmp/MyCrossCity/Yelp/spottrip_baseline_split.pkl \\
    --yelp_ori_data ../../Yelp/home.txt \\
    --yelp_dst_data ../../Yelp/oot.txt \\
    --yelp_trans_data ../../Yelp/travel.txt \\
    --yelp_ori_data_enriched ../../Yelp/extendData/enriched_home.txt \\
    --yelp_dst_data_enriched ../../Yelp/extendData/enriched_oot.txt \\
    --yelp_save_path ../../Yelp/model_save_new \\
    --fsq_data_split_path ../../Foursquare/data_split_new.pkl \\
    --fsq_ori_data ../../Foursquare/home.txt \\
    --fsq_dst_data ../../Foursquare/oot.txt \\
    --fsq_trans_data ../../Foursquare/travel.txt \\
    --fsq_ori_data_enriched ../../Foursquare/extendData/enriched_home.txt \\
    --fsq_dst_data_enriched ../../Foursquare/extendData/enriched_oot.txt \\
    --fsq_save_path ../../Foursquare/model_save_new
"""

import argparse
import glob
import json
import os
import shlex
import subprocess
import sys
from datetime import datetime
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# 通用工具函数（与 run_yelp_soft_pipeline.py 保持一致）
# ---------------------------------------------------------------------------

def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _shell_join(cmd: List[str]) -> str:
    return " ".join(shlex.quote(x) for x in cmd)


def _run_and_log(cmd: List[str], cwd: str, log_file: str) -> int:
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    with open(log_file, "w", encoding="utf-8") as f:
        f.write(f"[{_now()}] CWD: {cwd}\n")
        f.write(f"[{_now()}] CMD: {_shell_join(cmd)}\n\n")
        f.flush()

        proc = subprocess.Popen(
            cmd,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        assert proc.stdout is not None
        for line in proc.stdout:
            f.write(line)
        proc.wait()
        f.write(f"\n[{_now()}] EXIT_CODE: {proc.returncode}\n")
        return int(proc.returncode)


def _find_named_log(log_dir: str, run_name: str, seed: int) -> Optional[str]:
    pattern = os.path.join(log_dir, f"* {run_name}({seed}).log")
    files = glob.glob(pattern)
    if not files:
        return None
    files.sort(key=os.path.getmtime, reverse=True)
    return files[0]


def _read_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _to_str(v) -> str:
    return str(v)


def _get_best_param(best_params: Dict, key: str, default):
    return best_params.get(key, default)


# ---------------------------------------------------------------------------
# 报告写入
# ---------------------------------------------------------------------------

def _write_ds_section(lines: List[str], ds_name: str, ds_result: Dict):
    lines.append(f"## {ds_name} — 阶段汇总")
    lines.append("")
    lines.append("| 阶段 | 状态 | 日志 |")
    lines.append("|---|---|---|")
    for st in ds_result.get("stages", []):
        lines.append(f"| {st['name']} | {st['status']} | {st['log_file']} |")
    lines.append("")

    lines.append(f"## {ds_name} — 命令记录")
    lines.append("")
    for st in ds_result.get("stages", []):
        lines.append(f"### {st['name']}")
        lines.append("")
        lines.append("```bash")
        lines.append(st.get("command", "N/A"))
        lines.append("```")
        lines.append("")

    lines.append(f"## {ds_name} — 产出物")
    lines.append("")
    for k, v in ds_result.get("artifacts", {}).items():
        lines.append(f"- {k}: {v}")
    lines.append("")


def _write_combined_report(report_path: str, payload: Dict):
    lines = []
    lines.append("# Both Datasets Pipeline Report")
    lines.append("")
    lines.append(f"- 开始时间: {payload['start_time']}")
    lines.append(f"- 结束时间: {payload['end_time']}")
    lines.append(f"- 整体状态: {payload['status']}")
    lines.append("")

    for ds_key, ds_name in [("yelp", "Yelp"), ("foursquare", "Foursquare")]:
        if ds_key in payload:
            _write_ds_section(lines, ds_name, payload[ds_key])

    lines.append("## 说明")
    lines.append("")
    lines.append("- 两个数据集顺序运行，一个失败不影响另一个继续执行。")
    lines.append("- 调优阶段仅保存可训练参数（存储高效）；最终训练阶段保存完整 checkpoint。")

    os.makedirs(os.path.dirname(os.path.abspath(report_path)), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# 单数据集运行逻辑（抽象自 run_yelp_soft_pipeline.py 的 main）
# ---------------------------------------------------------------------------

def _run_single_dataset(args, ds_cfg: Dict) -> Dict:
    """
    对单个数据集执行三阶段流程。
    ds_cfg 字段: dataset_name, pipeline_name, data_split_path, save_path,
                 ori_data, dst_data, trans_data, ori_data_enriched,
                 dst_data_enriched
    返回: {'status': 'ok'|'failed', 'stages': [...], 'artifacts': {...}}
    """
    pipeline_name = ds_cfg["pipeline_name"]
    run_root      = os.path.join(args.workspace_dir, "pipeline_runs", pipeline_name)
    tuning_dir    = os.path.join(run_root, "tuning")
    final_dir     = os.path.join(run_root, "final_train")
    ablation_dir  = os.path.join(run_root, "ablation")

    os.makedirs(tuning_dir, exist_ok=True)
    os.makedirs(final_dir, exist_ok=True)
    os.makedirs(ablation_dir, exist_ok=True)

    stage_records: List[Dict] = []
    artifacts: Dict = {}

    study_name = f"{pipeline_name}_tune"
    best_json  = os.path.join(tuning_dir, f"{study_name}_best_params.json")
    best_params: Dict = {}

    # ── Stage A: 超参调优 ────────────────────────────────────────────────────
    if args.skip_tuning == 0:
        tune_log = os.path.join(tuning_dir, "stage_tuning.log")
        tune_cmd = [
            args.python_exec,
            "./optuna_tune.py",
            "--study_name", study_name,
            "--output_dir", tuning_dir,
            "--n_trials", _to_str(args.tune_trials),
            "--timeout", _to_str(args.tune_timeout),
            "--objective_metric", args.tune_objective_metric,
            "--objective_combo_beta_fixed", _to_str(args.objective_combo_beta_fixed),
            "--seed", _to_str(args.seed),
            "--main_script", "./main.py",
            "--dataset_name", ds_cfg["dataset_name"],
            "--data_split_path", ds_cfg["data_split_path"],
            "--save_path", ds_cfg["save_path"],
            "--ori_data", ds_cfg["ori_data"],
            "--dst_data", ds_cfg["dst_data"],
            "--trans_data", ds_cfg["trans_data"],
            "--ori_data_enriched", ds_cfg["ori_data_enriched"],
            "--dst_data_enriched", ds_cfg["dst_data_enriched"],
            "--device", args.device,
            "--epoch", _to_str(args.epoch),
            "--stop_epoch", _to_str(args.stop_epoch),
            "--use_enriched_data", _to_str(args.use_enriched_data),
            "--semantic_backend", args.semantic_backend,
            "--qwen_strict", _to_str(args.qwen_strict),
            "--llm_model_name", args.llm_model_name,
            "--llm_fallback_names", args.llm_fallback_names,
            "--llm_dtype", args.llm_dtype,
            "--use_mamba_backbone", _to_str(args.use_mamba_backbone),
            "--mamba_strict", _to_str(args.mamba_strict),
            "--mamba_d_conv", _to_str(args.mamba_d_conv),
            "--seq_num_layers_choices", args.seq_num_layers_choices,
            "--mamba_d_state_choices", args.mamba_d_state_choices,
            "--mamba_expand_choices", args.mamba_expand_choices,
            "--pair_max_future_choices", args.pair_max_future_choices,
            "--beam_size_choices", args.beam_size_choices,
            "--lr_min", _to_str(args.lr_min),
            "--lr_max", _to_str(args.lr_max),
            "--lambda_pair_min", _to_str(args.lambda_pair_min),
            "--lambda_pair_max", _to_str(args.lambda_pair_max),
            "--lambda_transition_min", _to_str(args.lambda_transition_min),
            "--lambda_transition_max", _to_str(args.lambda_transition_max),
            "--transition_logit_scale_min", _to_str(args.transition_logit_scale_min),
            "--transition_logit_scale_max", _to_str(args.transition_logit_scale_max),
            "--beam_len_penalty_min", _to_str(args.beam_len_penalty_min),
            "--beam_len_penalty_max", _to_str(args.beam_len_penalty_max),
            "--dropout_min", _to_str(args.dropout_min),
            "--dropout_max", _to_str(args.dropout_max),
            "--temperature_min", _to_str(args.temperature_min),
            "--temperature_max", _to_str(args.temperature_max),
            "--combo_beta_min", _to_str(args.combo_beta_min),
            "--combo_beta_max", _to_str(args.combo_beta_max),
            "--enable_pairwise_loss", "1",
            "--save_trainable_only", "1",
            "--save_optimizer_state", "0",
            "--save_dual_best", "1",
            "--use_f1_floor_filter", _to_str(args.use_f1_floor_filter),
            "--f1_floor_margin", _to_str(args.f1_floor_margin),
            "--use_no_repeat_mask", _to_str(args.use_no_repeat_mask),
            "--early_stop_metric", args.early_stop_metric_tune,
            "--decode_constraint_mode", args.decode_constraint_mode,
            "--enforce_start_end_constraints", _to_str(args.enforce_start_end_constraints),
            "--soft_constraint_scale_min", _to_str(args.soft_constraint_scale_min),
            "--soft_constraint_scale_max", _to_str(args.soft_constraint_scale_max),
            "--soft_constraint_dist_emb_dim_choices", args.soft_constraint_dist_emb_dim_choices,
        ]

        rc = _run_and_log(tune_cmd, cwd=args.workspace_dir, log_file=tune_log)
        stage_records.append({
            "name": "tuning",
            "status": "ok" if rc == 0 else f"failed({rc})",
            "log_file": tune_log,
            "command": _shell_join(tune_cmd),
        })
        if rc != 0:
            return {"status": "failed", "stages": stage_records, "artifacts": artifacts}

        artifacts["tuning_dir"] = tuning_dir
        artifacts["best_params_json"] = best_json
    else:
        stage_records.append({"name": "tuning", "status": "skipped", "log_file": "N/A", "command": "N/A"})

    if os.path.exists(best_json):
        best_info   = _read_json(best_json)
        best_params = best_info.get("best_params", {})
        artifacts["best_trial_log"]       = best_info.get("best_user_attrs", {}).get("log_file", "")
        artifacts["best_trial_model_dir"] = best_info.get("best_user_attrs", {}).get("model_dir", "")

    # ── Stage B: 最终训练 + 测试 ─────────────────────────────────────────────
    final_name     = f"{pipeline_name}_final_s{args.seed}"
    final_logs_dir = os.path.join(final_dir, "logs")
    os.makedirs(final_logs_dir, exist_ok=True)

    if args.skip_final_train == 0:
        final_log = os.path.join(final_dir, "stage_final_train.log")
        final_cmd = [
            args.python_exec,
            "./main.py",
            "--mode", "train",
            "--log",
            "--name", final_name,
            "--dataset_name", ds_cfg["dataset_name"],
            "--data_split_path", ds_cfg["data_split_path"],
            "--save_path", ds_cfg["save_path"],
            "--ori_data", ds_cfg["ori_data"],
            "--dst_data", ds_cfg["dst_data"],
            "--trans_data", ds_cfg["trans_data"],
            "--ori_data_enriched", ds_cfg["ori_data_enriched"],
            "--dst_data_enriched", ds_cfg["dst_data_enriched"],
            "--device", args.device,
            "--seed", _to_str(args.seed),
            "--semantic_backend", args.semantic_backend,
            "--llm_model_name", args.llm_model_name,
            "--llm_fallback_names", args.llm_fallback_names,
            "--llm_dtype", args.llm_dtype,
            "--use_mamba_backbone", _to_str(args.use_mamba_backbone),
            "--seq_num_layers", _to_str(_get_best_param(best_params, "seq_num_layers", args.base_seq_num_layers)),
            "--mamba_d_state", _to_str(_get_best_param(best_params, "mamba_d_state", args.base_mamba_d_state)),
            "--mamba_d_conv", _to_str(args.mamba_d_conv),
            "--mamba_expand", _to_str(_get_best_param(best_params, "mamba_expand", args.base_mamba_expand)),
            "--dropout", _to_str(_get_best_param(best_params, "dropout", args.base_dropout)),
            "--temperature", _to_str(_get_best_param(best_params, "temperature", args.base_temperature)),
            "--enable_pairwise_loss", "1",
            "--lambda_pair", _to_str(_get_best_param(best_params, "lambda_pair", args.base_lambda_pair)),
            "--pair_max_future", _to_str(_get_best_param(best_params, "pair_max_future", args.base_pair_max_future)),
            "--lambda_transition", _to_str(_get_best_param(best_params, "lambda_transition", args.base_lambda_transition)),
            "--transition_logit_scale", _to_str(_get_best_param(best_params, "transition_logit_scale", args.base_transition_logit_scale)),
            "--use_beam_search", "1",
            "--beam_size", _to_str(_get_best_param(best_params, "beam_size", args.base_beam_size)),
            "--beam_len_penalty", _to_str(_get_best_param(best_params, "beam_len_penalty", args.base_beam_len_penalty)),
            "--use_no_repeat_mask", _to_str(args.use_no_repeat_mask),
            "--pref_factor_k", _to_str(args.pref_factor_k),
            "--lambda_decouple", _to_str(args.lambda_decouple),
            "--lambda_semantic", _to_str(args.lambda_semantic),
            "--eta_fixed", _to_str(args.eta_fixed),
            "--decode_constraint_mode", args.decode_constraint_mode,
            "--soft_constraint_scale", _to_str(_get_best_param(best_params, "soft_constraint_scale", args.base_soft_constraint_scale)),
            "--soft_constraint_dist_emb_dim", _to_str(_get_best_param(best_params, "soft_constraint_dist_emb_dim", args.base_soft_constraint_dist_emb_dim)),
            "--enforce_start_end_constraints", _to_str(args.enforce_start_end_constraints),
            "--epoch", _to_str(args.epoch),
            "--stop_epoch", _to_str(args.stop_epoch),
            "--lr", _to_str(_get_best_param(best_params, "lr", args.base_lr)),
            "--l2", _to_str(args.l2),
            "--lr_dc", _to_str(args.lr_dc),
            "--lr_dc_step", _to_str(args.lr_dc_step),
            "--save_trainable_only", "0",
            "--save_optimizer_state", "0",
            "--run_final_test_after_train",
            "--early_stop_metric", args.early_stop_metric_final,
            "--combo_beta", _to_str(_get_best_param(best_params, "combo_beta", args.base_combo_beta)),
            "--use_f1_floor_filter", _to_str(args.use_f1_floor_filter),
            "--f1_floor_margin", _to_str(args.f1_floor_margin),
            "--save_dual_best", "1",
            "--log_path", final_logs_dir,
            "--best_save",
        ]

        if args.use_enriched_data == 1:
            final_cmd.append("--use_enriched_data")
        if args.qwen_strict == 1:
            final_cmd.append("--qwen_strict")
        if args.mamba_strict == 1:
            final_cmd.append("--mamba_strict")

        rc = _run_and_log(final_cmd, cwd=args.workspace_dir, log_file=final_log)
        stage_records.append({
            "name": "final_train",
            "status": "ok" if rc == 0 else f"failed({rc})",
            "log_file": final_log,
            "command": _shell_join(final_cmd),
        })
        if rc != 0:
            return {"status": "failed", "stages": stage_records, "artifacts": artifacts}

        named_log = _find_named_log(final_logs_dir, final_name, args.seed)
        artifacts["final_stage_log"] = final_log
        artifacts["final_model_log"] = named_log if named_log else ""
        artifacts["final_model_dir"] = os.path.abspath(
            os.path.join(args.workspace_dir, ds_cfg["save_path"], final_name)
        )
    else:
        stage_records.append({"name": "final_train", "status": "skipped", "log_file": "N/A", "command": "N/A"})

    # ── Stage C: 消融 + 超参实验 ──────────────────────────────────────────────
    batch_output_dir = os.path.join(ablation_dir, "batch_output")
    os.makedirs(batch_output_dir, exist_ok=True)
    batch_exp_prefix = f"{pipeline_name}_ablation"

    if args.skip_ablation == 0:
        ablation_log = os.path.join(ablation_dir, "stage_ablation.log")
        batch_cmd = [
            args.python_exec,
            "./batch_ablation_hparam.py",
            "--dataset_name", ds_cfg["dataset_name"],
            "--run_mode", "train",
            "--seed", _to_str(args.seed),
            "--device", args.device,
            "--output_dir", batch_output_dir,
            "--exp_prefix", batch_exp_prefix,
            "--log_path", final_logs_dir,
            "--data_split_path", ds_cfg["data_split_path"],
            "--save_path", ds_cfg["save_path"],
            "--ori_data", ds_cfg["ori_data"],
            "--dst_data", ds_cfg["dst_data"],
            "--trans_data", ds_cfg["trans_data"],
            "--ori_data_enriched", ds_cfg["ori_data_enriched"],
            "--dst_data_enriched", ds_cfg["dst_data_enriched"],
            "--semantic_backend", args.semantic_backend,
            "--llm_model_name", args.llm_model_name,
            "--llm_fallback_names", args.llm_fallback_names,
            "--llm_dtype", args.llm_dtype,
            "--qwen_strict", _to_str(args.qwen_strict),
            "--use_enriched_data", _to_str(args.use_enriched_data),
            "--use_mamba_backbone", _to_str(args.use_mamba_backbone),
            "--mamba_strict", _to_str(args.mamba_strict),
            "--seq_num_layers", _to_str(_get_best_param(best_params, "seq_num_layers", args.base_seq_num_layers)),
            "--mamba_d_state", _to_str(_get_best_param(best_params, "mamba_d_state", args.base_mamba_d_state)),
            "--mamba_d_conv", _to_str(args.mamba_d_conv),
            "--mamba_expand", _to_str(_get_best_param(best_params, "mamba_expand", args.base_mamba_expand)),
            "--dropout", _to_str(_get_best_param(best_params, "dropout", args.base_dropout)),
            "--temperature", _to_str(_get_best_param(best_params, "temperature", args.base_temperature)),
            "--enable_pairwise_loss", "1",
            "--lambda_pair", _to_str(_get_best_param(best_params, "lambda_pair", args.base_lambda_pair)),
            "--pair_max_future", _to_str(_get_best_param(best_params, "pair_max_future", args.base_pair_max_future)),
            "--lambda_transition", _to_str(_get_best_param(best_params, "lambda_transition", args.base_lambda_transition)),
            "--transition_logit_scale", _to_str(_get_best_param(best_params, "transition_logit_scale", args.base_transition_logit_scale)),
            "--use_beam_search", "1",
            "--beam_size", _to_str(_get_best_param(best_params, "beam_size", args.base_beam_size)),
            "--beam_len_penalty", _to_str(_get_best_param(best_params, "beam_len_penalty", args.base_beam_len_penalty)),
            "--use_no_repeat_mask", _to_str(args.use_no_repeat_mask),
            "--pref_factor_k", _to_str(args.pref_factor_k),
            "--lambda_decouple", _to_str(args.lambda_decouple),
            "--lambda_semantic", _to_str(args.lambda_semantic),
            "--eta_fixed", _to_str(args.eta_fixed),
            "--decode_constraint_mode", args.decode_constraint_mode,
            "--soft_constraint_scale", _to_str(_get_best_param(best_params, "soft_constraint_scale", args.base_soft_constraint_scale)),
            "--soft_constraint_dist_emb_dim", _to_str(_get_best_param(best_params, "soft_constraint_dist_emb_dim", args.base_soft_constraint_dist_emb_dim)),
            "--enforce_start_end_constraints", _to_str(args.enforce_start_end_constraints),
            "--epoch", _to_str(args.epoch),
            "--stop_epoch", _to_str(args.stop_epoch),
            "--lr", _to_str(_get_best_param(best_params, "lr", args.base_lr)),
            "--l2", _to_str(args.l2),
            "--lr_dc", _to_str(args.lr_dc),
            "--lr_dc_step", _to_str(args.lr_dc_step),
            "--write_incremental_csv", "1",
            "--save_trainable_only", "1",
            "--early_stop_metric", args.early_stop_metric_final,
            "--combo_beta", _to_str(_get_best_param(best_params, "combo_beta", args.base_combo_beta)),
            "--use_f1_floor_filter", _to_str(args.use_f1_floor_filter),
            "--f1_floor_margin", _to_str(args.f1_floor_margin),
            "--save_dual_best", "1",
            "--append_full_reference", "0",
            "--hparam_seq_num_layers", args.hparam_seq_num_layers,
            "--hparam_lambda_pair", args.hparam_lambda_pair,
            "--hparam_transition_strength", args.hparam_transition_strength,
            "--hparam_eta_fixed", args.hparam_eta_fixed,
        ]

        rc = _run_and_log(batch_cmd, cwd=args.workspace_dir, log_file=ablation_log)
        stage_records.append({
            "name": "ablation_hparam",
            "status": "ok" if rc == 0 else f"failed({rc})",
            "log_file": ablation_log,
            "command": _shell_join(batch_cmd),
        })
        if rc != 0:
            return {"status": "failed", "stages": stage_records, "artifacts": artifacts}

        artifacts["ablation_output_dir"] = batch_output_dir
        artifacts["ablation_summary_latest_csv"] = os.path.join(
            batch_output_dir, f"summary_{batch_exp_prefix}_latest.csv"
        )
    else:
        stage_records.append({"name": "ablation_hparam", "status": "skipped", "log_file": "N/A", "command": "N/A"})

    return {"status": "ok", "stages": stage_records, "artifacts": artifacts}


# ---------------------------------------------------------------------------
# 数据集配置构建
# ---------------------------------------------------------------------------

def _make_yelp_config(args) -> Dict:
    return {
        "dataset_name":      "Yelp",
        "pipeline_name":     args.yelp_pipeline_name,
        "data_split_path":   args.yelp_data_split_path,
        "save_path":         args.yelp_save_path,
        "ori_data":          args.yelp_ori_data,
        "dst_data":          args.yelp_dst_data,
        "trans_data":        args.yelp_trans_data,
        "ori_data_enriched": args.yelp_ori_data_enriched,
        "dst_data_enriched": args.yelp_dst_data_enriched,
    }


def _make_fsq_config(args) -> Dict:
    return {
        "dataset_name":      "Foursquare",
        "pipeline_name":     args.fsq_pipeline_name,
        "data_split_path":   args.fsq_data_split_path,
        "save_path":         args.fsq_save_path,
        "ori_data":          args.fsq_ori_data,
        "dst_data":          args.fsq_dst_data,
        "trans_data":        args.fsq_trans_data,
        "ori_data_enriched": args.fsq_ori_data_enriched,
        "dst_data_enriched": args.fsq_dst_data_enriched,
    }


# ---------------------------------------------------------------------------
# 参数解析
# ---------------------------------------------------------------------------

def build_parser():
    parser = argparse.ArgumentParser(
        description="一键双数据集实验脚本（Yelp + Foursquare）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # 运行控制
    parser.add_argument("--python_exec",   type=str, default=sys.executable)
    parser.add_argument("--workspace_dir", type=str, default=os.path.dirname(os.path.abspath(__file__)))
    parser.add_argument("--datasets",      type=str, default="yelp,foursquare",
                        help="要运行的数据集，逗号分隔，可选: yelp, foursquare")
    parser.add_argument("--seed",   type=int, default=2050)
    parser.add_argument("--device", type=str, default="cuda:0")

    # ── Yelp 数据集路径 ────────────────────────────────────────────────────
    parser.add_argument("--yelp_pipeline_name",     type=str, default="yelp_both_pipeline_v1")
    parser.add_argument("--yelp_data_split_path",   type=str,
                        default="/root/autodl-tmp/MyCrossCity/Yelp/spottrip_baseline_split.pkl")
    parser.add_argument("--yelp_save_path",         type=str, default="../../Yelp/model_save_new")
    parser.add_argument("--yelp_ori_data",          type=str, default="../../Yelp/home.txt")
    parser.add_argument("--yelp_dst_data",          type=str, default="../../Yelp/oot.txt")
    parser.add_argument("--yelp_trans_data",        type=str, default="../../Yelp/travel.txt")
    parser.add_argument("--yelp_ori_data_enriched", type=str, default="../../Yelp/extendData/enriched_home.txt")
    parser.add_argument("--yelp_dst_data_enriched", type=str, default="../../Yelp/extendData/enriched_oot.txt")

    # ── Foursquare 数据集路径 ──────────────────────────────────────────────
    parser.add_argument("--fsq_pipeline_name",     type=str, default="fsq_both_pipeline_v1")
    parser.add_argument("--fsq_data_split_path",   type=str, default="../../Foursquare/data_split_new.pkl")
    parser.add_argument("--fsq_save_path",         type=str, default="../../Foursquare/model_save_new")
    parser.add_argument("--fsq_ori_data",          type=str, default="../../Foursquare/home.txt")
    parser.add_argument("--fsq_dst_data",          type=str, default="../../Foursquare/oot.txt")
    parser.add_argument("--fsq_trans_data",        type=str, default="../../Foursquare/travel.txt")
    parser.add_argument("--fsq_ori_data_enriched", type=str, default="../../Foursquare/extendData/enriched_home.txt")
    parser.add_argument("--fsq_dst_data_enriched", type=str, default="../../Foursquare/extendData/enriched_oot.txt")

    # ── 公共模型 / 训练参数（两个数据集共用）─────────────────────────────
    parser.add_argument("--use_enriched_data",    type=int, default=1, choices=[0, 1])
    parser.add_argument("--semantic_backend",     type=str, default="qwen", choices=["qwen", "fallback"])
    parser.add_argument("--llm_model_name",       type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--llm_fallback_names",   type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--llm_dtype",            type=str, default="bfloat16",
                        choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--qwen_strict",          type=int, default=1, choices=[0, 1])
    parser.add_argument("--use_mamba_backbone",   type=int, default=1, choices=[0, 1])
    parser.add_argument("--mamba_strict",         type=int, default=1, choices=[0, 1])
    parser.add_argument("--mamba_d_conv",         type=int, default=4)
    parser.add_argument("--pref_factor_k",        type=int, default=4)
    parser.add_argument("--lambda_decouple",      type=float, default=0.1)
    parser.add_argument("--lambda_semantic",      type=float, default=0.1)
    parser.add_argument("--eta_fixed",            type=float, default=-1.0)
    parser.add_argument("--use_no_repeat_mask",   type=int, default=1, choices=[0, 1])
    parser.add_argument("--epoch",                type=int, default=30)
    parser.add_argument("--stop_epoch",           type=int, default=8)
    parser.add_argument("--l2",                   type=float, default=1e-5)
    parser.add_argument("--lr_dc",                type=float, default=0.3)
    parser.add_argument("--lr_dc_step",           type=int, default=8)
    parser.add_argument("--f1_floor_margin",      type=float, default=0.002)
    parser.add_argument("--use_f1_floor_filter",  type=int, default=0, choices=[0, 1])

    parser.add_argument("--decode_constraint_mode",          type=str, default="soft",
                        choices=["hard", "soft"])
    parser.add_argument("--enforce_start_end_constraints",   type=int, default=1, choices=[0, 1])

    # ── 调优参数 ───────────────────────────────────────────────────────────
    parser.add_argument("--tune_trials",              type=int, default=12)
    parser.add_argument("--tune_timeout",             type=int, default=0)
    parser.add_argument("--tune_objective_metric",    type=str, default="full_combo",
                        choices=["pairs_f1", "combo", "f1", "full_f1", "full_pairs_f1", "full_combo"])
    parser.add_argument("--objective_combo_beta_fixed", type=float, default=4.0)
    parser.add_argument("--early_stop_metric_tune",   type=str, default="full_combo",
                        choices=["combo", "f1", "pairs_f1", "full_f1", "full_pairs_f1", "full_combo"])
    parser.add_argument("--early_stop_metric_final",  type=str, default="full_pairs_f1",
                        choices=["combo", "f1", "pairs_f1", "full_f1", "full_pairs_f1", "full_combo"])
    parser.add_argument("--combo_beta_min", type=float, default=1.5)
    parser.add_argument("--combo_beta_max", type=float, default=4.0)

    parser.add_argument("--lr_min",                       type=float, default=3e-4)
    parser.add_argument("--lr_max",                       type=float, default=5e-4)
    parser.add_argument("--lambda_pair_min",              type=float, default=0.03)
    parser.add_argument("--lambda_pair_max",              type=float, default=0.2)
    parser.add_argument("--lambda_transition_min",        type=float, default=0.05)
    parser.add_argument("--lambda_transition_max",        type=float, default=0.2)
    parser.add_argument("--transition_logit_scale_min",   type=float, default=0.05)
    parser.add_argument("--transition_logit_scale_max",   type=float, default=0.3)
    parser.add_argument("--beam_len_penalty_min",         type=float, default=0.25)
    parser.add_argument("--beam_len_penalty_max",         type=float, default=0.6)
    parser.add_argument("--dropout_min",                  type=float, default=0.08)
    parser.add_argument("--dropout_max",                  type=float, default=0.2)
    parser.add_argument("--temperature_min",              type=float, default=0.06)
    parser.add_argument("--temperature_max",              type=float, default=0.12)
    parser.add_argument("--soft_constraint_scale_min",    type=float, default=0.0)
    parser.add_argument("--soft_constraint_scale_max",    type=float, default=0.08)

    parser.add_argument("--pair_max_future_choices",            type=str, default="4,6")
    parser.add_argument("--beam_size_choices",                  type=str, default="3")
    parser.add_argument("--seq_num_layers_choices",             type=str, default="2,3")
    parser.add_argument("--mamba_d_state_choices",              type=str, default="32")
    parser.add_argument("--mamba_expand_choices",               type=str, default="2")
    parser.add_argument("--soft_constraint_dist_emb_dim_choices", type=str, default="16,32")

    # ── 默认超参（调优失败时的 fallback）──────────────────────────────────
    parser.add_argument("--base_lr",                        type=float, default=0.0004421206523113628)
    parser.add_argument("--base_lambda_pair",               type=float, default=0.07415356011717347)
    parser.add_argument("--base_pair_max_future",           type=int,   default=4)
    parser.add_argument("--base_lambda_transition",         type=float, default=0.11150286871666779)
    parser.add_argument("--base_transition_logit_scale",    type=float, default=0.1134787841314744)
    parser.add_argument("--base_beam_size",                 type=int,   default=3)
    parser.add_argument("--base_beam_len_penalty",          type=float, default=0.4422800705989885)
    parser.add_argument("--base_combo_beta",                type=float, default=2.2634933830964377)
    parser.add_argument("--base_dropout",                   type=float, default=0.13266038732619795)
    parser.add_argument("--base_temperature",               type=float, default=0.09096176093766724)
    parser.add_argument("--base_seq_num_layers",            type=int,   default=2)
    parser.add_argument("--base_mamba_d_state",             type=int,   default=32)
    parser.add_argument("--base_mamba_expand",              type=int,   default=2)
    parser.add_argument("--base_soft_constraint_scale",     type=float, default=0.02)
    parser.add_argument("--base_soft_constraint_dist_emb_dim", type=int, default=32)

    # ── 消融 / 超参扫描维度 ───────────────────────────────────────────────
    parser.add_argument("--hparam_seq_num_layers",     type=str, default="1,2,3")
    parser.add_argument("--hparam_lambda_pair",       type=str, default="0.25,0.5,1.0,1.5,2.0")
    parser.add_argument("--hparam_transition_strength", type=str, default="1.0")
    parser.add_argument("--hparam_eta_fixed",          type=str, default="0.0,0.2,0.4,0.6,0.8")

    # ── 跳过控制 ──────────────────────────────────────────────────────────
    parser.add_argument("--skip_tuning",      type=int, default=0, choices=[0, 1])
    parser.add_argument("--skip_final_train", type=int, default=0, choices=[0, 1])
    parser.add_argument("--skip_ablation",    type=int, default=0, choices=[0, 1])

    return parser


# ---------------------------------------------------------------------------
# 主函数
# ---------------------------------------------------------------------------

def main():
    args = build_parser().parse_args()
    start_time = _now()

    datasets_to_run = [d.strip().lower() for d in args.datasets.split(",") if d.strip()]
    if not datasets_to_run:
        print("[ERROR] --datasets 为空，至少指定一个数据集（yelp 或 foursquare）")
        sys.exit(1)

    ds_key_map = {
        "yelp":        ("yelp",        _make_yelp_config),
        "foursquare":  ("foursquare",  _make_fsq_config),
        "fsq":         ("foursquare",  _make_fsq_config),
    }

    overall_status = "ok"
    results: Dict[str, Dict] = {}

    for ds_token in datasets_to_run:
        if ds_token not in ds_key_map:
            print(f"[WARN] 未知数据集 '{ds_token}'，跳过。可用: yelp, foursquare")
            continue

        result_key, cfg_fn = ds_key_map[ds_token]
        ds_cfg = cfg_fn(args)
        print(f"\n{'=' * 70}")
        print(f"[{_now()}]  开始处理数据集: {ds_cfg['dataset_name']}")
        print(f"{'=' * 70}\n")

        result = _run_single_dataset(args, ds_cfg)
        results[result_key] = result

        if result["status"] != "ok":
            overall_status = "partial_failure"
            print(f"\n[WARN] {ds_cfg['dataset_name']} 执行失败，继续处理下一个数据集（如有）。\n")
        else:
            print(f"\n[{_now()}]  {ds_cfg['dataset_name']} 所有阶段完成。\n")

    # 写汇总报告
    report_path = os.path.join(
        args.workspace_dir,
        "pipeline_runs",
        f"both_datasets_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
    )
    payload = {
        "start_time": start_time,
        "end_time":   _now(),
        "status":     overall_status,
        **results,
    }
    _write_combined_report(report_path, payload)

    print(f"\n{'=' * 70}")
    print(f"全部数据集处理完毕  |  整体状态: {overall_status}")
    print(f"汇总报告: {report_path}")
    for ds_key, ds_result in results.items():
        print(f"  [{ds_key}] status={ds_result['status']}")
        for k, v in ds_result.get("artifacts", {}).items():
            print(f"    {k}: {v}")
    print(f"{'=' * 70}\n")

    if overall_status != "ok":
        sys.exit(1)


if __name__ == "__main__":
    main()
