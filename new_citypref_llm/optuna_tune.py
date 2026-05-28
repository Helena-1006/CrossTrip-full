import argparse
import glob
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from datetime import datetime

import optuna


VAL_RE = re.compile(
    r"\[VAL\] Epoch (?P<epoch>\d+) F1: (?P<f1>[0-9.]+) \| Pairs_F1: (?P<pairs>[0-9.]+) "
    r"\| Full_F1: (?P<full_f1>[0-9.]+) \| Full_Pairs_F1: (?P<full_pairs>[0-9.]+)"
)


def parse_args():
    parser = argparse.ArgumentParser(description="Optuna tuner for new_citypref_llm")
    parser.add_argument("--n_trials", type=int, default=20)
    parser.add_argument("--timeout", type=int, default=0, help="seconds, 0 means no timeout")
    parser.add_argument("--study_name", type=str, default="citypref_optuna")
    parser.add_argument("--output_dir", type=str, default="./optuna_runs")
    parser.add_argument("--storage", type=str, default="")
    parser.add_argument("--objective_metric", type=str, default="pairs_f1",
                        choices=["pairs_f1", "combo", "f1", "full_f1", "full_pairs_f1", "full_combo"])
    parser.add_argument(
        "--objective_combo_beta_fixed",
        type=float,
        default=4.0,
        help="Fixed beta used only for Optuna objective scoring when objective_metric is combo/full_combo.",
    )
    parser.add_argument("--seed", type=int, default=2050)

    parser.add_argument("--main_script", type=str, default="./main.py")
    parser.add_argument("--dataset_name", type=str, default="Foursquare")
    parser.add_argument("--data_split_path", type=str, default="../../Foursquare/data_split_new.pkl")
    parser.add_argument("--save_path", type=str, default="../../Foursquare/model_save_new")
    parser.add_argument("--ori_data", type=str, default="")
    parser.add_argument("--dst_data", type=str, default="")
    parser.add_argument("--trans_data", type=str, default="")
    parser.add_argument("--ori_data_enriched", type=str, default="")
    parser.add_argument("--dst_data_enriched", type=str, default="")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--epoch", type=int, default=30)
    parser.add_argument("--stop_epoch", type=int, default=8)
    parser.add_argument("--split_strategy", type=str, default="legacy", choices=["legacy", "pair_robust"])
    parser.add_argument("--split_singleton_to_train", type=int, default=1, choices=[0, 1])

    parser.add_argument("--use_enriched_data", type=int, default=1, choices=[0, 1])
    parser.add_argument("--semantic_backend", type=str, default="qwen", choices=["qwen", "fallback"])
    parser.add_argument("--qwen_strict", type=int, default=1, choices=[0, 1])
    parser.add_argument("--qwen_train_soft_prompt", type=int, default=0, choices=[0, 1])
    parser.add_argument("--llm_model_name", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--llm_fallback_names", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--llm_dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"])

    parser.add_argument("--use_mamba_backbone", type=int, default=1, choices=[0, 1])
    parser.add_argument("--mamba_strict", type=int, default=1, choices=[0, 1])
    parser.add_argument("--seq_num_layers_choices", type=str, default="2,3")
    parser.add_argument("--mamba_d_state_choices", type=str, default="16,32")
    parser.add_argument("--mamba_d_conv", type=int, default=4)
    parser.add_argument("--mamba_expand_choices", type=str, default="2,3")

    parser.add_argument("--lr_min", type=float, default=1e-4)
    parser.add_argument("--lr_max", type=float, default=8e-4)
    parser.add_argument("--lambda_pair_min", type=float, default=0.1)
    parser.add_argument("--lambda_pair_max", type=float, default=1.5)
    parser.add_argument("--pair_max_future_choices", type=str, default="4,8,12")
    parser.add_argument("--enable_pairwise_loss", type=int, default=0, choices=[0, 1])
    parser.add_argument("--combo_beta_min", type=float, default=2.0)
    parser.add_argument("--combo_beta_max", type=float, default=8.0)
    parser.add_argument("--lambda_transition_min", type=float, default=0.1)
    parser.add_argument("--lambda_transition_max", type=float, default=1.0)
    parser.add_argument("--transition_logit_scale_min", type=float, default=0.2)
    parser.add_argument("--transition_logit_scale_max", type=float, default=1.0)
    parser.add_argument("--beam_size_choices", type=str, default="3,5")
    parser.add_argument("--beam_len_penalty_min", type=float, default=0.0)
    parser.add_argument("--beam_len_penalty_max", type=float, default=0.6)
    parser.add_argument("--dropout_min", type=float, default=0.05)
    parser.add_argument("--dropout_max", type=float, default=0.25)
    parser.add_argument("--temperature_min", type=float, default=0.05)
    parser.add_argument("--temperature_max", type=float, default=0.12)

    parser.add_argument("--save_dual_best", type=int, default=0, choices=[0, 1])
    parser.add_argument("--use_f1_floor_filter", type=int, default=0, choices=[0, 1])
    parser.add_argument("--f1_floor_margin", type=float, default=0.002)

    parser.add_argument("--copy_best_artifacts", type=int, default=1, choices=[0, 1])
    parser.add_argument("--save_trainable_only", type=int, default=1, choices=[0, 1])
    parser.add_argument("--save_optimizer_state", type=int, default=0, choices=[0, 1])
    parser.add_argument("--use_no_repeat_mask", type=int, default=1, choices=[0, 1])
    parser.add_argument("--early_stop_metric", type=str, default="combo",
                        choices=["combo", "f1", "pairs_f1", "full_f1", "full_pairs_f1", "full_combo"])

    parser.add_argument("--decode_constraint_mode", type=str, default="soft", choices=["hard", "soft"])
    parser.add_argument("--enforce_start_end_constraints", type=int, default=1, choices=[0, 1])
    parser.add_argument("--soft_constraint_scale_min", type=float, default=0.0)
    parser.add_argument("--soft_constraint_scale_max", type=float, default=0.2)
    parser.add_argument("--soft_constraint_dist_emb_dim_choices", type=str, default="16,32")
    return parser.parse_args()


def resolve_dataset_paths(args):
    name = str(args.dataset_name).strip().lower()
    if name == "yelp":
        base = "../../Yelp"
    elif name == "foursquare":
        base = "../../Foursquare"
    else:
        base = ""

    ori_data = args.ori_data if args.ori_data.strip() else (f"{base}/home.txt" if base else "")
    dst_data = args.dst_data if args.dst_data.strip() else (f"{base}/oot.txt" if base else "")
    trans_data = args.trans_data if args.trans_data.strip() else (f"{base}/travel.txt" if base else "")
    ori_data_enriched = (
        args.ori_data_enriched
        if args.ori_data_enriched.strip()
        else (f"{base}/extendData/enriched_home.txt" if base else "")
    )
    dst_data_enriched = (
        args.dst_data_enriched
        if args.dst_data_enriched.strip()
        else (f"{base}/extendData/enriched_oot.txt" if base else "")
    )

    missing = [
        key for key, value in {
            "ori_data": ori_data,
            "dst_data": dst_data,
            "trans_data": trans_data,
            "ori_data_enriched": ori_data_enriched,
            "dst_data_enriched": dst_data_enriched,
        }.items() if not value
    ]
    if missing:
        raise ValueError(
            f"Cannot resolve dataset paths for dataset_name={args.dataset_name}. "
            f"Please provide: {', '.join(missing)}"
        )

    return {
        "ori_data": ori_data,
        "dst_data": dst_data,
        "trans_data": trans_data,
        "ori_data_enriched": ori_data_enriched,
        "dst_data_enriched": dst_data_enriched,
    }


def parse_int_choices(raw):
    return [int(x.strip()) for x in raw.split(',') if x.strip()]


def bool_flag(v):
    return str(int(v))


def find_log_file(log_dir, trial_name, seed):
    pattern = os.path.join(log_dir, f"* {trial_name}({seed}).log")
    files = glob.glob(pattern)
    if not files:
        return None
    files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return files[0]


def parse_val_metrics(log_path):
    vals = []
    if log_path is None or (not os.path.exists(log_path)):
        return vals
    with open(log_path, 'r', encoding='utf-8') as f:
        for line in f:
            m = VAL_RE.search(line)
            if m is None:
                continue
            vals.append({
                "epoch": int(m.group("epoch")),
                "f1": float(m.group("f1")),
                "pairs_f1": float(m.group("pairs")),
                "full_f1": float(m.group("full_f1")),
                "full_pairs_f1": float(m.group("full_pairs")),
            })
    return vals


def get_objective_value(vals, metric, combo_beta_fixed):
    if not vals:
        return float("-inf"), -1, {}

    best_idx = -1
    best_v = float("-inf")
    for i, item in enumerate(vals):
        if metric == "pairs_f1":
            v = item["pairs_f1"]
        elif metric == "f1":
            v = item["f1"]
        elif metric == "full_f1":
            v = item["full_f1"]
        elif metric == "full_pairs_f1":
            v = item["full_pairs_f1"]
        elif metric == "full_combo":
            v = item["full_f1"] + combo_beta_fixed * item["full_pairs_f1"]
        else:
            v = item["f1"] + combo_beta_fixed * item["pairs_f1"]
        if v > best_v:
            best_v = v
            best_idx = i

    return best_v, vals[best_idx]["epoch"], vals[best_idx]


def run_trial(args, trial, output_dir, log_dir):
    trial_name = f"{args.study_name}_trial_{trial.number:03d}"
    data_paths = resolve_dataset_paths(args)

    seq_layers_choices = parse_int_choices(args.seq_num_layers_choices)
    mamba_state_choices = parse_int_choices(args.mamba_d_state_choices)
    mamba_expand_choices = parse_int_choices(args.mamba_expand_choices)
    pair_future_choices = parse_int_choices(args.pair_max_future_choices)
    beam_size_choices = parse_int_choices(args.beam_size_choices)
    soft_dist_choices = parse_int_choices(args.soft_constraint_dist_emb_dim_choices)
    if len(soft_dist_choices) == 0:
        soft_dist_choices = [32]

    if args.decode_constraint_mode == "soft":
        soft_constraint_scale = trial.suggest_float(
            "soft_constraint_scale", args.soft_constraint_scale_min, args.soft_constraint_scale_max
        )
    else:
        soft_constraint_scale = 0.0

    params = {
        "lr": trial.suggest_float("lr", args.lr_min, args.lr_max, log=True),
        "lambda_pair": trial.suggest_float("lambda_pair", args.lambda_pair_min, args.lambda_pair_max),
        "pair_max_future": trial.suggest_categorical("pair_max_future", pair_future_choices),
        "lambda_transition": trial.suggest_float("lambda_transition", args.lambda_transition_min, args.lambda_transition_max),
        "transition_logit_scale": trial.suggest_float("transition_logit_scale", args.transition_logit_scale_min, args.transition_logit_scale_max),
        "beam_size": trial.suggest_categorical("beam_size", beam_size_choices),
        "beam_len_penalty": trial.suggest_float("beam_len_penalty", args.beam_len_penalty_min, args.beam_len_penalty_max),
        "combo_beta": trial.suggest_float("combo_beta", args.combo_beta_min, args.combo_beta_max),
        "dropout": trial.suggest_float("dropout", args.dropout_min, args.dropout_max),
        "temperature": trial.suggest_float("temperature", args.temperature_min, args.temperature_max),
        "seq_num_layers": trial.suggest_categorical("seq_num_layers", seq_layers_choices),
        "mamba_d_state": trial.suggest_categorical("mamba_d_state", mamba_state_choices),
        "mamba_expand": trial.suggest_categorical("mamba_expand", mamba_expand_choices),
        "soft_constraint_scale": soft_constraint_scale,
        "soft_constraint_dist_emb_dim": trial.suggest_categorical("soft_constraint_dist_emb_dim", soft_dist_choices),
    }

    cmd = [
        sys.executable,
        args.main_script,
        "--mode", "train",
        "--dataset_name", args.dataset_name,
        "--ori_data", data_paths["ori_data"],
        "--dst_data", data_paths["dst_data"],
        "--trans_data", data_paths["trans_data"],
        "--ori_data_enriched", data_paths["ori_data_enriched"],
        "--dst_data_enriched", data_paths["dst_data_enriched"],
        "--log",
        "--save_trainable_only", bool_flag(args.save_trainable_only),
        "--save_optimizer_state", bool_flag(args.save_optimizer_state),
        "--best_save",
        "--log_path", log_dir,
        "--seed", str(args.seed),
        "--data_split_path", args.data_split_path,
        "--split_strategy", args.split_strategy,
        "--split_singleton_to_train", bool_flag(args.split_singleton_to_train),
        "--save_path", args.save_path,
        "--device", args.device,
        "--epoch", str(args.epoch),
        "--stop_epoch", str(args.stop_epoch),
        "--name", trial_name,
        "--semantic_backend", args.semantic_backend,
        "--qwen_train_soft_prompt", bool_flag(args.qwen_train_soft_prompt),
        "--llm_model_name", args.llm_model_name,
        "--llm_fallback_names", args.llm_fallback_names,
        "--llm_dtype", args.llm_dtype,
        "--use_mamba_backbone", bool_flag(args.use_mamba_backbone),
        "--mamba_d_conv", str(args.mamba_d_conv),
        "--save_dual_best", bool_flag(args.save_dual_best),
        "--use_f1_floor_filter", bool_flag(args.use_f1_floor_filter),
        "--f1_floor_margin", str(args.f1_floor_margin),
        "--early_stop_metric", args.early_stop_metric,
        "--enable_pairwise_loss", bool_flag(args.enable_pairwise_loss),
        "--lr", str(params["lr"]),
        "--lambda_pair", str(params["lambda_pair"]),
        "--pair_max_future", str(params["pair_max_future"]),
        "--lambda_transition", str(params["lambda_transition"]),
        "--transition_logit_scale", str(params["transition_logit_scale"]),
        "--beam_size", str(params["beam_size"]),
        "--beam_len_penalty", str(params["beam_len_penalty"]),
        "--use_beam_search", "1",
        "--use_no_repeat_mask", bool_flag(args.use_no_repeat_mask),
        "--combo_beta", str(params["combo_beta"]),
        "--dropout", str(params["dropout"]),
        "--temperature", str(params["temperature"]),
        "--seq_num_layers", str(params["seq_num_layers"]),
        "--mamba_d_state", str(params["mamba_d_state"]),
        "--mamba_expand", str(params["mamba_expand"]),
        "--decode_constraint_mode", args.decode_constraint_mode,
        "--enforce_start_end_constraints", bool_flag(args.enforce_start_end_constraints),
        "--soft_constraint_scale", str(params["soft_constraint_scale"]),
        "--soft_constraint_dist_emb_dim", str(params["soft_constraint_dist_emb_dim"]),
    ]

    if args.use_enriched_data == 1:
        cmd.append("--use_enriched_data")
    if args.qwen_strict == 1:
        cmd.append("--qwen_strict")
    if args.mamba_strict == 1:
        cmd.append("--mamba_strict")

    proc = subprocess.run(cmd, cwd=os.path.dirname(os.path.abspath(args.main_script)), capture_output=True, text=True)
    cmd_text = " ".join(shlex.quote(x) for x in cmd)

    log_file = find_log_file(log_dir, trial_name, args.seed)
    vals = parse_val_metrics(log_file)
    objective_value, best_val_epoch, best_val_item = get_objective_value(
        vals,
        args.objective_metric,
        args.objective_combo_beta_fixed,
    )

    trial_model_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(args.main_script)), args.save_path, trial_name))

    trial.set_user_attr("trial_name", trial_name)
    trial.set_user_attr("log_file", log_file if log_file is not None else "")
    trial.set_user_attr("model_dir", trial_model_dir)
    trial.set_user_attr("best_val_epoch", int(best_val_epoch))
    trial.set_user_attr("best_val_item", best_val_item)
    trial.set_user_attr("command", cmd_text)

    summary = {
        "trial_number": trial.number,
        "trial_name": trial_name,
        "objective_metric": args.objective_metric,
        "objective_combo_beta_fixed": args.objective_combo_beta_fixed,
        "objective_value": objective_value,
        "best_val_epoch": best_val_epoch,
        "best_val_item": best_val_item,
        "params": params,
        "return_code": proc.returncode,
        "log_file": log_file,
        "model_dir": trial_model_dir,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "command": cmd_text,
    }
    with open(os.path.join(output_dir, f"trial_{trial.number:03d}.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    if proc.returncode != 0:
        fail_path = os.path.join(output_dir, f"trial_{trial.number:03d}_stderr.log")
        with open(fail_path, "w", encoding="utf-8") as f:
            f.write(proc.stdout)
            f.write("\n\n===== STDERR =====\n")
            f.write(proc.stderr)
        raise optuna.exceptions.TrialPruned(f"trial failed, see {fail_path}")

    if not vals:
        raise optuna.exceptions.TrialPruned("no VAL lines parsed from log")

    return objective_value


def dump_study_results(args, study, output_dir):
    best = {
        "study_name": args.study_name,
        "objective_metric": args.objective_metric,
        "objective_combo_beta_fixed": args.objective_combo_beta_fixed,
        "best_trial_number": study.best_trial.number,
        "best_value": study.best_value,
        "best_params": study.best_params,
        "best_user_attrs": study.best_trial.user_attrs,
        "n_trials": len(study.trials),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }

    best_json = os.path.join(output_dir, f"{args.study_name}_best_params.json")
    with open(best_json, "w", encoding="utf-8") as f:
        json.dump(best, f, ensure_ascii=False, indent=2)

    all_trials = []
    for t in study.trials:
        all_trials.append({
            "number": t.number,
            "state": str(t.state),
            "value": t.value,
            "params": t.params,
            "user_attrs": t.user_attrs,
        })

    trials_json = os.path.join(output_dir, f"{args.study_name}_all_trials.json")
    with open(trials_json, "w", encoding="utf-8") as f:
        json.dump(all_trials, f, ensure_ascii=False, indent=2)

    if args.copy_best_artifacts == 1:
        src_dir = study.best_trial.user_attrs.get("model_dir", "")
        if src_dir and os.path.isdir(src_dir):
            dst_dir = os.path.join(output_dir, "best_trial_artifacts")
            if os.path.exists(dst_dir):
                shutil.rmtree(dst_dir)
            shutil.copytree(src_dir, dst_dir)

    print("=" * 80)
    print("Optuna finished")
    print(f"Study: {args.study_name}")
    print(f"Objective: {args.objective_metric}")
    print(f"Best value: {study.best_value:.6f}")
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best params json: {best_json}")
    print(f"All trials json: {trials_json}")
    print("Best trial model dir:", study.best_trial.user_attrs.get("model_dir", ""))
    print("Best trial log file:", study.best_trial.user_attrs.get("log_file", ""))
    print("=" * 80)


def main():
    args = parse_args()
    _ = resolve_dataset_paths(args)
    os.makedirs(args.output_dir, exist_ok=True)

    log_dir = os.path.join(args.output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)

    if args.storage.strip() == "":
        db_path = os.path.join(args.output_dir, f"{args.study_name}.db")
        storage = f"sqlite:///{db_path}"
    else:
        storage = args.storage

    sampler = optuna.samplers.TPESampler(seed=args.seed)

    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage,
        load_if_exists=True,
        direction="maximize",
        sampler=sampler,
    )

    def objective(trial):
        return run_trial(args, trial, args.output_dir, log_dir)

    timeout = None if args.timeout <= 0 else args.timeout
    study.optimize(objective, n_trials=args.n_trials, timeout=timeout)

    dump_study_results(args, study, args.output_dir)


if __name__ == "__main__":
    main()
