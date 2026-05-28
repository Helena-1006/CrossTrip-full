import argparse
import csv
import glob
import os
import re
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple


METRIC_RE = re.compile(
    r"\[(?P<split>VAL|TEST)\] Epoch (?P<epoch>[^ ]+) "
    r"F1: (?P<f1>[0-9.]+) \| "
    r"Pairs_F1: (?P<pairs_f1>[0-9.]+) \| "
    r"Full_F1: (?P<full_f1>[0-9.]+) \| "
    r"Full_Pairs_F1: (?P<full_pairs_f1>[0-9.]+) \| "
    r"REP: (?P<rep>[0-9.]+) \| "
    r"Full_REP: (?P<full_rep>[0-9.]+)"
)


@dataclass
class ExpResult:
    group: str
    exp_name: str
    status: str
    return_code: int
    duration_sec: float
    f1: float
    pairs_f1: float
    full_f1: float
    full_pairs_f1: float
    rep: float
    full_rep: float
    picked_split: str
    picked_epoch: str
    notes: str
    cmd: str


def parse_list(raw: str, cast_fn):
    return [cast_fn(x.strip()) for x in raw.split(',') if x.strip()]


def parse_set(raw: str) -> set:
    return {x.strip() for x in str(raw).split(',') if x.strip()}


def resolve_dataset_paths(args) -> Dict[str, str]:
    name = str(args.dataset_name).strip().lower()
    if name == 'yelp':
        base = '../../Yelp'
    elif name == 'foursquare':
        base = '../../Foursquare'
    else:
        base = ''

    ori_data = args.ori_data.strip() if str(args.ori_data).strip() else (f"{base}/home.txt" if base else '')
    dst_data = args.dst_data.strip() if str(args.dst_data).strip() else (f"{base}/oot.txt" if base else '')
    trans_data = args.trans_data.strip() if str(args.trans_data).strip() else (f"{base}/travel.txt" if base else '')
    ori_data_enriched = (
        args.ori_data_enriched.strip()
        if str(args.ori_data_enriched).strip()
        else (f"{base}/extendData/enriched_home.txt" if base else '')
    )
    dst_data_enriched = (
        args.dst_data_enriched.strip()
        if str(args.dst_data_enriched).strip()
        else (f"{base}/extendData/enriched_oot.txt" if base else '')
    )

    missing = [
        k for k, v in {
            'ori_data': ori_data,
            'dst_data': dst_data,
            'trans_data': trans_data,
            'ori_data_enriched': ori_data_enriched,
            'dst_data_enriched': dst_data_enriched,
        }.items() if not v
    ]
    if missing:
        raise ValueError(
            f"Cannot resolve data paths for dataset_name={args.dataset_name}. "
            f"Please provide explicit paths: {', '.join(missing)}"
        )

    return {
        'ori_data': ori_data,
        'dst_data': dst_data,
        'trans_data': trans_data,
        'ori_data_enriched': ori_data_enriched,
        'dst_data_enriched': dst_data_enriched,
    }


def score_row(item: ExpResult, metric: str, combo_beta: float) -> float:
    if metric == 'f1':
        return item.f1
    if metric == 'pairs_f1':
        return item.pairs_f1
    if metric == 'full_f1':
        return item.full_f1
    if metric == 'full_pairs_f1':
        return item.full_pairs_f1
    return item.f1 + combo_beta * item.pairs_f1


def sanitize_name(raw: str) -> str:
    out = []
    for c in raw:
        if c.isalnum() or c in ['_', '-', '.']:
            out.append(c)
        else:
            out.append('_')
    return ''.join(out)


def build_main_cmd(args, exp_name: str, overrides: Dict[str, object]) -> List[str]:
    kv = {
        'mode': args.run_mode,
        'log': True,
        'seed': args.seed,
        'data_split_path': args.data_split_path,
        'device': args.device,
        'name': exp_name,
        'save_path': args.save_path,
        'dataset_name': args.dataset_name,
        'ori_data': args.ori_data,
        'dst_data': args.dst_data,
        'trans_data': args.trans_data,
        'ori_data_enriched': args.ori_data_enriched,
        'dst_data_enriched': args.dst_data_enriched,
        'semantic_backend': args.semantic_backend,
        'llm_model_name': args.llm_model_name,
        'llm_fallback_names': args.llm_fallback_names,
        'llm_dtype': args.llm_dtype,
        'use_mamba_backbone': args.use_mamba_backbone,
        'seq_num_layers': args.seq_num_layers,
        'mamba_d_state': args.mamba_d_state,
        'mamba_d_conv': args.mamba_d_conv,
        'mamba_expand': args.mamba_expand,
        'dropout': args.dropout,
        'temperature': args.temperature,
        'enable_pairwise_loss': args.enable_pairwise_loss,
        'lambda_pair': args.lambda_pair,
        'pair_max_future': args.pair_max_future,
        'lambda_transition': args.lambda_transition,
        'transition_logit_scale': args.transition_logit_scale,
        'use_beam_search': args.use_beam_search,
        'beam_size': args.beam_size,
        'beam_len_penalty': args.beam_len_penalty,
        'use_no_repeat_mask': args.use_no_repeat_mask,
        'pref_factor_k': args.pref_factor_k,
        'lambda_decouple': args.lambda_decouple,
        'lambda_semantic': args.lambda_semantic,
        'ckpt_name': args.ckpt_name,
        'eta_fixed': args.eta_fixed,
        'enforce_start_end_constraints': args.enforce_start_end_constraints,
        'decode_constraint_mode': args.decode_constraint_mode,
        'soft_constraint_scale': args.soft_constraint_scale,
        'soft_constraint_dist_emb_dim': args.soft_constraint_dist_emb_dim,
        'ablate_generator_no_spatial_context': args.ablate_generator_no_spatial_context,
    }

    if args.run_mode == 'train':
        kv['epoch'] = args.epoch
        kv['stop_epoch'] = args.stop_epoch
        kv['lr'] = args.lr
        kv['l2'] = args.l2
        kv['lr_dc'] = args.lr_dc
        kv['lr_dc_step'] = args.lr_dc_step
        kv['disable_checkpoint_save'] = args.disable_checkpoint_save
        kv['best_save'] = True
        kv['save_trainable_only'] = args.save_trainable_only
        kv['save_optimizer_state'] = 0
        kv['run_final_test_after_train'] = True
        kv['early_stop_metric'] = args.early_stop_metric
        kv['combo_beta'] = args.combo_beta
        kv['use_f1_floor_filter'] = args.use_f1_floor_filter
        kv['f1_floor_margin'] = args.f1_floor_margin
        kv['save_dual_best'] = args.save_dual_best

    kv.update(overrides)

    cmd = [args.python_exec, args.main_script]

    bool_flags = {
        'log',
        'use_enriched_data',
        'qwen_strict',
        'mamba_strict',
        'llm_gradient_checkpointing',
        'best_save',
        'run_final_test_after_train',
    }

    # Fixed boolean defaults for reproducibility.
    kv.setdefault('use_enriched_data', bool(args.use_enriched_data))
    kv.setdefault('qwen_strict', bool(args.qwen_strict))
    kv.setdefault('mamba_strict', bool(args.mamba_strict))
    kv.setdefault('llm_gradient_checkpointing', bool(args.llm_gradient_checkpointing))

    for key, value in kv.items():
        if key in bool_flags:
            if bool(value):
                cmd.append(f"--{key}")
            continue
        cmd.extend([f"--{key}", str(value)])

    return cmd


def parse_metric_lines(text: str) -> List[Dict[str, object]]:
    rows = []
    for line in text.splitlines():
        m = METRIC_RE.search(line)
        if m is None:
            continue
        rows.append({
            'split': m.group('split'),
            'epoch': m.group('epoch'),
            'f1': float(m.group('f1')),
            'pairs_f1': float(m.group('pairs_f1')),
            'full_f1': float(m.group('full_f1')),
            'full_pairs_f1': float(m.group('full_pairs_f1')),
            'rep': float(m.group('rep')),
            'full_rep': float(m.group('full_rep')),
        })
    return rows


def pick_row(
    rows: List[Dict[str, object]],
    metric: str,
    combo_beta: float,
    result_split_policy: str,
) -> Tuple[str, str, Dict[str, object], bool]:
    test_rows = [r for r in rows if r['split'] == 'TEST']
    if test_rows:
        row = test_rows[-1]
        return str(row['split']), str(row['epoch']), row, True

    if result_split_policy == 'test_only':
        return 'NONE', 'NONE', {
            'f1': 0.0,
            'pairs_f1': 0.0,
            'full_f1': 0.0,
            'full_pairs_f1': 0.0,
            'rep': 0.0,
            'full_rep': 0.0,
        }, False

    val_rows = [r for r in rows if r['split'] == 'VAL']
    if not val_rows:
        return 'NONE', 'NONE', {
            'f1': 0.0,
            'pairs_f1': 0.0,
            'full_f1': 0.0,
            'full_pairs_f1': 0.0,
            'rep': 0.0,
            'full_rep': 0.0,
        }, False

    def score(item):
        if metric == 'f1':
            return item['f1']
        if metric == 'pairs_f1':
            return item['pairs_f1']
        if metric == 'full_f1':
            return item['full_f1']
        if metric == 'full_pairs_f1':
            return item['full_pairs_f1']
        return item['f1'] + combo_beta * item['pairs_f1']

    best = max(val_rows, key=score)
    return str(best['split']), str(best['epoch']), best, True


def find_log_file(log_path: str, exp_name: str, seed: int) -> Optional[str]:
    pattern = os.path.join(log_path, f"* {exp_name}({seed}).log")
    files = glob.glob(pattern)
    if not files:
        return None
    files.sort(key=os.path.getmtime, reverse=True)
    return files[0]


def find_log_file_multi(log_roots: List[str], exp_name: str, seed: int) -> Optional[str]:
    """Find the newest matching log across multiple root folders."""
    files = find_log_files_multi(log_roots, exp_name, seed)
    if not files:
        return None
    return files[0]


def find_log_files_multi(log_roots: List[str], exp_name: str, seed: int) -> List[str]:
    """Find all matching logs across multiple roots, newest first, deduplicated by absolute path."""
    all_files: List[str] = []
    for root in log_roots:
        if not root:
            continue
        pattern = os.path.join(root, f"* {exp_name}({seed}).log")
        all_files.extend(glob.glob(pattern))
    if not all_files:
        return []
    all_files = sorted((os.path.abspath(p) for p in all_files), key=os.path.getmtime, reverse=True)
    uniq: List[str] = []
    seen = set()
    for p in all_files:
        if p in seen:
            continue
        seen.add(p)
        uniq.append(p)
    return uniq


def parse_rows_from_log_file(log_file: str) -> List[Dict[str, object]]:
    if not log_file or (not os.path.exists(log_file)):
        return []
    with open(log_file, 'r', encoding='utf-8') as f:
        return parse_metric_lines(f.read())


def build_reused_result(
    args,
    group: str,
    exp_name: str,
    overrides: Dict[str, object],
    split_name: str,
    epoch_name: str,
    row: Dict[str, object],
    notes: str,
) -> ExpResult:
    cmd = build_main_cmd(args, exp_name=exp_name, overrides=overrides)
    cmd_text = ' '.join(shlex.quote(x) for x in cmd)
    return ExpResult(
        group=group,
        exp_name=exp_name,
        status='skipped_reused',
        return_code=0,
        duration_sec=0.0,
        f1=float(row['f1']),
        pairs_f1=float(row['pairs_f1']),
        full_f1=float(row['full_f1']),
        full_pairs_f1=float(row['full_pairs_f1']),
        rep=float(row['rep']),
        full_rep=float(row['full_rep']),
        picked_split=split_name,
        picked_epoch=epoch_name,
        notes=notes,
        cmd=cmd_text,
    )


def try_reuse_existing_result(
    args,
    group: str,
    tag: str,
    exp_name: str,
    overrides: Dict[str, object],
) -> Optional[ExpResult]:
    if args.resume_mode == 'none':
        return None

    if tag in args.force_rerun_tags_set or exp_name in args.force_rerun_tags_set:
        print(f"[resume] force rerun tag={tag} exp={exp_name}")
        return None

    log_roots: List[str] = []
    if str(args.existing_log_glob_root).strip():
        log_roots.append(args.existing_log_glob_root.strip())
    if str(args.log_path).strip():
        log_roots.append(args.log_path.strip())

    # Fallbacks: many runs write logs to cwd parent (e.g., ../) instead of --log_path.
    cwd = os.getcwd()
    log_roots.append(cwd)
    log_roots.append(os.path.abspath(os.path.join(cwd, '..')))

    # Deduplicate while keeping order.
    uniq_roots: List[str] = []
    seen = set()
    for p in log_roots:
        rp = os.path.abspath(p)
        if rp in seen:
            continue
        seen.add(rp)
        uniq_roots.append(rp)

    log_files = find_log_files_multi(uniq_roots, exp_name, args.seed)
    if not log_files:
        return None

    if args.resume_mode == 'skip_if_done_metric':
        # Scan from newest to oldest; interrupted latest logs may miss TEST lines.
        for log_file in log_files:
            rows = parse_rows_from_log_file(log_file)
            split_name, epoch_name, row, has_target_split = pick_row(
                rows,
                args.select_metric,
                args.combo_beta,
                args.result_split_policy,
            )
            if has_target_split:
                print(f"[resume] skip by done metric: {exp_name}")
                return build_reused_result(
                    args,
                    group,
                    exp_name,
                    overrides,
                    split_name,
                    epoch_name,
                    row,
                    f"reused_log={log_file}",
                )

        print(
            f"[resume] found logs but no target metric split for {exp_name}; rerun required "
            f"(policy={args.result_split_policy})"
        )
        return None

    log_file = log_files[0]
    rows = parse_rows_from_log_file(log_file)
    split_name, epoch_name, row, has_target_split = pick_row(
        rows,
        args.select_metric,
        args.combo_beta,
        args.result_split_policy,
    )

    if args.resume_mode == 'skip_if_log_exists':
        note = f"reused_log={log_file}"
        if not rows:
            note += '; no metrics parsed in log'
        elif not has_target_split:
            note += f'; no {args.result_split_policy} metrics parsed'
        print(f"[resume] skip by existing log: {exp_name}")
        return build_reused_result(
            args,
            group,
            exp_name,
            overrides,
            split_name,
            epoch_name,
            row,
            note,
        )

    print(
        f"[resume] found log but no target metric split for {exp_name}; rerun required "
        f"(policy={args.result_split_policy})"
    )
    return None


def run_one(args, group: str, exp_name: str, overrides: Dict[str, object]) -> ExpResult:
    cmd = build_main_cmd(args, exp_name=exp_name, overrides=overrides)
    cmd_text = ' '.join(shlex.quote(x) for x in cmd)

    t0 = time.time()
    proc = subprocess.run(
        cmd,
        cwd=os.path.dirname(os.path.abspath(args.main_script)),
        capture_output=True,
        text=True,
    )
    dt = time.time() - t0

    merged = (proc.stdout or '') + '\n' + (proc.stderr or '')
    rows = parse_metric_lines(merged)

    # If stdout has no metric lines, try parsing logger file.
    if not rows:
        log_file = find_log_file(args.log_path, exp_name, args.seed)
        rows = parse_rows_from_log_file(log_file) if log_file is not None else []

    split_name, epoch_name, row, has_target_split = pick_row(
        rows,
        args.select_metric,
        args.combo_beta,
        args.result_split_policy,
    )
    status = 'ok' if proc.returncode == 0 else 'failed'

    note = ''
    if proc.returncode != 0:
        err_file = os.path.join(args.output_dir, f"{exp_name}.stderr.log")
        with open(err_file, 'w', encoding='utf-8') as f:
            f.write(proc.stdout or '')
            f.write('\n\n===== STDERR =====\n')
            f.write(proc.stderr or '')
        note = f"stderr_saved={err_file}"
    elif not has_target_split:
        status = 'missing_target_split'
        note = f"no {args.result_split_policy} metrics parsed"

    return ExpResult(
        group=group,
        exp_name=exp_name,
        status=status,
        return_code=proc.returncode,
        duration_sec=dt,
        f1=float(row['f1']),
        pairs_f1=float(row['pairs_f1']),
        full_f1=float(row['full_f1']),
        full_pairs_f1=float(row['full_pairs_f1']),
        rep=float(row['rep']),
        full_rep=float(row['full_rep']),
        picked_split=split_name,
        picked_epoch=epoch_name,
        notes=note,
        cmd=cmd_text,
    )


def build_experiments(args) -> List[Tuple[str, str, Dict[str, object]]]:
    exps: List[Tuple[str, str, Dict[str, object]]] = []

    exps.append(('ablation', 'full', {}))

    # 1) remove LLM semantic preference extraction
    exps.append(('ablation', 'no_llm_semantic', {
        'semantic_backend': 'fallback',
    }))

    # 2) remove preference disentanglement + transfer
    exps.append(('ablation', 'no_disentangle_transfer', {
        'pref_factor_k': 1,
        'lambda_decouple': 0.0,
        'lambda_semantic': 0.0,
    }))

    # 3) remove city group preference (use personal transfer only)
    exps.append(('ablation', 'no_city_group_pref', {
        'eta_fixed': 1.0,
    }))

    # 4) remove spatial generation context in decoder input
    #    (drop prev-POI embedding and distance branch; keep z_final + query_h)
    exps.append(('ablation', 'no_generator_constraints', {
        'ablate_generator_no_spatial_context': 1,
    }))

    # 5) explicit hard-constraint decoding variant
    exps.append(('ablation', 'hard_constraint_variant', {
        'decode_constraint_mode': 'hard',
        'enforce_start_end_constraints': 1,
    }))

    # Hyperparam 1: Mamba layers
    for v in parse_list(args.hparam_seq_num_layers, int):
        exps.append(('hparam_seq_num_layers', f'seq_num_layers_{v}', {
            'seq_num_layers': v,
        }))

    # Hyperparam 2: pairwise loss weight multiplier
    for m in parse_list(args.hparam_lambda_pair, float):
        exps.append(('hparam_lambda_pair', f'lambda_pair_x{m:g}', {
            'lambda_pair': args.lambda_pair * m,
        }))

    # Hyperparam 3: transition strength multiplier
    for m in parse_list(args.hparam_transition_strength, float):
        exps.append(('hparam_transition_strength', f'trans_strength_x{m:g}', {
            'lambda_transition': args.lambda_transition * m,
            'transition_logit_scale': args.transition_logit_scale * m,
        }))

    # Hyperparam 4: fixed eta (personal transfer vs city group blend)
    for v in parse_list(args.hparam_eta_fixed, float):
        exps.append(('hparam_eta_fixed', f'eta_fixed_{v:g}', {
            'eta_fixed': v,
        }))

    return exps


def write_csv(path: str, rows: List[ExpResult]):
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            'group', 'exp_name', 'status', 'return_code', 'duration_sec',
            'f1', 'pairs_f1', 'full_f1', 'full_pairs_f1', 'rep', 'full_rep',
            'picked_split', 'picked_epoch', 'notes', 'cmd'
        ])
        for r in rows:
            writer.writerow([
                r.group, r.exp_name, r.status, r.return_code, f"{r.duration_sec:.2f}",
                f"{r.f1:.6f}", f"{r.pairs_f1:.6f}", f"{r.full_f1:.6f}", f"{r.full_pairs_f1:.6f}",
                f"{r.rep:.6f}", f"{r.full_rep:.6f}", r.picked_split, r.picked_epoch, r.notes, r.cmd,
            ])


def parse_args():
    parser = argparse.ArgumentParser(description='Batch runner for 4 ablations + 4 hyperparameter analyses')
    parser.add_argument('--python_exec', type=str, default=sys.executable)
    parser.add_argument('--main_script', type=str, default='./main.py')
    parser.add_argument('--output_dir', type=str, default='./batch_runs')
    parser.add_argument('--log_path', type=str, default='../')
    parser.add_argument('--run_mode', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--seed', type=int, default=2050)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--dataset_name', type=str, default='Foursquare')
    parser.add_argument('--ori_data', type=str, default='')
    parser.add_argument('--dst_data', type=str, default='')
    parser.add_argument('--trans_data', type=str, default='')
    parser.add_argument('--ori_data_enriched', type=str, default='')
    parser.add_argument('--dst_data_enriched', type=str, default='')
    parser.add_argument('--data_split_path', type=str, default='../../Foursquare/data_split_new.pkl')
    parser.add_argument('--save_path', type=str, default='../../Foursquare/model_save_new')
    parser.add_argument('--exp_prefix', type=str, default='batch_d3')
    parser.add_argument('--resume_mode', type=str, default='none',
                        choices=['none', 'skip_if_log_exists', 'skip_if_done_metric'])
    parser.add_argument('--existing_log_glob_root', type=str, default='')
    parser.add_argument('--force_rerun_tags', type=str, default='')
    parser.add_argument('--write_incremental_csv', type=int, default=1, choices=[0, 1])

    # Base config (your full experiment command as defaults)
    parser.add_argument('--use_enriched_data', type=int, default=1, choices=[0, 1])
    parser.add_argument('--semantic_backend', type=str, default='qwen', choices=['qwen', 'fallback'])
    parser.add_argument('--qwen_strict', type=int, default=1, choices=[0, 1])
    parser.add_argument('--llm_model_name', type=str, default='Qwen/Qwen2.5-1.5B-Instruct')
    parser.add_argument('--llm_fallback_names', type=str, default='Qwen/Qwen2.5-0.5B-Instruct')
    parser.add_argument('--llm_dtype', type=str, default='bfloat16', choices=['float16', 'bfloat16', 'float32'])
    parser.add_argument('--llm_gradient_checkpointing', type=int, default=0, choices=[0, 1])

    parser.add_argument('--use_mamba_backbone', type=int, default=1, choices=[0, 1])
    parser.add_argument('--mamba_strict', type=int, default=1, choices=[0, 1])
    parser.add_argument('--seq_num_layers', type=int, default=2)
    parser.add_argument('--mamba_d_state', type=int, default=32)
    parser.add_argument('--mamba_d_conv', type=int, default=4)
    parser.add_argument('--mamba_expand', type=int, default=3)

    parser.add_argument('--dropout', type=float, default=0.09006709541052905)
    parser.add_argument('--temperature', type=float, default=0.05767587816624931)
    parser.add_argument('--enable_pairwise_loss', type=int, default=0, choices=[0, 1])
    parser.add_argument('--lambda_pair', type=float, default=0.0)
    parser.add_argument('--pair_max_future', type=int, default=8)
    parser.add_argument('--lambda_transition', type=float, default=0.8273752580873036)
    parser.add_argument('--transition_logit_scale', type=float, default=1.1110770204640883)

    parser.add_argument('--use_beam_search', type=int, default=1, choices=[0, 1])
    parser.add_argument('--beam_size', type=int, default=3)
    parser.add_argument('--beam_len_penalty', type=float, default=0.5098784617538058)
    parser.add_argument('--use_no_repeat_mask', type=int, default=1, choices=[0, 1])

    parser.add_argument('--pref_factor_k', type=int, default=4)
    parser.add_argument('--lambda_decouple', type=float, default=0.1)
    parser.add_argument('--lambda_semantic', type=float, default=0.1)
    parser.add_argument('--eta_fixed', type=float, default=-1.0)
    parser.add_argument('--enforce_start_end_constraints', type=int, default=1, choices=[0, 1])
    parser.add_argument('--decode_constraint_mode', type=str, default='hard', choices=['hard', 'soft'])
    parser.add_argument('--soft_constraint_scale', type=float, default=0.2)
    parser.add_argument('--soft_constraint_dist_emb_dim', type=int, default=32)
    parser.add_argument('--ablate_generator_no_spatial_context', type=int, default=0, choices=[0, 1])

    # Train-only settings
    parser.add_argument('--disable_checkpoint_save', type=int, default=0, choices=[0, 1])
    parser.add_argument('--save_trainable_only', type=int, default=1, choices=[0, 1])
    parser.add_argument('--epoch', type=int, default=30)
    parser.add_argument('--stop_epoch', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--l2', type=float, default=1e-5)
    parser.add_argument('--lr_dc', type=float, default=0.3)
    parser.add_argument('--lr_dc_step', type=int, default=8)
    parser.add_argument(
        '--early_stop_metric',
        type=str,
        default='combo',
        choices=['combo', 'f1', 'pairs_f1', 'full_f1', 'full_pairs_f1', 'full_combo']
    )
    parser.add_argument('--combo_beta', type=float, default=4.0)
    parser.add_argument('--use_f1_floor_filter', type=int, default=1, choices=[0, 1])
    parser.add_argument('--f1_floor_margin', type=float, default=0.002)
    parser.add_argument('--save_dual_best', type=int, default=1, choices=[0, 1])

    # eval ckpt and row picking strategy
    parser.add_argument('--ckpt_name', type=str, default='model_best.xhr')
    parser.add_argument('--select_metric', type=str, default='combo',
                        choices=['combo', 'f1', 'pairs_f1', 'full_f1', 'full_pairs_f1'])
    parser.add_argument('--rank_metric', type=str, default='combo',
                        choices=['combo', 'f1', 'pairs_f1', 'full_f1', 'full_pairs_f1'])
    parser.add_argument('--result_split_policy', type=str, default='test_only',
                        choices=['test_only', 'test_or_val'])

    # 4 hyperparam analyses
    parser.add_argument('--hparam_seq_num_layers', type=str, default='1,2,3')
    parser.add_argument('--hparam_lambda_pair', type=str, default='0.25,0.5,1.0,1.5,2.0')
    parser.add_argument('--hparam_transition_strength', type=str, default='1.0')
    parser.add_argument('--hparam_eta_fixed', type=str, default='0.0,0.2,0.4,0.6,0.8')

    # include known full test result row as reference baseline
    parser.add_argument('--append_full_reference', type=int, default=1, choices=[0, 1])
    parser.add_argument('--full_ref_f1', type=float, default=0.0493)
    parser.add_argument('--full_ref_pairs_f1', type=float, default=0.0107)
    parser.add_argument('--full_ref_full_f1', type=float, default=0.4799)
    parser.add_argument('--full_ref_full_pairs_f1', type=float, default=0.1773)
    parser.add_argument('--full_ref_rep', type=float, default=0.0000)
    parser.add_argument('--full_ref_full_rep', type=float, default=0.0064)

    return parser.parse_args()


def main():
    args = parse_args()
    data_paths = resolve_dataset_paths(args)
    args.ori_data = data_paths['ori_data']
    args.dst_data = data_paths['dst_data']
    args.trans_data = data_paths['trans_data']
    args.ori_data_enriched = data_paths['ori_data_enriched']
    args.dst_data_enriched = data_paths['dst_data_enriched']
    args.force_rerun_tags_set = parse_set(args.force_rerun_tags)
    os.makedirs(args.output_dir, exist_ok=True)

    results: List[ExpResult] = []
    latest_csv = os.path.join(args.output_dir, f"summary_{args.exp_prefix}_latest.csv")

    if args.append_full_reference == 1:
        results.append(
            ExpResult(
                group='ablation',
                exp_name='full_reference_user_reported',
                status='reference',
                return_code=0,
                duration_sec=0.0,
                f1=args.full_ref_f1,
                pairs_f1=args.full_ref_pairs_f1,
                full_f1=args.full_ref_full_f1,
                full_pairs_f1=args.full_ref_full_pairs_f1,
                rep=args.full_ref_rep,
                full_rep=args.full_ref_full_rep,
                picked_split='TEST',
                picked_epoch='TEST_ONLY',
                notes='from user-reported full experiment result',
                cmd='N/A (reference row)',
            )
        )
        if args.write_incremental_csv == 1:
            write_csv(latest_csv, results)

    exps = build_experiments(args)
    total = len(exps)
    for idx, (group, tag, overrides) in enumerate(exps, start=1):
        exp_name = sanitize_name(f"{args.exp_prefix}_{tag}_s{args.seed}")
        reused = try_reuse_existing_result(args, group, tag, exp_name, overrides)
        if reused is not None:
            res = reused
            print(f"[{idx}/{total}] Skipped {exp_name} | group={group} | status={res.status}")
        else:
            print(f"[{idx}/{total}] Running {exp_name} | group={group}")
            res = run_one(args, group=group, exp_name=exp_name, overrides=overrides)
        results.append(res)
        if args.write_incremental_csv == 1:
            write_csv(latest_csv, results)
        print(
            f"[{idx}/{total}] {exp_name} done | status={res.status} | "
            f"F1={res.f1:.4f} Pairs={res.pairs_f1:.4f} FullPairs={res.full_pairs_f1:.4f}"
        )

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_csv = os.path.join(args.output_dir, f"summary_{args.exp_prefix}_{ts}.csv")
    write_csv(out_csv, results)
    if args.write_incremental_csv == 1:
        write_csv(latest_csv, results)

    best_ok = [r for r in results if r.status in ['ok', 'reference', 'skipped_reused']]
    best_ok.sort(key=lambda x: score_row(x, args.rank_metric, args.combo_beta), reverse=True)

    print('\n=== Batch finished ===')
    print(f"Summary CSV: {out_csv}")
    if best_ok:
        top = best_ok[0]
        print(
            f"Top by {args.rank_metric}: {top.exp_name} | "
            f"F1={top.f1:.4f} | Pairs={top.pairs_f1:.4f} | FullPairs={top.full_pairs_f1:.4f}"
        )


if __name__ == '__main__':
    main()
