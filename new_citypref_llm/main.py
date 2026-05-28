import argparse
import os


def _fix_omp_threads_env():
    omp_threads = os.environ.get('OMP_NUM_THREADS', None)
    if omp_threads is None:
        return
    try:
        if int(omp_threads) <= 0:
            os.environ['OMP_NUM_THREADS'] = '1'
            print('[WARN] OMP_NUM_THREADS<=0 detected, auto reset to 1 before torch import.')
    except Exception:
        os.environ['OMP_NUM_THREADS'] = '1'
        print('[WARN] Invalid OMP_NUM_THREADS detected, auto reset to 1 before torch import.')


def _fix_pytorch_cuda_alloc_conf_env():
    """
    Mitigate a known CUDACachingAllocator internal assert issue observed with
    expandable_segments in some torch/cuda driver combinations.
    """
    key = 'PYTORCH_CUDA_ALLOC_CONF'
    conf = os.environ.get(key, '').strip()
    if not conf:
        os.environ[key] = 'expandable_segments:False'
        print('[INFO] PYTORCH_CUDA_ALLOC_CONF is unset, auto set to expandable_segments:False for stability.')
        return

    conf_lower = conf.lower().replace(' ', '')
    if 'expandable_segments:true' in conf_lower:
        print(
            '[WARN] Detected expandable_segments:True in PYTORCH_CUDA_ALLOC_CONF. '
            'This may trigger allocator internal assert on some environments.'
        )


_fix_omp_threads_env()
_fix_pytorch_cuda_alloc_conf_env()

import torch
from torch.utils.data import DataLoader

from data import TravelDatasetV2, random_split
from model import CrossCityLLMCPR
from spot_utils import Logger, collate_fn, path_exist, set_seeds
from trainer import Trainer, load_checkpoint


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default='Foursquare')
    parser.add_argument('--ori_data', type=str, default='../../Foursquare/home.txt')
    parser.add_argument('--dst_data', type=str, default='../../Foursquare/oot.txt')
    parser.add_argument('--use_enriched_data', action='store_true')
    parser.add_argument('--ori_data_enriched', type=str, default='../../Foursquare/extendData/enriched_home.txt')
    parser.add_argument('--dst_data_enriched', type=str, default='../../Foursquare/extendData/enriched_oot.txt')
    parser.add_argument('--trans_data', type=str, default='../../Foursquare/travel.txt')
    parser.add_argument('--save_path', type=str, default='../../Foursquare/model_save_new')
    parser.add_argument('--data_split_path', type=str, default='../../Foursquare/data_split_new.pkl')
    parser.add_argument('--rebuild_split', action='store_true')
    parser.add_argument('--split_strategy', type=str, default='legacy', choices=['legacy', 'pair_robust'])
    parser.add_argument('--split_singleton_to_train', type=int, default=1, choices=[0, 1])

    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--seed', type=int, default=2050)

    parser.add_argument('--train_batch', type=int, default=16)
    parser.add_argument('--test_batch', type=int, default=16)
    parser.add_argument('--epoch', type=int, default=30)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--l2', type=float, default=1e-5)
    parser.add_argument('--lr_dc', type=float, default=0.3)
    parser.add_argument('--lr_dc_step', type=int, default=8)

    parser.add_argument('--save_step', type=int, default=1)
    parser.add_argument('--best_save', action='store_true')
    parser.add_argument('--disable_checkpoint_save', type=int, default=0, choices=[0, 1])
    parser.add_argument('--save_trainable_only', type=int, default=1, choices=[0, 1])
    parser.add_argument('--save_optimizer_state', type=int, default=0, choices=[0, 1])
    parser.add_argument('--stop_epoch', type=int, default=8)
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
    parser.add_argument('--run_final_test_after_train', action='store_true')
    parser.add_argument('--log_path', type=str, default='../')
    parser.add_argument('--log', action='store_true')
    parser.add_argument('--name', type=str, default='new_citypref_llm')

    parser.add_argument('--hidden_size', type=int, default=128)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--nhead', type=int, default=4)
    parser.add_argument('--semantic_layers', type=int, default=2)
    parser.add_argument('--seq_num_layers', type=int, default=2)    #用于控制使用mamba模块的层数，0表示不使用mamba模块
    parser.add_argument('--soft_prompt_len', type=int, default=8)
    parser.add_argument('--pref_factor_k', type=int, default=4)
    parser.add_argument('--use_mamba_backbone', type=int, default=1, choices=[0, 1])    #用于控制是否使用mamba作为背骨网络，0表示不使用mamba模块，改为普通的Transformer层
    parser.add_argument('--mamba_d_state', type=int, default=16)    #用于控制mamba模块中状态向量的维度，过大会增加显存和计算成本，过小可能无法充分建模偏好演变
    parser.add_argument('--mamba_d_conv', type=int, default=4)  #用于控制mamba模块中卷积层的维度，过大会增加显存和计算成本，过小可能无法充分建模状态转移
    parser.add_argument('--mamba_expand', type=int, default=2)  #用于控制mamba模块中卷积层的扩展倍数，过大会增加显存和计算成本，过小可能无法充分建模状态转移
    parser.add_argument('--mamba_strict', action='store_true')  #用于控制mamba模块的严格模式，启用后会严格按照论文实现状态更新和偏好建模，可能会增加训练难度但更贴近原始设计
    parser.add_argument('--semantic_backend', type=str, default='qwen', choices=['qwen', 'fallback'])
    parser.add_argument('--llm_model_name', type=str, default='Qwen/Qwen3.5-0.6B-Instruct')
    parser.add_argument('--llm_cache_dir', type=str, default='../../code/params')
    parser.add_argument('--llm_max_length', type=int, default=256)
    parser.add_argument('--llm_micro_batch', type=int, default=4)
    parser.add_argument('--llm_gradient_checkpointing', action='store_true')
    parser.add_argument('--llm_dtype', type=str, default='float16', choices=['float16', 'bfloat16', 'float32'])
    parser.add_argument('--llm_max_traj_tokens', type=int, default=64)
    parser.add_argument('--gen_loss_chunk_len', type=int, default=16)
    parser.add_argument('--llm_fallback_names', type=str,
                        default='Qwen/Qwen2.5-0.5B-Instruct,Qwen/Qwen2.5-1.5B-Instruct')
    parser.add_argument('--qwen_strict', action='store_true')
    parser.add_argument('--qwen_train_soft_prompt', type=int, default=0, choices=[0, 1])

    parser.add_argument('--temperature', type=float, default=0.07)
    parser.add_argument('--gamma_city', type=float, default=0.5)
    parser.add_argument('--lambda_decouple', type=float, default=0.1)
    parser.add_argument('--lambda_semantic', type=float, default=0.1)
    parser.add_argument('--lambda_gen', type=float, default=1.0)
    parser.add_argument('--lambda_transition', type=float, default=0.3)
    parser.add_argument('--transition_logit_scale', type=float, default=0.5)
    parser.add_argument('--enable_pairwise_loss', type=int, default=1, choices=[0, 1])
    parser.add_argument('--lambda_pair', type=float, default=0.2)   #用于pairs-F1的pair-wise损失
    parser.add_argument('--pair_max_future', type=int, default=4)   #用于pairs-F1的pair-wise损失
    parser.add_argument('--use_beam_search', type=int, default=1, choices=[0, 1])
    parser.add_argument('--beam_size', type=int, default=4)
    parser.add_argument('--beam_len_penalty', type=float, default=0.2)
    parser.add_argument('--use_no_repeat_mask', type=int, default=1, choices=[0, 1])
    parser.add_argument('--pop_bias_scale', type=float, default=0.1)
    parser.add_argument('--ellipse_filter', action='store_true')    #空间椭圆预筛选，不启用该模块
    parser.add_argument('--ellipse_beta', type=float, default=1.2)
    parser.add_argument('--city_memory_momentum', type=float, default=0.95)
    parser.add_argument('--city_memory_prior_k', type=float, default=20.0)
    parser.add_argument('--eta_fixed', type=float, default=-1.0)  # in [0,1] to fix user-vs-city blend; <0 uses learned eta_gate
    parser.add_argument('--enforce_start_end_constraints', type=int, default=1, choices=[0, 1])
    parser.add_argument('--decode_constraint_mode', type=str, default='hard', choices=['hard', 'soft'])
    parser.add_argument('--soft_constraint_scale', type=float, default=0.2)
    parser.add_argument('--soft_constraint_dist_emb_dim', type=int, default=32)
    parser.add_argument('--ablate_generator_no_spatial_context', type=int, default=0, choices=[0, 1])

    parser.add_argument('--profile_dim', type=int, default=7)
    parser.add_argument('--query_dim', type=int, default=7)
    parser.add_argument('--ckpt_name', type=str, default='model_best.xhr')
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    set_seeds(args.seed)
    args.save_path = os.path.join(args.save_path, args.name)
    path_exist(args.save_path)

    logger = Logger(args.log_path, args.name, args.seed, args.log)
    logger.log(str(args))

    ori_data_path = args.ori_data
    dst_data_path = args.dst_data
    if args.use_enriched_data:
        ori_data_path = args.ori_data_enriched
        dst_data_path = args.dst_data_enriched
        logger.log(f"Using enriched source files: {ori_data_path} | {dst_data_path}")

    dataset = TravelDatasetV2(args, ori_data_path, dst_data_path, args.trans_data)
    if args.rebuild_split and os.path.exists(args.data_split_path):
        os.remove(args.data_split_path)
        logger.log(f"Removed existing split file: {args.data_split_path}")
    train_data, valid_data, test_data = random_split(dataset, args.data_split_path, seed=args.seed, args=args)

    train_loader = DataLoader(train_data, batch_size=args.train_batch, shuffle=True, collate_fn=collate_fn)
    valid_loader = DataLoader(valid_data, batch_size=args.test_batch, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_data, batch_size=args.test_batch, shuffle=False, collate_fn=collate_fn)

    model = CrossCityLLMCPR(
        args,
        poi_num=dataset.poi_num,
        tag_num=dataset.tag_num,
        region_num=dataset.region_num,
        popularity_bias=dataset.poi_popularity,
        poi_coord_tensor=dataset.poi_coord_tensor,
        city_sample_count=dataset.region_sample_count_tensor,
    ).to(args.device)

    trainer = Trainer(model, args, logger)

    if args.mode == 'train':
        best_epoch = trainer.train(train_loader, valid_loader)
        logger.log(f"Training finished. Best epoch: {best_epoch}")

        if args.run_final_test_after_train:
            best_model_path = os.path.join(args.save_path, args.ckpt_name)
            if os.path.exists(best_model_path):
                load_checkpoint(model, best_model_path, map_location=args.device)
                trainer.validate(test_loader, 'FINAL', split_name='TEST')
            else:
                logger.log(f"Best model not found at {best_model_path}, skip final test.")

    else:
        best_model_path = os.path.join(args.save_path, args.ckpt_name)
        if os.path.exists(best_model_path):
            load_checkpoint(model, best_model_path, map_location=args.device)
            trainer.validate(test_loader, 'TEST_ONLY', split_name='TEST')
        else:
            logger.log(f"No model found at {best_model_path}")

    logger.close_log()


if __name__ == '__main__':
    main()
