import argparse
import os

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
    parser.add_argument('--trans_data', type=str, default='../../Foursquare/travel.txt')
    parser.add_argument('--save_path', type=str, default='../../Foursquare/model_save_new')
    parser.add_argument('--data_split_path', type=str, default='../../Foursquare/data_split_new.pkl')
    parser.add_argument('--rebuild_split', action='store_true')

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
    parser.add_argument('--save_trainable_only', type=int, default=1, choices=[0, 1])
    parser.add_argument('--save_optimizer_state', type=int, default=0, choices=[0, 1])
    parser.add_argument('--stop_epoch', type=int, default=8)
    parser.add_argument('--run_final_test_after_train', action='store_true')
    parser.add_argument('--log_path', type=str, default='../')
    parser.add_argument('--log', action='store_true')
    parser.add_argument('--name', type=str, default='new_citypref_llm')

    parser.add_argument('--hidden_size', type=int, default=128)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--nhead', type=int, default=4)
    parser.add_argument('--semantic_layers', type=int, default=2)
    parser.add_argument('--soft_prompt_len', type=int, default=8)
    parser.add_argument('--pref_factor_k', type=int, default=4)
    parser.add_argument('--semantic_backend', type=str, default='qwen', choices=['qwen', 'fallback'])
    parser.add_argument('--llm_model_name', type=str, default='Qwen/Qwen3.5-0.6B-Instruct')
    parser.add_argument('--llm_cache_dir', type=str, default='../../code/params')
    parser.add_argument('--llm_max_length', type=int, default=256)
    parser.add_argument('--llm_dtype', type=str, default='float16', choices=['float16', 'bfloat16', 'float32'])
    parser.add_argument('--llm_max_traj_tokens', type=int, default=64)
    parser.add_argument('--llm_fallback_names', type=str,
                        default='Qwen/Qwen2.5-0.5B-Instruct,Qwen/Qwen2.5-1.5B-Instruct')
    parser.add_argument('--qwen_strict', action='store_true')

    parser.add_argument('--temperature', type=float, default=0.07)
    parser.add_argument('--gamma_city', type=float, default=0.5)
    parser.add_argument('--lambda_decouple', type=float, default=0.1)
    parser.add_argument('--lambda_semantic', type=float, default=0.1)
    parser.add_argument('--lambda_gen', type=float, default=1.0)
    parser.add_argument('--pop_bias_scale', type=float, default=0.1)

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

    dataset = TravelDatasetV2(args, args.ori_data, args.dst_data, args.trans_data)
    if args.rebuild_split and os.path.exists(args.data_split_path):
        os.remove(args.data_split_path)
        logger.log(f"Removed existing split file: {args.data_split_path}")
    train_data, valid_data, test_data = random_split(dataset, args.data_split_path, seed=args.seed)

    train_loader = DataLoader(train_data, batch_size=args.train_batch, shuffle=True, collate_fn=collate_fn)
    valid_loader = DataLoader(valid_data, batch_size=args.test_batch, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_data, batch_size=args.test_batch, shuffle=False, collate_fn=collate_fn)

    model = CrossCityLLMCPR(
        args,
        poi_num=dataset.poi_num,
        tag_num=dataset.tag_num,
        region_num=dataset.region_num,
        popularity_bias=dataset.poi_popularity,
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
