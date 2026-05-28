import argparse
import os

from torch.utils.data import DataLoader

from data import TravelDatasetV2, random_split
from main import build_parser
from model import CrossCityLLMCPR
from spot_utils import Logger, collate_fn
from trainer import Trainer, load_checkpoint


def parse_args():
    parser = build_parser()
    parser.set_defaults(mode='test')
    parser.add_argument('--split', type=str, default='valid', choices=['valid', 'test'])
    return parser.parse_args()


def main():
    args = parse_args()
    args.save_path = os.path.join(args.save_path, args.name)

    logger = Logger(args.log_path, f"{args.name}_validate", args.seed, args.log)
    dataset = TravelDatasetV2(args, args.ori_data, args.dst_data, args.trans_data)
    train_data, valid_data, test_data = random_split(dataset, args.data_split_path, seed=args.seed)

    data = valid_data if args.split == 'valid' else test_data
    loader = DataLoader(data, batch_size=args.test_batch, shuffle=False, collate_fn=collate_fn)

    model = CrossCityLLMCPR(
        args,
        poi_num=dataset.poi_num,
        tag_num=dataset.tag_num,
        region_num=dataset.region_num,
        popularity_bias=dataset.poi_popularity,
    ).to(args.device)

    ckpt = os.path.join(args.save_path, args.ckpt_name)
    if not os.path.exists(ckpt):
        logger.log(f"Checkpoint not found: {ckpt}")
        logger.close_log()
        return

    load_checkpoint(model, ckpt, map_location=args.device)
    trainer = Trainer(model, args, logger)
    trainer.validate(loader, epoch_idx='EVAL', split_name=args.split.upper())
    logger.close_log()


if __name__ == '__main__':
    main()
