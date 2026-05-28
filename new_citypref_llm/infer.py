import argparse
import os

import torch
from torch.utils.data import DataLoader, Subset

from data import TravelDatasetV2, random_split
from main import build_parser
from model import CrossCityLLMCPR
from spot_utils import collate_fn
from trainer import load_checkpoint


def parse_args():
    parser = build_parser()
    parser.set_defaults(mode='test')
    parser.add_argument('--sample_index', type=int, default=0)
    parser.add_argument('--split', type=str, default='test', choices=['valid', 'test'])
    return parser.parse_args()


def main():
    args = parse_args()
    args.save_path = os.path.join(args.save_path, args.name)

    ori_data_path = args.ori_data_enriched if args.use_enriched_data else args.ori_data
    dst_data_path = args.dst_data_enriched if args.use_enriched_data else args.dst_data
    dataset = TravelDatasetV2(args, ori_data_path, dst_data_path, args.trans_data)
    _, valid_data, test_data = random_split(dataset, args.data_split_path, seed=args.seed)
    src_subset = valid_data if args.split == 'valid' else test_data

    safe_idx = max(0, min(args.sample_index, len(src_subset) - 1))
    single = Subset(src_subset, [safe_idx])
    loader = DataLoader(single, batch_size=1, shuffle=False, collate_fn=collate_fn)

    model = CrossCityLLMCPR(
        args,
        poi_num=dataset.poi_num,
        tag_num=dataset.tag_num,
        region_num=dataset.region_num,
        popularity_bias=dataset.poi_popularity,
        poi_coord_tensor=dataset.poi_coord_tensor,
        city_sample_count=dataset.region_sample_count_tensor,
    ).to(args.device)

    ckpt = os.path.join(args.save_path, args.ckpt_name)
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

    load_checkpoint(model, ckpt, map_location=args.device)
    model.eval()

    with torch.no_grad():
        for batch_data in loader:
            uid = batch_data[0].item()
            dst_gt = batch_data[2][0]

            batch = {
                'uid': batch_data[0].to(args.device),
                'ori_ck': batch_data[1].to(args.device),
                'dst_ck': batch_data[2].to(args.device),
                'masked_dst_ck': batch_data[3].to(args.device),
                'o_hour': batch_data[4].to(args.device),
                'd_hour': batch_data[5].to(args.device),
                'masked_d_h': batch_data[6].to(args.device),
                'ori_t': batch_data[7].to(args.device),
                'dst_t': batch_data[8].to(args.device),
                'ori_l': batch_data[9].to(args.device),
                'dst_l': batch_data[10].to(args.device),
                'ori_pad': batch_data[11].to(args.device),
                'dst_pad': batch_data[12].to(args.device),
                'ori_rg': batch_data[13].to(args.device),
                'dst_rg': batch_data[14].to(args.device),
                'ori_tag': batch_data[15].to(args.device),
                'dst_tag': batch_data[16].to(args.device),
                'query_start_poi': batch_data[17].to(args.device),
                'query_start_hour': batch_data[18].to(args.device),
                'query_end_poi': batch_data[19].to(args.device),
                'query_end_hour': batch_data[20].to(args.device),
                'query_len': batch_data[21].to(args.device),
                'user_profile': batch_data[22].to(args.device),
                'query_vec': batch_data[23].to(args.device),
                'home_prompt_text': batch_data[24],
            }

            pred = model.predict(batch)[0].detach().cpu().tolist()
            gt = [x for x in dst_gt.tolist() if x != 0]
            pred = [x for x in pred if x != 0]

            print(f"User(sample-id): {uid}")
            print(f"GT route: {gt}")
            print(f"Pred route: {pred}")


if __name__ == '__main__':
    main()
