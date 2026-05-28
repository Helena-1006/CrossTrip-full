import argparse
import json
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader, Subset

from data import TravelDatasetV2, random_split
from model import CrossCityLLMCPR
from spot_utils import collate_fn, set_seeds
from trainer import load_checkpoint

matplotlib.rcParams['font.family'] = 'serif'


def unpack_batch(batch_data, device):
    (
        uid,
        pad_ori_ck,
        pad_dst_ck,
        pad_masked_dst_ck,
        pad_o_hour,
        pad_d_hour,
        pad_masked_d_hour,
        pad_ori_t,
        pad_dst_t,
        pad_ori_l,
        pad_dst_l,
        ori_pad,
        dst_pad,
        ori_rg,
        dst_rg,
        pad_ori_tag,
        pad_dst_tag,
        query_start_poi,
        query_start_hour,
        query_end_poi,
        query_end_hour,
        query_len,
        user_profile,
        query_vec,
        home_prompt_text,
    ) = batch_data

    return {
        'uid': uid.to(device),
        'ori_ck': pad_ori_ck.to(device),
        'dst_ck': pad_dst_ck.to(device),
        'masked_dst_ck': pad_masked_dst_ck.to(device),
        'o_hour': pad_o_hour.to(device),
        'd_hour': pad_d_hour.to(device),
        'masked_d_h': pad_masked_d_hour.to(device),
        'ori_t': pad_ori_t.to(device),
        'dst_t': pad_dst_t.to(device),
        'ori_l': pad_ori_l.to(device),
        'dst_l': pad_dst_l.to(device),
        'ori_pad': ori_pad.to(device),
        'dst_pad': dst_pad.to(device),
        'ori_rg': ori_rg.to(device),
        'dst_rg': dst_rg.to(device),
        'ori_tag': pad_ori_tag.to(device),
        'dst_tag': pad_dst_tag.to(device),
        'query_start_poi': query_start_poi.to(device),
        'query_start_hour': query_start_hour.to(device),
        'query_end_poi': query_end_poi.to(device),
        'query_end_hour': query_end_hour.to(device),
        'query_len': query_len.to(device),
        'user_profile': user_profile.to(device),
        'query_vec': query_vec.to(device),
        'home_prompt_text': home_prompt_text,
    }


def extract_z_stack(model, dataloader, device):
    model.eval()
    all_z = []
    with torch.no_grad():
        for batch_data in dataloader:
            batch = unpack_batch(batch_data, device)
            _, _, z_stack, _ = model._encode_home(batch)
            all_z.append(z_stack.detach().cpu().numpy())
    if not all_z:
        raise RuntimeError('No z_stack extracted. Check test loader and split.')
    return np.concatenate(all_z, axis=0)


def build_tsne(perplexity, seed, max_iter):
    kwargs = dict(
        n_components=2,
        perplexity=perplexity,
        random_state=seed,
        init='pca',
        learning_rate='auto',
    )
    try:
        return TSNE(max_iter=max_iter, **kwargs)
    except TypeError:
        return TSNE(n_iter=max_iter, **kwargs)


def pairwise_cosine_stats(z):
    # z: [N, K, H]
    z_t = torch.from_numpy(z).float()
    z_t = torch.nn.functional.normalize(z_t, p=2, dim=-1)
    n, k, _ = z_t.shape
    out = []
    abs_vals = []
    raw_vals = []
    for i in range(k):
        for j in range(i + 1, k):
            cos_ij = (z_t[:, i, :] * z_t[:, j, :]).sum(dim=-1)
            mean_raw = float(cos_ij.mean().item())
            mean_abs = float(cos_ij.abs().mean().item())
            out.append({'pair': f'{i}-{j}', 'mean_cosine': mean_raw, 'mean_abs_cosine': mean_abs})
            raw_vals.append(mean_raw)
            abs_vals.append(mean_abs)
    return {
        'pairwise': out,
        'mean_pairwise_cosine': float(np.mean(raw_vals)) if raw_vals else 0.0,
        'mean_pairwise_abs_cosine': float(np.mean(abs_vals)) if abs_vals else 0.0,
        'N': int(n),
        'K': int(k),
    }


def run_tsne(z, perplexity, seed, max_iter):
    n, k, h = z.shape
    x = z.transpose(1, 0, 2).reshape(k * n, h)
    y = np.repeat(np.arange(k), n)
    tsne = build_tsne(perplexity=perplexity, seed=seed, max_iter=max_iter)
    x2d = tsne.fit_transform(x)
    return x2d, y


def plot_tsne_comparison(z_full, z_ablation, dataset_name, output_dir, perplexity, seed, max_iter):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6', '#f39c12', '#1abc9c']

    for ax, z_data, title in [
        (axes[0], z_full, '(a) w/ Decoupling Loss'),
        (axes[1], z_ablation, '(b) w/o Decoupling Loss'),
    ]:
        n, k, _ = z_data.shape
        x2d, y = run_tsne(z_data, perplexity=perplexity, seed=seed, max_iter=max_iter)
        for i in range(k):
            mask = y == i
            ax.scatter(
                x2d[mask, 0],
                x2d[mask, 1],
                c=colors[i % len(colors)],
                label=f'Factor {i + 1}',
                alpha=0.65,
                s=14,
                edgecolors='none',
            )
        ax.set_title(title, fontsize=13)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.legend(loc='upper right', fontsize=9, frameon=False)
        ax.text(0.01, 0.01, f'N={n}, K={k}', transform=ax.transAxes, fontsize=8)

    plt.suptitle(f't-SNE visualization of disentangled preference factors ({dataset_name})', fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    os.makedirs(output_dir, exist_ok=True)
    base = os.path.join(output_dir, f'tsne_decouple_{dataset_name}')
    plt.savefig(base + '.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(base + '.png', dpi=300, bbox_inches='tight')
    plt.close()
    return base + '.png', base + '.pdf'


def build_model(args, dataset, ckpt_path):
    model = CrossCityLLMCPR(
        args,
        poi_num=dataset.poi_num,
        tag_num=dataset.tag_num,
        region_num=dataset.region_num,
        popularity_bias=dataset.poi_popularity,
        poi_coord_tensor=dataset.poi_coord_tensor,
        city_sample_count=dataset.region_sample_count_tensor,
    )
    model = load_checkpoint(model, ckpt_path, map_location=args.device)
    model = model.to(args.device)
    return model


def parse_args():
    parser = argparse.ArgumentParser(description='t-SNE for preference decoupling (new_citypref_llm)')
    parser.add_argument('--dataset_name', type=str, required=True)
    parser.add_argument('--ckpt_full', type=str, required=True)
    parser.add_argument('--ckpt_ablation', type=str, required=True)
    parser.add_argument('--ori_data', type=str, required=True)
    parser.add_argument('--dst_data', type=str, required=True)
    parser.add_argument('--trans_data', type=str, required=True)
    parser.add_argument('--data_split_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='./pipeline_runs/tsne_decouple/figures')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--seed', type=int, default=2050)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--max_samples', type=int, default=0)
    parser.add_argument('--perplexity', type=float, default=30.0)
    parser.add_argument('--tsne_max_iter', type=int, default=1000)

    # Keep model/data args aligned with main.py defaults.
    parser.add_argument('--use_enriched_data', action='store_true')
    parser.add_argument('--ori_data_enriched', type=str, default='')
    parser.add_argument('--dst_data_enriched', type=str, default='')
    parser.add_argument('--split_strategy', type=str, default='legacy', choices=['legacy', 'pair_robust'])
    parser.add_argument('--split_singleton_to_train', type=int, default=1, choices=[0, 1])

    parser.add_argument('--hidden_size', type=int, default=128)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--nhead', type=int, default=4)
    parser.add_argument('--semantic_layers', type=int, default=2)
    parser.add_argument('--seq_num_layers', type=int, default=2)
    parser.add_argument('--soft_prompt_len', type=int, default=8)
    parser.add_argument('--pref_factor_k', type=int, default=4)
    parser.add_argument('--use_mamba_backbone', type=int, default=1, choices=[0, 1])
    parser.add_argument('--mamba_d_state', type=int, default=16)
    parser.add_argument('--mamba_d_conv', type=int, default=4)
    parser.add_argument('--mamba_expand', type=int, default=2)
    parser.add_argument('--mamba_strict', action='store_true')

    parser.add_argument('--semantic_backend', type=str, default='qwen', choices=['qwen', 'fallback'])
    parser.add_argument('--llm_model_name', type=str, default='Qwen/Qwen2.5-1.5B-Instruct')
    parser.add_argument('--llm_cache_dir', type=str, default='../../code/params')
    parser.add_argument('--llm_max_length', type=int, default=256)
    parser.add_argument('--llm_micro_batch', type=int, default=4)
    parser.add_argument('--llm_dtype', type=str, default='bfloat16', choices=['float16', 'bfloat16', 'float32'])
    parser.add_argument('--llm_max_traj_tokens', type=int, default=64)
    parser.add_argument('--llm_fallback_names', type=str, default='Qwen/Qwen2.5-1.5B-Instruct')
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
    parser.add_argument('--lambda_pair', type=float, default=0.2)
    parser.add_argument('--pair_max_future', type=int, default=4)
    parser.add_argument('--use_beam_search', type=int, default=1, choices=[0, 1])
    parser.add_argument('--beam_size', type=int, default=4)
    parser.add_argument('--beam_len_penalty', type=float, default=0.2)
    parser.add_argument('--use_no_repeat_mask', type=int, default=1, choices=[0, 1])
    parser.add_argument('--pop_bias_scale', type=float, default=0.1)
    parser.add_argument('--ellipse_filter', action='store_true')
    parser.add_argument('--ellipse_beta', type=float, default=1.2)
    parser.add_argument('--city_memory_momentum', type=float, default=0.95)
    parser.add_argument('--city_memory_prior_k', type=float, default=20.0)
    parser.add_argument('--eta_fixed', type=float, default=-1.0)
    parser.add_argument('--enforce_start_end_constraints', type=int, default=1, choices=[0, 1])
    parser.add_argument('--decode_constraint_mode', type=str, default='hard', choices=['hard', 'soft'])
    parser.add_argument('--soft_constraint_scale', type=float, default=0.2)
    parser.add_argument('--soft_constraint_dist_emb_dim', type=int, default=32)
    parser.add_argument('--ablate_generator_no_spatial_context', type=int, default=0, choices=[0, 1])
    parser.add_argument('--profile_dim', type=int, default=7)
    parser.add_argument('--query_dim', type=int, default=7)
    parser.add_argument('--gen_loss_chunk_len', type=int, default=16)

    return parser.parse_args()


def main():
    args = parse_args()
    set_seeds(args.seed)

    if not os.path.exists(args.ckpt_full):
        raise FileNotFoundError(f'ckpt_full not found: {args.ckpt_full}')
    if not os.path.exists(args.ckpt_ablation):
        raise FileNotFoundError(f'ckpt_ablation not found: {args.ckpt_ablation}')

    ori_data_path = args.ori_data_enriched if args.use_enriched_data and args.ori_data_enriched else args.ori_data
    dst_data_path = args.dst_data_enriched if args.use_enriched_data and args.dst_data_enriched else args.dst_data

    print(f'[INFO] Loading dataset: {args.dataset_name}')
    dataset = TravelDatasetV2(args, ori_data_path, dst_data_path, args.trans_data)
    _, _, test_data = random_split(dataset, args.data_split_path, seed=args.seed, args=args)

    if args.max_samples > 0 and len(test_data) > args.max_samples:
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(len(test_data), size=args.max_samples, replace=False)
        test_data = Subset(test_data, idx.tolist())

    if len(test_data) < 5:
        raise RuntimeError(f'Too few test samples ({len(test_data)}).')

    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    print(f'[INFO] Test set size: {len(test_data)}')

    print(f'[INFO] Loading full model: {args.ckpt_full}')
    model_full = build_model(args, dataset, args.ckpt_full)
    z_full = extract_z_stack(model_full, test_loader, args.device)
    del model_full
    torch.cuda.empty_cache()

    print(f'[INFO] Loading ablation model: {args.ckpt_ablation}')
    model_ab = build_model(args, dataset, args.ckpt_ablation)
    z_ab = extract_z_stack(model_ab, test_loader, args.device)
    del model_ab
    torch.cuda.empty_cache()

    n = min(z_full.shape[0], z_ab.shape[0])
    z_full = z_full[:n]
    z_ab = z_ab[:n]

    print('[INFO] Running t-SNE...')
    png_path, pdf_path = plot_tsne_comparison(
        z_full,
        z_ab,
        dataset_name=args.dataset_name,
        output_dir=args.output_dir,
        perplexity=args.perplexity,
        seed=args.seed,
        max_iter=args.tsne_max_iter,
    )

    full_stats = pairwise_cosine_stats(z_full)
    ab_stats = pairwise_cosine_stats(z_ab)
    payload = {
        'dataset_name': args.dataset_name,
        'ckpt_full': args.ckpt_full,
        'ckpt_ablation': args.ckpt_ablation,
        'metrics': {
            'full': full_stats,
            'ablation': ab_stats,
        },
    }
    os.makedirs(args.output_dir, exist_ok=True)
    json_path = os.path.join(args.output_dir, f'decouple_cosine_{args.dataset_name}.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print('[DONE] t-SNE completed.')
    print(f'[OUT] {png_path}')
    print(f'[OUT] {pdf_path}')
    print(f'[OUT] {json_path}')


if __name__ == '__main__':
    main()
