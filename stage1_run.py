import argparse
import torch
import os
import pickle
import numpy as np
from torch.utils.data import DataLoader
from torch.optim import Adam

from data import ContrastiveDataset
from stage1_model import ContrastiveModel
from stage1_utils import contrastive_collate_fn
# 引用工具库
import spot_utils

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default='Foursquare')
    parser.add_argument('--ori_data', type=str, default=f'../Foursquare/home.txt')
    parser.add_argument('--dst_data', type=str, default=f'../Foursquare/oot.txt')
    parser.add_argument('--trans_data', type=str, default=f'../Foursquare/travel.txt')
    parser.add_argument('--kg_path', type=str, default=f'../Foursquare/kg.txt') # dataset init需要
    # 注意这里必须要是第一步生成的 pairs 路径
    parser.add_argument('--pair_path', type=str, default=f'../Foursquare/stage1_pairs.pkl')
    
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--hidden_size', type=int, default=64)
    parser.add_argument('--epoch', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--device', type=str, default="cuda:0")
    parser.add_argument('--poi_num', type=int, default=23884) # 需要根据数据集实际情况调整
    
    # TravelDataset 需要的一些额外 dummy 参数
    parser.add_argument('--entity_num_per_poi', type=int, default=2) 
    
    args = parser.parse_args()
    device = torch.device(args.device)
    
    print("Loading Stage 1 Data...")
    # 初始化 Dataset
    # 注意：TravelDataset 内部可能会读取 kg_data，如果不使用 KG 可以注释掉 data.py 里的相关行
    # 这里假设 data.py 里的 init 逻辑是能够跑通的
    dataset = ContrastiveDataset(args, args.ori_data, args.dst_data, args.trans_data, args.pair_path)
    
    # 动态获取 POI 数量，防止 Embedding 越界 (TravelDataset里有计算 entity_count 等)
    # 假设 dataset.args.entity_count 或者类似的属性存在，或者简单点直接用 args 设置一个大值
    # 更好的是在 dataset 初始化后查看 dataset.poi_num (如果代码里有统计的话)
    # 根据 data.py 代码，可能没有显式存储 poi_num，通常做法是预设定一个最大ID
    
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=contrastive_collate_fn)
    
    print("Initializing Contrastive Model...")
    model = ContrastiveModel(args).to(device)
    optimizer = Adam(model.parameters(), lr=args.lr)
    
    print("Start Training...")
    model.train()
    
    # 用于收集纯净外地偏好的列表
    all_pure_oot_vectors = []
    
    for e in range(args.epoch):
        total_loss = 0
        for batch_data_a, batch_data_b in dataloader:
            # Unpack to GPU
            batch_data_a = [x.to(device) for x in batch_data_a]
            batch_data_b = [x.to(device) for x in batch_data_b]
            
            optimizer.zero_grad()
            
            loss, vec_a, vec_b = model(batch_data_a, batch_data_b)
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
            # 在最后一个 epoch，收集向量用于聚类
            if e == args.epoch - 1:
                all_pure_oot_vectors.append(vec_a.detach().cpu().numpy())
                all_pure_oot_vectors.append(vec_b.detach().cpu().numpy())
                
        print(f"Epoch {e}, Loss: {total_loss / len(dataloader):.4f}")

    # 保存模型
    torch.save(model.state_dict(), f'../{args.dataset_name}/stage1_model.pth')
    
    # 保存提取出的向量
    all_pure_oot_vectors = np.concatenate(all_pure_oot_vectors, axis=0)
    print(f"Collected {all_pure_oot_vectors.shape[0]} OOT preference vectors.")
    
    with open(f'../{args.dataset_name}/oot_vectors.pkl', 'wb') as f:
        pickle.dump(all_pure_oot_vectors, f)
        
    print("Stage 1 Finished. Vectors saved for clustering.")

if __name__ == '__main__':
    main()