import pickle
import torch
import numpy as np
from sklearn.cluster import KMeans
import argparse
import os

def run_clustering():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default='Foursquare')
    parser.add_argument('--n_clusters', type=int, default=16, help="外地偏好Pool的大小 (K值)")
    parser.add_argument('--seed', type=int, default=2050)
    args = parser.parse_args()
    
    # 路径设置
    data_dir = f'../{args.dataset_name}'
    vector_path = os.path.join(data_dir, 'oot_vectors.pkl')
    save_path = os.path.join(data_dir, 'preference_pool.pt')
    
    print(f"Loading vectors from {vector_path}...")
    with open(vector_path, 'rb') as f:
        vectors = pickle.load(f) # shape: [N_samples, Hidden]
        
    print(f"Loaded {vectors.shape[0]} vectors. Shape: {vectors.shape}")
    
    # 聚类
    print(f"Running K-Means with K={args.n_clusters}...")
    kmeans = KMeans(n_clusters=args.n_clusters, random_state=args.seed, n_init=10)
    kmeans.fit(vectors)
    
    # 获取聚类中心
    centroids = kmeans.cluster_centers_ # shape: [K, Hidden]
    
    # 转换为 Tensor 并保存
    pool_tensor = torch.tensor(centroids, dtype=torch.float32)
    torch.save(pool_tensor, save_path)
    
    print(f"Pool saved to {save_path}. Shape: {pool_tensor.shape}")

if __name__ == '__main__':
    run_clustering()