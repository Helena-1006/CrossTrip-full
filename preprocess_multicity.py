import os
import pickle
import pandas as pd
from collections import defaultdict
import random

def generate_multicity_pairs(dataset_path):
    print(f"正在处理数据集: {dataset_path}")
    
    travel_path = os.path.join(dataset_path, 'travel.txt')
    
    # 读取 travel.txt
    # 假设格式: index, user_id, home_city, oot_city
    # 注意：TravelDataset 中是用 list(map...) 读取的，这里我们模拟该过程
    with open(travel_path, 'r') as f:
        # 保存行号(dataset index), user_id, oot_city
        raw_data = []
        for line_idx, line in enumerate(f):
            parts = line.strip().split('\t')
            # travel.txt format verification based on your example:
            # col 0: index?, col 1: user_id, col 2: home_city, col 3: oot_city
            if len(parts) >= 4:
                uid = parts[1]
                oot_city = parts[3]
                raw_data.append({'idx': line_idx, 'uid': uid, 'oot_city': oot_city})

    # 按 User 分组
    user_records = defaultdict(list)
    for row in raw_data:
        user_records[row['uid']].append(row)
    # 在 raw_data 读取完成后添加
    unique_users = set(row['uid'] for row in raw_data)

    # 筛选多城市用户并构建 pair
    # 结构: lists of (idx_A, idx_B)
    stage1_pairs = []
    
    multi_city_user_count = 0
    
    for uid, records in user_records.items():
        # 获取该用户去过的所有不同城市
        visited_cities = set(r['oot_city'] for r in records)
        
        if len(visited_cities) < 2:
            continue
            
        multi_city_user_count += 1
        
        # 策略：即使该用户有3个外地城市，我们也可以两两组合，或者随机采样一对
        # 这里演示：随机采样两个不同的城市记录作为一对，用于对比学习
        # 为了数据充分利用，可以把所有不同城市的组合都加进去
        
        # 按城市分组记录索引
        city_to_indices = defaultdict(list)
        for r in records:
            city_to_indices[r['oot_city']].append(r['idx'])
            
        cities = list(city_to_indices.keys())
        
        # 简单的两两组合策略（Combinations）
        for i in range(len(cities)):
            for j in range(i + 1, len(cities)):
                city_a = cities[i]
                city_b = cities[j]
                
                # 从城市A的所有记录中随机选一条，从城市B的所有记录中随机选一条
                # (也可以遍历所有组合，但这会导致数据极其庞大，建议每个pair只采样一次或数次)
                idx_a = random.choice(city_to_indices[city_a])
                idx_b = random.choice(city_to_indices[city_b])
                
                stage1_pairs.append((idx_a, idx_b))

    print(f"处理完成！")
    print(f"原始记录总数: {len(raw_data)}")
    print(f"不重复的用户总数: {len(unique_users)}")
    print(f"多外地城市用户数: {multi_city_user_count}")
    print(f"生成的对比学习对 (Pairs) 总数: {len(stage1_pairs)}")
    
    save_file = os.path.join(dataset_path, 'stage1_pairs.pkl')
    with open(save_file, 'wb') as f:
        pickle.dump(stage1_pairs, f)
    print(f"索引对主要保存至: {save_file}")

if __name__ == '__main__':
    # 修改这里的路径为你实际的路径
    dataset_dir = r'../Foursquare' 
    generate_multicity_pairs(dataset_dir)