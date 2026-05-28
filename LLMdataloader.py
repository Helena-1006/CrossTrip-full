import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import pickle

class EnhancedDataLoader:
    """增强的数据加载器，包含POI类别树信息"""
    
    def __init__(self, home_path: str, oot_path: str):
        self.home_path = home_path
        self.oot_path = oot_path
        self.poi_cat_tree = {}  # POI ID -> 类别树映射
        self.poi_info = {}  # POI详细信息
        
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """加载home和oot数据，并提取类别树信息"""
        # 加载数据
        home_df = pd.read_csv(
            self.home_path, 
            sep='\t',
            header=None,
            names=['user_id', 'hometown', 'city', 'poi_id', 'timestamp', 
                   'poi_cat', 'poi_name', 'cat_tree']
        )
        
        oot_df = pd.read_csv(
            self.oot_path,
            sep='\t', 
            header=None,
            names=['user_id', 'hometown', 'city', 'poi_id', 'timestamp',
                   'poi_cat', 'poi_name', 'cat_tree']
        )
        
        # 构建POI信息字典
        self._build_poi_info(home_df)
        self._build_poi_info(oot_df)
        
        return home_df, oot_df
    
    def _build_poi_info(self, df: pd.DataFrame):
        """构建POI信息字典"""
        for _, row in df.iterrows():
            poi_id = row['poi_id']
            if poi_id not in self.poi_info:
                self.poi_info[poi_id] = {
                    'name': row['poi_name'],
                    'category': row['poi_cat'],
                    'cat_tree': row['cat_tree'],
                    'city': row['city']
                }
                self.poi_cat_tree[poi_id] = row['cat_tree']
    
    def get_poi_category_tree(self, poi_id: str) -> str:
        """获取POI的类别树"""
        return self.poi_cat_tree.get(poi_id, "Unknown")
    
    def get_poi_info(self, poi_id: str) -> Dict:
        """获取POI的完整信息"""
        return self.poi_info.get(poi_id, {})
    
    def build_trajectory_data(self, df: pd.DataFrame, user_id: int) -> List[Dict]:
        """构建用户轨迹数据，包含所需的所有信息"""
        user_data = df[df['user_id'] == user_id].sort_values('timestamp')
        
        trajectories = []
        for idx, row in user_data.iterrows():
            traj_info = {
                'poi_id': row['poi_id'],
                'poi_name': row['poi_name'],
                'category': row['poi_cat'],
                'cat_tree': row['cat_tree'],
                'timestamp': row['timestamp'],
                'city': row['city'],
                'hometown': row['hometown']
            }
            trajectories.append(traj_info)
        
        return trajectories