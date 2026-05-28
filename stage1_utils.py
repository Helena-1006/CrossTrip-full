import torch
from spot_utils import collate_fn as origin_collate_fn

def contrastive_collate_fn(batch):
    """
    Batch 结构: List corresponding to [ (item_a, item_b), (item_a, item_b), ... ]
    我们需要把它拆分成 Batch_A and Batch_B，分别调用原始的 padding 逻辑
    """
    batch_a = []
    batch_b = []
    
    for item_a, item_b in batch:
        batch_a.append(item_a)
        batch_b.append(item_b)
        
    #这就复用了 spot_utils 里复杂的 padding 逻辑
    res_a = origin_collate_fn(batch_a) 
    res_b = origin_collate_fn(batch_b)
    
    # 返回两个大包
    return res_a, res_b