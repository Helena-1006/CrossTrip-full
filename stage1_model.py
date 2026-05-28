import torch
import torch.nn as nn
import torch.nn.functional as F

class PreferenceEncoder(nn.Module):
    def __init__(self, args):
        super(PreferenceEncoder, self).__init__()
        self.poi_emb = nn.Embedding(args.poi_num, args.hidden_size)
        
        # 简单的序列编码器 (GRU)，用于提取 Home 和 OOT 的整体偏好
        self.rnn = nn.GRU(args.hidden_size, args.hidden_size, batch_first=True)
        
    def forward(self, input_seq):
        # input_seq: [Batch, Seq_Len] (POI IDs)
        # 简单实现：Embedding -> RNN -> Last Hidden State
        emb = self.poi_emb(input_seq)
        _, h_n = self.rnn(emb) # h_n: [1, Batch, Hidden]
        return h_n.squeeze(0) # [Batch, Hidden]

class ContrastiveModel(nn.Module):
    def __init__(self, args):
        super(ContrastiveModel, self).__init__()
        self.encoder = PreferenceEncoder(args)
        self.margin = 0.5 # Margin for triplet loss
        
    def forward(self, batch_a, batch_b):
        """
        batch_a/b 是包含 (uid, o_ck, d_ck...) 的大元组列表
        根据 data.py 和 spot_utils.py 的解包顺序:
        idx 1: pad_ori_ck (Home Sequence)
        idx 2: pad_dst_ck (OOT Sequence)
        注意：一定要确认 spot_utils.py 中 return 的顺序
        """
        # 此时 batch_a 已经是被 unpack 过的 list of tensors
        home_seq_a = batch_a[1] 
        oot_seq_a = batch_a[2]
        
        home_seq_b = batch_b[1]
        oot_seq_b = batch_b[2] # B 的 OOT，与 A 的 OOT 地点不同
        
        # 1. 编码
        # Home 偏好 (理论上 A 和 B 的 Home 是一样的，取一个即可)
        rep_home = self.encoder(home_seq_a) 
        
        # OOT 偏好
        rep_oot_a = self.encoder(oot_seq_a)
        rep_oot_b = self.encoder(oot_seq_b)
        
        # 2. 对比损失计算
        # 目标 A: 同一个用户的不同外地行为 (OOT_A, OOT_B) 应该相似 -> Pull
        loss_pos = 1 - F.cosine_similarity(rep_oot_a, rep_oot_b).mean()
        
        # 目标 B: 外地行为 (OOT) 与 本地行为 (Home) 应该疏远 (解耦) -> Push
        # 可以在 OOT_A 和 Home 之间，以及 OOT_B 和 Home 之间算
        cos_sim_home = F.cosine_similarity(rep_oot_a, rep_home)
        # 我们希望 cos_sim_home 越小越好 (越接近0或负数)
        # 使用 Hinge Loss: max(0, sim - margin) ??? 
        # 或者简单的: loss_neg = max(0, cos_sim_home).mean() 强迫正交
        # 这里使用一种软约束：让 OOT_A 和 OOT_B 的距离 比 OOT_A 和 Home 的距离 更近
        
        # Triplet Loss 思想: d(Anchor, Pos) < d(Anchor, Neg) + margin
        # Anchor: OOT_A, Pos: OOT_B, Neg: Home
        # Cosine Distance = 1 - Cosine Sim
        d_pos = 1 - F.cosine_similarity(rep_oot_a, rep_oot_b)
        d_neg = 1 - F.cosine_similarity(rep_oot_a, rep_home)
        
        loss_triplet = torch.clamp(d_pos - d_neg + self.margin, min=0.0).mean()
        
        return loss_triplet, rep_oot_a, rep_oot_b