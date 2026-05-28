import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=500):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1) # [Max_Len, 1, H]
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [Seq_Len, Batch, H]
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class MyRouteModel(nn.Module):
    def __init__(self, args, pool_path=None):
        super(MyRouteModel, self).__init__()
        self.args = args
        self.hidden_size = args.hidden_size
        self.poi_num = args.poi_num
        
        # ================= 1. Embeddings =================
        self.poi_emb = nn.Embedding(self.poi_num, self.hidden_size, padding_idx=0)
        self.pos_encoder = PositionalEncoding(self.hidden_size)
        
        # ================= 2. Preference Encoder (Home -> Local -> OOT) =================
        # 用于编码本地历史 (Home Seq)
        self.home_encoder = nn.GRU(self.hidden_size, self.hidden_size, batch_first=True)
        
        # Preference Pool (加载 Phase 3 聚类好的外地偏好原型)
        if pool_path:
            pool_tensor = torch.load(pool_path) # [K, H]
            self.preference_pool = nn.Parameter(pool_tensor)
        else:
            # Fallback for testing
            self.preference_pool = nn.Parameter(torch.randn(16, self.hidden_size))

        # Cross-Attention: 用本地偏好去 Pool 里“查询”合适的外地偏好
        self.pref_attn = nn.MultiheadAttention(embed_dim=self.hidden_size, num_heads=4, batch_first=True)
        
        # ================= 3. Route Decoder (Seq2Seq Transformer) =================
        # 我们使用 Transformer Decoder，它接受 "Query + User Pref" 作为 Memory
        decoder_layer = nn.TransformerDecoderLayer(d_model=self.hidden_size, nhead=4, 
                                                   dim_feedforward=self.hidden_size*4, 
                                                   batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=2)
        
        # Query Merge Layer: 将 OOT偏好, Start POI, End POI 融合成 Transformer 的 Context
        self.query_fusion = nn.Linear(self.hidden_size, self.hidden_size)

        # Output Head
        self.output_layer = nn.Linear(self.hidden_size, self.poi_num)

    def get_local_preference(self, home_seq):
        """编码本地历史，获取本地偏好向量"""
        emb = self.poi_emb(home_seq) # [Batch, Seq, H]
        # Mask掉 padding (index 0)
        # 简单 GRU 最后一个 Hidden state 作为 User Vector
        _, h_n = self.home_encoder(emb) 
        return h_n.squeeze(0) # [Batch, H]

    def get_context_memory(self, home_seq, start_poi, end_poi):
        """
        构建生成路线所需的 Context (OD Query + User Preference)
        这将作为 Transformer Decoder 的 Cross-Attention Source (Memory)
        """
        # A. 计算 Local Preference
        local_pref = self.get_local_preference(home_seq) # [Batch, H]
        
        # B. 与 Pool 交互生成外地偏好 (Disentangled OOT Pref)
        # Query: Local [B, 1, H], Key/Val: Pool [B, K, H]
        query = local_pref.unsqueeze(1)
        batch_size = local_pref.size(0)
        pool_expanded = self.preference_pool.unsqueeze(0).expand(batch_size, -1, -1)
        
        attn_out, _ = self.pref_attn(query, pool_expanded, pool_expanded)
        oot_pref = attn_out # [Batch, 1, H]
        
        # C. 编码 OD (Origin-Destination)
        start_emb = self.poi_emb(start_poi).unsqueeze(1) # [Batch, 1, H]
        end_emb = self.poi_emb(end_poi).unsqueeze(1)     # [Batch, 1, H]
        
        # D. 融合构建 Memory
        # Memory 结构: [User_OOT_Pref, Start_Node_Info, End_Node_Info]
        # transformer decoder 将会对这 3 个 token 做 attention
        # 也可以拼接更多信息，如时间 embedding 等
        memory = torch.cat([oot_pref, start_emb, end_emb], dim=1) # [Batch, 3, H]
        
        return memory

    def generate_square_subsequent_mask(self, sz, device):
        """生成 Causal Mask，防止 Transformer 看见未来"""
        mask = (torch.triu(torch.ones(sz, sz, device=device)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, inputs):
        """
        训练阶段 Forward (Teacher Forcing)
        inputs: standard list from collate_fn
        inputs[1]: Home Seq (User History)
        inputs[2]: Ground Truth Route (OD Trip), padded with 0
        """
        home_seq = inputs[1]  # [Batch, Home_Len]
        gt_route = inputs[2]  # [Batch, Route_Len] (包含 Start 和 End)
        device = home_seq.device

        # 1. 提取 OD Query (Start & End)
        # Start 是第一个点
        start_poi = gt_route[:, 0]
        # End 是最后一个非0点。为了方便，我们假设数据预处理保证了最后一个非0是End
        # 这里用一个小 trick 获取最后一个非零元素的索引
        mask = (gt_route != 0).long()
        lengths = mask.sum(dim=1)
        # gather 获取 end_poi
        end_poi = gt_route.gather(1, (lengths - 1).unsqueeze(1)).squeeze(1)

        # 2. 构建 Context Memory
        memory = self.get_context_memory(home_seq, start_poi, end_poi) # [Batch, 3, H]

        # 3. 准备 Transformer Decoder 输入
        # Input: [Start, p1, p2, ..., pN] (去掉最后一个End，或者作为输入序列)
        # Target: [p1, p2, ..., pN, End] (预测下一个)
        # 通常做法：dec_input = gt_route[:, :-1], target = gt_route[:, 1:]
        
        dec_input = gt_route[:, :-1] # [B, L-1]
        dec_target = gt_route[:, 1:] # [B, L-1]
        
        # Padding Mask for Decoder Input (0 is padding)
        tgt_padding_mask = (dec_input == 0) # [B, L-1]
        
        # Causal Mask (Autoregressive constraint)
        seq_len = dec_input.size(1)
        tgt_mask = self.generate_square_subsequent_mask(seq_len, device)

        # Embedding & Positional Encoding
        tgt_emb = self.poi_emb(dec_input) # [B, L-1, H]
        # TransformerDecoder (batch_first=True)
        # Memory Masking 通常不需要，因为 Memory 只有 3 个 token 且全有效
        
        # [Decoder Forward]
        # tgt: 目标序列输入, memory: 编码好的环境(User+OD)
        output = self.transformer_decoder(
            tgt=tgt_emb, 
            memory=memory, 
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_padding_mask
        ) # [B, L-1, H]
        
        logits = self.output_layer(output) # [B, L-1, POI_Num]
        
        # 4. 计算 Loss
        # Flatten for CrossEntropy
        loss = F.cross_entropy(logits.reshape(-1, self.poi_num), dec_target.reshape(-1), ignore_index=0)
        
        return loss

    def predict(self, inputs):
        """
        推理阶段 Predict (Auto-regressive Generation)
        生成整条路线
        """
        home_seq = inputs[1]
        gt_route = inputs[2] 
        device = home_seq.device
        batch_size = home_seq.size(0)
        
        # 1. 获取 OD
        start_poi = gt_route[:, 0]
        mask = (gt_route != 0).long()
        lengths = mask.sum(dim=1) # 获取每个样本的真实长度
        end_poi = gt_route.gather(1, (lengths - 1).unsqueeze(1)).squeeze(1)

        # 2. 构建 Context Memory
        memory = self.get_context_memory(home_seq, start_poi, end_poi)

        # 3. 自回归生成
        # 初始化输入序列为 [Start_POI]
        curr_seq = start_poi.unsqueeze(1) # [Batch, 1]
        
        # 4. [关键修改] 动态生成循环
        # 我们需要生成的最大步数 = Batch 中最长的序列长度 - 1 (减去起点)
        # 或者为了保险，稍微给多一点冗余，但在你的任务中，长度是已知的约束
        max_required_len = lengths.max().item() 
        
        # 这里我们生成到 Batch 中最长的那个长度为止
        # 在后面 metric 计算时，再截断多余的部分
        for step in range(max_required_len - 1):
            tgt_emb = self.poi_emb(curr_seq)
            output = self.transformer_decoder(tgt=tgt_emb, memory=memory)
            last_step_out = output[:, -1, :] 
            logits = self.output_layer(last_step_out) 
            next_token = torch.argmax(logits, dim=-1).unsqueeze(1)
            curr_seq = torch.cat([curr_seq, next_token], dim=1)
            
        return curr_seq

# import math
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.nn import TransformerEncoder, TransformerEncoderLayer
# import random
# #change random file name(utils->spot_utils)
# from spot_utils import _L2_loss_mean
# from GAT import GAT
# from einops import repeat
# from torchdiffeq import odeint
# import torchquad
# from torch.distributions import Normal, Independent
# from torch.nn.utils.rnn import pad_sequence
# import numpy as np
# from trainer import top_np_recommendation
# from LLMmodel import GPT2UserPrefEncoder
# from LLMprompt import PromptBuilder
# from LLMconfig import LLMConfig

# # 置顶添加一个安全封装
# def safe_odeint(func, z0, t, rtol, atol, prefer_method='dopri5'):
#     """
#     尝试自适应 dopri5，失败回退 rk4，再失败返回常量轨迹。
#     z0: [1,d], t: [T]
#     返回: [T,1,d]
#     """
#     # 双精度提高稳定性
#     t64 = t.to(torch.float64)
#     z064 = z0.to(torch.float64)
#     # 至少两个时间点才积分
#     if t64.numel() < 2:
#         return z064.unsqueeze(0)  # [1,1,d]
#     try:
#         sol = odeint(func, z064, t64, rtol=rtol, atol=atol, method=prefer_method)
#         return sol
#     except AssertionError:
#         # 回退固定步长 rk4
#         step = (t64[-1] - t64[0]) / (t64.numel() - 1)
#         try:
#             sol = odeint(func, z064, t64, method='rk4', options={'step_size': step})
#             return sol
#         except Exception:
#             # 最终回退：复制初值
#             return z064.expand(t64.numel(), -1).unsqueeze(1)
#     except Exception:
#         return z064.expand(t64.numel(), -1).unsqueeze(1)
    
# def clean_time_grid(t: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
#     """
#     清理时间网格，确保严格递增：
#     1. 转为 float64
#     2. 排序
#     3. 去除相邻重复
#     4. 对仍非递增的位置做微扰修正
#     """
#     if t.numel() == 0:
#         return t
#     t, _ = torch.sort(t)
#     # 去重
#     t = torch.unique_consecutive(t)
#     if t.numel() <= 1:
#         return t
#     diff = t[1:] - t[:-1]
#     if torch.any(diff <= 0):
#         # 用很小的扰动保证递增
#         adjust = torch.cumsum((diff <= 0).to(t.dtype) * t.new_tensor(eps), dim=0)
#         t = torch.cat([t[:1], t[1:] + adjust])
#     return t
# # def clean_time_grid(t: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
# #     """
# #     清理时间网格，确保严格递增：
# #     1. 转为 float64
# #     2. 排序
# #     3. 去除相邻重复
# #     4. 对仍非递增的位置做微扰修正
# #     """
# #     t = t.to(torch.float64)
# #     if t.numel() == 0:
# #         return t
# #     t, _ = torch.sort(t)
# #     t = torch.unique_consecutive(t)
# #     if t.numel() <= 1:
# #         return t
# #     diff = t[1:] - t[:-1]
# #     if torch.any(diff <= 0):
# #         # 生成累积微调，保证严格递增
# #         adjust = torch.cumsum((diff <= 0).double() * eps, dim=0)
# #         t[1:] = t[1:] + adjust
# #     return t

# class Encoder(nn.Module):
#     """Encoder mapping context sequences to parameters of the posterior q(z1)."""

#     def __init__(
#         self, poi_size,
#         d_z: int,
#         d_model: int, n_attn_heads: int, n_tf_layers: int, dropout_prob: float = 0.0,
#     ) -> None:
#         super().__init__()

#         self.time_proj = nn.Linear(1, d_model, bias=False)
#         self.space_proj = nn.Linear(2, d_model, bias=False)
#         self.poi_proj = nn.Linear(d_model, d_model, bias=False)
#         self.poi_emb = POIEmbeddings(poi_size, d_model)
#         self.transformer_stack = nn.ModuleList([
#             nn.TransformerEncoderLayer(
#                 d_model=d_model,
#                 nhead=n_attn_heads,
#                 dim_feedforward=2 * d_model,
#                 batch_first=True,
#                 dropout=dropout_prob,
#             ) for _ in range(n_tf_layers)
#         ])

#         self.gamma_proj = nn.Linear(d_model, d_z)
#         self.tau_proj = nn.Linear(d_model, d_z)

#         self.agg_token = nn.Parameter(torch.empty((1, 1, d_model)))
#         nn.init.xavier_uniform_(self.agg_token)

#     def forward(self, d_t, d_l, d_emb, d_pad):
#         """Maps context sequences `x` to parameters of the posterior q(z1)."""

#         t_emb = self.time_proj(d_t.to(torch.float32).unsqueeze(-1))
#         coords_emb = self.space_proj(d_l.to(torch.float32))
#         # poi_emb = self.poi_proj(d_emb)
#         poi_emb = self.poi_emb(d_emb)

#         x = torch.cat(
#             [
#                 t_emb + coords_emb + poi_emb,
#                 repeat(self.agg_token, "() () d -> b () d", b=d_t.shape[0]),
#             ],
#             dim=1,
#         )
        
#         # 修正 mask 长度不匹配问题
#         # x 的长度是 d_t.shape[1] + 1
#         # d_pad 的长度可能比 d_t 长 (例如包含了 target 的位置)
#         # 我们只取 d_pad 前 d_t.shape[1] 位
        
#         seq_len = d_t.shape[1]
#         current_pad = d_pad[:, :seq_len]
        
#         batch_size = d_pad.size(0)
#         agg_mask = torch.zeros((batch_size, 1), dtype=torch.bool, device=d_pad.device) # False = keep
        
#         # d_pad=True 是有效，Transformer 需要 True=Mask(Drop)，所以取反 ~current_pad
#         tf_mask = torch.cat([~current_pad, agg_mask], dim=1)

#         for layer in self.transformer_stack:
#             x = layer(x, src_key_padding_mask=tf_mask)

#         x = x[:, -1, :]

#         return x, self.gamma_proj(x), torch.nn.functional.softplus(self.tau_proj(x))
#     # def forward(self, d_t, d_l, d_emb, d_pad):
#     #     """Maps context sequences `x` to parameters of the posterior q(z1)."""

#     #     t_emb = self.time_proj(d_t.to(torch.float32).unsqueeze(-1))
#     #     coords_emb = self.space_proj(d_l.to(torch.float32))
#     #     # poi_emb = self.poi_proj(d_emb)
#     #     poi_emb = self.poi_emb(d_emb)

#     #     x = torch.cat(
#     #         [
#     #             t_emb + coords_emb + poi_emb,
#     #             repeat(self.agg_token, "() () d -> b () d", b=d_t.shape[0]),
#     #         ],
#     #         dim=1,
#     #     )
#     #     # x = torch.cat(
#     #     #     [
#     #     #         t_emb + poi_emb,
#     #     #         repeat(self.agg_token, "() () d -> b () d", b=d_t.shape[0]),
#     #     #     ],
#     #     #     dim=1,
#     #     # )

#     #     for layer in self.transformer_stack:
#     #         x = layer(x, src_key_padding_mask=d_pad)

#     #     x = x[:, -1, :]

#     #     return x, self.gamma_proj(x), torch.nn.functional.softplus(self.tau_proj(x))  # 更稳定

# def _nearest_interpolate(t_eval, t, z, ind_left, ind_right):
#     dist_left = torch.abs(t_eval - t[ind_left])
#     dist_right = torch.abs(t_eval - t[ind_right])
#     nearer_right = dist_right < dist_left
#     return torch.where(nearer_right.unsqueeze(1), z[ind_right], z[ind_left])


# def _linear_interpolate(t_eval, t, z, ind_left, ind_right):
#     t_left = t[ind_left]
#     t_right = t[ind_right]
#     weight_right = (t_eval - t_left) / (t_right - t_left + 1e-3)
#     weight_left = 1 - weight_right
#     return weight_left.unsqueeze(1) * z[ind_left] + weight_right.unsqueeze(1) * z[ind_right]

# def interpolate(t_eval, t, z, method: str = "nearest"):
#     """
#     Interpolates values at specified evaluation points.

#     Args:
#         t_eval (Tensor): The evaluation time points, shape (n,).
#         t (Tensor): The trajectory time points, shape (time,).
#         z (Tensor): The trajectory values at time points `t`, shape (time, d_z).
#         method (str, optional): The interpolation method ('nearest' or 'linear'). Defaults to 'nearest'.

#     Returns:
#         Tensor: Interpolated values at `t_eval`.
#     """
#     if method not in {"nearest", "linear"}:
#         raise ValueError(f"Interpolation method {method} is not supported.")

#     ind_right = torch.searchsorted(t, t_eval) # 查找 t_eval 在时间序列 t 中的插入位置，返回的 ind_right 是右侧的索引
#     ind_left = ind_right - 1
#     ind_left.clamp_(min=0)
#     ind_right.clamp_(max=len(t) - 1)

#     if method == "nearest":
#         return _nearest_interpolate(t_eval, t, z, ind_left, ind_right)
#     else:  # method == "linear"
#         return _linear_interpolate(t_eval, t, z, ind_left, ind_right)

# def kl_norm_norm(mu0, mu1, sig0, sig1):
#     """Calculates KL divergence between two K-dimensional Normal
#         distributions with diagonal covariance matrices.

#     Args:
#         mu0: Mean of the first distribution. Has shape (*, K).
#         mu1: Mean of the second distribution. Has shape (*, K).
#         sig0: Diagonal of the covatiance matrix of the first distribution. Has shape (*, K).
#         sig1: Diagonal of the covatiance matrix of the second distribution. Has shape (*, K).

#     Returns:
#         KL divergence between the distributions. Has shape (*, 1).
#     """
#     assert mu0.shape == mu1.shape == sig0.shape == sig1.shape, (f"{mu0.shape=} {mu1.shape=} {sig0.shape=} {sig1.shape=}")
#     a = (sig0 / sig1).pow(2).sum(-1, keepdim=True)
#     b = ((mu1 - mu0).pow(2) / sig1**2).sum(-1, keepdim=True)
#     c = 2 * (torch.log(sig1) - torch.log(sig0)).sum(-1, keepdim=True)
#     kl = 0.5 * (a + b + c - mu0.shape[-1])
#     return kl

# def create_mlp(
#         input_size,
#         output_size,
#         hidden_size,
#         num_hidden_layers,
#         activation_func,
#         use_layer_norm=False,
#         use_dropout=False,
#         dropout_prob=0.5,
# ):
#     """
#     Create MLP with optional layer normalization and dropout.

#     Args:
#         input_size (int): The size of the input layer.
#         output_size (int): The size of the output layer.
#         hidden_size (int): The size of the hidden layers.
#         num_hidden_layers (int): The number of hidden layers.
#         activation_func (function): The nonlinear activation function to use.
#         use_layer_norm (bool): Whether to use layer normalization (default: False).
#         use_dropout (bool): Whether to use dropout (default: False).
#         dropout_prob (float): Dropout probability, used if use_dropout is True (default: 0.5).

#     Returns:
#         nn.Sequential: The constructed MLP model.
#     """
#     layers = []
#     for i in range(num_hidden_layers):
#         if i == 0:
#             layers.append(nn.Linear(input_size, hidden_size))
#         else:
#             layers.append(nn.Linear(hidden_size, hidden_size))
#         if use_layer_norm:
#             layers.append(nn.LayerNorm(hidden_size))
#         layers.append(activation_func())
#         if use_dropout:
#             layers.append(nn.Dropout(dropout_prob))

#     layers.append(nn.Linear(hidden_size, output_size))
#     return nn.Sequential(*layers)

# # class DynamicsFunction(nn.Module):
# #     def __init__(self, f):
# #         super().__init__()
# #         self.f = f

# #     def forward(self, t, z):
# #         return self.f(z)

# # =============== 新的稳定动力学封装 ===============
# class DynamicsFunction(nn.Module):
#     def __init__(self, f, clamp_val=10.0):
#         super().__init__()
#         self.f = f
#         self.clamp_val = clamp_val

#     def forward(self, t, z):
#         out = self.f(z)
#         return torch.clamp(out, -self.clamp_val, self.clamp_val)

# #add(LXT)    
# # def integrate_fixed_rk4(func, z0, t_grid):
# #     """
# #     手写固定步长 RK4，避免 torchdiffeq 自适应 underflow。
# #     z0: [d]
# #     t_grid: [T] 递增
# #     返回: [T, d]
# #     """
# #     if t_grid.numel() == 1:
# #         return z0.unsqueeze(0)
# #     z = z0
# #     traj = [z0]
# #     for i in range(1, t_grid.numel()):
# #         dt = (t_grid[i] - t_grid[i-1])
# #         k1 = func(t_grid[i-1], z)
# #         k2 = func(t_grid[i-1] + 0.5*dt, z + 0.5*dt*k1)
# #         k3 = func(t_grid[i-1] + 0.5*dt, z + 0.5*dt*k2)
# #         k4 = func(t_grid[i-1] + dt, z + dt*k3)
# #         z = z + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)
# #         # z = z + dt * func(t_grid[i-1], z)
# #         # 防爆
# #         if torch.isnan(z).any() or torch.isinf(z).any():
# #             z = traj[0].clone()  # 回退初值
# #         traj.append(z)
# #     return torch.stack(traj, dim=0)
# def integrate_fixed_rk4(func, z0, t_grid):
#     if t_grid.numel() == 1:
#         return z0.unsqueeze(0)
#     z = z0
#     traj = [z0]
#     for i in range(1, t_grid.numel()):
#         dt = (t_grid[i] - t_grid[i-1])
#         k1 = func(t_grid[i-1], z)
#         if torch.isnan(k1).any() or torch.isinf(k1).any():
#             return z0.unsqueeze(0).expand(t_grid.numel(), -1)
#         k2 = func(t_grid[i-1] + 0.5*dt, z + 0.5*dt*k1)
#         if torch.isnan(k2).any() or torch.isinf(k2).any():
#             return z0.unsqueeze(0).expand(t_grid.numel(), -1)
#         k3 = func(t_grid[i-1] + 0.5*dt, z + 0.5*dt*k2)
#         if torch.isnan(k3).any() or torch.isinf(k3).any():
#             return z0.unsqueeze(0).expand(t_grid.numel(), -1)
#         k4 = func(t_grid[i-1] + dt, z + dt*k3)
#         if torch.isnan(k4).any() or torch.isinf(k4).any():
#             return z0.unsqueeze(0).expand(t_grid.numel(), -1)
#         z = z + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)
#         if torch.isnan(z).any() or torch.isinf(z).any():
#             return z0.unsqueeze(0).expand(t_grid.numel(), -1)
#         traj.append(z)
#     return torch.stack(traj, dim=0)
    
# class DynamicTimeGenerator(nn.Module):

#     def __init__(self, input_dim, hidden_dim):
#         super(DynamicTimeGenerator, self).__init__()
#         self.hidden_dim = hidden_dim
#         self.rnn_cell = nn.GRUCell(input_dim, hidden_dim)
#         self.fc_interval = nn.Linear(hidden_dim, 1)
#         self.init_input = nn.Parameter(torch.zeros(1, input_dim), requires_grad=False)

#     def forward(self, context, num_pred_list):
#         batch_size = context.size(0)
#         device = context.device

#         full_times_list = []
#         seq_lengths = []

#         for i in range(batch_size):
#             n_pred = num_pred_list[i]
#             hidden = context[i:i + 1]
#             rnn_input = self.init_input
#             cum_time = torch.zeros(1, device=device)
#             times = []
#             for _ in range(n_pred):
#                 hidden = self.rnn_cell(rnn_input, hidden)
#                 delta = F.softplus(self.fc_interval(hidden))
#                 cum_time = cum_time + delta.squeeze(1)
#                 times.append(cum_time.clone())
#             if len(times) > 0:
#                 times_tensor = torch.cat(times, dim=0)
#             else:
#                 times_tensor = torch.tensor([], device=device)
#             if times_tensor.numel() > 0:
#                 times_tensor = times_tensor / (times_tensor[-1] + 1e-6)
#             full_time = torch.cat(
#                 [torch.tensor([0.0], device=device), times_tensor, torch.tensor([1.0], device=device)], dim=0)
#             full_times_list.append(full_time)
#             seq_lengths.append(full_time.numel())

#         max_len = max(seq_lengths)
#         full_times_padded = torch.zeros(batch_size, max_len, device=device)
#         for i, t_seq in enumerate(full_times_list):
#             length = t_seq.numel()
#             full_times_padded[i, :length] = t_seq

#         return full_times_padded

# class ContinuousDecoder(nn.Module):
#     """Maps latent state z(t) and spatial coordinate x to u(t, x).

#     Attributes:
#         d_z (int): Dimensionality of the latent state.
#         d_x (int): Dimensionality of the spatial coodinates.
#         d_u (int): Dimensionality of the latent spatiotemporal state.
#         f (Module): Mapping from (z(t), x) to u(t, x).
#     """
#     def __init__(self, d_z, d_x, f, interp_method):
#         super().__init__()
#         # self.space_proj = nn.Linear(d_x, d_z, bias=False)
#         self.f = f # mlp
#         self.interp_method = interp_method

#     def forward(self, t_eval, t, z):
#         """Evaluates the latent spatiotemporal state u(t, x) for a single trajectory t, z.

#         Args:
#             t_eval: Evaluation time points, has shape (n, ).
#             t: Trajectory time points, has shape (time, ).
#             z: Trajectory values at time points `t`, has shape (time, d_z).

#         Returns:
#             Latent spatiotemporals state at (t_eval, x_eval). Has shape (n, d_u).
#         """
#         if t_eval.ndim != 1 or t.ndim != 1:
#             raise ValueError("t and t_eval should be a 1-dimensional arrays.")
#         if z.ndim != 2:
#             raise ValueError("z should be a 2-dimensional arrays.")
#         if t.shape[0] != z.shape[0]:
#             raise ValueError("t and z must have matching first dimension.")

#         z_eval = interpolate(t_eval, t, z, method=self.interp_method)
#         # return self.f(z_eval + self.space_proj(x_eval))
#         return self.f(z_eval)
# class IntensityCorrection(nn.Module):
#     def __init__(self, val=0):
#         super().__init__()
#         self.val = val

#     def forward(self, x):
#         # return torch.pow(x, 2) + self.val
#         return torch.exp(x) + self.val

# class POIEmbeddings(nn.Module):
#     def __init__(self, poi_size, poi_embed_dim):
#         super(POIEmbeddings, self).__init__()
#         self.emb = nn.Embedding(poi_size, poi_embed_dim)

#     def forward(self, traj):
#         x = self.emb(traj)
#         return x

# # =================Transformer framework================== #
# class TransformerModel(nn.Module):
#     def __init__(self, embed_size, nhead, nhid, nlayers, dropout=0.3):
#         super(TransformerModel, self).__init__()

#         # self.pos_encoder = PositionalEncoding(embed_size, dropout)
#         encoder_layers = TransformerEncoderLayer(embed_size, nhead, nhid, dropout, batch_first=True)
#         self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
#         self.embed_size = embed_size

#     def generate_square_subsequent_mask(self, sz):
#         mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
#         mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
#         return mask

#     def forward(self, src):
#         # src = src * math.sqrt(self.embed_size)
#         # src = self.pos_encoder(src)
#         x = self.transformer_encoder(src)

#         return x


# class Recommender(nn.Module):
#     def __init__(self, out_dim, poi_size):
#         super(Recommender, self).__init__()
#         self.fc = nn.Linear(out_dim, poi_size)
#         self.leaky_relu = nn.LeakyReLU(0.2)

#     def forward(self, outputs):
#         x = self.fc(outputs)
#         x = self.leaky_relu(x)
#         return x


# # ============================Penalty=============================== #
# class Drifting(nn.Module):
#     def __init__(self, beta):
#         super(Drifting, self).__init__()
#         self.beta = beta

#     def forward(self, fix_outputs, region_mask):

#         batch_size, seq_len, _ = fix_outputs.size()
#         max_num_moves = seq_len - 1
#         total_similarity = 0.0

#         count = 0

#         for num_moves in range(1, max_num_moves + 1):
#             for i in range(batch_size):
#                 valid_indices = region_mask[i]  # [poi_size]
#                 for t in range(seq_len - num_moves):
#                     vec1 = fix_outputs[i, t, :][valid_indices]
#                     vec2 = fix_outputs[i, t + num_moves, :][valid_indices]
#                     sim = F.cosine_similarity(vec1.unsqueeze(0), vec2.unsqueeze(0))
#                     total_similarity += sim.item()
#                     count += 1
#         avg_similarity = torch.tensor(total_similarity / count)
#         # for num_moves in range(1, max_num_moves + 1):
#         #     shift_outputs = fix_outputs[:, num_moves:]
#         #     similarity = F.cosine_similarity(fix_outputs[:, :-num_moves], shift_outputs, dim=-1)
#         #     total_similarity += similarity.mean()
#         # avg_similarity = total_similarity / (seq_len - 1)
#         repetition_penalty_loss = -torch.log(1 - 0.5 * (avg_similarity + 1)) * self.beta

#         return repetition_penalty_loss


# class Guiding(nn.Module):
#     def __init__(self, out_dim, poi_size):
#         super(Guiding, self).__init__()
#         self.predictor = Recommender(out_dim, poi_size)
#         # self.confidence = nn.Linear(poi_size, 1)

#     def forward(self, outputs, AM, PM):
#         fix_outputs = self.predictor(outputs)  # [b,l,d] -> [b,l,v]
#         clipped_PM = PM[:, :fix_outputs.shape[1]]  # [v,l_max] -> [v,l]
#         clipped_outputs = fix_outputs * (clipped_PM.T.unsqueeze(0).expand(fix_outputs.shape[0], -1, -1))  # [b,l,v]

#         return clipped_outputs

# # Construct total framework(AR-Trip)
# class SPOTModel(nn.Module):
#     def __init__(self, args, poi_size, region_poi, poi_coord_tensor,
#                  max_length_venue_id=100, d_model=128, n_head=4, num_encoder_layers=1, n_tf_layers=4, d_z=128, kg_dataset=None, poi_info_map=None):

#         super(SPOTModel, self).__init__()
#         # initial hyperparameter
#         self.hidden_size = d_model
#         self.args = args
#         self.poi_info_map = poi_info_map # 需要从 dataset 传进来
#         # ==================== 新增/修改代码 START ====================
#         # 注册坐标张量为模型的 buffer，它会自动被移动到正确的设备
#         self.register_buffer('poi_coords', poi_coord_tensor)

#         # 创建一个 MLP 来将距离值 (标量) embedding 成一个向量
#         self.distance_mlp = nn.Sequential(
#             nn.Linear(1, self.hidden_size // 2),
#             nn.ReLU(),
#             nn.Linear(self.hidden_size // 2, self.hidden_size)
#         )
        
#         # [CHANGE] 初始化 LLM 模块
#         if args.use_llm_pref:
#             print("Initializing LLM for Static User Preference...")
#             llm_conf = LLMConfig(embedding_dim=self.hidden_size, use_lora=True) # 这里的emb_dim要和模型hidden size一致
#             self.llm_encoder = GPT2UserPrefEncoder(llm_conf).to(self.args.device)
#             self.prompt_builder = PromptBuilder()
            
#             # 为了能在 train/forward 过程中反向传播更新 LLM 参数（端到端）
#             # 确保 LLM 参数在 optimizer 中（通常只要是 submodule 就会自动加入）

#         # 动态调整预测器的输入维度
#         predictor_input_dim = 0
#         base_dim = self.hidden_size * 2
#         pd_dim = self.hidden_size if self.args.ode else 0
#         # [FIX] 如果启用了 s_infer 或者 use_llm_pref，都需要加上 ps_dim
#         if self.args.s_infer or self.args.use_llm_pref:
#             ps_dim = self.hidden_size
#         else:
#             ps_dim = 0
#         dist_dim = self.hidden_size if self.args.use_distance_feature else 0 # <-- 新增距离维度

#         predictor_input_dim = base_dim + pd_dim + ps_dim + dist_dim
        
#         self.predictor = Recommender(predictor_input_dim, poi_size)
#         # ===================== 新增/修改代码 END =====================
#         # add(LXT)
#         self.tau = self.args.tau  # 用于 info_nce_loss_overall
#         # model setting
#         self.poi_embedding = POIEmbeddings(poi_size, self.hidden_size)
#         self.poi_size = poi_size
#         self.pos_emb = nn.Embedding(max_length_venue_id, self.hidden_size)
#         self.fusion_mlp = nn.Sequential(
#             nn.Linear(self.hidden_size * 4, self.hidden_size * 4),
#             nn.SiLU()
#         )
#         if self.args.kg:
#             self.kg_dataset = kg_dataset
#             self.n_entities = self.kg_dataset.entity_count
#             self.n_relations = self.kg_dataset.relation_count
#             self.entity_embedding = nn.Embedding(self.n_entities + 1, d_model)
#             self.relations_embedding = nn.Embedding(self.n_relations + 1, d_model)
#             self.kg_dict, self.poi2relations = self.kg_dataset.get_kg_dict(self.poi_size)
#             self.gat = GAT(self.hidden_size, self.hidden_size, dropout=0.4, alpha=0.2).train()
#             if self.args.trans == 'transr':
#                 self.projection_matrix = nn.Linear(self.hidden_size, self.args.projection_dim)

#         self.encoder = Encoder(poi_size, d_z, d_model, n_head, n_tf_layers)
#         self.dyf = DynamicsFunction(
#             f=create_mlp(input_size=args.hidden_size,
#             output_size=args.hidden_size,
#             hidden_size=args.dyn_latent_dim,
#             num_hidden_layers=args.dyn_hid_layers,
#             activation_func=nn.GELU))
#         self.time_generator = DynamicTimeGenerator(self.hidden_size, self.hidden_size)
#         self.lm = nn.Sequential(
#             create_mlp(
#                 input_size=args.hidden_size,
#                 output_size=1,
#                 hidden_size=args.lm_latent_dim,
#                 num_hidden_layers=args.lm_hid_layers,
#                 activation_func=nn.GELU,
#             ),
#             IntensityCorrection(0.0000001),
#         )

#         self.transformer_encoder = TransformerModel(embed_size=self.hidden_size * 2, nhead=n_head,
#                                                     nhid=2048, nlayers=num_encoder_layers)
#         self.infer_layer = nn.Sequential(
#             nn.Linear(self.hidden_size, self.hidden_size),
#             nn.SiLU()
#         )
#         # if self.args.ode and self.args.s_infer:
#         #     self.predictor = Recommender(self.hidden_size * 4, poi_size)
#         # elif self.args.ode or self.args.s_infer:
#         #     self.predictor = Recommender(self.hidden_size * 3, poi_size)
#         # else:
#         #     self.predictor = Recommender(self.hidden_size * 2, poi_size)
#         self.region_poi = region_poi
#         self.region_embedding = nn.Embedding(len(self.region_poi), self.hidden_size)
#         self.region_masks = {}
#         for region, poi_list in region_poi.items():
#             mask = torch.zeros(poi_size, dtype=torch.bool)
#             mask[list(poi_list)] = True
#             self.region_masks[region] = mask
#         self.head_linear = nn.Linear(self.hidden_size, self.hidden_size)
#         self.tail_linear = nn.Linear(self.hidden_size, self.hidden_size)
#         self.criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding index during loss calculation

#     def calc_kg_loss_transE(self, h, r, pos_t, neg_t):
#         """
#         Calculates the loss for the model using the TransE approach.
#         Args:
#             h:      (kg_batch_size)
#             r:      (kg_batch_size)
#             pos_t:  (kg_batch_size)
#             neg_t:  (kg_batch_size)
#         Returns:
#             loss
#         """
#         # Each sample corresponds to an index of a relation type, and embedding_relation converts the index of each relation type into the corresponding embedding vector.
#         r_embed = self.relations_embedding(r)
#         h_embed = self.poi_embedding(h) # (kg_batch_size, entity_dim)
#         pos_t_embed = self.entity_embedding(pos_t) # (kg_batch_size, entity_dim)
#         neg_t_embed = self.entity_embedding(neg_t) # (kg_batch_size, entity_dim)
#         pos_score = torch.sum(torch.pow(h_embed + r_embed - pos_t_embed, 2), dim=1) # (kg_batch_size) As per the formula f_d in the paper.
#         neg_score = torch.sum(torch.pow(h_embed + r_embed - neg_t_embed, 2), dim=1) # (kg_batch_size)
#         kg_loss = (-1.0) * F.logsigmoid(neg_score - pos_score)
#         kg_loss = torch.mean(kg_loss)

#         # This value can be considered as the "energy" of the input samples.
#         # This code is typically used for calculating regularization terms in the loss function.
#         l2_loss = _L2_loss_mean(h_embed) + _L2_loss_mean(r_embed) + _L2_loss_mean(pos_t_embed) + _L2_loss_mean(neg_t_embed)
#         # # TODO: optimize L2 weight
#         loss = kg_loss + 1e-3 * l2_loss
#         return loss

#     def calc_kg_loss_transR(self, h, r, pos_t, neg_t):
#         """
#         Calculates the loss for the model using the TransR approach.
#         Args:
#             h:      (kg_batch_size)
#             r:      (kg_batch_size)
#             pos_t:  (kg_batch_size)
#             neg_t:  (kg_batch_size)
#         Returns:
#             loss
#         """
#         r_embed = self.projection_matrix(self.relations_embedding(r))
#         h_embed = self.projection_matrix(self.poi_embedding(h))
#         pos_t_embed = self.projection_matrix(self.entity_embedding(pos_t))
#         neg_t_embed = self.projection_matrix(self.entity_embedding(neg_t))
#         pos_score = torch.sum(torch.pow(h_embed + r_embed - pos_t_embed, 2), dim=1)
#         neg_score = torch.sum(torch.pow(h_embed + r_embed - neg_t_embed, 2), dim=1)
#         kg_loss = (-1.0) * F.logsigmoid(neg_score - pos_score)
#         kg_loss = torch.mean(kg_loss)

#         l2_loss = _L2_loss_mean(h_embed) + _L2_loss_mean(r_embed) + _L2_loss_mean(pos_t_embed) + _L2_loss_mean(neg_t_embed)
#         # # TODO: optimize L2 weight
#         loss = kg_loss + 1e-3 * l2_loss
#         return loss

#     def calc_kg_loss_SEEK(self, h, r, pos_t, neg_t):
#         """
#         Calculates the loss using the SEEK approach for knowledge graph embeddings.
#         Args:
#             h:      (kg_batch_size)
#             r:      (kg_batch_size)
#             pos_t:  (kg_batch_size)
#             neg_t:  (kg_batch_size)
#         Returns:
#             loss
#         """
#         # Each sample corresponds to an index of a relation type, and the embedding_relation converts the index of each relation type into the corresponding embedding vector.
#         r_embed = self.relations_embedding(r)        # (kg_batch_size, relation_dim)
#         h_embed = self.poi_embedding(h)               # (kg_batch_size, entity_dim)
#         pos_t_embed = self.entity_embedding(pos_t)      # (kg_batch_size, entity_dim)
#         neg_t_embed = self.entity_embedding(neg_t)      # (kg_batch_size, entity_dim)

#         k_num = self.args.segments
#         rank = int(self.hidden_size / k_num)
#         h = [h_embed[i * rank : (i + 1) * rank] for i in range(k_num)]
#         h = tuple(h)
#         r = [r_embed[i * rank : (i + 1) * rank] for i in range(k_num)]
#         r = tuple(r)
#         pos_t = [pos_t_embed[i * rank : (i + 1) * rank] for i in range(k_num)]
#         pos_t = tuple(pos_t)
#         neg_t = [neg_t_embed[i * rank : (i + 1) * rank] for i in range(k_num)]
#         neg_t = tuple(neg_t)
#         pos_tmp = 0
#         neg_tmp = 0

#         for x in range(k_num):
#             for y in range(k_num):
#                 s = -1 if x % 2 != 0 and x + y >= k_num else 1
#                 w = y if x % 2 == 0 else (x + y) % k_num
#                 pos_tmp += s * r[x] * h[y] * pos_t[w]
#                 neg_tmp += s * r[x] * h[y] * neg_t[w]
#         pos_score = torch.sum(pos_tmp, 1)
#         neg_score = torch.sum(neg_tmp, 1)
#         kg_loss = (-1.0) * F.logsigmoid(neg_score - pos_score)
#         kg_loss = torch.mean(kg_loss)

#         # This value can be considered as the "energy" of the input samples.
#         # This code is typically used for calculating regularization terms in the loss function.
#         l2_loss = _L2_loss_mean(h_embed) + _L2_loss_mean(r_embed) + _L2_loss_mean(pos_t_embed) + _L2_loss_mean(neg_t_embed)
#         # # TODO: optimize L2 weight
#         loss = kg_loss + 1e-3 * l2_loss
#         # loss = kg_loss
#         return loss

#     def _alias(self, bids, n_poi):
#         """
#         Creates an alias tensor mapping original POI indices to a new set of indices.
#         Returns:
#             torch.Tensor
#         """
#         alias = torch.zeros(n_poi).long()
#         for idx, b in enumerate(bids):
#             alias[b] = idx
#         alias = alias.to(self.args.device)
#         return alias

#     def _avg_pooling(self, ck, emb):
#         """
#         Applies average pooling to embeddings.
#         Returns:
#             Tensor
#         """
#         emb_sum = torch.sum(emb, axis=1)
#         #add(LXT)
#         row_count = torch.sum(ck != 0, axis=-1).clamp_min(1)  # 防 0
#         # row_count = torch.sum(ck != 0, axis=-1)
#         emb_agg = emb_sum / row_count.unsqueeze(1).expand_as(emb_sum)
#         return emb_agg
    

#     def _relation(self, emb, r, rel_embs, mode='transr'):
#         """
#         Applies a relation transformation to the embeddings, Dynamic Mapping.
#         Args:
#             o_emb: b x h
#             d_emb: b x l x h
#         Returns:
#             Tensor
#         """
#         if mode == 'transr':
#             relation = rel_embs[r].view(-1, self.hidden_size, self.hidden_size) # b x (h x h)
#             if len(emb.shape) == 2:
#                 emb_r = torch.bmm(emb.unsqueeze(1), relation).squeeze(1)
#                 return emb_r
#             elif len(emb.shape) == 3:
#                 emb_r = torch.matmul(emb.unsqueeze(2), relation.unsqueeze(1).expand(-1, emb.size(1), -1, -1)).squeeze(2)
#                 return emb_r
#         if mode == 'transd':
#             relation = rel_embs[r] # b x h Embeddings of 64 regions (cities) visited by users
#             if len(emb.shape) == 2:
#                 # b x h x h matrix multiplication
#                 # equivalent to the embedding weights of the region (city) multiplied by the node embedding that has passed through a linear layer.
#                 trans_mat = torch.matmul(relation.unsqueeze(2), self.head_linear(emb).unsqueeze(1)) # b x h x h
#                 # torch.bmm() might be faster, but both are matrix-level multiplication
#                 emb_r = torch.bmm(emb.unsqueeze(1), trans_mat).squeeze(1)
#                 return emb_r
#             elif len(emb.shape) == 3:
#                 # b x h x h (64, 13, 128, 1) * (64, 13, 1, 128)
#                 trans_mat = torch.matmul(relation.view(relation.size(0), 1, -1, 1).expand(-1, emb.size(1), -1, -1), self.tail_linear(emb).unsqueeze(2)) # b x h x h
#                 emb_r = torch.matmul(emb.unsqueeze(2), trans_mat).squeeze(2)
#                 return emb_r
#         if mode == 'transe':
#             return emb

#     def drop_edge_random(self, poi2entities, p_drop, padding):
#         """
#         Randomly drops edges from the POI to entity mappings.
#         Returns:
#             dict
#         """
#         res = dict()
#         for item, es in poi2entities.items():
#             new_es = list()
#             for e in es.tolist():
#                 if (random.random() > p_drop):
#                     new_es.append(e)
#                 else:
#                     new_es.append(padding)
#             res[item] = torch.IntTensor(new_es).to(self.args.device)
#         return res

#     def get_kg_views(self):
#         """
#         Generates two views of the knowledge graph by randomly dropping edges.
#         Returns:
#             tuple
#         """
#         kg = self.kg_dict
#         view1 = self.drop_edge_random(kg, self.args.kg_p_drop, self.n_entities)
#         view2 = self.drop_edge_random(kg, self.args.kg_p_drop, self.n_entities)
#         return view1, view2

#     def cal_poi_embedding_mean(self, kg: dict):
#         """
#         Calculates the mean embeddings of POIs based on their associated entities.
#         Returns:
#             Tensor
#         """
#         poi_embs = self.poi_embedding(torch.IntTensor(list(kg.keys())).to(self.args.device)) #poi_num, emb_dim
#         poi_entities = torch.stack(list(kg.values())) # poi_num, entity_num_each
#         entity_embs = self.entity_embedding(poi_entities) # poi_num, entity_num_each, emb_dim
#         # item_num, entity_num_each
#         padding_mask = torch.where(poi_entities!=self.n_entities, torch.ones_like(poi_entities), torch.zeros_like(poi_entities)).float()
#         # padding is zero
#         entity_embs = entity_embs * padding_mask.unsqueeze(-1).expand(entity_embs.size())
#         # poi_num, emb_dim
#         entity_embs_sum = entity_embs.sum(1)
#         entity_embs_mean = entity_embs_sum / padding_mask.sum(-1).unsqueeze(-1).expand(entity_embs_sum.size())
#         # replace nan with zeros
#         entity_embs_mean = torch.nan_to_num(entity_embs_mean)
#         # poi_num, emb_dim
#         return poi_embs+entity_embs_mean

#     def cal_poi_embedding_gat(self, kg:dict):
#         """
#         Calculates the POI embeddings using a Graph Attention Network (GAT) based on the associated entities.
#         Returns:
#             Tensor
#         """
#         poi_embs = self.poi_embedding(torch.IntTensor(list(kg.keys())).to(self.args.device)) #poi_num, emb_dim
#         poi_entities = torch.stack(list(kg.values())) # poi_num, entity_num_each
#         entity_embs = self.entity_embedding(poi_entities) # poi_num, entity_num_each, emb_dim
#         # poi_num, entity_num_each
#         padding_mask = torch.where(poi_entities!=self.n_entities, torch.ones_like(poi_entities), torch.zeros_like(poi_entities)).float()
#         return self.gat(poi_embs, entity_embs, padding_mask)

#     def cal_poi_embedding_rgat(self, kg:dict):
#         """
#         Calculates POI embeddings using a Relational Graph Attention Network (RGAT).
#         Returns:
#             Tensor
#         """
#         poi_embs = self.poi_embedding(torch.IntTensor(list(kg.keys())).to(self.args.device)) #poi_num, emb_dim
#         poi_entities = torch.stack(list(kg.values())) # poi_num, entity_num_each
#         poi_relations = torch.stack(list(self.poi2relations.values()))
#         entity_embs = self.entity_embedding(poi_entities) # poi_num, entity_num_each, emb_dim
#         relation_embs = self.relations_embedding(poi_relations) # poi_num, entity_num_each, emb_dim
#         padding_mask = torch.where(poi_entities!=self.n_entities, torch.ones_like(poi_entities), torch.zeros_like(poi_entities)).float()
#         return self.gat.forward_relation(poi_embs, entity_embs, relation_embs, padding_mask)

#     def cal_poi_embedding_from_kg(self, kg: dict):
#         """
#         Calculates POI embeddings based on the specified knowledge graph convolution method.
#         Returns:
#             Tensor
#         """
#         if kg is None:
#             kg = self.kg_dict

#         if(self.args.kgcn=="GAT"):
#             return self.cal_poi_embedding_gat(kg)
#         elif self.args.kgcn=="RGAT":
#             return self.cal_poi_embedding_rgat(kg)
#         elif(self.args.kgcn=="MEAN"):
#             return self.cal_poi_embedding_mean(kg)
#         elif(self.args.kgcn=="NO"):
#             return self.poi_embedding.weight

#     # def get_ui_views_weighted(self, poi_stabilities, stab_weight):
#     #     """
#     #     Calculates weighted POI views based on stability scores.
#     #     Returns:
#     #         Tensor
#     #     """
#     #     # kg probability of keep
#     #     poi_stabilities = torch.exp(poi_stabilities)
#     #     kg_weights = (poi_stabilities - poi_stabilities.min()) / (poi_stabilities.max() - poi_stabilities.min())
#     #     # Replace elements in kg_weights less than or equal to 0.3 with 0.3, keep elements greater than 0.3 unchanged.
#     #     kg_weights = kg_weights.where(kg_weights > 0.3, torch.ones_like(kg_weights) * 0.3)
#     #     weights = (1-self.args.ui_p_drop)/torch.mean(stab_weight*kg_weights)*(stab_weight*kg_weights)
#     #     # weights = weights.where(weights>0.3, torch.ones_like(weights) * 0.3)
#     #     # Replace elements in weights greater than or equal to 0.95 with 0.95, keep elements less than 0.95 unchanged.
#     #     weights = weights.where(weights<0.95, torch.ones_like(weights) * 0.95)
#     #     # Perform Bernoulli sampling to get a mask tensor poi_mask of the same dimension as weights,
#     #     # where the probability of an element being True is the corresponding value in weights.
#     #     # Values are chosen as 1 or 0 with probabilities p and 1-p, respectively.
#     #     poi_mask = torch.bernoulli(weights).to(torch.bool)
#     #     # drop
#     #     poi_mask.requires_grad = False
#     #     return poi_mask
    
#     #add(LXT)
#     def get_ui_views_weighted(self, poi_stabilities, stab_weight):
#         poi_stabilities = torch.exp(poi_stabilities.clamp(max=10))
#         kg_weights = (poi_stabilities - poi_stabilities.min()) / (poi_stabilities.max() - poi_stabilities.min() + 1e-6)
#         kg_weights = kg_weights.where(kg_weights > 0.3, torch.full_like(kg_weights, 0.3))
#         weights = (1 - self.args.ui_p_drop) / torch.mean(stab_weight * kg_weights) * (stab_weight * kg_weights)
#         weights = weights.where(weights < 0.95, torch.full_like(weights, 0.95))
#         poi_mask = torch.bernoulli(weights).to(torch.bool)
#         poi_mask.requires_grad = False
#         return poi_mask

#     def sim(self, z1: torch.Tensor, z2: torch.Tensor):
#         """
#         Calculates the similarity between two tensors.
#         Returns:
#             Tensor
#         """
#         if z1.size()[0] == z2.size()[0]:
#             return F.cosine_similarity(z1,z2)
#         else:
#             z1 = F.normalize(z1)
#             z2 = F.normalize(z2)
#             return torch.mm(z1, z2.t())

#     def poi_kg_stability(self, view1, view2):
#         """
#         Computes the stability of POI embeddings across two views of the knowledge graph.
#         Returns:
#             Tuple
#         """
#         kgv1_ro = self.cal_poi_embedding_from_kg(view1)
#         kgv2_ro = self.cal_poi_embedding_from_kg(view2)
#         sim = self.sim(kgv1_ro, kgv2_ro)
#         return kgv1_ro, kgv2_ro, sim

#     # 辅助函数：将 Tensor 转换成 Prompt 列表
#     def _generate_prompts(self, history, mask):
#         prompts = []
#         batch_size, seq_len = history.shape
#         history_cpu = history.cpu().numpy()
        
#         if mask.shape[1] > seq_len:
#             mask = mask[:, :seq_len]

#         for i in range(batch_size):
#             # 还原轨迹
#             valid_mask = mask[i].bool().cpu()
#             traj_indices = history_cpu[i][valid_mask] 
#             trajectory_dicts = []
            
#             # 假设 hometown 信息在 batch 的 metadata 里，或者简化为 "Unknown City"
#             # 这里为了跑通，如果无法获得每个用户的 hometown，可以设为默认值
#             hometown_str = "Unknown City" 
            
#             for idx in traj_indices:
#                 # 注意：这里的 idx 是模型映射后的 ID，需要转回原始 POI ID 才能查到 info
#                 # 如果 poi_info_map key 是 mapped_id (0~N)，则直接查
#                 info = self.poi_info_map.get(int(idx), {})
#                 # 构建 PromptBuilder 需要的字典格式
#                 trajectory_dicts.append({
#                     'poi_name': info.get('name', 'Unknown'),
#                     'cat_tree': info.get('cat_tree', 'Unknown'),
#                     'timestamp': 1672531200 + i*3600, # Mock timestamp 如果 tensor 里没传时间
#                     'city': info.get('city', 'Unknown')
#                 })
#                 # 如果 dataset 里有真实 hometown，更新 hometown_str
#                 if 'hometown' in info: hometown_str = info['hometown']

#             # 调用 PromptBuilder
#             p = self.prompt_builder.build_prompt(hometown_str, trajectory_dicts)
#             prompts.append(p)
#         return prompts

#     def get_views(self, aug_side="both"):
#         """
#         Generates augmented views for contrastive learning.
#         Returns:
#             Dict
#         """
#         # drop (epoch based)
#         # kg drop -> 2 views -> view similarity for item
#         # Randomly remove tail entities and fill in the removed parts.
#         kgv1, kgv2 = self.get_kg_views()
#         # [item_num]
#         kgv1, kgv2, stability = self.poi_kg_stability(kgv1, kgv2)  # Calculate consistency
#         kgv1 = kgv1.to(self.args.device)
#         kgv2 = kgv2.to(self.args.device)
#         stability = stability.to(self.args.device)
#         # item drop -> 2 views
#         # Delete the user-item interaction edges (deleting edges with item nodes as index) from the interaction graph.
#         v1_mask = self.get_ui_views_weighted(stability, 1)
#         # uiv2 = self.ui_drop_random(world.ui_p_drop)
#         v2_mask = self.get_ui_views_weighted(stability, 1)

#         contrast_views = {
#             "kgv1": kgv1,
#             "kgv2": kgv2,
#             "uiv1": v1_mask,
#             "uiv2": v2_mask
#         }
#         return contrast_views

#     def info_nce_loss_overall(self, z1, z2):
#         """
#         Calculates the InfoNCE loss, a contrastive loss used for learning efficient embeddings.
#         Returns:
#             torch.Tensor
#         """
#         criterion = torch.nn.CrossEntropyLoss(reduction='mean')
#         batch_size, d_model = z1.shape
#         features = torch.cat([z1, z2], dim=0)  # (batch_size * 2, d_model)

#         labels = torch.cat([torch.arange(batch_size) for i in range(2)], dim=0)
#         labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
#         labels = labels.to(self.args.device)

#         features = F.normalize(features, dim=1)
#         similarity_matrix = torch.matmul(features, features.T)

#         # discard the main diagonal from both: labels and similarities matrix
#         mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.args.device)
#         labels = labels[~mask].view(labels.shape[0], -1)
#         similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
#         # assert similarity_matrix.shape == labels.shape

#         # select and combine multiple positives
#         positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)  # [batch_size * 2, 1]

#         # select only the negatives
#         negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)  # [batch_size * 2, 2N-2]

#         logits = torch.cat([positives, negatives], dim=1)  # (batch_size * 2, batch_size * 2 - 1)
#         labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.args.device)  # (batch_size * 2, 1)
#         logits = logits / self.tau

#         loss = criterion(logits, labels)
#         return loss

#     def pad_with_embedding(self, seq_list, pad_vector):
#         # seq_list: list of tensors [seq_len, d]
#         # pad_vector: tensor of shape [d]
#         max_len = max(seq.shape[0] for seq in seq_list)
#         padded_list = []
#         for seq in seq_list:
#             pad_len = max_len - seq.shape[0]
#             if pad_len > 0:
#                 pad_tensor = pad_vector.unsqueeze(0).expand(pad_len, -1)
#                 padded_seq = torch.cat([seq, pad_tensor], dim=0)
#             else:
#                 padded_seq = seq
#             padded_list.append(padded_seq)
#         return torch.stack(padded_list, dim=0)
    
#     #add(LXT)
#     def _prepare_ode_times(self, raw_times: torch.Tensor):
#         """
#         统一处理 ODE 时间序列：排序->去重->严格递增修复
#         """
#         t = raw_times.to(torch.float32)
#         if t.numel() < 2:
#             return t[:1]  # 至少返回一个点
#         t, _ = torch.sort(t)
#         t = torch.unique_consecutive(t)
#         if t.numel() < 2:
#             return t[:1]
#         diff = t[1:] - t[:-1]
#         if torch.any(diff <= 0):
#             span = max((t[-1] - t[0]).item(), 1e-4)
#             return torch.linspace(t[0], t[0] + span, raw_times.shape[0], device=t.device)
#         # 插值回原长度
#         return torch.linspace(t[0], t[-1], raw_times.shape[0], device=t.device)

#     #add(LXT)
#     # [修改] 增加 logit_mask 参数
#     # ==================== 替换整个 forward 方法 START ====================
#     def forward(self, o_ck, query, o_t, d_t, o_l, d_l, o_pad, d_pad, d_ck, o_rg, d_rg, target_seq=None, logit_mask=None):
#         batch_size, seq_length = query.size()
        
#         # --- 1. 准备 Embedding 和 Mask ---
#         region_mask = torch.stack([self.region_masks[int(r)] for r in d_rg], dim=0).to(query.device)
#         region_mask_uns = region_mask.unsqueeze(1).expand(-1, seq_length, -1)

#         if self.args.kg:
#             poi_embedding = self.cal_poi_embedding_from_kg(self.kg_dict)
#             query_emb = poi_embedding[query]
#             o_emb = poi_embedding[o_ck]
#             d_target_emb = poi_embedding[d_ck]
#             pad_vec = poi_embedding[0]
#         else:
#             query_emb = self.poi_embedding(query)
#             o_emb = self.poi_embedding(o_ck)
#             d_target_emb = self.poi_embedding(d_ck)
#             pad_vec = self.poi_embedding.emb.weight[0]

#         # --- 2. 计算 P_D (动态偏好) 和 P_S (静态偏好) ---
#         P_D, P_S = None, None
#         elbo_loss = torch.zeros(1, device=query.device)
#         infer_loss = torch.zeros(1, device=query.device)

#         if self.args.ode:
#             #add(LXT)
#             _, gamma, tau = self.encoder(o_t, o_l, o_ck, o_pad)
            
#             # 强力清洗 Encoder 输出
#             if torch.isnan(gamma).any() or torch.isnan(tau).any():
#                 # print("[Warn] Encoder output NaN, resetting to defaults")
#                 gamma = torch.nan_to_num(gamma, nan=0.0)
#                 tau = torch.nan_to_num(tau, nan=1.0)

#             tau = tau.clamp(1e-3, 10.)
#             gamma = gamma.clamp(-50., 50.)
#             z_0 = (gamma + tau * torch.randn_like(tau)).clamp(-100., 100.)
            
#             # 再次清洗 z_0
#             if torch.isnan(z_0).any():
#                 z_0 = torch.nan_to_num(z_0, nan=0.0)

#             dynamic_d_emb = self.encoder.time_proj(d_t.to(torch.float32).unsqueeze(-1)) + \
#                             self.encoder.space_proj(d_l.to(torch.float32)) + self.encoder.poi_emb(d_ck)
#             P_D_list = []
#             process_loglik = torch.zeros(1, device=query.device)
#             obs_loglik = torch.zeros(1, device=query.device)

#             for j in range(batch_size):
#                 # 正确的有效位置（True 是有效）, 去掉最后聚合 token
#                 valid_idx = torch.nonzero(d_pad[j], as_tuple=True)[0][:-1]
#                 if valid_idx.numel() < 2:
#                     P_D_list.append(z_0[j].unsqueeze(0))
#                     continue
#                 raw_times = d_t[j][valid_idx].to(torch.float32)
#                 t_fixed = clean_time_grid(raw_times)
#                 if t_fixed.numel() < 2:
#                     P_D_list.append(z_0[j].unsqueeze(0))
#                     continue
#                 # 统一拉伸到与 valid_idx 同长度
#                 t_grid = torch.linspace(t_fixed[0], t_fixed[-1], valid_idx.numel(), device=raw_times.device)
#                 sol = integrate_fixed_rk4(self.dyf, z_0[j], t_grid)
#                 if torch.isnan(sol).any() or torch.isinf(sol).any():
#                     sol = z_0[j].unsqueeze(0).expand(t_grid.numel(), -1)
#                 u_hat = sol.clamp(-100, 100)
#                 if torch.isnan(u_hat).any():
#                     u_hat = z_0[j].unsqueeze(0).expand(t_grid.numel(), -1)
#                 P_D_list.append(u_hat)
#                 if target_seq is not None:
#                     lm_hat = self.lm(u_hat.clamp(-20, 20))
#                     lm_hat = torch.nan_to_num(lm_hat, nan=1e-6, posinf=1e6, neginf=1e-6)
#                     safe_lm = torch.clamp(lm_hat.squeeze(-1), min=1e-8)
#                     process_loglik += safe_lm.log().sum()
#                     f_values = self.lm(sol.clamp(-20, 20)).squeeze(-1)
#                     f_values = torch.nan_to_num(f_values, nan=0.0)
#                     # 均匀网格近似积分
#                     integrated_value = f_values.mean() * (t_grid[-1] - t_grid[0])
#                     process_loglik -= integrated_value
#                     v = dynamic_d_emb[j][valid_idx]
#                     L = min(u_hat.size(0), v.size(0))
#                     v = v[:L]
#                     mu = u_hat[:L]
#                     obs_loglik += Normal(mu, self.args.sig_v).log_prob(v).sum()

#             P_D = self.pad_with_embedding(P_D_list, pad_vec)
#             if target_seq is not None:
#                 kl_qp = 2 * kl_norm_norm(gamma, torch.zeros_like(gamma), tau, torch.ones_like(tau)).sum()
#                 elbo_loss = - (obs_loglik + process_loglik - kl_qp)

#         # ================= Static Preference Calculation =================
#         if self.args.use_llm_pref:
#             # [NEW] 使用 LLM 替换原有的 infer_layer
#             # 1. 构造 Text Prompts (这一步涉及 CPU 操作，可能影响训练速度，建议后续移到 DataLoader)
#             prompts = self._generate_prompts(o_ck, o_pad)
            
#             # 2. 调用 LLM
#             # 注意：p_s 的维度必须是 [Batch, Hidden_Size]
#             P_S = self.llm_encoder(prompts)
#             P_S = P_S.unsqueeze(1).expand(-1, seq_length, -1)
#         elif self.args.s_infer:
#             u_o_emb_s = self._avg_pooling(o_ck, o_emb)
#             infer = self.infer_layer(u_o_emb_s)
#             P_S = infer.unsqueeze(1).expand_as(d_target_emb)
#             if target_seq is not None:
#                 u_d_emb_s = self._avg_pooling(d_ck, d_target_emb)
#                 infer_loss = F.mse_loss(infer, u_d_emb_s)

#         # --- 3. 训练 和 推理 分支 ---
#         if target_seq is not None:
#             # --- 3.1 训练逻辑 ---
            
#             # a. 计算基础序列特征
#             position_ids = torch.arange(seq_length, dtype=torch.long, device=query.device).unsqueeze(0).expand(batch_size, -1)
#             position_embedded = self.pos_emb(position_ids)
#             model_input = torch.cat([query_emb, position_embedded], dim=2)
#             encoder_output = self.transformer_encoder(model_input)

#             # b. 融合 P_D 和 P_S
#             feat = encoder_output
#             if P_D is not None and P_S is not None:
#                 feat = torch.cat([encoder_output, P_D, P_S], dim=2)
#             elif P_D is not None:
#                 feat = torch.cat([encoder_output, P_D], dim=2)
#             elif P_S is not None:
#                 feat = torch.cat([encoder_output, P_S], dim=2)
            
#             # c. (如果启用) 融合距离特征
#             if self.args.use_distance_feature:
#                 end_point_ids = []
#                 for i in range(batch_size):
#                     valid_indices = torch.nonzero(d_ck[i], as_tuple=True)[0]
#                     end_point_ids.append(d_ck[i][valid_indices[-1]])
#                 end_point_ids = torch.stack(end_point_ids)
#                 end_coords = self.poi_coords[end_point_ids]

#                 prev_point_ids = torch.roll(d_ck, shifts=1, dims=1)
#                 prev_point_ids[:, 0] = d_ck[:, 0]
#                 prev_coords = self.poi_coords[prev_point_ids]

#                 distances = torch.norm(prev_coords - end_coords.unsqueeze(1), p=2, dim=-1) + 1e-8
#                 distance_feature = self.distance_mlp(distances.unsqueeze(-1))
#                 feat = torch.cat([feat, distance_feature], dim=2)

#             # d. 最终预测和损失计算
#             poi_output = self.predictor(feat)
#             masked_poi_output = poi_output.masked_fill(~region_mask_uns, -1e9)
            
#             # [新增] 应用椭圆过滤 (训练阶段通常不需要，但为了逻辑一致性加上)
#             if logit_mask is not None:
#                 curr_mask = logit_mask
#                 if curr_mask.size(1) > masked_poi_output.size(-1):
#                     curr_mask = curr_mask[:, :masked_poi_output.size(-1)]
#                 # 扩展到序列维度 [Batch, 1, N_POI]
#                 masked_poi_output = masked_poi_output.masked_fill(~curr_mask.unsqueeze(1), -1e9)

#             cls_loss = self.criterion(masked_poi_output.view(-1, self.poi_size), d_ck.flatten())
#             total_loss = cls_loss + elbo_loss + infer_loss
#             if torch.isnan(total_loss):
#                 total_loss = cls_loss
#             return total_loss
#         else:
#             # --- 3.2 推理逻辑 ---
#             if self.args.generation_mode == 'iterative':
#                 generated_ids = query.clone()
#                 end_point_ids = []
#                 for i in range(batch_size):
#                     valid_indices = torch.nonzero(d_ck[i], as_tuple=True)[0]
#                     end_point_ids.append(d_ck[i][valid_indices[-1]])
#                 end_point_ids = torch.stack(end_point_ids)

#                 for t in range(1, seq_length - 1):
#                     if self.args.kg:
#                         current_poi_embedding = self.cal_poi_embedding_from_kg(self.kg_dict)
#                         current_emb = current_poi_embedding[generated_ids]
#                     else:
#                         current_emb = self.poi_embedding(generated_ids)
                    
#                     position_ids = torch.arange(seq_length, device=query.device).unsqueeze(0).expand(batch_size, -1)
#                     position_embedded = self.pos_emb(position_ids)
#                     model_input = torch.cat([current_emb, position_embedded], dim=2)
#                     encoder_output = self.transformer_encoder(model_input)

#                     feat_iter = encoder_output
#                     if P_D is not None and P_S is not None:
#                         feat_iter = torch.cat([encoder_output, P_D, P_S], dim=2)
#                     elif P_D is not None:
#                         feat_iter = torch.cat([encoder_output, P_D], dim=2)
#                     elif P_S is not None:
#                         feat_iter = torch.cat([encoder_output, P_S], dim=2)

#                     if self.args.use_distance_feature:
#                         prev_point_ids = generated_ids[:, t-1]
#                         prev_coords = self.poi_coords[prev_point_ids]
#                         end_coords = self.poi_coords[end_point_ids]
#                         distances = torch.norm(prev_coords - end_coords, p=2, dim=-1, keepdim=True) + 1e-8
#                         distance_feature = self.distance_mlp(distances)
#                         distance_feature_expanded = distance_feature.unsqueeze(1).expand(-1, seq_length, -1)
#                         feat_iter = torch.cat([feat_iter, distance_feature_expanded], dim=2)

#                     poi_logits_iter = self.predictor(feat_iter)
#                     logits_at_t = poi_logits_iter[:, t, :]
#                     masked_logits_at_t = logits_at_t.masked_fill(~region_mask, -1e9)
                    
#                     # [新增] 应用椭圆过滤 (Iterative 模式)
#                     if logit_mask is not None:
#                         curr_mask = logit_mask
#                         if curr_mask.size(1) > masked_logits_at_t.size(-1):
#                             curr_mask = curr_mask[:, :masked_logits_at_t.size(-1)]
#                         masked_logits_at_t = masked_logits_at_t.masked_fill(~curr_mask, -1e9)

#                     _, next_poi_id = torch.max(masked_logits_at_t, dim=-1)
#                     generated_ids[:, t] = next_poi_id
                
#                 return generated_ids
            
#             elif self.args.generation_mode == 'parallel':
#                 # 并行模式也需要融合所有特征
#                 # feat_parallel = encoder_output # 这是错误的，应该重新计算
#                 position_ids = torch.arange(seq_length, dtype=torch.long, device=query.device).unsqueeze(0).expand(batch_size, -1)
#                 position_embedded = self.pos_emb(position_ids)
#                 model_input = torch.cat([query_emb, position_embedded], dim=2)
#                 encoder_output_p = self.transformer_encoder(model_input)

#                 feat_parallel = encoder_output_p
#                 if P_D is not None and P_S is not None:
#                     feat_parallel = torch.cat([encoder_output_p, P_D, P_S], dim=2)
#                 elif P_D is not None:
#                     feat_parallel = torch.cat([encoder_output_p, P_D], dim=2)
#                 elif P_S is not None:
#                     feat_parallel = torch.cat([encoder_output_p, P_S], dim=2)
                
#                 # 并行模式下无法使用距离特征，因为不知道上一个点是什么
                
#                 poi_logits = self.predictor(feat_parallel)
#                 masked_logits = poi_logits.masked_fill(~region_mask_uns, -1e9)
                
#                 # [新增] 应用椭圆过滤 (Parallel 模式)
#                 if logit_mask is not None:
#                     curr_mask = logit_mask
#                     if curr_mask.size(1) > masked_logits.size(-1):
#                         curr_mask = curr_mask[:, :masked_logits.size(-1)]
#                     # 扩展到序列维度 [Batch, 1, N_POI]
#                     masked_logits = masked_logits.masked_fill(~curr_mask.unsqueeze(1), -1e9)

#                 sim_ratio, cand_ids = torch.topk(masked_logits, k=masked_logits.shape[1], dim=2)
#                 pred_ids = top_np_recommendation(cand_ids, sim_ratio,
#                                                  confidence=torch.tensor(self.args.confidence),
#                                                  threshold=0.8)
#                 return pred_ids
#     # def forward(self, o_ck, query, o_t, d_t, o_l, d_l, o_pad, d_pad, d_ck, o_rg, d_rg, target_seq=None, logit_mask=None):
#     #     batch_size, seq_length = query.size()
        
#     #     # --- 1. 准备 Embedding 和 Mask ---
#     #     region_mask = torch.stack([self.region_masks[int(r)] for r in d_rg], dim=0).to(query.device)
#     #     region_mask_uns = region_mask.unsqueeze(1).expand(-1, seq_length, -1)

#     #     if self.args.kg:
#     #         poi_embedding = self.cal_poi_embedding_from_kg(self.kg_dict)
#     #         query_emb = poi_embedding[query]
#     #         o_emb = poi_embedding[o_ck]
#     #         d_target_emb = poi_embedding[d_ck]
#     #         pad_vec = poi_embedding[0]
#     #     else:
#     #         query_emb = self.poi_embedding(query)
#     #         o_emb = self.poi_embedding(o_ck)
#     #         d_target_emb = self.poi_embedding(d_ck)
#     #         pad_vec = self.poi_embedding.emb.weight[0]

#     #     # --- 2. 计算 P_D (动态偏好) 和 P_S (静态偏好) ---
#     #     P_D, P_S = None, None
#     #     elbo_loss = torch.zeros(1, device=query.device)
#     #     infer_loss = torch.zeros(1, device=query.device)

#     #     if self.args.ode:
#     #         # _, gamma, tau = self.encoder(o_t, o_l, o_ck, o_pad)
#     #         # tau = tau.clamp(1e-3, 10.)
#     #         # gamma = gamma.clamp(-50., 50.)
#     #         # z_0 = (gamma + tau * torch.randn_like(tau)).clamp(-100., 100.)

#     #         # dynamic_d_emb = self.encoder.time_proj(d_t.to(torch.float32).unsqueeze(-1)) + \
#     #         #                 self.encoder.space_proj(d_l.to(torch.float32)) + self.encoder.poi_emb(d_ck)
#     #         #add(LXT)
#     #         _, gamma, tau = self.encoder(o_t, o_l, o_ck, o_pad)
            
#     #         # 强力清洗 Encoder 输出
#     #         if torch.isnan(gamma).any() or torch.isnan(tau).any():
#     #             # print("[Warn] Encoder output NaN, resetting to defaults")
#     #             gamma = torch.nan_to_num(gamma, nan=0.0)
#     #             tau = torch.nan_to_num(tau, nan=1.0)

#     #         tau = tau.clamp(1e-3, 10.)
#     #         gamma = gamma.clamp(-50., 50.)
#     #         z_0 = (gamma + tau * torch.randn_like(tau)).clamp(-100., 100.)
            
#     #         # 再次清洗 z_0
#     #         if torch.isnan(z_0).any():
#     #             z_0 = torch.nan_to_num(z_0, nan=0.0)

#     #         dynamic_d_emb = self.encoder.time_proj(d_t.to(torch.float32).unsqueeze(-1)) + \
#     #                         self.encoder.space_proj(d_l.to(torch.float32)) + self.encoder.poi_emb(d_ck)
#     #         P_D_list = []
#     #         process_loglik = torch.zeros(1, device=query.device)
#     #         obs_loglik = torch.zeros(1, device=query.device)

#     #         for j in range(batch_size):
#     #             # 正确的有效位置（True 是有效）, 去掉最后聚合 token
#     #             valid_idx = torch.nonzero(d_pad[j], as_tuple=True)[0][:-1]
#     #             if valid_idx.numel() < 2:
#     #                 P_D_list.append(z_0[j].unsqueeze(0))
#     #                 continue
#     #             raw_times = d_t[j][valid_idx].to(torch.float32)
#     #             t_fixed = clean_time_grid(raw_times)
#     #             if t_fixed.numel() < 2:
#     #                 P_D_list.append(z_0[j].unsqueeze(0))
#     #                 continue
#     #             # 统一拉伸到与 valid_idx 同长度
#     #             t_grid = torch.linspace(t_fixed[0], t_fixed[-1], valid_idx.numel(), device=raw_times.device)
#     #             sol = integrate_fixed_rk4(self.dyf, z_0[j], t_grid)
#     #             if torch.isnan(sol).any() or torch.isinf(sol).any():
#     #                 sol = z_0[j].unsqueeze(0).expand(t_grid.numel(), -1)
#     #             u_hat = sol.clamp(-100, 100)
#     #             if torch.isnan(u_hat).any():
#     #                 u_hat = z_0[j].unsqueeze(0).expand(t_grid.numel(), -1)
#     #             P_D_list.append(u_hat)
#     #             if target_seq is not None:
#     #                 lm_hat = self.lm(u_hat.clamp(-20, 20))
#     #                 lm_hat = torch.nan_to_num(lm_hat, nan=1e-6, posinf=1e6, neginf=1e-6)
#     #                 safe_lm = torch.clamp(lm_hat.squeeze(-1), min=1e-8)
#     #                 process_loglik += safe_lm.log().sum()
#     #                 f_values = self.lm(sol.clamp(-20, 20)).squeeze(-1)
#     #                 f_values = torch.nan_to_num(f_values, nan=0.0)
#     #                 # 均匀网格近似积分
#     #                 integrated_value = f_values.mean() * (t_grid[-1] - t_grid[0])
#     #                 process_loglik -= integrated_value
#     #                 v = dynamic_d_emb[j][valid_idx]
#     #                 L = min(u_hat.size(0), v.size(0))
#     #                 v = v[:L]
#     #                 mu = u_hat[:L]
#     #                 obs_loglik += Normal(mu, self.args.sig_v).log_prob(v).sum()

#     #         P_D = self.pad_with_embedding(P_D_list, pad_vec)
#     #         if target_seq is not None:
#     #             kl_qp = 2 * kl_norm_norm(gamma, torch.zeros_like(gamma), tau, torch.ones_like(tau)).sum()
#     #             elbo_loss = - (obs_loglik + process_loglik - kl_qp)

#     #     if self.args.s_infer:
#     #         u_o_emb_s = self._avg_pooling(o_ck, o_emb)
#     #         infer = self.infer_layer(u_o_emb_s)
#     #         P_S = infer.unsqueeze(1).expand_as(d_target_emb)
#     #         if target_seq is not None:
#     #             u_d_emb_s = self._avg_pooling(d_ck, d_target_emb)
#     #             infer_loss = F.mse_loss(infer, u_d_emb_s)

#     #     # --- 3. 训练 和 推理 分支 ---
#     #     if target_seq is not None:
#     #         # --- 3.1 训练逻辑 ---
            
#     #         # a. 计算基础序列特征
#     #         position_ids = torch.arange(seq_length, dtype=torch.long, device=query.device).unsqueeze(0).expand(batch_size, -1)
#     #         position_embedded = self.pos_emb(position_ids)
#     #         model_input = torch.cat([query_emb, position_embedded], dim=2)
#     #         encoder_output = self.transformer_encoder(model_input)

#     #         # b. 融合 P_D 和 P_S
#     #         feat = encoder_output
#     #         if P_D is not None and P_S is not None:
#     #             feat = torch.cat([encoder_output, P_D, P_S], dim=2)
#     #         elif P_D is not None:
#     #             feat = torch.cat([encoder_output, P_D], dim=2)
#     #         elif P_S is not None:
#     #             feat = torch.cat([encoder_output, P_S], dim=2)
            
#     #         # c. (如果启用) 融合距离特征
#     #         if self.args.use_distance_feature:
#     #             end_point_ids = []
#     #             for i in range(batch_size):
#     #                 valid_indices = torch.nonzero(d_ck[i], as_tuple=True)[0]
#     #                 end_point_ids.append(d_ck[i][valid_indices[-1]])
#     #             end_point_ids = torch.stack(end_point_ids)
#     #             end_coords = self.poi_coords[end_point_ids]

#     #             prev_point_ids = torch.roll(d_ck, shifts=1, dims=1)
#     #             prev_point_ids[:, 0] = d_ck[:, 0]
#     #             prev_coords = self.poi_coords[prev_point_ids]

#     #             distances = torch.norm(prev_coords - end_coords.unsqueeze(1), p=2, dim=-1) + 1e-8
#     #             distance_feature = self.distance_mlp(distances.unsqueeze(-1))
#     #             feat = torch.cat([feat, distance_feature], dim=2)

#     #         # d. 最终预测和损失计算
#     #         poi_output = self.predictor(feat)
#     #         masked_poi_output = poi_output.masked_fill(~region_mask_uns, -1e9)
            
#     #         cls_loss = self.criterion(masked_poi_output.view(-1, self.poi_size), d_ck.flatten())
#     #         total_loss = cls_loss + elbo_loss + infer_loss
#     #         if torch.isnan(total_loss):
#     #             total_loss = cls_loss
#     #         return total_loss
#     #     else:
#     #         # --- 3.2 推理逻辑 ---
#     #         if self.args.generation_mode == 'iterative':
#     #             generated_ids = query.clone()
#     #             end_point_ids = []
#     #             for i in range(batch_size):
#     #                 valid_indices = torch.nonzero(d_ck[i], as_tuple=True)[0]
#     #                 end_point_ids.append(d_ck[i][valid_indices[-1]])
#     #             end_point_ids = torch.stack(end_point_ids)

#     #             for t in range(1, seq_length - 1):
#     #                 if self.args.kg:
#     #                     current_poi_embedding = self.cal_poi_embedding_from_kg(self.kg_dict)
#     #                     current_emb = current_poi_embedding[generated_ids]
#     #                 else:
#     #                     current_emb = self.poi_embedding(generated_ids)
                    
#     #                 position_ids = torch.arange(seq_length, device=query.device).unsqueeze(0).expand(batch_size, -1)
#     #                 position_embedded = self.pos_emb(position_ids)
#     #                 model_input = torch.cat([current_emb, position_embedded], dim=2)
#     #                 encoder_output = self.transformer_encoder(model_input)

#     #                 feat_iter = encoder_output
#     #                 if P_D is not None and P_S is not None:
#     #                     feat_iter = torch.cat([encoder_output, P_D, P_S], dim=2)
#     #                 elif P_D is not None:
#     #                     feat_iter = torch.cat([encoder_output, P_D], dim=2)
#     #                 elif P_S is not None:
#     #                     feat_iter = torch.cat([encoder_output, P_S], dim=2)

#     #                 if self.args.use_distance_feature:
#     #                     prev_point_ids = generated_ids[:, t-1]
#     #                     prev_coords = self.poi_coords[prev_point_ids]
#     #                     end_coords = self.poi_coords[end_point_ids]
#     #                     distances = torch.norm(prev_coords - end_coords, p=2, dim=-1, keepdim=True) + 1e-8
#     #                     distance_feature = self.distance_mlp(distances)
#     #                     distance_feature_expanded = distance_feature.unsqueeze(1).expand(-1, seq_length, -1)
#     #                     feat_iter = torch.cat([feat_iter, distance_feature_expanded], dim=2)

#     #                 poi_logits_iter = self.predictor(feat_iter)
#     #                 logits_at_t = poi_logits_iter[:, t, :]
#     #                 masked_logits_at_t = logits_at_t.masked_fill(~region_mask, -1e9)
#     #                 _, next_poi_id = torch.max(masked_logits_at_t, dim=-1)
#     #                 generated_ids[:, t] = next_poi_id
                
#     #             return generated_ids
            
#     #         elif self.args.generation_mode == 'parallel':
#     #             # 并行模式也需要融合所有特征
#     #             feat_parallel = encoder_output # 这是错误的，应该重新计算
#     #             position_ids = torch.arange(seq_length, dtype=torch.long, device=query.device).unsqueeze(0).expand(batch_size, -1)
#     #             position_embedded = self.pos_emb(position_ids)
#     #             model_input = torch.cat([query_emb, position_embedded], dim=2)
#     #             encoder_output_p = self.transformer_encoder(model_input)

#     #             feat_parallel = encoder_output_p
#     #             if P_D is not None and P_S is not None:
#     #                 feat_parallel = torch.cat([encoder_output_p, P_D, P_S], dim=2)
#     #             elif P_D is not None:
#     #                 feat_parallel = torch.cat([encoder_output_p, P_D], dim=2)
#     #             elif P_S is not None:
#     #                 feat_parallel = torch.cat([encoder_output_p, P_S], dim=2)
                
#     #             # 并行模式下无法使用距离特征，因为不知道上一个点是什么
                
#     #             poi_logits = self.predictor(feat_parallel)
#     #             masked_logits = poi_logits.masked_fill(~region_mask_uns, -1e9)
                
#     #             sim_ratio, cand_ids = torch.topk(masked_logits, k=masked_logits.shape[1], dim=2)
#     #             pred_ids = top_np_recommendation(cand_ids, sim_ratio,
#     #                                              confidence=torch.tensor(self.args.confidence),
#     #                                              threshold=0.8)
#     #             return pred_ids
    