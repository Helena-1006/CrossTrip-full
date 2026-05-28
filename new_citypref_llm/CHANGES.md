# 代码修改说明

> 本文档记录了针对论文描述与代码实现之间两处出入所做的修改，以及新增的双数据集实验脚本。

---

## 修改文件一览

| 文件 | 操作 | 说明 |
|---|---|---|
| `model.py` | 修改 | 修复两处与论文描述不一致的实现 |
| `run_both_datasets_pipeline.py` | 新建 | 一键双数据集（Yelp + Foursquare）实验脚本 |

---

## 一、`model.py` 修改详情

### 修改背景

**出入1**：论文中描述 `query_h` 应和本地/外地序列一样，将 query（起始POI、起始时间、终点POI、终点时间、预期POI数）作为一个包含两个签到的序列，经过签到嵌入和 Mamba 序列编码后得到。但代码中直接将 query 的几个数值拼接为 7 维向量后通过一个 Linear 层投影。

**出入2**：论文中描述路线生成模块的空间上下文（上一步 POI Embedding + 上一步 POI 距起终点的距离编码）应与 `z_final` 和 `query_h` 拼接后作为 Mamba 解码器的输入。但代码中是在 Mamba 解码生成 logits 之后，再用空间上下文对 logits 进行 re-scoring。

---

### 修改一：`query_h` 的生成方式

#### 1.1 删除 `self.query_proj`（原 line 402）

**修改前：**
```python
self.profile_proj = nn.Linear(args.profile_dim, args.hidden_size)
self.query_proj = nn.Linear(args.query_dim, args.hidden_size)   # ← 删除此行
self.decode_constraint_mode = ...
```

**修改后：**
```python
self.profile_proj = nn.Linear(args.profile_dim, args.hidden_size)
self.decode_constraint_mode = ...
```

**原因**：query 不再通过 Linear 投影，改为通过 BiMamba 编码器编码。

---

#### 1.2 新增 `self.query_seq_encoder`（在 `tour_seq_encoder` 初始化之后）

**新增代码：**
```python
# Query sequence encoder: 独立的BiMamba编码器，将query视为2个签到的序列（起始POI+终点POI）
if use_mamba and Mamba is not None:
    self.query_seq_encoder = CheckinBiMambaEncoder(
        poi_num=poi_num,
        hidden_size=args.hidden_size,
        nlayers=max(1, int(getattr(args, 'seq_num_layers', args.semantic_layers))),
        dropout=args.dropout,
        d_state=int(getattr(args, 'mamba_d_state', 16)),
        d_conv=int(getattr(args, 'mamba_d_conv', 4)),
        expand=int(getattr(args, 'mamba_expand', 2)),
    )
else:
    self.query_seq_encoder = CheckinTransformerEncoder(
        poi_num=poi_num,
        hidden_size=args.hidden_size,
        nhead=args.nhead,
        nlayers=max(1, args.semantic_layers),
        dropout=args.dropout,
    )
```

**原因**：使用独立 encoder 而非共享 `tour_seq_encoder` 权重，因为 query 序列固定为 2 个 token，而 tour 序列长度可变，共享权重会相互干扰。

---

#### 1.3 新增 `_encode_query` 方法（在 `_encode_tour` 之后）

**新增代码：**
```python
def _encode_query(self, batch):
    """
    将Query视为包含2个签到的序列 [起始签到, 终点签到]，经过签到嵌入和BiMamba编码，
    mean-pool后得到 query_h，与本地/外地序列的编码方式保持一致。
    """
    B = batch['query_start_poi'].size(0)
    device = batch['query_start_poi'].device

    # poi_seq: [B, 2]
    poi_seq = torch.stack([
        batch['query_start_poi'],
        batch['query_end_poi'],
    ], dim=1)

    # hour_seq: [B, 2]
    hour_seq = torch.stack([
        batch['query_start_hour'],
        batch['query_end_hour'],
    ], dim=1)

    # coord_seq: [B, 2, 2] — 从 poi_coords buffer 查表
    start_coord = self.poi_coords[batch['query_start_poi']]  # [B, 2]
    end_coord   = self.poi_coords[batch['query_end_poi']]    # [B, 2]
    coord_seq   = torch.stack([start_coord, end_coord], dim=1)  # [B, 2, 2]

    # valid_mask: [B, 2] — 始终有效
    valid_mask = torch.ones(B, 2, dtype=torch.bool, device=device)

    # 编码 → [B, 2, hidden_size]
    enc = self.query_seq_encoder(poi_seq, hour_seq, coord_seq, valid_mask)
    enc = torch.nan_to_num(enc, nan=0.0, posinf=1e3, neginf=-1e3)

    # mean-pool → [B, hidden_size]
    query_h = enc.mean(dim=1)
    return torch.nan_to_num(query_h, nan=0.0, posinf=1e3, neginf=-1e3)
```

**坐标来源**：使用 `self.poi_coords` buffer（训练/推理均可用），而非 `batch['dst_l']`（推理时不可用）。

---

#### 1.4 修改 `_adaptive_transfer` 签名

**修改前：**
```python
def _adaptive_transfer(self, z_stack, user_profile, query_vec):
    ...
    query_h = torch.nan_to_num(self.query_proj(query_vec), nan=0.0, posinf=1e3, neginf=-1e3)
```

**修改后：**
```python
def _adaptive_transfer(self, z_stack, user_profile, query_h):
    ...
    query_h = torch.nan_to_num(query_h, nan=0.0, posinf=1e3, neginf=-1e3)
```

**原因**：`query_h` 现在由调用方传入（已经过 BiMamba 编码），不再内部投影。

---

#### 1.5 修改 `forward` 方法

**修改前：**
```python
def forward(self, batch):
    h_seq, s_u, z_stack, mix = self._encode_home(batch)
    z_trans, alpha, profile_h, query_h = self._adaptive_transfer(
        z_stack, batch['user_profile'], batch['query_vec']
    )
```

**修改后：**
```python
def forward(self, batch):
    h_seq, s_u, z_stack, mix = self._encode_home(batch)
    query_h = self._encode_query(batch)
    z_trans, alpha, profile_h, query_h = self._adaptive_transfer(
        z_stack, batch['user_profile'], query_h
    )
```

---

#### 1.6 修改 `predict` 方法

**修改前：**
```python
def predict(self, batch):
    _, _, z_stack, _ = self._encode_home(batch)
    z_trans, _, profile_h, query_h = self._adaptive_transfer(z_stack, batch['user_profile'], batch['query_vec'])
```

**修改后：**
```python
def predict(self, batch):
    _, _, z_stack, _ = self._encode_home(batch)
    query_h = self._encode_query(batch)
    z_trans, _, profile_h, query_h = self._adaptive_transfer(z_stack, batch['user_profile'], query_h)
```

---

### 修改二：空间上下文作为 Mamba 解码输入

#### 2.1 修改 decoder `input_size`

**修改前：**
```python
self.decoder = CausalMambaDecoder(
    input_size=args.hidden_size * 2,   # [poi_emb, z_final]
    ...
)
# GRU fallback:
self.decoder = nn.GRU(args.hidden_size * 2, args.hidden_size, batch_first=True)
```

**修改后：**
```python
self.decoder = CausalMambaDecoder(
    input_size=args.hidden_size * 4,   # [poi_emb, dist_enc, z_final, query_h]
    ...
)
# GRU fallback:
self.decoder = nn.GRU(args.hidden_size * 4, args.hidden_size, batch_first=True)
```

**原因**：decoder 输入现在由 4 个 `hidden_size` 维向量拼接而成：
1. `prev_poi_emb`：上一步 POI 嵌入
2. `dist_enc`：上一步 POI 距起终点的距离编码（经过 `_build_distance_cond`）
3. `z_final`：用户跨城市偏好
4. `query_h`：BiMamba 编码的查询表示

---

#### 2.2 修改 `_generator_loss` 方法（训练阶段）

**签名修改：**
```python
# 修改前
def _generator_loss(self, z_final, dst_ck, query_vec=None, start_poi=None, end_poi=None):
# 修改后
def _generator_loss(self, z_final, dst_ck, query_h=None, start_poi=None, end_poi=None):
```

**Decoder 输入构建修改前：**
```python
dec_emb = self.tour_poi_emb(dec_in)
context = z_final.unsqueeze(1).expand(-1, dec_emb.size(1), -1)
dec_feat = torch.cat([dec_emb, context], dim=-1)
dec_out = self._decoder_forward(self.decoder, dec_feat)
```

**修改后：**
```python
dec_emb = self.tour_poi_emb(dec_in)                                        # [B, T, H]

# 空间距离编码：上一步POI距起终点的距离
if start_poi is not None and end_poi is not None:
    dist_enc = self._build_distance_cond(dec_in, start_poi, end_poi)       # [B, T, H]
else:
    dist_enc = torch.zeros_like(dec_emb)

context   = z_final.unsqueeze(1).expand(-1, dec_emb.size(1), -1)          # [B, T, H]
query_ctx = query_h.unsqueeze(1).expand(-1, dec_emb.size(1), -1) if query_h is not None \
            else torch.zeros_like(dec_emb)                                  # [B, T, H]

dec_feat = torch.cat([dec_emb, dist_enc, context, query_ctx], dim=-1)     # [B, T, 4H]
dec_out  = self._decoder_forward(self.decoder, dec_feat)
```

**同时删除** loop 内的 `soft_constraint_bias` 调用块（8行）：
```python
# 删除以下代码：
soft_bias = self._soft_constraint_bias(
    prev_ids=dec_in[:, st:ed],
    query_vec=query_vec,
    start_poi=start_poi,
    end_poi=end_poi,
)
if soft_bias is not None:
    logits = logits + soft_bias
```

**原因**：空间上下文已经作为 decoder 的输入，不需要再对 logits 做 re-scoring。

---

#### 2.3 修改 `forward` 中 `_generator_loss` 的调用

**修改前：**
```python
loss_gen, loss_pair, loss_transition = self._generator_loss(
    z_final,
    batch['dst_ck'],
    query_vec=batch.get('query_vec', None),
    ...
)
```

**修改后：**
```python
loss_gen, loss_pair, loss_transition = self._generator_loss(
    z_final,
    batch['dst_ck'],
    query_h=query_h,
    ...
)
```

---

#### 2.4 修改 `_next_step_logits` 方法（推理阶段每步）

**签名修改：**
```python
# 修改前
def _next_step_logits(self, curr_seq, z_final_vec, query_vec=None, start_poi=None, end_poi=None):
# 修改后
def _next_step_logits(self, curr_seq, z_final_vec, query_h=None, start_poi=None, end_poi=None):
```

**方法体修改前：**
```python
dec_emb = self.tour_poi_emb(curr_seq)
context = z_final_vec.unsqueeze(1).expand(-1, dec_emb.size(1), -1)
dec_feat = torch.cat([dec_emb, context], dim=-1)
dec_out = self._decoder_forward(self.decoder, dec_feat)
logits = self.decoder_out(dec_out[:, -1, :])
logits = logits + self.args.pop_bias_scale * self.popularity_bias.unsqueeze(0)

soft_bias = self._soft_constraint_bias(
    prev_ids=curr_seq[:, -1],
    query_vec=query_vec,
    start_poi=start_poi,
    end_poi=end_poi,
)
if soft_bias is not None:
    logits = logits + soft_bias
```

**修改后：**
```python
dec_emb = self.tour_poi_emb(curr_seq)                                          # [B, T, H]

# 空间距离编码：序列中每个位置的POI距起终点的距离
if start_poi is not None and end_poi is not None:
    dist_enc = self._build_distance_cond(curr_seq, start_poi, end_poi)         # [B, T, H]
else:
    dist_enc = torch.zeros_like(dec_emb)

context   = z_final_vec.unsqueeze(1).expand(-1, dec_emb.size(1), -1)
query_ctx = query_h.unsqueeze(1).expand(-1, dec_emb.size(1), -1) if query_h is not None \
            else torch.zeros_like(dec_emb)

dec_feat = torch.cat([dec_emb, dist_enc, context, query_ctx], dim=-1)
dec_out  = self._decoder_forward(self.decoder, dec_feat)
logits   = self.decoder_out(dec_out[:, -1, :])
logits   = logits + self.args.pop_bias_scale * self.popularity_bias.unsqueeze(0)
# 空间上下文已作为decoder输入，不再对logits做re-scoring
```

---

#### 2.5 修改 `_predict_greedy` 签名和内部调用

**签名修改：**
```python
# 修改前
def _predict_greedy(self, z_final, query_vec, start_poi, end_poi, lengths, max_len, ellipse_mask):
# 修改后
def _predict_greedy(self, z_final, query_h, start_poi, end_poi, lengths, max_len, ellipse_mask):
```

内部 `_next_step_logits` 调用从 `query_vec=query_vec` 改为 `query_h=query_h`。

---

#### 2.6 修改 `_predict_beam_single` 签名和内部调用

**签名修改：**
```python
# 修改前
def _predict_beam_single(self, z_final_vec, query_vec_single, start, end, route_len, ...):
# 修改后
def _predict_beam_single(self, z_final_vec, query_h_single, start, end, route_len, ...):
```

内部变量从 `query_vec_batch` 改为 `query_h_batch`，传给 `_next_step_logits` 的参数从 `query_vec=query_vec_batch` 改为 `query_h=query_h_batch`。

---

#### 2.7 修改 `predict` 中对 `_predict_greedy` / `_predict_beam_single` 的调用

**修改前：**
```python
return self._predict_greedy(z_final, batch['query_vec'], ...)

seq = self._predict_beam_single(
    z_final_vec=z_final[b],
    query_vec_single=batch['query_vec'][b],
    ...
)
```

**修改后：**
```python
return self._predict_greedy(z_final, query_h, ...)   # query_h 已在 predict 中通过 _encode_query 得到

seq = self._predict_beam_single(
    z_final_vec=z_final[b],
    query_h_single=query_h[b],
    ...
)
```

---

### 不需要修改的文件

| 文件 | 原因 |
|---|---|
| `data.py` | batch 中已有所有需要字段：`query_start_poi`, `query_end_poi`, `query_start_hour`, `query_end_hour` |
| `trainer.py` | `unpack_batch` 保持不变；`query_vec` 字段仍在 batch 中（供 `_blend_user_city` 使用）|
| `main.py` | 无新增 CLI 参数；`query_seq_encoder` 复用现有的 `seq_num_layers`, `mamba_d_state` 等 args |
| `spot_utils.py` | collate_fn 不变 |

---

### 注意：checkpoint 兼容性

decoder 的 `input_size` 从 `hidden_size*2` 改为 `hidden_size*4`，`CausalMambaDecoder.in_proj` 的权重形状改变。**旧 checkpoint 无法直接 `strict=True` 加载**，需要从头重新训练。

---

## 二、`run_both_datasets_pipeline.py` 新建说明

参考 `run_yelp_soft_pipeline.py` 实现，支持对 Yelp 和 Foursquare 两个数据集顺序执行完整三阶段流程。

### 主要特性

- `--datasets` 参数（默认 `yelp,foursquare`）控制运行哪些数据集
- Yelp 参数统一加 `--yelp_` 前缀，Foursquare 参数加 `--fsq_` 前缀
- 核心三阶段逻辑抽取为 `_run_single_dataset(args, ds_cfg)` 函数，失败时返回 `status='failed'` 而非直接退出，确保一个数据集失败不影响另一个继续执行
- 最终输出合并报告 `pipeline_runs/both_datasets_report_{timestamp}.md`

---

## 三、服务器实验命令

### 3.1 完整跑两个数据集（推荐）

```bash
cd /root/autodl-tmp/MyCrossCity/code/new_citypref_llm

python run_both_datasets_pipeline.py \
  --datasets yelp,foursquare \
  --device cuda:0 \
  --yelp_data_split_path /root/autodl-tmp/MyCrossCity/Yelp/spottrip_baseline_split.pkl \
  --yelp_ori_data ../../Yelp/home.txt \
  --yelp_dst_data ../../Yelp/oot.txt \
  --yelp_trans_data ../../Yelp/travel.txt \
  --yelp_ori_data_enriched ../../Yelp/extendData/enriched_home.txt \
  --yelp_dst_data_enriched ../../Yelp/extendData/enriched_oot.txt \
  --yelp_save_path ../../Yelp/model_save_new \
  --fsq_data_split_path ../../Foursquare/data_split_new.pkl \
  --fsq_ori_data ../../Foursquare/home.txt \
  --fsq_dst_data ../../Foursquare/oot.txt \
  --fsq_trans_data ../../Foursquare/travel.txt \
  --fsq_ori_data_enriched ../../Foursquare/extendData/enriched_home.txt \
  --fsq_dst_data_enriched ../../Foursquare/extendData/enriched_oot.txt \
  --fsq_save_path ../../Foursquare/model_save_new
```

### 3.2 只跑 Yelp

```bash
cd /root/autodl-tmp/MyCrossCity/code/new_citypref_llm

python run_both_datasets_pipeline.py \
  --datasets yelp \
  --device cuda:0 \
  --yelp_data_split_path /root/autodl-tmp/MyCrossCity/Yelp/spottrip_baseline_split.pkl \
  --yelp_ori_data ../../Yelp/home.txt \
  --yelp_dst_data ../../Yelp/oot.txt \
  --yelp_trans_data ../../Yelp/travel.txt \
  --yelp_ori_data_enriched ../../Yelp/extendData/enriched_home.txt \
  --yelp_dst_data_enriched ../../Yelp/extendData/enriched_oot.txt \
  --yelp_save_path ../../Yelp/model_save_new
```

### 3.3 只跑 Foursquare

```bash
cd /root/autodl-tmp/MyCrossCity/code/new_citypref_llm

python run_both_datasets_pipeline.py \
  --datasets foursquare \
  --device cuda:0 \
  --fsq_data_split_path ../../Foursquare/data_split_new.pkl \
  --fsq_ori_data ../../Foursquare/home.txt \
  --fsq_dst_data ../../Foursquare/oot.txt \
  --fsq_trans_data ../../Foursquare/travel.txt \
  --fsq_ori_data_enriched ../../Foursquare/extendData/enriched_home.txt \
  --fsq_dst_data_enriched ../../Foursquare/extendData/enriched_oot.txt \
  --fsq_save_path ../../Foursquare/model_save_new
```

### 3.4 跳过调优直接训练（使用内置默认超参）

```bash
python run_both_datasets_pipeline.py \
  --datasets yelp,foursquare \
  --skip_tuning 1 \
  [... 数据集路径参数 ...]
```

### 3.5 只跑最终训练（跳过调优和消融）

```bash
python run_both_datasets_pipeline.py \
  --datasets yelp,foursquare \
  --skip_tuning 1 \
  --skip_ablation 1 \
  [... 数据集路径参数 ...]
```

> **注意**：如果服务器路径与默认值一致，可省略对应的路径参数。实验结果位于 `pipeline_runs/{yelp|fsq}_both_pipeline_v1/` 目录，汇总报告位于 `pipeline_runs/both_datasets_report_{timestamp}.md`。
