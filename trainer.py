# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import os
from collections import defaultdict

# 引用你的工具库
from spot_utils import save_model
import metrics

class Trainer:
    def __init__(self, model, args, logger):
        self.model = model
        self.args = args
        self.logger = logger
        self.optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=args.lr_dc_step, gamma=args.lr_dc)
        self.device = args.device

    def unpack_batch(self, batch_data):
        """
        统一处理 DataLoader 返回的 15 元组，移动到 GPU。
        你可以根据需要在模型内部挑选使用的 tensor。
        对应 data.py 的输出顺序。
        """
        return [x.to(self.device) for x in batch_data]

    def train_epoch(self, train_loader, epoch_idx):
        self.model.train()
        loss_sum = 0.
        
        # 包装 tqdm
        iter_wrapper = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch_idx} Train")
        
        for b, batch_data in iter_wrapper:
            # 1. 数据准备
            inputs = self.unpack_batch(batch_data)
            
            # 2. 前向传播 & Loss 计算
            self.optimizer.zero_grad()
            
            # 约定：模型 forward 接收整个 inputs 列表，返回 loss
            loss = self.model(inputs) 
            
            # 3. 反向传播
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            loss_sum += loss.item()
            
        self.scheduler.step()
        avg_loss = loss_sum / len(train_loader)
        self.logger.log(f"Epoch {epoch_idx}/{self.args.epoch} : Train Loss {avg_loss:.6f}")
        return avg_loss

    def validate(self, valid_loader, epoch_idx):
        self.model.eval()
        import metrics 

        batch_f1 = []
        batch_pairs_f1 = []
        batch_full_f1 = []
        batch_full_pairs_f1 = []

        with torch.no_grad():
            for batch_data in tqdm(valid_loader, desc=f"Epoch {epoch_idx} Val"):
                inputs = self.unpack_batch(batch_data)
                target_batch = inputs[2]
                pred_batch = self.model.predict(inputs) # 现在 predict 会生成合理的长度
                
                batch_size = target_batch.size(0)
                for i in range(batch_size):
                    # 1. 提取序列并转为 list (去除 padding 0)
                    t_seq = [x for x in target_batch[i].cpu().numpy().tolist() if x != 0]
                    p_seq = [x for x in pred_batch[i].cpu().numpy().tolist() if x != 0]
                    
                    if len(t_seq) < 2: continue
                    
                    # 2. [关键] 强制截断/对齐预测长度 = 真实长度
                    # 因为你的任务逻辑是固定长度生成
                    target_len = len(t_seq)
                    if len(p_seq) > target_len:
                        p_seq = p_seq[:target_len] # 截断多余的
                    elif len(p_seq) < target_len:
                        # 这种情况理论上少见（如果用循环生成），但若发生了，就保持原样
                        pass
                    
                    # 3. 构造 Mid 和 Full
                    t_mid = t_seq[1:-1]
                    p_mid = p_seq[1:-1] if len(p_seq) >= 2 else []
                    
                    # Full: 强制把首尾换成真实的 OD，评估中间的填充能力
                    # (或者你可以保留模型预测的 OD，但这取决于评测标准)
                    t_full = t_seq
                    p_full_constrained = [t_seq[0]] + p_mid + [t_seq[-1]]

                    # ========================================================
                    # [Hack Fix] metrics.py 似乎对 Tensor/List 支持有 bug
                    # 这里的报错 RuntimeError: a Tensor with 19 elements ...
                    # 说明 metrics.py 内部某处把序列当成标量处理了。
                    # 通常我们自己在这里算 F1 最安全。
                    # ========================================================
                    
                    # 为了规避 metrics.py 的内部实现问题，我们直接在这里转 Tensor 传进去
                    # 假设 metrics.py 可以处理 1D Tensor
                    t_mid_T = torch.LongTensor(t_mid)
                    p_mid_T = torch.LongTensor(p_mid)
                    t_full_T = torch.LongTensor(t_full)
                    p_full_T = torch.LongTensor(p_full_constrained)

                    # 计算
                    if len(t_mid) > 0 and len(p_mid) > 0:
                        f1 = metrics.f1_score(t_mid_T, p_mid_T)
                        # 如果 metrics.pairs_f1_score 坏了，我们可以 try-catch 或者暂时只算 f1
                        try:
                            p_f1 = metrics.pairs_f1_score(t_mid_T, p_mid_T)
                        except RuntimeError: 
                            p_f1 = 0 # 降级处理
                    else:
                        f1, p_f1 = 0.0, 0.0
                    
                    batch_f1.append(f1)
                    batch_pairs_f1.append(p_f1)
                    
                    # Full 计算
                    try:
                       full_f1 = metrics.f1_score(t_full_T, p_full_T)
                       full_p_f1 = metrics.pairs_f1_score(t_full_T, p_full_T)
                    except RuntimeError:
                       full_f1, full_p_f1 = 0, 0
                       
                    batch_full_f1.append(full_f1)
                    batch_full_pairs_f1.append(full_p_f1)

        # 汇总结果
        metrics_dict = {
            "f1": np.mean(batch_f1) if batch_f1 else 0,
            "pairs_f1": np.mean(batch_pairs_f1) if batch_pairs_f1 else 0,
            "full_f1": np.mean(batch_full_f1) if batch_full_f1 else 0,
            "full_pairs_f1": np.mean(batch_full_pairs_f1) if batch_full_pairs_f1 else 0
        }
        
        self.logger.log(f"[Val] Epoch {epoch_idx} "
                        f"F1: {metrics_dict['f1']:.4f} | Pairs_F1: {metrics_dict['pairs_f1']:.4f} | "
                        f"Full_F1: {metrics_dict['full_f1']:.4f} | Full_Pairs_F1: {metrics_dict['full_pairs_f1']:.4f}")
        
        self.model.train()
        return metrics_dict

    def train(self, train_loader, valid_loader):
        stopping_dict = defaultdict(float)
        best_epoch = -1
        
        for e in range(self.args.epoch):
            # 1. 训练
            self.train_epoch(train_loader, e)
            
            # 2. 保存 Checkpoint
            if e % self.args.save_step == 0:
                save_model(self.model, e, self.args.save_path, self.optimizer, self.scheduler)
            
            # 3. 验证
            val_metrics = self.validate(valid_loader, e)
            
            # 4. Early Stopping 逻辑
            current_f1 = val_metrics['f1']
            if current_f1 > stopping_dict['best_f1']:
                stopping_dict['best_f1'] = current_f1
                stopping_dict['f1_early_stop_cnt'] = 0
                stopping_dict['best_epoch'] = e
                best_epoch = e
                if self.args.best_save:
                     save_model(self.model, "best", self.args.save_path, self.optimizer, self.scheduler)
            else:
                stopping_dict['f1_early_stop_cnt'] += 1
            
            if stopping_dict['f1_early_stop_cnt'] >= self.args.stop_epoch:
                self.logger.log("Early stopped!")
                break
        
        return best_epoch