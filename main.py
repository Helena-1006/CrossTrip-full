# -*- coding: utf-8 -*-
import argparse
import os
import torch
from torch.utils.data import DataLoader
import numpy as np

# 引用工具库 (包含 collate_fn, Logger, set_seeds, save_model 等)
from spot_utils import *
from data import TravelDataset, random_split

# 引用你的新 Trainer 和 Model
from trainer import Trainer
from model import MyRouteModel

def main():
    parser = argparse.ArgumentParser()
    # ================= 1. 数据集路径设置 =================
    parser.add_argument('--dataset_name', type=str, default='Foursquare')
    parser.add_argument('--ori_data', type=str, default=f'../Foursquare/home.txt')
    parser.add_argument('--dst_data', type=str, default=f'../Foursquare/oot.txt')
    parser.add_argument('--trans_data', type=str, default=f'../Foursquare/travel.txt')
    parser.add_argument('--save_path', type=str, default=f'../Foursquare/model_save')
    parser.add_argument('--data_split_path', type=str, default=f'../Foursquare/data_split.pkl')

    # ================= 2. 训练超参数 =================
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--device', type=str, default="cuda:0")
    parser.add_argument('--seed', type=int, default=2050)
    
    parser.add_argument('--train_batch', type=int, default=16)
    parser.add_argument('--test_batch', type=int, default=16)
    parser.add_argument('--epoch', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--l2', type=float, default=1e-5)
    parser.add_argument('--lr_dc', type=float, default=0.2)
    parser.add_argument('--lr_dc_step', type=int, default=10)
    
    # 保存与日志
    parser.add_argument('--save_step', type=int, default=1)
    parser.add_argument("--best_save", action="store_true", help="是否保存效果最好的模型")
    parser.add_argument("--stop_epoch", type=int, default=10, help="早停轮数")
    parser.add_argument('--log_path', type=str, default='../')
    parser.add_argument('--log', action="store_true")
    parser.add_argument('--name', type=str, default="demo_experiment")

    # ================= 3. 模型参数 (根据你的 MyNewModel 调整) =================
    parser.add_argument('--hidden_size', type=int, default=64)
    # 你可以在这里添加更多你模型需要的参数，例如 dropout, layer_num 等
    parser.add_argument('--poi_num', type=int, default=23884) # 记得修改这个值！
    parser.add_argument('--pool_path', type=str, default=f'../Foursquare/preference_pool.pt')


    args = parser.parse_args()

    # 环境初始化
    set_seeds(args.seed)
    args.save_path = os.path.join(args.save_path, args.name)
    path_exist(args.save_path)

    # 日志初始化
    logger = Logger(args.log_path, args.name, args.seed, args.log)
    logger.log(str(args))
    logger.log(f"Experiment name: {args.name}")

    # ================= 4. 数据加载 (保留核心逻辑) =================
    logger.log("Loading data...")
    # 初始化 Dataset
    data = TravelDataset(args, args.ori_data, args.dst_data, args.trans_data)
    
    # 划分数据集
    train_data, valid_data, test_data = random_split(data, dataset_name=args.dataset_name, split_path=args.data_split_path)
    
    # 构建 DataLoader (注意: collate_fn 来自 spot_utils, 用于处理 padding)
    train_loader = DataLoader(train_data, args.train_batch, shuffle=True, collate_fn=collate_fn)
    valid_loader = DataLoader(valid_data, args.test_batch, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_data, args.test_batch, shuffle=False, collate_fn=collate_fn)

    # ================= 5. 模型与训练器初始化 =================
    logger.log("Initializing model...")
    model = MyRouteModel(args).to(args.device)
    
    trainer = Trainer(model, args, logger)

    # ================= 6. 执行训练或测试 =================
    if args.mode == 'train':
        logger.log("Start training...")
        best_epoch = trainer.train(train_loader, valid_loader)
        logger.log(f"Training finished. Best epoch: {best_epoch}")
        
        # 训练结束后，加载最佳模型在测试集上跑一遍
        best_model_path = os.path.join(args.save_path, "model_best.xhr")
        if os.path.exists(best_model_path):
            logger.log("Loading best model for testing...")
            checkpoint = torch.load(best_model_path)
            model.load_state_dict(checkpoint['state_dict'])
            trainer.validate(test_loader, "FINAL_TEST")
            
    elif args.mode == 'test':
        logger.log("Loading model for testing...")
        best_model_path = os.path.join(args.save_path, "model_best.xhr")
        if os.path.exists(best_model_path):
            checkpoint = torch.load(best_model_path)
            model.load_state_dict(checkpoint['state_dict'])
            trainer.validate(test_loader, "TEST_ONLY")
        else:
            logger.log(f"Error: No model found at {best_model_path}")

    logger.close_log()

if __name__ == "__main__":
    main()