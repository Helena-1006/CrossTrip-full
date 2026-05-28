## new_citypref_llm

本目录是**独立于旧代码**的新模型实现，覆盖训练、验证、推理完整流程。

### 文件说明

- `main.py`：训练/测试入口
- `validate.py`：单独验证入口
- `infer.py`：单样本推理入口
- `data.py`：数据集读取与特征构建
- `model.py`：新模型核心结构（soft prompt + 解耦偏好 + 对齐 + 约束生成）
- `trainer.py`：训练与验证循环
- `spot_utils.py`：日志、seed、保存模型、collate
- `metrics.py`：轨迹评估指标

### 运行方式

> 默认路径以当前文件所在目录为基准，建议先进入本目录执行。

```bash
cd /root/autodl-tmp/MyCrossCity/code/new_citypref_llm

# 训练（仅训练+验证，不触发测试集评估）
python main.py --mode train --log

# 重建划分并训练（会删除已有data_split_path）
python main.py --mode train --rebuild_split --seed 2050

# 最终测试（单独运行，读取 best checkpoint）
python main.py --mode test --ckpt_name model_best.xhr

# 如需在训练结束后立即跑一次测试（不推荐用于严格科研对比）
python main.py --mode train --run_final_test_after_train

# 验证集评估
python validate.py --split valid --ckpt_name model_best.xhr

# 测试集单样本推理
python infer.py --split test --sample_index 0 --ckpt_name model_best.xhr
```

### 关键提示

- 若需要切换数据集（例如 Yelp），请同步修改：
  - `--dataset_name`
  - `--ori_data`
  - `--dst_data`
  - `--trans_data`
  - `--data_split_path`
- 若 GPU 不可用，请设置 `--device cpu`。
- 数据划分由 `random_split` 负责：若 `data_split_path` 已存在则直接复用；不存在时按 `--seed` 一次性生成并落盘。
- `best checkpoint` 会在验证集 `F1` 提升时自动保存到：`{save_path}/{name}/model_best.xhr`。
- query 按任务设定构建：`<start_poi, start_time, end_poi, end_time, route_len>`（并拼接OD城市特征）。
- 推理阶段生成仅依赖 `home_traj + target_city + query`，不再读取完整目标轨迹中间POI。
- 关键新增逻辑均在代码中用 `NEW:` 注释标注。

### Qwen 语义编码

- 默认启用 `--semantic_backend qwen`，会加载 `--llm_model_name` 指定的 HuggingFace Qwen 模型。
- 若加载失败，会自动回退到结构化 frozen-transformer 语义编码器。
- 建议提前安装依赖：

```bash
pip install transformers accelerate sentencepiece
```

- 若你本地没有 `Qwen/Qwen3.5-0.6B-Instruct`，可直接替换为可用的 Qwen 模型名：

```bash
python main.py --llm_model_name Qwen/Qwen2.5-1.5B-Instruct
```
