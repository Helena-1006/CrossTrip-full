# CrossTrip 复现说明（`new_citypref_llm`）

本文档说明如何复现 `code/new_citypref_llm` 下的跨城旅游偏好迁移算法实验，并说明如何配合 **GitHub + TeraBox** 组织代码与数据。

## 1. 项目内容概述

`new_citypref_llm` 是当前实验主目录，包含以下核心内容：

- `main.py`：单次训练 / 测试入口
- `optuna_tune.py`：超参数搜索
- `run_both_datasets_pipeline.py`：一键顺序运行 Yelp 与 Foursquare 两个数据集的完整流程
- `model.py`：模型主体，包含 soft prompt、跨城偏好迁移、约束解码等模块
- `trainer.py`：训练、验证与 checkpoint 保存逻辑
- `pipeline_runs/`：历史实验输出目录

其中，已经跑出的 Foursquare 双数据集流水线实验结果位于：

```text
code/new_citypref_llm/pipeline_runs/fsq_both_pipeline_v1
```

如果本地仓库中也保留了 Yelp 的实验输出，则通常位于：

```text
code/new_citypref_llm/pipeline_runs/yelp_both_pipeline_v1
```

## 2. 代码与数据的组织方式

建议在服务器或本地统一组织为下面的目录结构：

```text
MyCrossCity/
├── code/
│   ├── ReadMe.md
│   └── new_citypref_llm/
│       ├── main.py
│       ├── model.py
│       ├── trainer.py
│       ├── optuna_tune.py
│       ├── run_both_datasets_pipeline.py
│       └── pipeline_runs/
├── Foursquare/
│   ├── home.txt
│   ├── oot.txt
│   ├── travel.txt
│   ├── data_split_new.pkl
│   ├── extendData/
│   │   ├── enriched_home.txt
│   │   └── enriched_oot.txt
│   └── model_save_new/
└── Yelp/
    ├── home.txt
    ├── oot.txt
    ├── travel.txt
    ├── spottrip_baseline_split.pkl
    ├── extendData/
    │   ├── enriched_home.txt
    │   └── enriched_oot.txt
    └── model_save_new/
```

也就是说：

- **GitHub**：保存 `code/` 目录
- **TeraBox**：保存 `Foursquare/` 与 `Yelp/` 两个大目录
- 下载数据后，将 `Foursquare/` 和 `Yelp/` 与 `code/` 放在同一级目录下

## 3. TeraBox 数据说明

请将你上传到 TeraBox 的数据链接补充到这里：

- Foursquare：`<请填写你的 TeraBox 链接>`
- Yelp：`<请填写你的 TeraBox 链接>`

建议在 TeraBox 中直接保留目录压缩包，或按目录分别上传。下载后解压到 `MyCrossCity/` 根目录，保证最终路径满足上一节的目录结构。

## 4. 环境配置

仓库中目前没有单独提供 `requirements.txt` 或 `environment.yml`，可以按下面方式手动创建环境。

### 4.1 建议环境

- Linux
- Python 3.10
- CUDA 11.8 或兼容版本
- 建议使用独立 conda 环境

### 4.2 创建环境

```bash
conda create -n crosstrip python=3.10 -y
conda activate crosstrip
```

### 4.3 安装依赖

先安装 PyTorch（请根据你的 CUDA 版本选择官方对应命令；下面以 CUDA 11.8 为例）：

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

再安装其余 Python 依赖：

```bash
pip install numpy tqdm optuna transformers accelerate sentencepiece pytz
```

如果你需要启用 `model.py` 中的 Mamba 主干，还需要额外安装：

```bash
pip install mamba-ssm
```

如果 `mamba-ssm` 安装失败，可以先不装，并在实验时关闭相关选项；但当前流水线默认会使用 Mamba 主干，因此更建议安装完成后再复现。

## 5. 模型与数据路径约定

`new_citypref_llm` 中的默认相对路径是按如下结构写的：

- Foursquare 默认读取：
  - `../../Foursquare/home.txt`
  - `../../Foursquare/oot.txt`
  - `../../Foursquare/travel.txt`
  - `../../Foursquare/extendData/enriched_home.txt`
  - `../../Foursquare/extendData/enriched_oot.txt`
- Yelp 默认读取：
  - `../../Yelp/home.txt`
  - `../../Yelp/oot.txt`
  - `../../Yelp/travel.txt`
  - `../../Yelp/extendData/enriched_home.txt`
  - `../../Yelp/extendData/enriched_oot.txt`

因此，**只要保证 `Foursquare/`、`Yelp/` 与 `code/` 同级放置，就不需要额外修改脚本路径。**

## 6. 复现实验命令

进入实验目录后，直接运行：

```bash
cd /root/autodl-tmp/MyCrossCity/code/new_citypref_llm
python run_both_datasets_pipeline.py
```

这条命令会顺序执行两个数据集的完整流程，包含：

1. 超参数调优（Optuna）
2. 最终训练与测试
3. 消融实验与敏感性实验

如果你当前目录结构与本文一致，脚本会自动使用默认数据路径。

## 7. 结果输出位置

运行完成后，结果会保存在：

```text
code/new_citypref_llm/pipeline_runs/
```

典型输出包括：

- `fsq_both_pipeline_v1/`
- `yelp_both_pipeline_v1/`
- `both_datasets_report_*.md`

其中各实验目录下通常会包含：

- `tuning/`：Optuna 搜索记录、最优参数、trial 明细
- `final_train/`：最终训练阶段日志
- `ablation/`：消融实验日志

## 8. 已有结果复查

如果你已经保留了历史运行结果，可以优先查看：

```text
code/new_citypref_llm/pipeline_runs/fsq_both_pipeline_v1
```

该目录下保存了 Foursquare 相关流水线实验的调参记录、训练日志和消融实验日志，可用于核对复现结果是否一致。

## 9. 复现建议

- 首次复现前，先确认 `Foursquare/` 和 `Yelp/` 已完整解压
- 确认 `extendData/` 子目录存在
- 确认 GPU 可用；若无 GPU，需要自行将相关脚本中的设备参数改为 CPU
- 若 Qwen 模型需要从 Hugging Face 下载，请提前保证网络或缓存目录可用
- 若只希望保留论文复现所需结果，建议只备份最终结果目录与最优 checkpoint，不保留中间 checkpoint

## 10. 推荐备份方式

为了后续服务器到期后仍能完整复现，建议按以下方式保存：

- **GitHub**：上传 `code/`
- **TeraBox**：上传 `Foursquare/`、`Yelp/` 压缩包或目录
- **README**：保留本文档，并补上你自己的 TeraBox 链接

这样新的使用者只需要：

1. 从 GitHub 拉取 `code/`
2. 从 TeraBox 下载 `Foursquare/` 和 `Yelp/`
3. 按本文档组织目录
4. 执行复现命令即可
