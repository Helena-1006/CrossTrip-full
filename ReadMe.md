### 代码修改说明
为了快速构建自己的模型，我在SPOT-Trip的基础上提取出了与模型无关的训练测试框架。
其中
- data.py, metrics.py, spot_utils.py文件保留
- 与模型相关的文件均注释掉了，并在model.py中写了一个Mymodel的模型框架，便于后续填充我自己的模型
- main和trainer需要重写。