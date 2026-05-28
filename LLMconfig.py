# import os
# from typing import Optional

# class LLMConfig:
#     """LLM配置类 - 使用本地GPT-2模型"""
#     def __init__(
#         self,
#         model_name: str = "uer/gpt2-chinese-cluecorpussmall",
#         cache_dir: Optional[str] = None,
#         embedding_dim: int = 128,
#         max_length: int = 512,
#         use_lora: bool = True,
#         lora_r: int = 8,
#         lora_alpha: int = 32,
#         lora_dropout: float = 0.02,
#     ):
#         self.model_name = model_name
#         self.cache_dir = cache_dir or os.path.join(os.getcwd(), 'params/gpt2-chinese')
#         self.embedding_dim = embedding_dim
#         self.max_length = max_length
#         self.use_lora = use_lora
#         self.lora_r = lora_r
#         self.lora_alpha = lora_alpha
#         self.lora_dropout = lora_dropout