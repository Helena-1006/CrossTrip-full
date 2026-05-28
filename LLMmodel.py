# import torch
# import torch.nn as nn
# from transformers import AutoTokenizer, GPT2LMHeadModel
# from peft import LoraConfig, get_peft_model, TaskType
# from typing import List, Dict
# import os

# class GPT2UserPrefEncoder(nn.Module):
#     """基于GPT-2的LLM编码器，用于获取静态用户偏好向量"""
    
#     def __init__(self, config):
#         super().__init__()
#         self.config = config
        
#         print(f"Loading GPT-2 model (User Pref Mode) from {config.model_name}...")
#         self.tokenizer = AutoTokenizer.from_pretrained(
#             config.model_name,
#             cache_dir=config.cache_dir,
#             trust_remote_code=True
#         )
        
#         # 设置pad_token
#         if self.tokenizer.pad_token is None:
#             self.tokenizer.pad_token = self.tokenizer.eos_token
        
#         self.model = GPT2LMHeadModel.from_pretrained(
#             config.model_name,
#             cache_dir=config.cache_dir,
#             trust_remote_code=True
#         )
        
#         self.gpt2_hidden_size = self.model.config.n_embd
        
#         # [CHANGE] 添加 UserPrefToken
#         self._add_special_tokens()
        
#         # 应用LoRA
#         if config.use_lora:
#             self._apply_lora()
#         else:
#             for param in self.model.parameters():
#                 param.requires_grad = False
        
#         # 投影层：将GPT-2 hidden size 映射到 SPOT-Trip 的 hidden_size (例如32或64)
#         self.pref_projection = nn.Sequential(
#             nn.Linear(self.gpt2_hidden_size, config.embedding_dim),
#             nn.Tanh() # 偏好向量通常归一化或限制范围较好
#         )
        
#     def _add_special_tokens(self):
#         """添加特殊的 User Preference token"""
#         self.pref_token = "[UserPrefToken]"
#         special_tokens = {'additional_special_tokens': [self.pref_token]}
#         self.tokenizer.add_special_tokens(special_tokens)
#         self.model.resize_token_embeddings(len(self.tokenizer))
#         # 记录 ID 以便快速查找
#         self.pref_token_id = self.tokenizer.convert_tokens_to_ids(self.pref_token)
#         print(f"Added special token: {self.pref_token} (ID: {self.pref_token_id})")
    
#     def _apply_lora(self):
#         lora_config = LoraConfig(
#             task_type=TaskType.CAUSAL_LM,
#             r=self.config.lora_r,
#             lora_alpha=self.config.lora_alpha,
#             lora_dropout=self.config.lora_dropout,
#             target_modules=["c_attn", "c_proj"],
#             bias="none",
#         )
#         self.model = get_peft_model(self.model, lora_config)
    
#     def forward(self, prompts: List[str]) -> torch.Tensor:
#         """
#         Batch 处理 Prompts 并返回 User Preference Vectors
#         """
#         # Batch Tokenize
#         inputs = self.tokenizer(
#             prompts,
#             return_tensors='pt',
#             max_length=self.config.max_length,
#             truncation=True,
#             padding=True
#         ).to(self.model.device)
        
#         # Forward
#         outputs = self.model(**inputs, output_hidden_states=True)
#         last_hidden_state = outputs.hidden_states[-1] # [Batch, Seq, Hidden]
        
#         # 提取 [UserPrefToken] 的向量
#         # 逻辑：找到 input_ids 中等于 pref_token_id 的位置
#         batch_size = inputs['input_ids'].shape[0]
#         pref_vectors = []
        
#         for i in range(batch_size):
#             # 找到当前样本中 token id 的位置
#             indices = (inputs['input_ids'][i] == self.pref_token_id).nonzero(as_tuple=True)[0]
#             if len(indices) > 0:
#                 # 取最后一个匹配的 (通常只有一个)
#                 idx = indices[-1]
#                 vec = last_hidden_state[i, idx, :]
#             else:
#                 # Fallback: 如果被截断导致没有token，取最后一个token
#                 vec = last_hidden_state[i, -1, :]
#             pref_vectors.append(vec)
            
#         pref_tensor = torch.stack(pref_vectors) # [Batch, GPT_Hidden]
        
#         # 投影到目标维度
#         return self.pref_projection(pref_tensor) # [Batch, Embedding_Dim]