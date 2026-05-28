# from jinja2 import Template
# from typing import List, Dict
# import numpy as np
# from datetime import datetime

# class PromptBuilder:
#     """LLM Prompt构建器 - User Preference Mode"""

#     def __init__(self):
#         # [NEW] 定义需要忽略的通勤/日常类别关键词 (大小写不敏感匹配)
#         self.ignore_categories = {
#             'airport', 'office', 'train station', 'residential', 
#             'subway', 'bus station', 'work', 'home', 'building', 'transport hub'
#         }
#         # 4段式模板
#         self.template_str = """
# {# (1) Head Part #}
# A user from {{hometown}} on {{start_date}} at {{start_hour}} o'clock, 

# {# (2) POI-Context Part (First 3 POIs) #}
# often visits around
# {%- for poi in context_pois -%}
#  {{poi.name}} ({{poi.cat_tree}}){% if not loop.last %}, {% endif %}
# {%- endfor -%}, 

# {# (3) Sequence Part (Remaining POIs) #}
# and then checks in 
# {%- for poi in sequence_pois -%}
#  {{poi.name}} ({{poi.cat_tree}}, {{poi.chain}}, {{poi.hour}}){% if not loop.last %}, {% endif %}
# {%- endfor %}.

# {# (4) Suffix #}
# The user preference is [UserPrefToken].
# """
#         self.template = Template(self.template_str)
    
#     def build_prompt(
#         self,
#         hometown: str,
#         trajectory: List[Dict],
#         embedding_dim: int = None # 兼容接口，暂未用到
#     ) -> str:
#         """
#         构建 User Preference Prompt
#         """
#         if not trajectory:
#             return ""

#         # 处理时间
#         first_ts = trajectory[0]['timestamp']
#         dt_obj = datetime.fromtimestamp(first_ts)
#         start_date = dt_obj.strftime('%Y-%m-%d')
#         start_hour = dt_obj.hour
        
#         # 划分 Context (前3个) 和 Sequence (剩余)
#         # 如果序列长度小于3，Context取全部，Sequence为空
#         split_idx = min(3, len(trajectory))
        
#         context_data = []
#         for i in range(split_idx):
#             poi = trajectory[i]
#             context_data.append({
#                 'name': poi.get('poi_name', 'Unknown'),
#                 'cat_tree': poi.get('cat_tree', 'Unknown')
#             })
            
#         sequence_data = []
#         for i in range(split_idx, len(trajectory)):
#             poi = trajectory[i]
#             # [NEW] 过滤逻辑: 检查是否为通勤无关POI
#             cat_name = str(poi.get('cat_tree', '')).lower()
#             poi_name = str(poi.get('poi_name', '')).lower()
#             # 检查是否包含任何黑名单关键词
#             is_commute = False
#             for pad_word in self.ignore_categories:
#                 if pad_word in cat_name or pad_word in poi_name:
#                     is_commute = True
#                     break
            
#             # 如果是通勤类地点，跳过不放入Prompt序列
#             if is_commute:
#                 continue

#             ts = poi.get('timestamp', 0)
#             hour = datetime.fromtimestamp(ts).hour
#             sequence_data.append({
#                 'name': poi.get('poi_name', 'Unknown'),
#                 'cat_tree': poi.get('cat_tree', 'Unknown'),
#                 'chain': poi.get('city', 'Unknown'), # 暂时用Client city代替Chain，如果数据里有Chain更好
#                 'hour': f"{hour}:00"
#             })
            
#         # 渲染prompt
#         prompt = self.template.render(
#             hometown=hometown,
#             start_date=start_date,
#             start_hour=start_hour,
#             context_pois=context_data,
#             sequence_pois=sequence_data
#         )
        
#         # 清理多余空格和换行，保持紧凑
#         return " ".join(prompt.split())