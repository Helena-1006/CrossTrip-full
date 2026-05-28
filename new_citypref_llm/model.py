import math

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from transformers import AutoModel, AutoTokenizer
except Exception:
    AutoModel = None
    AutoTokenizer = None

try:
    from mamba_ssm import Mamba  # type: ignore
except Exception:
    Mamba = None


class FrozenSemanticEncoder(nn.Module):
    """
    Structured semantic encoder with frozen transformer blocks and trainable soft prompt.
    """

    def __init__(self, poi_num, tag_num, hidden_size, prompt_len=8, nhead=4, nlayers=2, dropout=0.1):
        super().__init__()
        self.prompt_len = prompt_len
        self.hidden_size = hidden_size

        self.poi_emb = nn.Embedding(poi_num, hidden_size, padding_idx=0)
        self.tag_emb = nn.Embedding(tag_num, hidden_size, padding_idx=0)
        self.hour_emb = nn.Embedding(25, hidden_size, padding_idx=0)
        self.coord_proj = nn.Linear(2, hidden_size)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=nhead,
            dim_feedforward=hidden_size * 4,
            batch_first=True,
            dropout=dropout,
            activation='gelu',
        )
        self.frozen_encoder = nn.TransformerEncoder(encoder_layer, num_layers=nlayers)
        self.soft_prompt = nn.Parameter(torch.randn(prompt_len, hidden_size) * 0.02)

        for p in self.frozen_encoder.parameters():
            p.requires_grad = False

    def forward(self, poi_seq, tag_seq, hour_seq, coord_seq, valid_mask):
        token_emb = self.poi_emb(poi_seq) + self.tag_emb(tag_seq) + self.hour_emb(hour_seq.clamp(min=0, max=24))
        token_emb = token_emb + self.coord_proj(coord_seq)

        batch_size = token_emb.size(0)
        prompt = self.soft_prompt.unsqueeze(0).expand(batch_size, -1, -1)
        x = torch.cat([prompt, token_emb], dim=1)

        prompt_mask = torch.ones((batch_size, self.prompt_len), dtype=torch.bool, device=valid_mask.device)
        full_valid = torch.cat([prompt_mask, valid_mask], dim=1)
        src_key_padding_mask = ~full_valid

        h = self.frozen_encoder(x, src_key_padding_mask=src_key_padding_mask)
        token_h = h[:, self.prompt_len:, :]

        denom = valid_mask.float().sum(dim=1, keepdim=True).clamp(min=1.0)
        pooled = (token_h * valid_mask.unsqueeze(-1).float()).sum(dim=1) / denom
        return pooled


class QwenSoftPromptEncoder(nn.Module):
    """
    Qwen encoder with frozen LLM parameters and trainable soft prompt.
    """

    def __init__(self, args):
        super().__init__()
        if AutoModel is None or AutoTokenizer is None:
            raise ImportError("transformers is required for Qwen semantic encoder")

        self.args = args
        self.train_soft_prompt = bool(getattr(args, 'qwen_train_soft_prompt', 0))
        self.prompt_len = args.soft_prompt_len
        model_name = self._resolve_model_name(args)
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=args.llm_cache_dir,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        dtype_map = {
            'float16': torch.float16,
            'bfloat16': torch.bfloat16,
            'float32': torch.float32,
        }
        llm_dtype = dtype_map.get(args.llm_dtype, torch.float16)
        self.llm = AutoModel.from_pretrained(
            model_name,
            cache_dir=args.llm_cache_dir,
            trust_remote_code=True,
            torch_dtype=llm_dtype,
        )
        self.llm.config.use_cache = False
        if getattr(args, 'llm_gradient_checkpointing', False):
            if self.train_soft_prompt:
                try:
                    self.llm.gradient_checkpointing_enable()
                except Exception as e:
                    print(f"[WARN] Failed to enable gradient checkpointing: {e}")
            else:
                print("[INFO] Skip LLM gradient checkpointing because qwen_train_soft_prompt=0 (inference-only Qwen).")
        self.llm.eval()
        self.llm_hidden = int(self.llm.config.hidden_size)

        for p in self.llm.parameters():
            p.requires_grad = False

        self.soft_prompt = nn.Parameter(torch.randn(self.prompt_len, self.llm_hidden) * 0.02)
        if not self.train_soft_prompt:
            self.soft_prompt.requires_grad_(False)
        self.proj = nn.Sequential(
            nn.Linear(self.llm_hidden, args.hidden_size),
            nn.Tanh(),
        )
        self.norm = nn.LayerNorm(args.hidden_size)

    def _forward_impl(self, prompt_text_list, device):
        inputs = self.tokenizer(
            prompt_text_list,
            return_tensors='pt',
            truncation=True,
            padding=True,
            max_length=self.args.llm_max_length,
        )
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)

        token_emb = self.llm.get_input_embeddings()(input_ids)
        bsz = token_emb.size(0)
        prompt = torch.tanh(self.soft_prompt).unsqueeze(0).expand(bsz, -1, -1)
        input_emb = torch.cat([prompt.to(token_emb.dtype), token_emb], dim=1)

        prompt_mask = torch.ones((bsz, self.prompt_len), dtype=attention_mask.dtype, device=device)
        full_mask = torch.cat([prompt_mask, attention_mask], dim=1)

        if self.train_soft_prompt:
            out = self.llm(inputs_embeds=input_emb, attention_mask=full_mask)
        else:
            with torch.no_grad():
                out = self.llm(inputs_embeds=input_emb, attention_mask=full_mask)
        hs = out.last_hidden_state[:, self.prompt_len:, :]
        hs = torch.nan_to_num(hs, nan=0.0, posinf=1e4, neginf=-1e4)
        denom = attention_mask.sum(dim=1, keepdim=True).clamp(min=1)
        hs_f = hs.float()
        pooled_mean = (hs_f * attention_mask.unsqueeze(-1)).sum(dim=1) / denom

        last_pos = (attention_mask.sum(dim=1) - 1).clamp(min=0)
        batch_idx = torch.arange(hs.size(0), device=hs.device)
        pooled_last = hs_f[batch_idx, last_pos, :]

        pooled = 0.7 * pooled_mean + 0.3 * pooled_last
        pooled = torch.nan_to_num(pooled, nan=0.0, posinf=1e4, neginf=-1e4)
        out = self.proj(pooled)
        out = self.norm(out)
        return torch.nan_to_num(out, nan=0.0, posinf=1e3, neginf=-1e3)

    @staticmethod
    def _resolve_model_name(args):
        candidate = [args.llm_model_name]
        fallback_names = [x.strip() for x in str(getattr(args, 'llm_fallback_names', '')).split(',') if x.strip()]
        candidate.extend(fallback_names)
        last_err = None
        for name in candidate:
            try:
                AutoTokenizer.from_pretrained(name, cache_dir=args.llm_cache_dir, trust_remote_code=True)
                return name
            except Exception as e:
                last_err = e
        raise RuntimeError(f"No available LLM model from candidates: {candidate}. Last error: {last_err}")

    def forward(self, prompt_text_list, device):
        micro_batch = int(getattr(self.args, 'llm_micro_batch', 0))
        if micro_batch <= 0 or micro_batch >= len(prompt_text_list):
            return self._forward_impl(prompt_text_list, device)

        outs = []
        for i in range(0, len(prompt_text_list), micro_batch):
            chunk = prompt_text_list[i:i + micro_batch]
            outs.append(self._forward_impl(chunk, device))
        return torch.cat(outs, dim=0)


class CheckinTransformerEncoder(nn.Module):
    """
    Transformer fallback sequence encoder for check-ins.
    """

    def __init__(self, poi_num, hidden_size, nhead=4, nlayers=2, dropout=0.1, max_len=512):
        super().__init__()
        self.poi_emb = nn.Embedding(poi_num, hidden_size, padding_idx=0)
        self.hour_emb = nn.Embedding(25, hidden_size, padding_idx=0)
        self.coord_proj = nn.Linear(2, hidden_size)
        self.pos_emb = nn.Embedding(max_len, hidden_size)

        layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=nhead,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            batch_first=True,
            activation='gelu',
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=nlayers)

    def forward(self, poi_seq, hour_seq, coord_seq, valid_mask):
        seq_len = poi_seq.size(1)
        pos_idx = torch.arange(seq_len, device=poi_seq.device).unsqueeze(0).expand_as(poi_seq)
        pos_idx = pos_idx.clamp(max=self.pos_emb.num_embeddings - 1)

        coord_seq = torch.nan_to_num(coord_seq, nan=0.0, posinf=1e3, neginf=-1e3)
        x = self.poi_emb(poi_seq)
        x = x + self.hour_emb(hour_seq.clamp(min=0, max=24))
        x = x + self.coord_proj(coord_seq)
        x = x + self.pos_emb(pos_idx)
        x = torch.nan_to_num(x, nan=0.0, posinf=1e3, neginf=-1e3)
        x = self.encoder(x, src_key_padding_mask=~valid_mask)
        return torch.nan_to_num(x, nan=0.0, posinf=1e3, neginf=-1e3)


class MambaResidualBlock(nn.Module):
    def __init__(self, hidden_size, d_state=16, d_conv=4, expand=2, dropout=0.1):
        super().__init__()
        if Mamba is None:
            raise ImportError("mamba_ssm is required for Mamba blocks. Please `pip install mamba-ssm`.")
        self.norm = nn.LayerNorm(hidden_size)
        self.mamba = Mamba(d_model=hidden_size, d_state=d_state, d_conv=d_conv, expand=expand)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = self.mamba(self.norm(x))
        out = self.dropout(out)
        return x + out


class CheckinBiMambaEncoder(nn.Module):
    """
    Bidirectional Mamba encoder for check-ins using POI + time + geo embeddings.
    """

    def __init__(self, poi_num, hidden_size, nlayers=2, dropout=0.1, max_len=512, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.poi_emb = nn.Embedding(poi_num, hidden_size, padding_idx=0)
        self.hour_emb = nn.Embedding(25, hidden_size, padding_idx=0)
        self.coord_proj = nn.Linear(2, hidden_size)
        self.pos_emb = nn.Embedding(max_len, hidden_size)
        self.fwd_blocks = nn.ModuleList(
            [MambaResidualBlock(hidden_size, d_state, d_conv, expand, dropout) for _ in range(max(1, nlayers))]
        )
        self.bwd_blocks = nn.ModuleList(
            [MambaResidualBlock(hidden_size, d_state, d_conv, expand, dropout) for _ in range(max(1, nlayers))]
        )
        self.out_norm = nn.LayerNorm(hidden_size)

    @staticmethod
    def _reverse_valid(x, valid_mask):
        seq_len = x.size(1)
        idx = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(x.size(0), -1)
        lengths = valid_mask.long().sum(dim=1, keepdim=True)
        rev_idx = torch.where(idx < lengths, lengths - 1 - idx, idx)
        rev_idx = rev_idx.clamp(min=0, max=seq_len - 1)
        return x.gather(1, rev_idx.unsqueeze(-1).expand_as(x))

    def forward(self, poi_seq, hour_seq, coord_seq, valid_mask):
        seq_len = poi_seq.size(1)
        pos_idx = torch.arange(seq_len, device=poi_seq.device).unsqueeze(0).expand_as(poi_seq)
        pos_idx = pos_idx.clamp(max=self.pos_emb.num_embeddings - 1)

        coord_seq = torch.nan_to_num(coord_seq, nan=0.0, posinf=1e3, neginf=-1e3)
        x = self.poi_emb(poi_seq)
        x = x + self.hour_emb(hour_seq.clamp(min=0, max=24))
        x = x + self.coord_proj(coord_seq)
        x = x + self.pos_emb(pos_idx)
        x = torch.nan_to_num(x, nan=0.0, posinf=1e3, neginf=-1e3)
        x = x * valid_mask.unsqueeze(-1).to(x.dtype)

        x_f = x
        for blk in self.fwd_blocks:
            x_f = blk(x_f)
            x_f = x_f * valid_mask.unsqueeze(-1).to(x_f.dtype)

        x_b = self._reverse_valid(x, valid_mask)
        for blk in self.bwd_blocks:
            x_b = blk(x_b)
            x_b = x_b * valid_mask.unsqueeze(-1).to(x_b.dtype)
        x_b = self._reverse_valid(x_b, valid_mask)

        out = 0.5 * (x_f + x_b)
        out = self.out_norm(out)
        out = out * valid_mask.unsqueeze(-1).to(out.dtype)
        return torch.nan_to_num(out, nan=0.0, posinf=1e3, neginf=-1e3)


class CausalMambaDecoder(nn.Module):
    """
    Causal Mamba decoder block stack for route generation.
    """

    def __init__(self, input_size, hidden_size, nlayers=2, dropout=0.1, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.in_proj = nn.Linear(input_size, hidden_size)
        self.blocks = nn.ModuleList(
            [MambaResidualBlock(hidden_size, d_state, d_conv, expand, dropout) for _ in range(max(1, nlayers))]
        )
        self.out_norm = nn.LayerNorm(hidden_size)

    def forward(self, x):
        h = self.in_proj(x)
        h = torch.nan_to_num(h, nan=0.0, posinf=1e3, neginf=-1e3)
        for blk in self.blocks:
            h = blk(h)
        h = self.out_norm(h)
        return torch.nan_to_num(h, nan=0.0, posinf=1e3, neginf=-1e3)


class CrossCityLLMCPR(nn.Module):
    def __init__(self, args, poi_num, tag_num, region_num, popularity_bias=None, poi_coord_tensor=None, city_sample_count=None):
        super().__init__()
        self.args = args
        self.hidden_size = args.hidden_size
        self.pref_num = args.pref_factor_k
        self.poi_num = poi_num
        self.region_num = region_num

        self.semantic_backend = args.semantic_backend
        self.semantic_encoder_fallback = FrozenSemanticEncoder(
            poi_num=poi_num,
            tag_num=tag_num,
            hidden_size=args.hidden_size,
            prompt_len=args.soft_prompt_len,
            nhead=args.nhead,
            nlayers=args.semantic_layers,
            dropout=args.dropout,
        )
        self.semantic_encoder_qwen = None
        if self.semantic_backend == 'qwen':
            try:
                self.semantic_encoder_qwen = QwenSoftPromptEncoder(args)
                print(
                    f"[INFO] Qwen semantic encoder loaded: {self.semantic_encoder_qwen.model_name} | "
                    f"train_soft_prompt={int(self.semantic_encoder_qwen.train_soft_prompt)}"
                )
            except Exception as e:
                if getattr(args, 'qwen_strict', False):
                    raise RuntimeError(f"Qwen semantic encoder init failed under strict mode: {e}")
                print(f"[WARN] Qwen semantic encoder init failed: {e}. Fallback to frozen transformer encoder.")
                self.semantic_backend = 'fallback'

        self.semantic_fusion_gate = nn.Sequential(
            nn.Linear(args.hidden_size, args.hidden_size // 2),
            nn.ReLU(),
            nn.Linear(args.hidden_size // 2, 1),
            nn.Sigmoid(),
        )

        use_mamba = bool(getattr(args, 'use_mamba_backbone', 1))
        if use_mamba and Mamba is None and bool(getattr(args, 'mamba_strict', False)):
            raise RuntimeError("Mamba backbone requested but mamba_ssm is unavailable. Please install mamba-ssm.")

        if use_mamba and Mamba is not None:
            self.home_seq_encoder = CheckinBiMambaEncoder(
                poi_num=poi_num,
                hidden_size=args.hidden_size,
                nlayers=max(1, int(getattr(args, 'seq_num_layers', args.semantic_layers))),
                dropout=args.dropout,
                d_state=int(getattr(args, 'mamba_d_state', 16)),
                d_conv=int(getattr(args, 'mamba_d_conv', 4)),
                expand=int(getattr(args, 'mamba_expand', 2)),
            )
        else:
            if use_mamba and Mamba is None:
                print("[WARN] mamba_ssm unavailable, fallback to Transformer encoder/GRU generator.")
            self.home_seq_encoder = CheckinTransformerEncoder(
                poi_num=poi_num,
                hidden_size=args.hidden_size,
                nhead=args.nhead,
                nlayers=max(1, args.semantic_layers),
                dropout=args.dropout,
            )

        self.disentangle_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(args.hidden_size * 2, args.hidden_size),
                    nn.GELU(),
                    nn.Linear(args.hidden_size, args.hidden_size),
                )
                for _ in range(self.pref_num)
            ]
        )

        self.profile_proj = nn.Linear(args.profile_dim, args.hidden_size)
        self.decode_constraint_mode = str(getattr(args, 'decode_constraint_mode', 'hard')).lower()
        self.soft_query_proj = nn.Linear(args.query_dim, args.hidden_size)
        soft_dist_dim = int(getattr(args, 'soft_constraint_dist_emb_dim', 32))
        soft_dist_dim = max(4, soft_dist_dim)
        self.soft_dist_proj = nn.Sequential(
            nn.Linear(2, soft_dist_dim),
            nn.GELU(),
            nn.Linear(soft_dist_dim, args.hidden_size),
        )
        self.soft_cond_fuse = nn.Sequential(
            nn.Linear(args.hidden_size * 2, args.hidden_size),
            nn.GELU(),
            nn.Linear(args.hidden_size, args.hidden_size),
        )
        self.soft_poi_key = nn.Linear(args.hidden_size, args.hidden_size, bias=False)
        self.transfer_gate = nn.Sequential(
            nn.Linear(args.hidden_size * 3, args.hidden_size),
            nn.GELU(),
            nn.Linear(args.hidden_size, self.pref_num),
        )

        if use_mamba and Mamba is not None:
            self.tour_seq_encoder = CheckinBiMambaEncoder(
                poi_num=poi_num,
                hidden_size=args.hidden_size,
                nlayers=max(1, int(getattr(args, 'seq_num_layers', args.semantic_layers))),
                dropout=args.dropout,
                d_state=int(getattr(args, 'mamba_d_state', 16)),
                d_conv=int(getattr(args, 'mamba_d_conv', 4)),
                expand=int(getattr(args, 'mamba_expand', 2)),
            )
        else:
            self.tour_seq_encoder = CheckinTransformerEncoder(
                poi_num=poi_num,
                hidden_size=args.hidden_size,
                nhead=args.nhead,
                nlayers=max(1, args.semantic_layers),
                dropout=args.dropout,
            )

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

        self.user_proj = nn.Linear(args.hidden_size, args.hidden_size)
        self.route_proj = nn.Linear(args.hidden_size, args.hidden_size)
        self.city_proj = nn.Linear(args.hidden_size, args.hidden_size)
        self.city_residual = nn.Embedding(region_num, args.hidden_size)
        self.register_buffer("city_base_memory", torch.zeros(region_num, args.hidden_size))

        if city_sample_count is None:
            city_sample_count = torch.ones(region_num, dtype=torch.float)
        self.register_buffer("city_sample_count", city_sample_count.float().clamp(min=0.0))

        self.eta_gate = nn.Sequential(
            nn.Linear(args.hidden_size * 2, args.hidden_size),
            nn.ReLU(),
            nn.Linear(args.hidden_size, 1),
            nn.Sigmoid(),
        )

        self.tour_poi_emb = nn.Embedding(poi_num, args.hidden_size, padding_idx=0)
        if use_mamba and Mamba is not None:
            self.decoder = CausalMambaDecoder(
                input_size=args.hidden_size * 4,  # [poi_emb, dist_enc, z_final, query_h] 各 H 维
                hidden_size=args.hidden_size,
                nlayers=max(1, int(getattr(args, 'seq_num_layers', args.semantic_layers))),
                dropout=args.dropout,
                d_state=int(getattr(args, 'mamba_d_state', 16)),
                d_conv=int(getattr(args, 'mamba_d_conv', 4)),
                expand=int(getattr(args, 'mamba_expand', 2)),
            )
        else:
            self.decoder = nn.GRU(args.hidden_size * 4, args.hidden_size, batch_first=True)
        self.decoder_out = nn.Linear(args.hidden_size, poi_num)

        self.transition_prev = nn.Linear(args.hidden_size, args.hidden_size, bias=False)
        self.transition_next = nn.Linear(args.hidden_size, args.hidden_size, bias=False)
        self.transition_ctx = nn.Linear(args.hidden_size, args.hidden_size, bias=False)
        self.transition_bias = nn.Parameter(torch.zeros(poi_num))

        if popularity_bias is None:
            popularity_bias = torch.zeros(poi_num)
        self.register_buffer("popularity_bias", popularity_bias)

        if poi_coord_tensor is None:
            poi_coord_tensor = torch.zeros((poi_num, 2), dtype=torch.float)
        self.register_buffer("poi_coords", poi_coord_tensor.float())

    def _update_city_base_memory(self, r_tour, dst_rg):
        momentum = float(getattr(self.args, 'city_memory_momentum', 0.95))
        momentum = max(0.0, min(0.9999, momentum))
        with torch.no_grad():
            r_tour = torch.nan_to_num(r_tour, nan=0.0, posinf=1e3, neginf=-1e3)
            unique_city = torch.unique(dst_rg)
            for city_id in unique_city.tolist():
                city_mask = (dst_rg == city_id)
                if city_mask.any():
                    city_mean = torch.nan_to_num(r_tour[city_mask].mean(dim=0), nan=0.0, posinf=1e3, neginf=-1e3)
                    if not torch.isfinite(city_mean).all():
                        continue
                    self.city_base_memory[city_id] = momentum * self.city_base_memory[city_id] + (1.0 - momentum) * city_mean

    def _compose_city_vectors(self):
        base_plus_res = torch.nan_to_num(self.city_base_memory + self.city_residual.weight, nan=0.0, posinf=1e3, neginf=-1e3)
        count = torch.nan_to_num(self.city_sample_count, nan=0.0, posinf=1e6, neginf=0.0)
        k = float(getattr(self.args, 'city_memory_prior_k', 20.0))
        reliability = (count / (count + k)).unsqueeze(-1)
        reliability = torch.nan_to_num(reliability, nan=0.0, posinf=1.0, neginf=0.0)

        weight_sum = count.sum().clamp(min=1.0)
        global_city = (base_plus_res * count.unsqueeze(-1)).sum(dim=0, keepdim=True) / weight_sum
        global_city = torch.nan_to_num(global_city, nan=0.0, posinf=1e3, neginf=-1e3)
        city_vec = reliability * base_plus_res + (1.0 - reliability) * global_city
        return torch.nan_to_num(city_vec, nan=0.0, posinf=1e3, neginf=-1e3)

    @staticmethod
    def _masked_last_from_encoded(encoded_seq, valid_mask):
        lengths = valid_mask.long().sum(dim=1).clamp(min=1)
        last_idx = (lengths - 1).unsqueeze(1).unsqueeze(2).expand(-1, 1, encoded_seq.size(-1))
        return encoded_seq.gather(1, last_idx).squeeze(1)

    @staticmethod
    def _calculate_ellipse_mask(start_idx, end_idx, all_coords, beta):
        s_coords = all_coords[start_idx]
        e_coords = all_coords[end_idx]
        dist_od = torch.norm(s_coords - e_coords, dim=1)

        s_coords = s_coords.unsqueeze(1)
        e_coords = e_coords.unsqueeze(1)
        all_coords_exp = all_coords.unsqueeze(0)

        dist_s_p = torch.norm(all_coords_exp - s_coords, dim=2)
        dist_e_p = torch.norm(all_coords_exp - e_coords, dim=2)

        mask = (dist_s_p + dist_e_p) <= (dist_od.unsqueeze(1) * beta)
        mask.scatter_(1, start_idx.unsqueeze(1), True)
        mask.scatter_(1, end_idx.unsqueeze(1), True)
        return mask

    def _encode_home(self, batch):
        home_seq = self.home_seq_encoder(batch['ori_ck'], batch['o_hour'], batch['ori_l'], batch['ori_pad'])
        h_seq = self._masked_last_from_encoded(home_seq, batch['ori_pad'])
        h_seq = torch.nan_to_num(h_seq, nan=0.0, posinf=1e3, neginf=-1e3)
        s_struct = self.semantic_encoder_fallback(
            batch['ori_ck'], batch['ori_tag'], batch['o_hour'], batch['ori_l'], batch['ori_pad']
        )
        s_struct = torch.nan_to_num(s_struct, nan=0.0, posinf=1e3, neginf=-1e3)
        if self.semantic_backend == 'qwen' and self.semantic_encoder_qwen is not None:
            s_llm = self.semantic_encoder_qwen(batch['home_prompt_text'], device=batch['ori_ck'].device)
            s_llm = torch.nan_to_num(s_llm, nan=0.0, posinf=1e3, neginf=-1e3)
            finite_row = torch.isfinite(s_llm).all(dim=1, keepdim=True)
            s_llm = torch.where(finite_row, s_llm, s_struct)
            mix = self.semantic_fusion_gate(h_seq)
            mix = torch.nan_to_num(mix, nan=0.0, posinf=1.0, neginf=0.0)
            mix = torch.where(finite_row, mix, torch.zeros_like(mix))
            s_u = mix * s_llm + (1.0 - mix) * s_struct
        else:
            mix = torch.zeros((h_seq.size(0), 1), device=h_seq.device)
            s_u = s_struct
        s_u = torch.nan_to_num(s_u, nan=0.0, posinf=1e3, neginf=-1e3)
        h_u = torch.cat([h_seq, s_u], dim=-1)
        z_list = [head(h_u) for head in self.disentangle_heads]
        z_stack = torch.stack(z_list, dim=1)
        z_stack = torch.nan_to_num(z_stack, nan=0.0, posinf=1e3, neginf=-1e3)
        return h_seq, s_u, z_stack, mix

    def _adaptive_transfer(self, z_stack, user_profile, query_h):
        z_stack = torch.nan_to_num(z_stack, nan=0.0, posinf=1e3, neginf=-1e3)
        profile_h = torch.nan_to_num(self.profile_proj(user_profile), nan=0.0, posinf=1e3, neginf=-1e3)
        query_h = torch.nan_to_num(query_h, nan=0.0, posinf=1e3, neginf=-1e3)
        z_mean = torch.nan_to_num(z_stack.mean(dim=1), nan=0.0, posinf=1e3, neginf=-1e3)
        gate_in = torch.cat([z_mean, profile_h, query_h], dim=-1)
        gate_logits = torch.nan_to_num(self.transfer_gate(gate_in), nan=0.0, posinf=1e3, neginf=-1e3)
        alpha = F.softmax(gate_logits, dim=-1)
        alpha = torch.nan_to_num(alpha, nan=1.0 / float(self.pref_num), posinf=1.0, neginf=0.0)
        z_trans = torch.sum(alpha.unsqueeze(-1) * z_stack, dim=1)
        z_trans = torch.nan_to_num(z_trans, nan=0.0, posinf=1e3, neginf=-1e3)
        return z_trans, alpha, profile_h, query_h

    def _encode_tour(self, batch):
        tour_seq = self.tour_seq_encoder(batch['dst_ck'], batch['d_hour'], batch['dst_l'], batch['dst_pad'])
        tour_seq = torch.nan_to_num(tour_seq, nan=0.0, posinf=1e3, neginf=-1e3)
        tour_last = self._masked_last_from_encoded(tour_seq, batch['dst_pad'])
        return torch.nan_to_num(tour_last, nan=0.0, posinf=1e3, neginf=-1e3)

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

    def _decouple_semantic_loss(self, z_stack, s_u):
        decouple = z_stack.new_zeros(())
        for i in range(self.pref_num):
            for j in range(i + 1, self.pref_num):
                decouple = decouple + torch.mean(torch.abs(F.cosine_similarity(z_stack[:, i, :], z_stack[:, j, :], dim=-1)))

        sem = z_stack.new_zeros(())
        for i in range(self.pref_num):
            sem = sem + torch.mean(1.0 - F.cosine_similarity(z_stack[:, i, :], s_u, dim=-1))
        return decouple, sem

    def _alignment_loss(self, z_trans, r_tour, dst_rg):
        tau = max(float(self.args.temperature), 1e-6)
        z_trans = torch.nan_to_num(z_trans, nan=0.0, posinf=1e3, neginf=-1e3)
        r_tour = torch.nan_to_num(r_tour, nan=0.0, posinf=1e3, neginf=-1e3)
        z_u = F.normalize(self.user_proj(z_trans), dim=-1)
        r_u = F.normalize(self.route_proj(r_tour), dim=-1)
        z_u = torch.nan_to_num(z_u, nan=0.0, posinf=1e3, neginf=-1e3)
        r_u = torch.nan_to_num(r_u, nan=0.0, posinf=1e3, neginf=-1e3)

        logits_user = torch.matmul(z_u, r_u.t()) / tau
        logits_user = torch.nan_to_num(logits_user, nan=0.0, posinf=1e4, neginf=-1e4)
        labels = torch.arange(logits_user.size(0), device=logits_user.device)
        loss_user = F.cross_entropy(logits_user, labels)

        city_all = F.normalize(self.city_proj(self._compose_city_vectors()), dim=-1)
        city_all = torch.nan_to_num(city_all, nan=0.0, posinf=1e3, neginf=-1e3)
        logits_city = torch.matmul(z_u, city_all.t()) / tau
        logits_city = torch.nan_to_num(logits_city, nan=0.0, posinf=1e4, neginf=-1e4)
        loss_city = F.cross_entropy(logits_city, dst_rg)
        return loss_user, loss_city

    @staticmethod
    def _decoder_forward(decoder, dec_feat):
        out = decoder(dec_feat)
        if isinstance(out, tuple):
            return out[0]
        return out

    @staticmethod
    def _build_future_negatives(target_row, start_t, pos_id, max_future):
        future = target_row[start_t + 1:]
        future = future[future != 0]
        if future.numel() == 0:
            return []
        picked = []
        seen = set()
        for v in future.tolist():
            if v == pos_id or v in seen:
                continue
            seen.add(v)
            picked.append(v)
            if len(picked) >= max_future:
                break
        return picked

    def _transition_logits(self, prev_ids, z_final):
        prev_emb = self.tour_poi_emb(prev_ids)
        all_next = self.transition_next(self.tour_poi_emb.weight)
        scale = 1.0 / math.sqrt(max(self.hidden_size, 1))
        prev_emb = torch.nan_to_num(prev_emb, nan=0.0, posinf=1e3, neginf=-1e3)
        all_next = torch.nan_to_num(all_next, nan=0.0, posinf=1e3, neginf=-1e3)
        z_final = torch.nan_to_num(z_final, nan=0.0, posinf=1e3, neginf=-1e3)

        if prev_emb.dim() == 2:
            query = self.transition_prev(prev_emb) + self.transition_ctx(z_final)
            logits = torch.matmul(query, all_next.t()) * scale
            logits = logits + self.transition_bias.unsqueeze(0)
            return torch.nan_to_num(logits, nan=0.0, posinf=1e4, neginf=-1e4)

        query = self.transition_prev(prev_emb) + self.transition_ctx(z_final).unsqueeze(1)
        logits = torch.einsum('bth,ph->btp', query, all_next) * scale
        logits = logits + self.transition_bias.view(1, 1, -1)
        return torch.nan_to_num(logits, nan=0.0, posinf=1e4, neginf=-1e4)

    def _build_distance_cond(self, prev_ids, start_poi, end_poi):
        prev_coord = self.poi_coords[prev_ids]
        start_coord = self.poi_coords[start_poi]
        end_coord = self.poi_coords[end_poi]

        if prev_ids.dim() == 1:
            dist_start = torch.norm(prev_coord - start_coord, dim=-1, keepdim=True)
            dist_end = torch.norm(prev_coord - end_coord, dim=-1, keepdim=True)
            dist_feat = torch.cat([dist_start, dist_end], dim=-1)
            dist_feat = torch.nan_to_num(dist_feat, nan=0.0, posinf=1e3, neginf=0.0)
            return torch.nan_to_num(self.soft_dist_proj(dist_feat), nan=0.0, posinf=1e3, neginf=-1e3)

        start_coord = start_coord.unsqueeze(1)
        end_coord = end_coord.unsqueeze(1)
        dist_start = torch.norm(prev_coord - start_coord, dim=-1, keepdim=True)
        dist_end = torch.norm(prev_coord - end_coord, dim=-1, keepdim=True)
        dist_feat = torch.cat([dist_start, dist_end], dim=-1)
        dist_feat = torch.nan_to_num(dist_feat, nan=0.0, posinf=1e3, neginf=0.0)
        return torch.nan_to_num(self.soft_dist_proj(dist_feat), nan=0.0, posinf=1e3, neginf=-1e3)

    def _soft_constraint_bias(self, prev_ids, query_vec, start_poi, end_poi):
        if self.decode_constraint_mode != 'soft':
            return None

        if query_vec is None or start_poi is None or end_poi is None:
            return None

        query_cond = torch.nan_to_num(self.soft_query_proj(query_vec), nan=0.0, posinf=1e3, neginf=-1e3)
        dist_cond = self._build_distance_cond(prev_ids, start_poi, end_poi)
        poi_key = torch.nan_to_num(self.soft_poi_key(self.tour_poi_emb.weight), nan=0.0, posinf=1e3, neginf=-1e3)
        scale = 1.0 / math.sqrt(max(self.hidden_size, 1))
        gain = float(getattr(self.args, 'soft_constraint_scale', 0.0))

        if prev_ids.dim() == 1:
            cond = torch.cat([query_cond, dist_cond], dim=-1)
            cond = torch.nan_to_num(self.soft_cond_fuse(cond), nan=0.0, posinf=1e3, neginf=-1e3)
            bias = torch.matmul(cond, poi_key.t()) * scale
            return gain * torch.nan_to_num(bias, nan=0.0, posinf=1e4, neginf=-1e4)

        query_expand = query_cond.unsqueeze(1).expand(-1, prev_ids.size(1), -1)
        cond = torch.cat([query_expand, dist_cond], dim=-1)
        cond = torch.nan_to_num(self.soft_cond_fuse(cond), nan=0.0, posinf=1e3, neginf=-1e3)
        bias = torch.einsum('bth,ph->btp', cond, poi_key) * scale
        return gain * torch.nan_to_num(bias, nan=0.0, posinf=1e4, neginf=-1e4)

    def _generator_loss(self, z_final, dst_ck, query_h=None, start_poi=None, end_poi=None):
        dec_in = dst_ck[:, :-1]
        dec_target = dst_ck[:, 1:]
        dec_emb = self.tour_poi_emb(dec_in)                                        # [B, T, H]

        no_spatial_ctx = bool(getattr(self.args, 'ablate_generator_no_spatial_context', 0))
        if no_spatial_ctx:
            # Ablation: remove spatial generation context by zeroing
            # previous-POI embedding and distance encoding branches.
            dec_emb = torch.zeros_like(dec_emb)
            dist_enc = torch.zeros_like(dec_emb)
        else:
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
        dec_out  = torch.nan_to_num(dec_out, nan=0.0, posinf=1e3, neginf=-1e3)

        chunk_len = int(getattr(self.args, 'gen_loss_chunk_len', 16))
        if chunk_len <= 0:
            chunk_len = dec_out.size(1)

        loss_sum = dec_out.new_zeros(())
        token_cnt = dec_out.new_zeros(())
        pair_loss_sum = dec_out.new_zeros(())
        pair_cnt = dec_out.new_zeros(())
        trans_loss_sum = dec_out.new_zeros(())
        trans_cnt = dec_out.new_zeros(())
        pop_bias = self.args.pop_bias_scale * self.popularity_bias.unsqueeze(0).unsqueeze(0)
        pair_max_future = int(getattr(self.args, 'pair_max_future', 4))
        use_pairwise = bool(getattr(self.args, 'enable_pairwise_loss', 1))
        transition_scale = float(getattr(self.args, 'transition_logit_scale', 0.0))

        for st in range(0, dec_out.size(1), chunk_len):
            ed = min(st + chunk_len, dec_out.size(1))
            logits = self.decoder_out(dec_out[:, st:ed, :])
            logits = logits + pop_bias

            trans_logits = self._transition_logits(dec_in[:, st:ed], z_final)
            if transition_scale != 0.0:
                logits = logits + transition_scale * trans_logits
            logits = torch.nan_to_num(logits, nan=0.0, posinf=1e4, neginf=-1e4)
            target = dec_target[:, st:ed]

            loss_chunk = F.cross_entropy(
                logits.reshape(-1, self.poi_num),
                target.reshape(-1),
                ignore_index=0,
                reduction='sum',
            )
            valid_tokens = (target != 0).sum().to(dec_out.dtype)
            loss_sum = loss_sum + loss_chunk
            token_cnt = token_cnt + valid_tokens

            trans_loss_chunk = F.cross_entropy(
                trans_logits.reshape(-1, self.poi_num),
                target.reshape(-1),
                ignore_index=0,
                reduction='sum',
            )
            trans_loss_sum = trans_loss_sum + trans_loss_chunk
            trans_cnt = trans_cnt + valid_tokens

            if use_pairwise and pair_max_future > 0:
                for b in range(logits.size(0)):
                    target_row = dec_target[b]
                    for lt in range(ed - st):
                        t = st + lt
                        pos_id = int(target_row[t].item())
                        if pos_id == 0:
                            continue
                        neg_ids = self._build_future_negatives(target_row, t, pos_id, pair_max_future)
                        if len(neg_ids) == 0:
                            continue
                        pos_logit = logits[b, lt, pos_id]
                        neg_idx = torch.tensor(neg_ids, dtype=torch.long, device=logits.device)
                        neg_logits = logits[b, lt].index_select(0, neg_idx)
                        pair_loss_sum = pair_loss_sum + F.softplus(neg_logits - pos_logit).mean()
                        pair_cnt = pair_cnt + 1.0

        token_cnt = token_cnt.clamp(min=1.0)
        pair_cnt = pair_cnt.clamp(min=1.0)
        trans_cnt = trans_cnt.clamp(min=1.0)
        return loss_sum / token_cnt, pair_loss_sum / pair_cnt, trans_loss_sum / trans_cnt

    def _blend_user_city(self, z_trans, city_vec, profile_h, query_h):
        eta_fixed = float(getattr(self.args, 'eta_fixed', -1.0))
        if 0.0 <= eta_fixed <= 1.0:
            eta = torch.full((z_trans.size(0), 1), eta_fixed, dtype=z_trans.dtype, device=z_trans.device)
        else:
            eta = self.eta_gate(torch.cat([profile_h, query_h], dim=-1))
        z_final = eta * z_trans + (1.0 - eta) * city_vec
        return z_final, eta

    def forward(self, batch):
        h_seq, s_u, z_stack, mix = self._encode_home(batch)
        query_h = self._encode_query(batch)
        z_trans, alpha, profile_h, query_h = self._adaptive_transfer(
            z_stack, batch['user_profile'], query_h
        )
        r_tour = self._encode_tour(batch)

        if self.training:
            self._update_city_base_memory(r_tour.detach(), batch['dst_rg'])

        loss_decouple, loss_semantic = self._decouple_semantic_loss(z_stack, s_u)
        loss_user, loss_city = self._alignment_loss(z_trans, r_tour, batch['dst_rg'])
        loss_align = loss_user + self.args.gamma_city * loss_city

        city_vec = self._compose_city_vectors()[batch['dst_rg']]
        z_final, eta = self._blend_user_city(z_trans, city_vec, profile_h, query_h)

        loss_gen, loss_pair, loss_transition = self._generator_loss(
            z_final,
            batch['dst_ck'],
            query_h=query_h,
            start_poi=batch.get('query_start_poi', None),
            end_poi=batch.get('query_end_poi', None),
        )
        loss_align = torch.nan_to_num(loss_align, nan=0.0, posinf=1e4, neginf=-1e4)
        loss_decouple = torch.nan_to_num(loss_decouple, nan=0.0, posinf=1e4, neginf=-1e4)
        loss_semantic = torch.nan_to_num(loss_semantic, nan=0.0, posinf=1e4, neginf=-1e4)
        loss_gen = torch.nan_to_num(loss_gen, nan=0.0, posinf=1e4, neginf=-1e4)
        loss_pair = torch.nan_to_num(loss_pair, nan=0.0, posinf=1e4, neginf=-1e4)
        loss_transition = torch.nan_to_num(loss_transition, nan=0.0, posinf=1e4, neginf=-1e4)
        total_loss = (
            loss_align
            + self.args.lambda_decouple * loss_decouple
            + self.args.lambda_semantic * loss_semantic
            + self.args.lambda_gen * loss_gen
            + float(getattr(self.args, 'lambda_pair', 0.0)) * loss_pair
            + float(getattr(self.args, 'lambda_transition', 0.0)) * loss_transition
        )

        return {
            "loss": total_loss,
            "align": loss_align.detach(),
            "decouple": loss_decouple.detach(),
            "semantic": loss_semantic.detach(),
            "gen": loss_gen.detach(),
            "pair": loss_pair.detach(),
            "transition": loss_transition.detach(),
            "eta": eta.mean().detach(),
            "sem_mix": torch.nan_to_num(mix, nan=0.0, posinf=1.0, neginf=0.0).mean().detach(),
            "alpha_mean": alpha.mean(dim=0).detach(),
        }

    def _next_step_logits(self, curr_seq, z_final_vec, query_h=None, start_poi=None, end_poi=None):
        dec_emb = self.tour_poi_emb(curr_seq)                                          # [B, T, H]

        no_spatial_ctx = bool(getattr(self.args, 'ablate_generator_no_spatial_context', 0))
        if no_spatial_ctx:
            # Keep decoder input shape fixed while ablating spatial context.
            dec_emb = torch.zeros_like(dec_emb)
            dist_enc = torch.zeros_like(dec_emb)
        else:
            # 空间距离编码：序列中每个位置的POI距起终点的距离
            if start_poi is not None and end_poi is not None:
                dist_enc = self._build_distance_cond(curr_seq, start_poi, end_poi)         # [B, T, H]
            else:
                dist_enc = torch.zeros_like(dec_emb)

        context   = z_final_vec.unsqueeze(1).expand(-1, dec_emb.size(1), -1)          # [B, T, H]
        query_ctx = query_h.unsqueeze(1).expand(-1, dec_emb.size(1), -1) if query_h is not None \
                    else torch.zeros_like(dec_emb)                                      # [B, T, H]

        dec_feat = torch.cat([dec_emb, dist_enc, context, query_ctx], dim=-1)         # [B, T, 4H]
        dec_out  = self._decoder_forward(self.decoder, dec_feat)
        logits   = self.decoder_out(dec_out[:, -1, :])
        logits   = logits + self.args.pop_bias_scale * self.popularity_bias.unsqueeze(0)
        # 空间上下文已作为decoder输入，不再对logits做re-scoring

        transition_scale = float(getattr(self.args, 'transition_logit_scale', 0.0))
        if transition_scale != 0.0:
            prev_ids = curr_seq[:, -1]
            logits = logits + transition_scale * self._transition_logits(prev_ids, z_final_vec)
        return torch.nan_to_num(logits, nan=0.0, posinf=1e4, neginf=-1e4)

    def _predict_greedy(self, z_final, query_h, start_poi, end_poi, lengths, max_len, ellipse_mask):
        use_constraints = (
            self.decode_constraint_mode == 'hard'
            and bool(getattr(self.args, 'enforce_start_end_constraints', 1))
        )
        init_token = start_poi if use_constraints else torch.zeros_like(start_poi)
        curr = init_token.unsqueeze(1)
        batch_size = curr.size(0)
        use_no_repeat_mask = bool(getattr(self.args, 'use_no_repeat_mask', 1))
        visited = torch.zeros(batch_size, self.poi_num, dtype=torch.bool, device=curr.device)
        if use_constraints:
            visited.scatter_(1, start_poi.unsqueeze(1), True)

        for step in range(max_len - 1):
            logits = self._next_step_logits(
                curr,
                z_final,
                query_h=query_h,
                start_poi=start_poi,
                end_poi=end_poi,
            )
            logits[:, 0] = -1e9
            if ellipse_mask is not None:
                logits = logits.masked_fill(~ellipse_mask, -1e9)
            if use_no_repeat_mask:
                logits = logits.masked_fill(visited, -1e9)
            next_token = torch.argmax(logits, dim=-1)

            if use_constraints:
                force_end_mask = (step == (lengths - 2))
                after_end_mask = (step > (lengths - 2))
                next_token = torch.where(force_end_mask, end_poi, next_token)
                next_token = torch.where(after_end_mask, torch.zeros_like(next_token), next_token)

            curr = torch.cat([curr, next_token.unsqueeze(1)], dim=1)
            valid_token_mask = next_token != 0
            if use_no_repeat_mask and valid_token_mask.any():
                visited.scatter_(1, next_token.unsqueeze(1), True)

        return curr

    def _predict_beam_single(self, z_final_vec, query_h_single, start, end, route_len, ellipse_mask_row=None):
        device = z_final_vec.device
        beam_size = max(1, int(getattr(self.args, 'beam_size', 4)))
        len_penalty = float(getattr(self.args, 'beam_len_penalty', 0.0))
        use_no_repeat_mask = bool(getattr(self.args, 'use_no_repeat_mask', 1))
        use_constraints = (
            self.decode_constraint_mode == 'hard'
            and bool(getattr(self.args, 'enforce_start_end_constraints', 1))
        )

        init_token = int(start) if use_constraints else 0
        init_seq = torch.tensor([init_token], dtype=torch.long, device=device)
        init_visited = torch.zeros(self.poi_num, dtype=torch.bool, device=device)
        if use_constraints and start != 0:
            init_visited[start] = True
        beams = [(init_seq, 0.0, init_visited)]

        for step in range(max(route_len - 1, 0)):
            cand = []
            for seq, score, visited in beams:
                if use_constraints:
                    if step > (route_len - 2):
                        tok = 0
                        nseq = torch.cat([seq, torch.tensor([tok], dtype=torch.long, device=device)], dim=0)
                        cand.append((nseq, score, visited))
                        continue

                    if step == (route_len - 2):
                        tok = int(end)
                        nseq = torch.cat([seq, torch.tensor([tok], dtype=torch.long, device=device)], dim=0)
                        nvisited = visited.clone()
                        if tok != 0:
                            nvisited[tok] = True
                        cand.append((nseq, score, nvisited))
                        continue

                query_h_batch = query_h_single.unsqueeze(0) if query_h_single is not None else None
                start_batch = torch.tensor([start], dtype=torch.long, device=device)
                end_batch = torch.tensor([end], dtype=torch.long, device=device)
                logits = self._next_step_logits(
                    seq.unsqueeze(0),
                    z_final_vec.unsqueeze(0),
                    query_h=query_h_batch,
                    start_poi=start_batch,
                    end_poi=end_batch,
                )[0]
                logits[0] = -1e9
                if ellipse_mask_row is not None:
                    logits = logits.masked_fill(~ellipse_mask_row, -1e9)
                if use_no_repeat_mask:
                    logits = logits.masked_fill(visited, -1e9)

                if not torch.isfinite(logits).any():
                    tok = int(end)
                    nseq = torch.cat([seq, torch.tensor([tok], dtype=torch.long, device=device)], dim=0)
                    nvisited = visited.clone()
                    nvisited[tok] = True
                    cand.append((nseq, score, nvisited))
                    continue

                logp = F.log_softmax(logits, dim=-1)
                k = min(beam_size, logp.numel())
                topv, topi = torch.topk(logp, k=k)
                for i in range(k):
                    tok = int(topi[i].item())
                    nseq = torch.cat([seq, torch.tensor([tok], dtype=torch.long, device=device)], dim=0)
                    nvisited = visited.clone()
                    if tok != 0:
                        nvisited[tok] = True
                    cand.append((nseq, score + float(topv[i].item()), nvisited))

            def rank_key(item):
                seq, score, _ = item
                if len_penalty <= 0.0:
                    return score
                denom = float(max(seq.numel() - 1, 1)) ** len_penalty
                return score / denom

            cand.sort(key=rank_key, reverse=True)
            beams = cand[:beam_size]

        if len(beams) == 0:
            return torch.tensor([start, end], dtype=torch.long, device=device)

        beams.sort(key=lambda x: x[1], reverse=True)
        return beams[0][0]

    def predict(self, batch):
        _, _, z_stack, _ = self._encode_home(batch)
        query_h = self._encode_query(batch)
        z_trans, _, profile_h, query_h = self._adaptive_transfer(z_stack, batch['user_profile'], query_h)
        city_vec = self._compose_city_vectors()[batch['dst_rg']]
        z_final, _ = self._blend_user_city(z_trans, city_vec, profile_h, query_h)

        start_poi = batch['query_start_poi']
        end_poi = batch['query_end_poi']
        lengths = batch['query_len'].long().clamp(min=2)
        max_len = int(lengths.max().item())

        ellipse_mask = None
        if getattr(self.args, 'ellipse_filter', False):
            ellipse_mask = self._calculate_ellipse_mask(
                start_idx=start_poi.long(),
                end_idx=end_poi.long(),
                all_coords=self.poi_coords,
                beta=float(getattr(self.args, 'ellipse_beta', 1.2)),
            )

        use_beam = bool(getattr(self.args, 'use_beam_search', 1))
        beam_size = int(getattr(self.args, 'beam_size', 4))
        if (not use_beam) or beam_size <= 1:
            return self._predict_greedy(z_final, query_h, start_poi, end_poi, lengths, max_len, ellipse_mask)

        batch_size = start_poi.size(0)
        out = torch.zeros((batch_size, max_len), dtype=torch.long, device=start_poi.device)
        for b in range(batch_size):
            mask_row = ellipse_mask[b] if ellipse_mask is not None else None
            seq = self._predict_beam_single(
                z_final_vec=z_final[b],
                query_h_single=query_h[b],
                start=int(start_poi[b].item()),
                end=int(end_poi[b].item()),
                route_len=int(lengths[b].item()),
                ellipse_mask_row=mask_row,
            )
            use_len = min(seq.numel(), max_len)
            out[b, :use_len] = seq[:use_len]
        return out
