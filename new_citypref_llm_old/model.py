import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from transformers import AutoModel, AutoTokenizer
except Exception:
    AutoModel = None
    AutoTokenizer = None


class FrozenSemanticEncoder(nn.Module):
    """
    NEW: LLM驱动语义编码（soft prompt + frozen encoder）。
    这里用冻结 Transformer 近似 LLM 编码器，以保持依赖与旧工程一致。
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

        # NEW: 冻结“LLM主体”参数，只训练 soft prompt。
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
    NEW: 使用Qwen模型进行语义编码（冻结LLM参数 + 可训练soft prompt）。
    """

    def __init__(self, args):
        super().__init__()
        if AutoModel is None or AutoTokenizer is None:
            raise ImportError("transformers is required for Qwen semantic encoder")

        self.args = args
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
        self.llm.eval()
        self.llm_hidden = int(self.llm.config.hidden_size)

        # NEW: 冻结Qwen参数，仅训练soft prompt和projection。
        for p in self.llm.parameters():
            p.requires_grad = False

        self.soft_prompt = nn.Parameter(torch.randn(self.prompt_len, self.llm_hidden) * 0.02)
        self.proj = nn.Sequential(
            nn.Linear(self.llm_hidden, args.hidden_size),
            nn.Tanh(),
        )
        self.norm = nn.LayerNorm(args.hidden_size)

    @staticmethod
    def _resolve_model_name(args):
        """
        NEW: 模型名解析，优先用户指定，其次尝试候选Qwen模型，降低加载失败概率。
        """
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


class CrossCityLLMCPR(nn.Module):
    def __init__(self, args, poi_num, tag_num, region_num, popularity_bias=None):
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
                print(f"[INFO] Qwen semantic encoder loaded: {self.semantic_encoder_qwen.model_name}")
            except Exception as e:
                # NEW: 若Qwen加载失败，回退到结构化编码器，保证训练不中断。
                if getattr(args, 'qwen_strict', False):
                    raise RuntimeError(f"Qwen semantic encoder init failed under strict mode: {e}")
                print(f"[WARN] Qwen semantic encoder init failed: {e}. Fallback to frozen transformer encoder.")
                self.semantic_backend = 'fallback'

        # NEW: LLM语义与结构化语义的自适应融合门控，提升LLM分布偏移下的稳健性。
        self.semantic_fusion_gate = nn.Sequential(
            nn.Linear(args.hidden_size, args.hidden_size // 2),
            nn.ReLU(),
            nn.Linear(args.hidden_size // 2, 1),
            nn.Sigmoid(),
        )

        self.home_poi_emb = nn.Embedding(poi_num, args.hidden_size, padding_idx=0)
        self.home_seq_encoder = nn.GRU(args.hidden_size, args.hidden_size, batch_first=True)

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
        self.query_proj = nn.Linear(args.query_dim, args.hidden_size)
        self.transfer_gate = nn.Sequential(
            nn.Linear(args.hidden_size * 3, args.hidden_size),
            nn.GELU(),
            nn.Linear(args.hidden_size, self.pref_num),
        )

        self.tour_poi_emb = nn.Embedding(poi_num, args.hidden_size, padding_idx=0)
        self.tour_encoder = nn.GRU(args.hidden_size, args.hidden_size, batch_first=True)

        self.user_proj = nn.Linear(args.hidden_size, args.hidden_size)
        self.route_proj = nn.Linear(args.hidden_size, args.hidden_size)
        self.city_proj = nn.Linear(args.hidden_size, args.hidden_size)
        self.city_memory = nn.Embedding(region_num, args.hidden_size)

        self.eta_gate = nn.Sequential(
            nn.Linear(args.hidden_size * 2, args.hidden_size),
            nn.ReLU(),
            nn.Linear(args.hidden_size, 1),
            nn.Sigmoid(),
        )

        self.decoder = nn.GRU(args.hidden_size * 2, args.hidden_size, batch_first=True)
        self.decoder_out = nn.Linear(args.hidden_size, poi_num)

        if popularity_bias is None:
            popularity_bias = torch.zeros(poi_num)
        self.register_buffer("popularity_bias", popularity_bias)

    @staticmethod
    def _masked_last_hidden(gru, emb, valid_mask):
        out, _ = gru(emb)
        lengths = valid_mask.long().sum(dim=1).clamp(min=1)
        last_idx = (lengths - 1).unsqueeze(1).unsqueeze(2).expand(-1, 1, out.size(-1))
        last_hidden = out.gather(1, last_idx).squeeze(1)
        return last_hidden

    def _encode_home(self, batch):
        seq_emb = self.home_poi_emb(batch['ori_ck'])
        h_seq = self._masked_last_hidden(self.home_seq_encoder, seq_emb, batch['ori_pad'])
        s_struct = self.semantic_encoder_fallback(
            batch['ori_ck'], batch['ori_tag'], batch['o_hour'], batch['ori_l'], batch['ori_pad']
        )
        if self.semantic_backend == 'qwen' and self.semantic_encoder_qwen is not None:
            s_llm = self.semantic_encoder_qwen(batch['home_prompt_text'], device=batch['ori_ck'].device)
            mix = self.semantic_fusion_gate(h_seq)
            s_u = mix * s_llm + (1.0 - mix) * s_struct
        else:
            mix = torch.zeros((h_seq.size(0), 1), device=h_seq.device)
            s_u = s_struct
        h_u = torch.cat([h_seq, s_u], dim=-1)
        z_list = [head(h_u) for head in self.disentangle_heads]
        z_stack = torch.stack(z_list, dim=1)
        return h_seq, s_u, z_stack, mix

    def _adaptive_transfer(self, z_stack, user_profile, query_vec):
        profile_h = self.profile_proj(user_profile)
        query_h = self.query_proj(query_vec)
        z_mean = z_stack.mean(dim=1)
        gate_in = torch.cat([z_mean, profile_h, query_h], dim=-1)
        alpha = F.softmax(self.transfer_gate(gate_in), dim=-1)
        z_trans = torch.sum(alpha.unsqueeze(-1) * z_stack, dim=1)
        return z_trans, alpha, profile_h, query_h

    def _encode_tour(self, batch):
        tour_emb = self.tour_poi_emb(batch['dst_ck'])
        r_tour = self._masked_last_hidden(self.tour_encoder, tour_emb, batch['dst_pad'])
        return r_tour

    def _decouple_semantic_loss(self, z_stack, s_u):
        decouple = 0.0
        for i in range(self.pref_num):
            for j in range(i + 1, self.pref_num):
                decouple = decouple + torch.mean(torch.abs(F.cosine_similarity(z_stack[:, i, :], z_stack[:, j, :], dim=-1)))

        sem = 0.0
        for i in range(self.pref_num):
            sem = sem + torch.mean(1.0 - F.cosine_similarity(z_stack[:, i, :], s_u, dim=-1))
        return decouple, sem

    def _alignment_loss(self, z_trans, r_tour, dst_rg):
        tau = self.args.temperature
        z_u = F.normalize(self.user_proj(z_trans), dim=-1)
        r_u = F.normalize(self.route_proj(r_tour), dim=-1)

        logits_user = torch.matmul(z_u, r_u.t()) / tau
        labels = torch.arange(logits_user.size(0), device=logits_user.device)
        loss_user = F.cross_entropy(logits_user, labels)

        city_all = F.normalize(self.city_proj(self.city_memory.weight), dim=-1)
        logits_city = torch.matmul(z_u, city_all.t()) / tau
        loss_city = F.cross_entropy(logits_city, dst_rg)
        return loss_user, loss_city

    def _generator_loss(self, z_final, dst_ck):
        dec_in = dst_ck[:, :-1]
        dec_target = dst_ck[:, 1:]
        dec_emb = self.tour_poi_emb(dec_in)
        context = z_final.unsqueeze(1).expand(-1, dec_emb.size(1), -1)
        dec_feat = torch.cat([dec_emb, context], dim=-1)
        dec_out, _ = self.decoder(dec_feat)
        logits = self.decoder_out(dec_out)

        # NEW: 显式注入热门度偏置，满足“城市热门度与特色偏置”约束。
        logits = logits + self.args.pop_bias_scale * self.popularity_bias.unsqueeze(0).unsqueeze(0)
        loss_gen = F.cross_entropy(logits.reshape(-1, self.poi_num), dec_target.reshape(-1), ignore_index=0)
        return loss_gen

    def forward(self, batch):
        h_seq, s_u, z_stack, mix = self._encode_home(batch)
        z_trans, alpha, profile_h, query_h = self._adaptive_transfer(
            z_stack, batch['user_profile'], batch['query_vec']
        )
        r_tour = self._encode_tour(batch)

        loss_decouple, loss_semantic = self._decouple_semantic_loss(z_stack, s_u)
        loss_user, loss_city = self._alignment_loss(z_trans, r_tour, batch['dst_rg'])
        loss_align = loss_user + self.args.gamma_city * loss_city

        city_vec = self.city_memory(batch['dst_rg'])
        eta = self.eta_gate(torch.cat([profile_h, query_h], dim=-1))
        z_final = eta * z_trans + (1.0 - eta) * city_vec

        loss_gen = self._generator_loss(z_final, batch['dst_ck'])
        total_loss = loss_align + self.args.lambda_decouple * loss_decouple + self.args.lambda_semantic * loss_semantic + self.args.lambda_gen * loss_gen

        return {
            "loss": total_loss,
            "align": loss_align.detach(),
            "decouple": loss_decouple.detach(),
            "semantic": loss_semantic.detach(),
            "gen": loss_gen.detach(),
            "eta": eta.mean().detach(),
            "sem_mix": mix.mean().detach(),
            "alpha_mean": alpha.mean(dim=0).detach(),
        }

    def predict(self, batch):
        _, _, z_stack, _ = self._encode_home(batch)
        z_trans, _, profile_h, query_h = self._adaptive_transfer(z_stack, batch['user_profile'], batch['query_vec'])
        city_vec = self.city_memory(batch['dst_rg'])
        eta = self.eta_gate(torch.cat([profile_h, query_h], dim=-1))
        z_final = eta * z_trans + (1.0 - eta) * city_vec

        start_poi = batch['query_start_poi']
        end_poi = batch['query_end_poi']
        lengths = batch['query_len'].long().clamp(min=2)
        max_len = int(lengths.max().item())

        curr = start_poi.unsqueeze(1)
        batch_size = curr.size(0)
        visited = torch.zeros(batch_size, self.poi_num, dtype=torch.bool, device=curr.device)
        visited.scatter_(1, start_poi.unsqueeze(1), True)

        for step in range(max_len - 1):
            dec_emb = self.tour_poi_emb(curr)
            context = z_final.unsqueeze(1).expand(-1, dec_emb.size(1), -1)
            dec_feat = torch.cat([dec_emb, context], dim=-1)
            dec_out, _ = self.decoder(dec_feat)
            logits = self.decoder_out(dec_out[:, -1, :])
            logits = logits + self.args.pop_bias_scale * self.popularity_bias.unsqueeze(0)

            # NEW: 约束感知生成，避免重复访问并屏蔽padding id。
            logits[:, 0] = -1e9
            logits = logits.masked_fill(visited, -1e9)
            next_token = torch.argmax(logits, dim=-1)

            force_end_mask = (step == (lengths - 2))
            after_end_mask = (step > (lengths - 2))
            next_token = torch.where(force_end_mask, end_poi, next_token)
            next_token = torch.where(after_end_mask, torch.zeros_like(next_token), next_token)

            curr = torch.cat([curr, next_token.unsqueeze(1)], dim=1)
            valid_token_mask = next_token != 0
            if valid_token_mask.any():
                visited.scatter_(1, next_token.unsqueeze(1), True)

        return curr
