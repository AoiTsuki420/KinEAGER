import math
import torch
import torch.nn as nn
import torch.utils.checkpoint as cp
import torch.nn.functional as F
from .encoders import ProteinEncoder, LigandEncoder
from .interactions import CrossAttentionBlock, ResidueMask,LigandMask
from utils import masked_mean
import json
import loralib as lora
from peft import LoraConfig, get_peft_model
from transformers import AutoModel
from transformers import AutoModelForSeq2SeqLM

class PredictorConfig:
    def __init__(self,
                 d_model=1024,
                 num_heads=8,
                 num_interaction_layers=6,
                 num_mamba_layers=0,
                 num_token_transformer_layers=None,
                 rate=0.1,
                 transformer_ffn_dim=2048,
                 head_hidden=768,
                 dropout=0.1,
                 use_attention_pool=False,
                 use_checkpoint=False,
                 num_stages=1,                    # 多 stage
                 num_interaction_layers_per_stage=None,  # 每个 stage 的交互层数
                 num_heads_in_head=1,             # 多塔 head 数
                 head_hidden_dim=None,      # 每塔隐藏维度
                 head_depth=3,
                 active_heads=None,             # 任务顺序决定 forward 输出顺序
                 head_specs=None,               # 每个 head 的结构超参
                 share_fusion=True,             # 共享融合向量（蛋白+配体）
                 fusion_norm=True,              # 融合前是否 LayerNorm
                 fusion_proj_dim=None,          # 可选：把 concat 后维度投影
                 use_moe_in_head=False,         # MoE 开关（按每个 head 生效）
                 moe_num_experts=4,
                 moe_top_k=1,
                 use_struct_branch=False,          # 是否启用结构分支
                 struct_in_dim_prot=None,         # 蛋白结构向量维度
                 struct_in_dim_lig=None,          # 配体结构向量维度
                 struct_fusion_mode="concat",     # "concat" 或 "add"
                 enable_avail_gate=False,         # 结构可用性感知门控
                 avail_gate_hidden=256,
                 use_phys_branch=False,
                 phys_in_dim=0,
                 phys_hidden=128,
                 domain_uncertainty=False,        # 按域×任务不确定性
                 num_domains=3,
                 use_source_residual_head=False,
                 source_head_hidden=256,
                 source_head_dropout=0.1,
                 source_head_tasks=None,
                 default_source_id=0,
                 use_interactions=True,   # 是否启用 cross-attention 堆栈
                 use_gate_p=True,         # 是否启用 protein gate
                 use_gate_l=True,         # 是否启用 ligand gate         
                ):
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_interaction_layers = num_interaction_layers
        self.num_mamba_layers = num_mamba_layers
        if num_token_transformer_layers is None:
            self.num_token_transformer_layers = num_mamba_layers
        else:
            self.num_token_transformer_layers = num_token_transformer_layers
            self.num_mamba_layers = num_token_transformer_layers
        self.rate = rate
        self.transformer_ffn_dim = transformer_ffn_dim
        self.head_hidden = head_hidden
        self.dropout = dropout
        self.use_attention_pool = use_attention_pool
        self.use_checkpoint = use_checkpoint
        self.num_stages = num_stages
        self.num_interaction_layers_per_stage = num_interaction_layers_per_stage or num_interaction_layers
        self.num_heads_in_head = num_heads_in_head
        self.head_hidden_dim = head_hidden_dim or head_hidden
        self.head_depth = head_depth  # 保存参数
        
        self.active_heads = active_heads or ["kcat", "Km", "ratio"]

        default_spec = {
            "type": "mlp",
            "hidden_dim": self.head_hidden_dim,
            "depth": self.head_depth,
            "out_dim": 1,
            "dropout": self.dropout,
        }
        self.head_specs = head_specs or {
            "kcat":  {**default_spec},
            "Km":    {**default_spec},
            "ratio": {**default_spec},
        }

        self.share_fusion = share_fusion
        self.fusion_norm = fusion_norm
        self.fusion_proj_dim = fusion_proj_dim  # e.g. 512 降维；None 表示不投影

        self.use_moe_in_head = use_moe_in_head
        self.moe_num_experts = moe_num_experts
        self.moe_top_k = moe_top_k
        
        self.use_struct_branch = use_struct_branch
        self.struct_in_dim_prot = struct_in_dim_prot
        self.struct_in_dim_lig = struct_in_dim_lig
        self.struct_fusion_mode = struct_fusion_mode
        self.enable_avail_gate = enable_avail_gate
        self.avail_gate_hidden = avail_gate_hidden
        self.use_phys_branch = use_phys_branch
        self.phys_in_dim = phys_in_dim
        self.phys_hidden = phys_hidden
        self.domain_uncertainty = domain_uncertainty
        self.num_domains = num_domains
        self.use_source_residual_head = use_source_residual_head
        self.source_head_hidden = source_head_hidden
        self.source_head_dropout = source_head_dropout
        self.source_head_tasks = source_head_tasks or ["kcat"]
        self.default_source_id = default_source_id
        
        self.use_interactions = use_interactions
        self.use_gate_p = use_gate_p
        self.use_gate_l = use_gate_l
        
        
class AttentionPool(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.query = nn.Parameter(torch.randn(d_model))
        self.scale = 1.0 / (d_model ** 0.5)

    def forward(self, x, mask=None):
        q = self.query.unsqueeze(0).unsqueeze(0)  # (1,1,D)
        scores = torch.sum(x * q, dim=-1) * self.scale  # (B, L)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn = torch.softmax(scores, dim=-1)  # (B, L)
        pooled = torch.sum(attn.unsqueeze(-1) * x, dim=1)  # (B, D)
        
        return pooled


class ProteinTransformerLayer(nn.Module):
    def __init__(self, d_model, num_heads, ffn_dim=2048, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
        )

    def forward(self, x, token_mask=None):
        key_padding_mask = None
        if token_mask is not None:
            key_padding_mask = ~token_mask.bool()

        x_norm = self.norm1(x)
        attn_out, _ = self.self_attn(
            x_norm,
            x_norm,
            x_norm,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        x = x + self.dropout(attn_out)
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x
    
    
class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, drop=0.1, depth=3):
        super().__init__()
        layers = []
        dim = in_dim
        for i in range(depth-1):
            layers.append(nn.Linear(dim, hidden_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(drop))
            dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, out_dim))
        layers.append(nn.GELU())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class ResidualMLP(nn.Module):
    """
    支持纵向残差的 MLP，每个中间层都可以加残差连接
    """
    def __init__(self, in_dim, hidden_dim, out_dim, drop=0.1, depth=3):
        super().__init__()
        self.depth = depth
        self.layers = nn.ModuleList()
        self.drop = nn.Dropout(drop)
        
        dim = in_dim
        for i in range(depth-1):
            self.layers.append(nn.Linear(dim, hidden_dim))
            self.layers.append(nn.GELU())
            dim = hidden_dim
        
        self.layers.append(nn.Linear(hidden_dim, out_dim))
        self.layers.append(nn.GELU())
        
        self.net = nn.Sequential(*self.layers)
        
        if in_dim != out_dim:
            self.res_proj = nn.Linear(in_dim, out_dim)
        else:
            self.res_proj = nn.Identity()

    def forward(self, x):
        residual = self.res_proj(x)
        out = x
        for layer in self.layers:
            out = layer(out)
            if isinstance(layer, nn.GELU):
                out = out + residual
                residual = out
        out = self.drop(out)
        return out

class TaskHeadBase(nn.Module):
    def forward(self, x):  # (B, D) -> (B, out_dim)
        raise NotImplementedError

class MLPHead(TaskHeadBase):
    def __init__(self, in_dim, hidden_dim, out_dim=1, depth=3, drop=0.1):
        super().__init__()
        layers = []
        d = in_dim
        for i in range(depth-1):
            layers += [nn.Linear(d, hidden_dim), nn.GELU(), nn.Dropout(drop)]
            d = hidden_dim
        layers += [nn.Linear(d, out_dim)]
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)

class LinearHead(TaskHeadBase):
    def __init__(self, in_dim, out_dim=1):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)
    def forward(self, x):
        return self.fc(x)

class MoEMLPHead(TaskHeadBase):
    """
    极简门控 MoE：top-k 路由到若干 MLP 专家；适合 head 侧提升表达力且计算可控。
    """
    def __init__(self, in_dim, hidden_dim, out_dim=1, depth=3, drop=0.1, num_experts=4, top_k=1):
        super().__init__()
        self.experts = nn.ModuleList([
            MLPHead(in_dim, hidden_dim, out_dim, depth, drop) for _ in range(num_experts)
        ])
        self.gate = nn.Linear(in_dim, num_experts)  # 简单门控
        self.top_k = top_k

    def forward(self, x):
        logits = self.gate(x)                       # (B, E)
        topk_val, topk_idx = torch.topk(logits, k=self.top_k, dim=-1)  # (B, K)
        weights = torch.softmax(topk_val, dim=-1)   # (B, K)

        out = 0.0
        for k in range(self.top_k):
            idx = topk_idx[:, k]                    # (B,)
            w = weights[:, k].unsqueeze(-1)        # (B,1)
            parts = []
            for b in range(x.size(0)):
                parts.append(self.experts[idx[b].item()](x[b:b+1, :]))  # (1, out_dim)
            parts = torch.cat(parts, dim=0)         # (B, out_dim)
            out = out + w * parts
        return out

class FiLMHead(TaskHeadBase):
    """
    FiLM: Feature-wise Linear Modulation.
    这里假设输入 x = concat([p_vec, l_vec]) 的同时，我们用 l_vec 生成 gamma/beta 调制 p_vec（或反之）。
    为保持通用，只要传入 x 与一个“条件”向量 cond（下方 MultiHead 负责提供）。
    """
    def __init__(self, base_in_dim, cond_dim, hidden_dim, out_dim=1, depth=3, drop=0.1):
        super().__init__()
        self.gamma = nn.Linear(cond_dim, base_in_dim)
        self.beta  = nn.Linear(cond_dim, base_in_dim)
        self.backbone = MLPHead(base_in_dim, hidden_dim, out_dim, depth, drop)

    def forward(self, x, cond=None):
        assert cond is not None, "FiLMHead 需要 cond"
        g = self.gamma(cond)
        b = self.beta(cond)
        x_mod = x * (1 + g) + b
        return self.backbone(x_mod)

class MultiHead(nn.Module):
    """
    任务名 -> head 的注册容器，forward 返回 (list_outputs, dict_outputs)
    list_outputs 顺序与 active_heads 一致，便于兼容原训练脚本。
    """
    def __init__(self, in_dim, cfg: PredictorConfig, cond_split=None):
        super().__init__()
        self.active = list(cfg.active_heads)
        self.cond_split = cond_split  # 用于 FiLM：把 concat 向量切成 (p_vec, l_vec)
        self.heads = nn.ModuleDict()

        for name in self.active:
            spec = cfg.head_specs.get(name, {})
            typ = spec.get("type", "mlp")
            hidden = spec.get("hidden_dim", cfg.head_hidden_dim)
            depth  = spec.get("depth", cfg.head_depth)
            out_dim = spec.get("out_dim", 1)
            drop = spec.get("dropout", cfg.dropout)

            if typ == "linear":
                self.heads[name] = LinearHead(in_dim, out_dim)
            elif typ == "moe_mlp" or (cfg.use_moe_in_head and typ == "mlp"):
                self.heads[name] = MoEMLPHead(
                    in_dim, hidden, out_dim, depth, drop,
                    num_experts=cfg.moe_num_experts, top_k=cfg.moe_top_k
                )
            elif typ == "film_mlp":
                assert self.cond_split is not None, "film_mlp 需要 cond_split=(p_dim, l_dim)"
                p_dim, l_dim = self.cond_split
                self.heads[name] = FiLMHead(p_dim, l_dim, hidden, out_dim, depth, drop)
            else:
                self.heads[name] = MLPHead(in_dim, hidden, out_dim, depth, drop)

    def forward(self, x, cond=None):
        """
        x: (B, D) 或者当 film 时传 concat 向量，并在内部拆分
        cond: 可选，为 film head 提供条件向量
        """
        outs_list = []
        outs_dict = {}
        for name in self.active:
            head = self.heads[name]
            if isinstance(head, FiLMHead):
                assert cond is not None and self.cond_split is not None
                p_dim, l_dim = self.cond_split
                p_vec = x[..., :p_dim]
                l_vec = x[..., p_dim:p_dim+l_dim]
                y = head(p_vec, cond=l_vec)
            else:
                y = head(x)
            outs_dict[name] = y
            outs_list.append(y)
        return outs_list, outs_dict


class SourceResidualHead(nn.Module):
    def __init__(self, in_dim, num_domains, hidden_dim=256, drop=0.1):
        super().__init__()
        self.num_domains = int(max(1, num_domains))
        self.heads = nn.ModuleList([
            MLPHead(in_dim, hidden_dim, out_dim=1, depth=2, drop=drop)
            for _ in range(self.num_domains)
        ])

    def forward(self, x, source_ids):
        source_ids = source_ids.to(device=x.device, dtype=torch.long).reshape(-1)
        source_ids = source_ids.clamp(min=0, max=max(self.num_domains - 1, 0))
        delta = torch.zeros(x.size(0), 1, device=x.device, dtype=x.dtype)
        for sid, head in enumerate(self.heads):
            mask = (source_ids == sid)
            if mask.any():
                delta[mask] = head(x[mask])
        return delta.squeeze(-1)


class AvailabilityAwareFusion(nn.Module):
    """
    样本级标量门控：
      g = sigmoid(MLP([p_vec, l_vec, prot_s, lig_s, avail]))
      g_eff = avail * g
      p_out = p_vec + g_eff * (prot_s - p_vec)
      l_out = l_vec + g_eff * (lig_s  - l_vec)
    """
    def __init__(self, d_model: int, hidden_dim: int = 256):
        super().__init__()
        in_dim = d_model * 4 + 1
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, p_vec, l_vec, prot_s, lig_s, struct_avail_mask):
        if struct_avail_mask is None:
            struct_avail_mask = torch.ones(
                p_vec.size(0),
                device=p_vec.device,
                dtype=p_vec.dtype,
            )
        avail = struct_avail_mask.reshape(-1, 1).to(dtype=p_vec.dtype)
        gate_in = torch.cat([p_vec, l_vec, prot_s, lig_s, avail], dim=-1)
        g_logits = self.net(gate_in)
        g = torch.sigmoid(g_logits)
        g_eff = g * avail
        p_out = p_vec + g_eff * (prot_s - p_vec)
        l_out = l_vec + g_eff * (lig_s - l_vec)
        return p_out, l_out, g_eff.squeeze(-1), g_logits.squeeze(-1), avail.squeeze(-1)


class QualityAwarePhysFusion(nn.Module):
    def __init__(self, d_model: int, phys_in_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.phys_adapter = nn.Sequential(
            nn.LayerNorm(phys_in_dim),
            nn.Linear(phys_in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, d_model),
        )
        gate_in = d_model * 3 + 2
        self.gate_net = nn.Sequential(
            nn.Linear(gate_in, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, p_vec, l_vec, phys_feat, phys_mask, phys_quality):
        if phys_feat is None:
            return p_vec, l_vec, None, None, None
        emb = self.phys_adapter(phys_feat)
        if phys_mask is None:
            phys_mask = torch.ones(p_vec.size(0), device=p_vec.device, dtype=p_vec.dtype)
        if phys_quality is None:
            phys_quality = phys_mask
        m = phys_mask.reshape(-1, 1).to(dtype=p_vec.dtype)
        q = phys_quality.reshape(-1, 1).to(dtype=p_vec.dtype)
        gate_in = torch.cat([p_vec, l_vec, emb, m, q], dim=-1)
        logits = self.gate_net(gate_in)
        alpha = torch.sigmoid(logits)
        alpha_eff = alpha * m * q
        p_out = p_vec + alpha_eff * emb
        l_out = l_vec + alpha_eff * emb
        return p_out, l_out, alpha_eff.squeeze(-1), logits.squeeze(-1), m.squeeze(-1)
    

class KineticsPredictor(nn.Module):
    def __init__(self,
                 device="cuda",
                 esm_model="/root/autodl-tmp/models/esm2_t33_650M_UR50D",
                 molt5_model="/root/autodl-tmp/models/molt5-base-smiles2caption",
                 cfg: PredictorConfig = PredictorConfig(),
                 use_lora=True,
                 lora_r=8,
                 lora_alpha=16,
                 lora_dropout=0.05,
                 struct_in_dim_prot=None,
                 struct_in_dim_lig=None,
                ):
        super().__init__()
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.cfg = cfg
        self.use_lora = use_lora
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        
        self.num_stages = getattr(cfg, 'num_stages', 1)
        self.num_interaction_layers_per_stage = getattr(cfg, 'num_interaction_layers_per_stage', 1)
        self.num_heads_in_head = getattr(cfg, 'num_heads_in_head', 1)
        self.head_hidden_dim = getattr(cfg, 'head_hidden_dim', 128)

        
        C = cfg.d_model
        
        self.esm = AutoModel.from_pretrained(esm_model)
        self.molt5 = AutoModelForSeq2SeqLM.from_pretrained(molt5_model)
        
        self.use_struct_branch = cfg.use_struct_branch
        if self.use_struct_branch:
            assert cfg.struct_in_dim_prot is not None, "use_struct_branch=True 时必须设置 struct_in_dim_prot"
            assert cfg.struct_in_dim_lig is not None, "use_struct_branch=True 时必须设置 struct_in_dim_lig"

            self.prot_struct_proj = nn.Sequential(
                nn.LayerNorm(cfg.struct_in_dim_prot),
                nn.Linear(cfg.struct_in_dim_prot, C),
                nn.GELU(),
            )
            self.lig_struct_proj = nn.Sequential(
                nn.LayerNorm(cfg.struct_in_dim_lig),
                nn.Linear(cfg.struct_in_dim_lig, C),
                nn.GELU(),
            )

            if cfg.struct_fusion_mode == "concat":
                self.prot_struct_fuse = nn.Linear(2 * C, C)
                self.lig_struct_fuse  = nn.Linear(2 * C, C)
            elif cfg.struct_fusion_mode == "add":
                self.prot_struct_fuse = None
                self.lig_struct_fuse  = None
            else:
                raise ValueError(f"Unknown struct_fusion_mode: {cfg.struct_fusion_mode}")       

            self.enable_avail_gate = bool(getattr(cfg, "enable_avail_gate", False))
            if self.enable_avail_gate:
                self.struct_avail_fusion = AvailabilityAwareFusion(
                    d_model=C,
                    hidden_dim=int(getattr(cfg, "avail_gate_hidden", 256)),
                )
            else:
                self.struct_avail_fusion = None
        else:
            self.enable_avail_gate = False
            self.struct_avail_fusion = None

        self.use_phys_branch = bool(getattr(cfg, "use_phys_branch", False))
        if self.use_phys_branch:
            phys_in_dim = int(getattr(cfg, "phys_in_dim", 0))
            if phys_in_dim <= 0:
                raise ValueError("use_phys_branch=True requires phys_in_dim > 0")
            self.phys_fusion = QualityAwarePhysFusion(
                d_model=C,
                phys_in_dim=phys_in_dim,
                hidden_dim=int(getattr(cfg, "phys_hidden", 128)),
            )
        else:
            self.phys_fusion = None

        if self.use_lora:
            def _count_lora_params(model):
                n_params, n_tensors = 0, 0
                for name, p in model.named_parameters():
                    if "lora_" in name:
                        n_params += p.numel()
                        n_tensors += 1
                return n_params, n_tensors

            def _list_linear_names(m):
                return [n for n, mod in m.named_modules() if isinstance(mod, nn.Linear)]

            def _num_layers_from_model(m):
                L = getattr(getattr(m, "config", object()), "num_hidden_layers", None)
                if L is None and hasattr(m, "encoder") and hasattr(m.encoder, "layer"):
                    L = len(m.encoder.layer)
                assert L is not None, "无法推断 ESM 层数"
                return int(L)

            def _select_layers(L, policy="sandwich", last_k=2, explicit=None):
                if policy in ("all", "full", "everything"):
                    return list(range(max(0, L)))
                if policy == "last_k":
                    k = max(1, min(last_k, L))
                    return list(range(L - k, L))
                elif policy == "sandwich":
                    picks = {0, L // 2, max(L - 4, 0), max(L - 2, 0), L - 1}
                    return sorted(i for i in picks if 0 <= i < L)
                elif policy == "explicit":
                    assert explicit, "explicit 需提供层号"
                    return sorted(i for i in explicit if 0 <= i < L)
                
                else:
                    raise ValueError(f"Unknown policy: {policy}")

            def _build_esm_targets_exact(model: nn.Module, layers):
                names = set(_list_linear_names(model))
                targets = []
                suffixes = [
                    "attention.self.query",
                    "attention.self.key",
                    "attention.self.value",
                    "attention.output.dense",
                    "intermediate.dense",
                    "output.dense",
                ]
                for i in layers:
                    stem = f"encoder.layer.{i}"  # 你的模型是 encoder.layer.* 这一种
                    for sfx in suffixes:
                        key = f"{stem}.{sfx}"
                        if key in names:
                            targets.append(key)
                targets = sorted(set(targets))
                assert targets, "没有命中任何 ESM 线性层，请检查层号或命名。"
                return targets
            
            def _num_t5_encoder_layers(t5):
                if hasattr(t5, "encoder") and hasattr(t5.encoder, "block"):
                    return len(t5.encoder.block)
                return int(getattr(getattr(t5, "config", object()), "num_layers", 0))

            def _build_t5_encoder_targets_exact(t5_model: nn.Module):
                names = set(_list_linear_names(t5_model))
                L = _num_t5_encoder_layers(t5_model)
                targets = []
                sfx = [
                    "layer.0.SelfAttention.q",
                    "layer.0.SelfAttention.k",
                    "layer.0.SelfAttention.v",
                    "layer.0.SelfAttention.o",
                    "layer.1.DenseReluDense.wi_0",
                    "layer.1.DenseReluDense.wi_1",
                    "layer.1.DenseReluDense.wo",
                ]
                for i in range(L):
                    stem = f"encoder.block.{i}"
                    for tail in sfx:
                        key = f"{stem}.{tail}"
                        if key in names:
                            targets.append(key)
                targets = sorted(set(targets))
                assert targets, "没有命中任何 MolT5 encoder 线性层，请检查命名。"
                return targets

            L = _num_layers_from_model(self.esm)
            selected_layers = _select_layers(L, policy="all")  # -> L=12 时为 [0,6,8,10,11]
            print(f"[ESM] total layers = {L}, selected layers = {selected_layers}")

            all_linear = _list_linear_names(self.esm)
            probe = [n for n in all_linear if any(f"encoder.layer.{i}." in n for i in selected_layers)]
            print("[ESM Linear sample near selected layers]")
            for n in sorted(probe)[:20]:
                print("  ", n)

            target_modules_esm = _build_esm_targets_exact(self.esm, layers=selected_layers)
            print("[ESM target_modules] count =", len(target_modules_esm))
            for t in target_modules_esm[:24]:
                print("  ", t)

            lora_cfg_esm = LoraConfig(
                r=self.lora_r,
                lora_alpha=self.lora_alpha,
                target_modules=target_modules_esm,
                lora_dropout=self.lora_dropout,
                bias="none",
                task_type="FEATURE_EXTRACTION",
            )
            self.esm = get_peft_model(self.esm, lora_cfg_esm)

            n_lora_params, n_lora_tensors = _count_lora_params(self.esm)
            print(f"[ESM injected] LoRA tensors={n_lora_tensors}, params={n_lora_params}")
            assert n_lora_params > 0, "LoRA 没注进 self.esm（请检查 target_modules 是否命中真实层名）"
            
            
            
            target_modules_molt5 = _build_t5_encoder_targets_exact(self.molt5)
            print("[MolT5 encoder target_modules] count =", len(target_modules_molt5))
            for t in target_modules_molt5[:24]:
                print("  ", t)
                
            lora_cfg_molt5 = LoraConfig(
               r=self.lora_r,
               lora_alpha=self.lora_alpha,
               target_modules=target_modules_molt5,
               lora_dropout=self.lora_dropout,
                bias="none",
                task_type="FEATURE_EXTRACTION",
            )
            self.molt5 = get_peft_model(self.molt5, lora_cfg_molt5)
            n_lora_params_m5, n_lora_tensors_m5 = _count_lora_params(self.molt5)
            print(f"[MolT5 encoder injected] LoRA tensors={n_lora_tensors_m5}, params={n_lora_params_m5}")
            assert n_lora_params_m5 > 0, "MolT5 LoRA 注入失败（没有命中 q/v）"
            

        self.protein = ProteinEncoder(
            esm_model=esm_model,
            backbone_obj=self.esm,
            d_model=C,
            device=self.device,
            train_backbone=True,
        )
        self.ligand = LigandEncoder(
            molt5_model=molt5_model,
            backbone_obj=self.molt5.get_encoder(),
            d_model=C,
            device=self.device,
            train_backbone=True,
        )
        

        self.token_transformer_layers = nn.ModuleList([
            ProteinTransformerLayer(
                d_model=C,
                num_heads=cfg.num_heads,
                ffn_dim=cfg.transformer_ffn_dim,
                dropout=cfg.dropout,
            )
            for _ in range(cfg.num_token_transformer_layers)
        ])
        self.mamba_layers = self.token_transformer_layers

        self.interactions = nn.ModuleList([
            nn.ModuleList([
                CrossAttentionBlock(d_model=C, num_heads=cfg.num_heads)
                for _ in range(self.num_interaction_layers_per_stage)
            ])
            for _ in range(self.num_stages)
        ])

        self.resmask = ResidueMask(d_model=C, droprate=0.5)

        self.ligmask = LigandMask(d_model=C, dropout=cfg.dropout)

        self.norm_p = nn.LayerNorm(C)
        self.norm_l = nn.LayerNorm(C)

        self.use_attention_pool = cfg.use_attention_pool
        if self.use_attention_pool:
            self.att_pool_p = AttentionPool(C)
            self.att_pool_l = AttentionPool(C)
        else:
            self.att_pool_p = None
            self.att_pool_l = None

        self.proj = nn.Identity()  # if desired, replace with nn.Linear(C, C)

        H = cfg.head_hidden
        drop = cfg.dropout

        

        
        
        fusion_in_dim = C * 2
        self.fusion_ln = nn.LayerNorm(fusion_in_dim) if cfg.fusion_norm else nn.Identity()
        if cfg.fusion_proj_dim is not None:
            self.fusion_proj = nn.Linear(fusion_in_dim, cfg.fusion_proj_dim)
            fused_dim = cfg.fusion_proj_dim
        else:
            self.fusion_proj = nn.Identity()
            fused_dim = fusion_in_dim

        cond_split = (C, C)  # p_vec 与 l_vec 的原始维度
        self.multi_head = MultiHead(
            in_dim=fused_dim,
            cfg=cfg,
            cond_split=cond_split  # 若不用 film_mlp 也无妨
        )
        self.kcat_source_residual = None
        if bool(getattr(cfg, "use_source_residual_head", False)):
            tasks = set(getattr(cfg, "source_head_tasks", ["kcat"]))
            if "kcat" in tasks:
                self.kcat_source_residual = SourceResidualHead(
                    in_dim=fused_dim,
                    num_domains=int(getattr(cfg, "num_domains", 3)),
                    hidden_dim=int(getattr(cfg, "source_head_hidden", 256)),
                    drop=float(getattr(cfg, "source_head_dropout", cfg.dropout)),
                )

        self.last_pred_dict = None
        

        
        self.last_gate_p = None   # (B, Lp)
        self.last_gate_l = None   # (B, Ll)
        self.last_align  = None   # (B, Lp, Ll) alignment matrix (soft contacts)
        self.last_gate_struct = None          # (B,)
        self.last_gate_struct_logits = None   # (B,)
        self.last_struct_avail = None         # (B,)
        self.last_gate_phys = None
        self.last_gate_phys_logits = None
        self.last_phys_avail = None

        self.s_kcat  = nn.Parameter(torch.zeros(1))
        self.s_Km    = nn.Parameter(torch.zeros(1))
        self.s_ratio = nn.Parameter(torch.zeros(1))
        if bool(getattr(cfg, "domain_uncertainty", False)):
            self.s_domain_task = nn.Parameter(torch.zeros(int(getattr(cfg, "num_domains", 3)), 3))
        else:
            self.s_domain_task = None

        self._reset_parameters()
        self.to(self.device)
        
    def _reset_parameters(self):
        skip_prefixes = ("protein.backbone", "ligand.backbone")
        for name, m in self.named_modules():
            if any(name.startswith(pfx) or f".{pfx}" in name for pfx in skip_prefixes):
                continue
            if "lora" in name.lower():
                continue
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _pool(self, tok, mask, which='p'):
        if self.use_attention_pool:
            if which == 'p':
                return self.att_pool_p(tok, mask)
            else:
                return self.att_pool_l(tok, mask)
        else:
            if which == 'p':
                return masked_mean(self.norm_p(tok), mask)
            else:
                return masked_mean(self.norm_l(tok), mask)

    def compute_alignment(self, p_tok, l_tok, p_mask=None, l_mask=None):
        """
        Compute a soft alignment matrix between protein residues and ligand tokens.
        Returns A of shape (B, Lp, Ll) where rows are residues and columns are ligand tokens.
        Implementation: scaled dot-product of L2-normalized token embeddings followed by softmax over ligand tokens.
        """
        B, Lp, D = p_tok.shape
        Ll = l_tok.shape[1]

        p_norm = p_tok / (p_tok.norm(dim=-1, keepdim=True) + 1e-9)  # (B, Lp, D)
        l_norm = l_tok / (l_tok.norm(dim=-1, keepdim=True) + 1e-9)  # (B, Ll, D)

        sim = torch.matmul(p_norm, l_norm.transpose(1,2))  # (B, Lp, Ll)
        sim = sim / math.sqrt(D)

        if l_mask is not None:
            mask = l_mask.unsqueeze(1)  # (B, 1, Ll)
            sim = sim.masked_fill(mask == 0, float('-inf'))

        align = torch.softmax(sim, dim=-1)  # normalize across ligand tokens
        return align

    def get_last_gates(self):
        """
        Returns a tuple: (gate_p, gate_l, align)
        gate_p: (B, Lp) protein gate (same as old 'gate' returned by forward)
        gate_l: (B, Ll) ligand gate (new)
        align:  (B, Lp, Ll) alignment matrix
        """
        return self.last_gate_p, self.last_gate_l, self.last_align

    def forward(self, seqs, smiles, use_mask: bool = True,
        bias_tuple=(0.0, 0.0, 0.0), caps=(0.5, 0.5, 0.4), floor_m: float = -1e9,prot_struct=None,
        lig_struct=None, struct_avail_mask=None, phys_feat=None, phys_mask=None, phys_quality=None,
        source_ids=None):
        """
        输入:
            seqs   : 蛋白序列列表或张量
            smiles : 配体 SMILES 列表或张量
            use_mask: 是否在训练时使用稀疏掩码
        输出:
            p1, p2, p3: 预测值（**正数**，已通过 softplus）
            gate_p    : 蛋白 token gate (B, Lp)
            gate_s    : 配体 token gate (B, Ll)
        """
        p_tok, p_mask = self.protein(seqs)
        l_tok, l_mask = self.ligand(smiles)

        if p_mask is None:
            p_mask = torch.ones(p_tok.shape[:2], device=p_tok.device, dtype=torch.long)
        if l_mask is None:
            l_mask = torch.ones(l_tok.shape[:2], device=l_tok.device, dtype=torch.long)

        for m in self.token_transformer_layers:
            if self.cfg.use_checkpoint:
                p_tok = cp.checkpoint(m, p_tok, p_mask, use_reentrant=False)
            else:
                p_tok = m(p_tok, p_mask)

        if getattr(self.cfg, "use_gate_p", True):
            gate_p, erase_p = self.resmask(p_tok, train_mask=use_mask)
            self.last_gate_p = gate_p.detach()
            if use_mask and self.training:
                p_tok = p_tok * (1.0 - erase_p)
        else:
            gate_p = torch.ones(p_tok.shape[:2], device=p_tok.device, dtype=p_tok.dtype)
            self.last_gate_p = gate_p.detach()

        if getattr(self.cfg, "use_gate_l", True):
            gate_s, erase_s = self.ligmask(l_tok, train_mask=use_mask)
            self.last_gate_l = gate_s.detach()
            if use_mask and self.training:
                l_tok = l_tok * (1.0 - erase_s)
        else:
            gate_s = torch.ones(l_tok.shape[:2], device=l_tok.device, dtype=l_tok.dtype)
            self.last_gate_l = gate_s.detach()    
        

        if getattr(self.cfg, "use_interactions", True) and len(self.interactions) > 0:
            for stage in self.interactions:
                for blk in stage:               # ✅ 遍历每个 block
                    if self.cfg.use_checkpoint:
                        p_tok, l_tok = cp.checkpoint(blk, p_tok, l_tok, p_mask, l_mask, use_reentrant=False)
                    else:
                        p_tok, l_tok = blk(p_tok, l_tok, p_mask, l_mask)
        else:
            pass

        self.last_align = self.compute_alignment(p_tok, l_tok, p_mask, l_mask)
        if getattr(self.cfg, "use_interactions", True) and len(self.interactions) > 0:
            last_blk = self.interactions[-1][-1]
            self.last_pocket_loss = getattr(last_blk, 'last_pocket_loss', torch.tensor(0.0, device=p_tok.device))
            self.last_ortho = getattr(last_blk, 'last_ortho', torch.tensor(0.0, device=p_tok.device))
        else:
            self.last_pocket_loss = torch.tensor(0.0, device=p_tok.device)
            self.last_ortho = torch.tensor(0.0, device=p_tok.device)

        p_vec = self._pool(p_tok, p_mask, which='p')   # (B, C)
        l_vec = self._pool(l_tok, l_mask, which='l')   # (B, C)
        
        self.last_gate_struct = torch.zeros(p_vec.size(0), device=p_vec.device, dtype=p_vec.dtype)
        self.last_gate_struct_logits = torch.zeros(p_vec.size(0), device=p_vec.device, dtype=p_vec.dtype)
        if struct_avail_mask is None:
            self.last_struct_avail = torch.zeros(p_vec.size(0), device=p_vec.device, dtype=p_vec.dtype)
        else:
            self.last_struct_avail = struct_avail_mask.reshape(-1).to(device=p_vec.device, dtype=p_vec.dtype)

        if self.use_struct_branch and (prot_struct is not None) and (lig_struct is not None):
            prot_s = self.prot_struct_proj(prot_struct)   # (B, C)
            lig_s  = self.lig_struct_proj(lig_struct)     # (B, C)

            if self.enable_avail_gate and self.struct_avail_fusion is not None:
                p_vec, l_vec, gate_struct, gate_logits, avail = self.struct_avail_fusion(
                    p_vec,
                    l_vec,
                    prot_s,
                    lig_s,
                    struct_avail_mask,
                )
                self.last_gate_struct = gate_struct.detach()
                self.last_gate_struct_logits = gate_logits.detach()
                self.last_struct_avail = avail.detach()
            else:
                if self.cfg.struct_fusion_mode == "concat":
                    p_vec = self.prot_struct_fuse(torch.cat([p_vec, prot_s], dim=-1))
                    l_vec = self.lig_struct_fuse(torch.cat([l_vec, lig_s], dim=-1))
                elif self.cfg.struct_fusion_mode == "add":
                    p_vec = 0.5 * (p_vec + prot_s)
                    l_vec = 0.5 * (l_vec + lig_s)
                else:
                    raise ValueError(f"Unknown struct_fusion_mode: {self.cfg.struct_fusion_mode}")
                if struct_avail_mask is None:
                    self.last_struct_avail = torch.ones(p_vec.size(0), device=p_vec.device, dtype=p_vec.dtype)
                else:
                    self.last_struct_avail = struct_avail_mask.reshape(-1).to(device=p_vec.device, dtype=p_vec.dtype)

        self.last_gate_phys = torch.zeros(p_vec.size(0), device=p_vec.device, dtype=p_vec.dtype)
        self.last_gate_phys_logits = torch.zeros(p_vec.size(0), device=p_vec.device, dtype=p_vec.dtype)
        self.last_phys_avail = torch.zeros(p_vec.size(0), device=p_vec.device, dtype=p_vec.dtype)
        if self.use_phys_branch and (phys_feat is not None) and (self.phys_fusion is not None):
            p_vec, l_vec, gate_phys, gate_phys_logits, p_avail = self.phys_fusion(
                p_vec,
                l_vec,
                phys_feat,
                phys_mask,
                phys_quality,
            )
            if gate_phys is not None:
                self.last_gate_phys = gate_phys.detach()
                self.last_gate_phys_logits = gate_phys_logits.detach()
                self.last_phys_avail = p_avail.detach()
        

        x = torch.cat([p_vec, l_vec], dim=-1)          # (B, 2C)
        if self.cfg.fusion_norm:                        # 与 __init__ 的 cfg.fusion_norm 保持一致
            x = self.fusion_ln(x)                       # LayerNorm(2C)
        x = self.fusion_proj(x)                         # (B, fused_dim)；若未配置降维则是恒等映射

        outs_list, outs_dict = self.multi_head(x, cond=x)
        self.last_pred_dict = outs_dict  # 方便你在外部拿到完整字典，比如 outs_dict["kcat"] shape=(B,1)

        heads = self.cfg.active_heads  # e.g. ["kcat","Km","ratio"]
        outs_list = [outs_dict[name].squeeze(-1) for name in heads]

        pred_kcat_log, pred_Km_log, pred_ratio_log = outs_list[:3]

        if self.kcat_source_residual is not None:
            if source_ids is None:
                source_ids = torch.full(
                    (x.size(0),),
                    int(getattr(self.cfg, "default_source_id", 0)),
                    device=x.device,
                    dtype=torch.long,
                )
            elif not torch.is_tensor(source_ids):
                source_ids = torch.as_tensor(source_ids, device=x.device, dtype=torch.long)
            else:
                source_ids = source_ids.to(device=x.device, dtype=torch.long)

            source_ids = source_ids.reshape(-1)
            if source_ids.numel() == 1 and x.size(0) > 1:
                source_ids = source_ids.expand(x.size(0))
            if source_ids.numel() != x.size(0):
                source_ids = source_ids[:x.size(0)]
            pred_kcat_log = pred_kcat_log + self.kcat_source_residual(x, source_ids)
        
        b_k, b_m, b_r = bias_tuple
        cap_k, cap_m, cap_r = caps

        use_domain_s = bool(getattr(self.cfg, "domain_uncertainty", False)) and (self.s_domain_task is not None) and (source_ids is not None)
        if use_domain_s:
            if not torch.is_tensor(source_ids):
                source_ids = torch.as_tensor(source_ids, device=x.device)
            source_ids = source_ids.to(device=x.device, dtype=torch.long).reshape(-1)
            num_domains = int(self.s_domain_task.shape[0])
            source_ids = source_ids.clamp(min=0, max=max(num_domains - 1, 0))
            s_dom_raw = self.s_domain_task[source_ids]  # (B, 3)

            s_k_raw = s_dom_raw[:, 0]
            s_m_raw = s_dom_raw[:, 1]
            s_r_raw = s_dom_raw[:, 2]
            s_k_eff = torch.clamp(s_k_raw + b_k, min=-1e9, max=cap_k)
            s_m_eff = torch.clamp(s_m_raw + b_m, min=floor_m, max=cap_m)
            s_r_eff = torch.clamp(s_r_raw + b_r, min=-1e9, max=cap_r)
        else:
            s_k_raw = self.s_kcat     # nn.Parameter
            s_m_raw = self.s_Km
            s_r_raw = self.s_ratio
            s_k_eff = torch.clamp(s_k_raw + b_k, min=-1e9, max=cap_k)
            s_m_eff = torch.clamp(s_m_raw + b_m, min=floor_m, max=cap_m)
            s_r_eff = torch.clamp(s_r_raw + b_r, min=-1e9, max=cap_r)

        s_raw = (s_k_raw, s_m_raw, s_r_raw)
        s_eff = (s_k_eff, s_m_eff, s_r_eff)
        
        return pred_kcat_log, pred_Km_log, pred_ratio_log, gate_p, gate_s,s_raw, s_eff


    def saliency(self, seqs, smiles, target='kcat'):
            """
            Compute simple gradient-based saliency on protein residues.
            Returns (sal_protein, p_mask).
            """
            self.eval()
            for p in self.parameters():
                p.requires_grad_(True)
            p_tok, p_mask = self.protein(seqs)
            l_tok, l_mask = self.ligand(smiles)

            p_tok.requires_grad_(True)

            for stage in self.interactions:
                for blk in stage:
                    p_tok, l_tok = blk(p_tok, l_tok, p_mask, l_mask)

            p_vec = masked_mean(self.norm_p(p_tok), p_mask)
            l_vec = masked_mean(self.norm_l(l_tok), l_mask)
            x = torch.cat([p_vec, l_vec], dim=-1)
            outs_list, outs_dict = self.multi_head(x, cond=x)
            head_idx = {'kcat': 0, 'Km': 1, 'kcat_Km': 2}[target]
            score = outs_list[head_idx].sum()
            score.backward()
            sal = (p_tok.grad ** 2).sum(dim=-1)
            return sal.detach(), p_mask
