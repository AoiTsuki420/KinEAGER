"""
kcat-OOD 专家模型：结构感知、单任务（只预测 log10 kcat）。
与主模型在 log 空间做精度加权 ensemble。

输入约定（batch dict）:
  prot_ids      : LongTensor [B, L_p]          ESM2 token ids
  prot_mask     : BoolTensor [B, L_p]          True=valid
  prot_dist     : FloatTensor [B, L_p, L_p]    Ca-Ca 距离图 (Å)，pad 处填 0
  prot_struct_mask: BoolTensor [B, L_p]        True=该残基有结构
  lig_ids       : LongTensor [B, L_l]          MolT5 token ids
  lig_mask      : BoolTensor [B, L_l]
  lig_atom_feat : FloatTensor [B, N_a, F_a]    原子特征（元素 onehot + 电荷 + 杂化等）
  lig_atom_dist : FloatTensor [B, N_a, N_a]    原子对距离 (Å)
  lig_atom_mask : BoolTensor [B, N_a]
  geom_feats    : FloatTensor [B, 16]          手工几何特征（口袋体积/SASA/logP…）
  has_struct    : BoolTensor [B]               样本是否有结构（用于 gate）
  y_kcat        : FloatTensor [B]              log10(kcat) 标签
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F



@dataclass
class KcatExpertConfig:
    d_prot_seq: int = 1280         # ESM2-650M hidden
    d_lig_seq: int = 768           # MolT5-base hidden
    d_prot_struct: int = 256
    d_lig_struct: int = 128
    d_model: int = 512
    n_cross_blocks: int = 2
    n_heads: int = 8
    d_geom: int = 16
    mlp_hidden: int = 256
    dropout: float = 0.2
    lig_atom_feat_dim: int = 44
    max_prot_len: int = 1024
    max_lig_atoms: int = 128



class ProtStructEncoderV1(nn.Module):
    """Cα 距离图 -> 残基级 embedding。
    做法：把每行距离做 radial basis 展开 -> 1D CNN 压缩 -> 得到每残基 d 维。
    """

    def __init__(self, d_out: int = 256, n_rbf: int = 16, rbf_max: float = 20.0):
        super().__init__()
        self.n_rbf = n_rbf
        self.register_buffer("rbf_centers", torch.linspace(0.0, rbf_max, n_rbf))
        self.rbf_sigma = rbf_max / n_rbf
        self.proj = nn.Sequential(
            nn.Conv1d(n_rbf, d_out, kernel_size=7, padding=3),
            nn.GELU(),
            nn.Conv1d(d_out, d_out, kernel_size=5, padding=2),
            nn.GELU(),
        )
        self.ln = nn.LayerNorm(d_out)

    def forward(self, dist: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        d = dist.unsqueeze(-1)  # [B,L,L,1]
        centers = self.rbf_centers.view(1, 1, 1, -1)
        rbf = torch.exp(-((d - centers) ** 2) / (2 * self.rbf_sigma ** 2))
        valid = mask.unsqueeze(1) & mask.unsqueeze(2)  # [B,L,L]
        rbf = rbf * valid.unsqueeze(-1).float()
        denom = valid.float().sum(dim=2, keepdim=True).clamp_min(1.0)
        feat = rbf.sum(dim=2) / denom  # [B, L, n_rbf]
        x = feat.transpose(1, 2)  # [B, n_rbf, L]
        x = self.proj(x).transpose(1, 2)  # [B, L, d]
        return self.ln(x)


class LigStructEncoderV1(nn.Module):
    """原子特征 + 距离矩阵 -> set-transformer 风格编码。"""

    def __init__(self, atom_in: int, d_out: int = 128, n_heads: int = 4, n_layers: int = 2, n_rbf: int = 16, rbf_max: float = 10.0):
        super().__init__()
        self.atom_proj = nn.Linear(atom_in, d_out)
        self.n_rbf = n_rbf
        self.register_buffer("rbf_centers", torch.linspace(0.0, rbf_max, n_rbf))
        self.rbf_sigma = rbf_max / n_rbf
        self.edge_proj = nn.Linear(n_rbf, n_heads)
        self.layers = nn.ModuleList([
            _PairBiasedAttn(d_out, n_heads) for _ in range(n_layers)
        ])
        self.ln = nn.LayerNorm(d_out)

    def forward(self, atom_feat: torch.Tensor, atom_dist: torch.Tensor, atom_mask: torch.Tensor) -> torch.Tensor:
        h = self.atom_proj(atom_feat)  # [B, N, d]
        d = atom_dist.unsqueeze(-1)
        centers = self.rbf_centers.view(1, 1, 1, -1)
        rbf = torch.exp(-((d - centers) ** 2) / (2 * self.rbf_sigma ** 2))
        pair_bias = self.edge_proj(rbf)  # [B, N, N, n_heads]
        pair_bias = pair_bias.permute(0, 3, 1, 2)  # [B, heads, N, N]
        for layer in self.layers:
            h = layer(h, pair_bias, atom_mask)
        return self.ln(h)


class _PairBiasedAttn(nn.Module):
    def __init__(self, d: int, n_heads: int):
        super().__init__()
        assert d % n_heads == 0
        self.h = n_heads
        self.dh = d // n_heads
        self.qkv = nn.Linear(d, 3 * d)
        self.out = nn.Linear(d, d)
        self.ln1 = nn.LayerNorm(d)
        self.ln2 = nn.LayerNorm(d)
        self.ffn = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x, pair_bias, mask):
        B, N, D = x.shape
        h = self.ln1(x)
        qkv = self.qkv(h).reshape(B, N, 3, self.h, self.dh).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B, heads, N, dh]
        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.dh) + pair_bias
        m = mask.unsqueeze(1).unsqueeze(1)  # [B,1,1,N]
        attn = attn.masked_fill(~m, -1e4)
        attn = attn.softmax(dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0, posinf=0.0, neginf=0.0)
        o = (attn @ v).transpose(1, 2).reshape(B, N, D)
        x = x + self.out(o)
        x = x + self.ffn(self.ln2(x))
        return x



class CrossAttnBlock(nn.Module):
    def __init__(self, d: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.p2l = nn.MultiheadAttention(d, n_heads, dropout=dropout, batch_first=True)
        self.l2p = nn.MultiheadAttention(d, n_heads, dropout=dropout, batch_first=True)
        self.ln_p1 = nn.LayerNorm(d)
        self.ln_l1 = nn.LayerNorm(d)
        self.ln_p2 = nn.LayerNorm(d)
        self.ln_l2 = nn.LayerNorm(d)
        self.ffn_p = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Dropout(dropout), nn.Linear(4 * d, d))
        self.ffn_l = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Dropout(dropout), nn.Linear(4 * d, d))

    def forward(self, H_p, H_l, mask_p, mask_l):
        kpm_p = ~mask_p  # key_padding_mask expects True=pad
        kpm_l = ~mask_l
        p2, _ = self.p2l(self.ln_p1(H_p), self.ln_l1(H_l), self.ln_l1(H_l), key_padding_mask=kpm_l, need_weights=False)
        l2, _ = self.l2p(self.ln_l1(H_l), self.ln_p1(H_p), self.ln_p1(H_p), key_padding_mask=kpm_p, need_weights=False)
        H_p = H_p + p2
        H_l = H_l + l2
        H_p = H_p + self.ffn_p(self.ln_p2(H_p))
        H_l = H_l + self.ffn_l(self.ln_l2(H_l))
        return H_p, H_l


def masked_mean(x, mask):
    m = mask.unsqueeze(-1).float()
    return (x * m).sum(1) / m.sum(1).clamp_min(1.0)



class AvailGate(nn.Module):
    """seq 特征始终用；struct 特征在 has_struct=False 时门控为 0。
    用可学习的门控权重（sigmoid），以 has_struct 标志做 hard mask 相乘。"""

    def __init__(self, d_seq: int, d_struct: int, d_out: int):
        super().__init__()
        self.gate = nn.Linear(d_seq, d_struct)
        self.merge = nn.Linear(d_seq + d_struct, d_out)

    def forward(self, h_seq, h_struct, has_struct):
        g = torch.sigmoid(self.gate(h_seq))
        mask = has_struct.view(-1, *([1] * (h_struct.dim() - 1))).float()
        h_struct_g = h_struct * g * mask
        return self.merge(torch.cat([h_seq, h_struct_g], dim=-1))



class KcatExpert(nn.Module):
    def __init__(self, cfg: KcatExpertConfig, esm_encoder: nn.Module, molt5_encoder: nn.Module):
        """esm_encoder / molt5_encoder 传入已冻结（或带 LoRA）的 HF 模型封装，
        需要实现 .forward(ids, mask) -> [B, L, d]。外部负责 LoRA / 冻结策略。"""
        super().__init__()
        self.cfg = cfg
        self.esm = esm_encoder
        self.molt5 = molt5_encoder

        self.prot_struct = ProtStructEncoderV1(d_out=cfg.d_prot_struct)
        self.lig_struct = LigStructEncoderV1(atom_in=cfg.lig_atom_feat_dim, d_out=cfg.d_lig_struct)

        self.prot_merge = AvailGate(cfg.d_prot_seq, cfg.d_prot_struct, cfg.d_model)
        self.lig_merge = AvailGate(cfg.d_lig_seq, cfg.d_lig_struct, cfg.d_model)

        self.blocks = nn.ModuleList([
            CrossAttnBlock(cfg.d_model, cfg.n_heads, cfg.dropout) for _ in range(cfg.n_cross_blocks)
        ])

        head_in = 2 * cfg.d_model + cfg.d_geom
        self.dropout = nn.Dropout(cfg.dropout)
        self.head = nn.Sequential(
            nn.Linear(head_in, cfg.mlp_hidden * 2), nn.GELU(), nn.Dropout(cfg.dropout),
            nn.Linear(cfg.mlp_hidden * 2, cfg.mlp_hidden), nn.GELU(), nn.Dropout(cfg.dropout),
            nn.Linear(cfg.mlp_hidden, 2),  # (mu, log_sigma2)
        )

    def encode(self, batch):
        cfg = self.cfg
        h_p_seq = self.esm(batch["prot_ids"], batch["prot_mask"])              # [B, L_p, d_prot_seq]
        h_l_seq = self.molt5(batch["lig_ids"], batch["lig_mask"])              # [B, L_l, d_lig_seq]

        h_p_struct_res = self.prot_struct(batch["prot_dist"], batch["prot_struct_mask"])  # [B, L_p, d_ps]
        if h_p_struct_res.size(1) != h_p_seq.size(1):
            L = h_p_seq.size(1)
            h_p_struct_res = F.pad(h_p_struct_res, (0, 0, 0, max(0, L - h_p_struct_res.size(1))))[:, :L]

        h_l_struct_atom = self.lig_struct(batch["lig_atom_feat"], batch["lig_atom_dist"], batch["lig_atom_mask"])  # [B, N_a, d_ls]
        lig_struct_vec = masked_mean(h_l_struct_atom, batch["lig_atom_mask"])  # [B, d_ls]
        h_l_struct_seq = lig_struct_vec.unsqueeze(1).expand(-1, h_l_seq.size(1), -1)

        has_struct = batch["has_struct"]
        H_p = self.prot_merge(h_p_seq, h_p_struct_res, has_struct)
        H_l = self.lig_merge(h_l_seq, h_l_struct_seq, has_struct)
        return H_p, H_l

    def forward(self, batch):
        H_p, H_l = self.encode(batch)
        mask_p = batch["prot_mask"]
        mask_l = batch["lig_mask"]
        for blk in self.blocks:
            H_p, H_l = blk(H_p, H_l, mask_p, mask_l)
        pool_p = masked_mean(H_p, mask_p)
        pool_l = masked_mean(H_l, mask_l)
        feats = torch.cat([pool_p, pool_l, batch["geom_feats"]], dim=-1)
        out = self.head(self.dropout(feats))
        mu, log_var = out[..., 0], out[..., 1]
        return mu, log_var

    @torch.no_grad()
    def predict_mc(self, batch, n_samples: int = 5):
        """MC-Dropout 推理，返回均值和不确定度（aleatoric + epistemic）。"""
        was_training = self.training
        self.train()  # 打开 dropout
        mus, lvs = [], []
        for _ in range(n_samples):
            mu, lv = self.forward(batch)
            mus.append(mu); lvs.append(lv)
        self.train(was_training)
        mu_stack = torch.stack(mus, 0)
        lv_stack = torch.stack(lvs, 0)
        mu_mean = mu_stack.mean(0)
        aleatoric = lv_stack.exp().mean(0)
        epistemic = mu_stack.var(0, unbiased=False)
        sigma2 = aleatoric + epistemic
        return mu_mean, sigma2



def _ccc_loss(mu: torch.Tensor, y: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """1 - CCC（Concordance Correlation Coefficient）。
    CCC 同时优化 Pearson r 和 scale/location 匹配。
    CCC = 2 * r * sx * sy / (sx² + sy² + (mx-my)²)
    """
    mx, my = mu.mean(), y.mean()
    sx = mu.std(unbiased=False).clamp_min(eps)
    sy = y.std(unbiased=False).clamp_min(eps)
    cov = ((mu - mx) * (y - my)).mean()
    ccc = 2 * cov / (sx ** 2 + sy ** 2 + (mx - my) ** 2 + eps)
    return 1.0 - ccc


def kcat_expert_loss(
    mu: torch.Tensor,
    log_var: torch.Tensor,
    y: torch.Tensor,
    *,
    huber_beta: float = 1.0,
    rank_weight: float = 0.3,
    rank_pairs: int = 512,
    rank_margin: float = 0.1,
    var_reg_weight: float = 0.1,
    nll_weight: float = 0.05,
    ccc_weight: float = 0.0,
) -> tuple[torch.Tensor, dict]:
    mu = torch.nan_to_num(mu, nan=0.0, posinf=1e4, neginf=-1e4)
    y = torch.nan_to_num(y, nan=0.0, posinf=1e4, neginf=-1e4)
    log_var = torch.nan_to_num(log_var, nan=0.0, posinf=10.0, neginf=-10.0).clamp(-10.0, 10.0)

    logs = {}
    loss_huber = F.smooth_l1_loss(mu, y, beta=huber_beta)
    logs["huber"] = loss_huber.detach()

    loss_rank = mu.new_zeros(())
    n = y.numel()
    if rank_weight > 0 and n >= 2:
        n_pairs = min(rank_pairs, n * (n - 1))
        i = torch.randint(0, n, (n_pairs,), device=y.device)
        j = torch.randint(0, n, (n_pairs,), device=y.device)
        keep = i != j
        i, j = i[keep], j[keep]
        sign = torch.sign(y[i] - y[j])
        nz = sign != 0
        if nz.any():
            loss_rank = F.margin_ranking_loss(mu[i[nz]], mu[j[nz]], sign[nz], margin=rank_margin)
    logs["rank"] = loss_rank.detach()

    if y.numel() > 1:
        loss_varreg = (mu.std(unbiased=False) - y.std(unbiased=False)).abs()
    else:
        loss_varreg = mu.new_zeros(())
    logs["var_reg"] = loss_varreg.detach()

    loss_nll = 0.5 * (log_var + (y - mu) ** 2 / log_var.exp().clamp_min(1e-6))
    loss_nll = loss_nll.mean()
    logs["nll"] = loss_nll.detach()

    loss_ccc = mu.new_zeros(())
    if ccc_weight > 0 and n > 1:
        loss_ccc = _ccc_loss(mu, y)
    logs["ccc"] = loss_ccc.detach()

    total = (loss_huber
             + rank_weight * loss_rank
             + var_reg_weight * loss_varreg
             + nll_weight * loss_nll
             + ccc_weight * loss_ccc)
    logs["total"] = total.detach()
    return total, logs



@torch.no_grad()
def precision_weighted_merge(
    mu_main: torch.Tensor, sigma2_main: torch.Tensor,
    mu_expert: torch.Tensor, sigma2_expert: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    w_main = 1.0 / sigma2_main.clamp_min(eps)
    w_exp = 1.0 / sigma2_expert.clamp_min(eps)
    z = w_main + w_exp
    return (w_main * mu_main + w_exp * mu_expert) / z
