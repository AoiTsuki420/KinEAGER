"""KinEAGER: OOD-aware mixture of (generalist) main predictor + (specialist) kcat expert.

路由信号：查询序列的 ESM 平均池化向量 vs. expert 训练集索引的余弦距离。
距离越大（越 OOD），expert 权重越高。

组件:
  - OODRouter: 输入查询 embedding → 输出 (w_expert, ood_distance)
  - KinEAGER  : 持有 main / expert / router，forward(batch) 返回融合结果与所有中间量

仅做推理封装；所有子模块内部如何 MC-Dropout 由本模块包装。
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F



class OODRouter(nn.Module):
    """基于 train set ESM embedding 的 k-NN 余弦距离路由。

    d_cos_nn = 1 - max_top_k(cosine(query, train_emb))
    w_expert = sigmoid((d_cos_nn - d0) / tau)

    参数:
      train_emb : [N, D]，外部传入，内部会再做一次 L2 归一化。
      d0        : 阈值；查询距离 > d0 时偏向 expert（w_expert > 0.5）。
      tau       : 温度；越小越硬（极限为阶跃函数）。
      k         : top-k 邻居；k=1 为最近邻，k>1 则取 top-k 平均相似度。
    """

    def __init__(self, train_emb: torch.Tensor, d0: float = 0.15,
                 tau: float = 0.05, k: int = 1):
        super().__init__()
        if train_emb.dim() != 2:
            raise ValueError(f"train_emb must be [N, D], got {tuple(train_emb.shape)}")
        emb = F.normalize(train_emb.float(), dim=-1)
        self.register_buffer("train_emb", emb, persistent=False)
        self.d0 = float(d0)
        self.tau = float(tau)
        self.k = int(k)

    def set_params(self, d0: Optional[float] = None,
                   tau: Optional[float] = None, k: Optional[int] = None) -> None:
        if d0 is not None:
            self.d0 = float(d0)
        if tau is not None:
            self.tau = float(tau)
        if k is not None:
            self.k = int(k)

    @torch.no_grad()
    def nn_distance(self, query: torch.Tensor) -> torch.Tensor:
        q = F.normalize(query.float().to(self.train_emb.device), dim=-1)
        sim = q @ self.train_emb.t()  # [B, N]
        if self.k <= 1:
            max_sim = sim.max(dim=-1).values
        else:
            top = sim.topk(min(self.k, sim.size(-1)), dim=-1).values
            max_sim = top.mean(dim=-1)
        return (1.0 - max_sim).clamp_min(0.0)

    @torch.no_grad()
    def route(self, query: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        d = self.nn_distance(query)
        w_expert = torch.sigmoid((d - self.d0) / max(self.tau, 1e-6))
        return w_expert, d



class KinEAGER(nn.Module):
    """统一的 MoE 推理壳。query_encoder 用于算 OOD 路由的 embedding（通常复用 expert 的 ESM）。"""

    def __init__(
        self,
        main_model: nn.Module,
        expert_model: nn.Module,
        router: OODRouter,
        query_encoder: nn.Module,
        main_mc_samples: int = 5,
        expert_mc_samples: int = 5,
        hard_gate_on_no_struct: bool = True,
        use_precision_weighting: bool = False,
        source_id: int = 0,
    ):
        super().__init__()
        self.main = main_model
        self.expert = expert_model
        self.router = router
        self.query_encoder = query_encoder
        self.main_mc = int(main_mc_samples)
        self.expert_mc = int(expert_mc_samples)
        self.hard_gate_on_no_struct = bool(hard_gate_on_no_struct)
        self.use_precision_weighting = bool(use_precision_weighting)
        self.source_id = int(source_id)

    @torch.no_grad()
    def _route_embed(self, seqs: list[str]) -> torch.Tensor:
        backbone = getattr(self.query_encoder, "backbone", None)
        disable_ctx = None
        if backbone is not None and hasattr(backbone, "disable_adapter"):
            disable_ctx = backbone.disable_adapter()
        if disable_ctx is not None:
            with disable_ctx:
                tok, mask = self.query_encoder.encode_tokens(seqs)
        else:
            tok, mask = self.query_encoder.encode_tokens(seqs)
        m = mask.float().unsqueeze(-1)
        return (tok * m).sum(1) / m.sum(1).clamp_min(1.0)

    @torch.no_grad()
    def _main_mc(self, seqs, smiles, device) -> tuple[torch.Tensor, torch.Tensor]:
        src = torch.full((len(seqs),), self.source_id, device=device, dtype=torch.long)
        mus = []
        for _ in range(self.main_mc):
            out = self.main(list(seqs), list(smiles), source_ids=src, use_mask=False)
            pred = out[0]
            if pred.dim() > 1:
                pred = pred.squeeze(-1)
            mus.append(pred)
        stack = torch.stack(mus, 0)
        return stack.mean(0), stack.var(0, unbiased=False).clamp_min(1e-6)

    @torch.no_grad()
    def _expert_mc(self, batch) -> tuple[torch.Tensor, torch.Tensor]:
        was = self.expert.training
        self.expert.train()  # 打开 dropout
        mus, lvs = [], []
        for _ in range(self.expert_mc):
            mu, lv = self.expert(batch)
            mus.append(mu); lvs.append(lv)
        self.expert.train(was)
        mu_stack = torch.stack(mus, 0)
        lv_stack = torch.stack(lvs, 0)
        mu_mean = mu_stack.mean(0)
        aleatoric = lv_stack.exp().mean(0)
        epistemic = mu_stack.var(0, unbiased=False)
        return mu_mean, (aleatoric + epistemic).clamp_min(1e-6)

    @torch.no_grad()
    def _fuse(self, mu_m, s2_m, mu_e, s2_e, w_e, has_struct=None):
        if self.hard_gate_on_no_struct and has_struct is not None:
            w_e = w_e * has_struct.to(w_e.dtype)
        if self.use_precision_weighting:
            p_m = 1.0 / s2_m.clamp_min(1e-6)
            p_e = 1.0 / s2_e.clamp_min(1e-6)
            w_m_raw = (1.0 - w_e) * p_m
            w_e_raw = w_e * p_e
            z = (w_m_raw + w_e_raw).clamp_min(1e-8)
            w_e = w_e_raw / z
        w_m = 1.0 - w_e
        mu_ens = w_m * mu_m + w_e * mu_e
        s2_ens = w_m * (s2_m + mu_m ** 2) + w_e * (s2_e + mu_e ** 2) - mu_ens ** 2
        return mu_ens, s2_ens.clamp_min(1e-6), w_m, w_e

    @torch.no_grad()
    def predict(self, batch) -> dict:
        """batch 需包含 kcat_expert 的输入字段以及 'seqs'/'smiles' 列表。返回所有中间量。"""
        seqs = batch["seqs"]
        smiles = batch["smiles"]
        device = self.router.train_emb.device

        mu_m, s2_m = self._main_mc(seqs, smiles, device=device)
        mu_e, s2_e = self._expert_mc(batch)
        query = self._route_embed(seqs)
        w_e, ood = self.router.route(query)
        has_struct = batch.get("has_struct")
        mu_ens, s2_ens, w_m, w_e = self._fuse(mu_m, s2_m, mu_e, s2_e, w_e, has_struct)

        return {
            "mu_main": mu_m, "s2_main": s2_m,
            "mu_expert": mu_e, "s2_expert": s2_e,
            "ood_score": ood,
            "w_main": w_m, "w_expert": w_e,
            "mu_ensemble": mu_ens, "s2_ensemble": s2_ens,
        }



def load_train_embed_index(path: str | Path, device: str = "cpu") -> torch.Tensor:
    """加载预计算的 L2-normalized 训练集 embedding（.npy）。"""
    arr = np.load(str(path))
    t = torch.from_numpy(arr).to(device).float()
    return F.normalize(t, dim=-1)


def build_router_from_npy(
    npy_path: str | Path, d0: float = 0.15, tau: float = 0.05, k: int = 1,
    device: str = "cpu",
) -> OODRouter:
    emb = load_train_embed_index(npy_path, device=device)
    router = OODRouter(emb, d0=d0, tau=tau, k=k)
    return router.to(device)
