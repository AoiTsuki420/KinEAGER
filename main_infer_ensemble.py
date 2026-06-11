"""KinEAGER 推理: main predictor + kcat expert + OOD 路由。

默认路由:
  OOD score = 1 - max_cosine( ESM(query_seq), train_index )
  w_expert  = sigmoid((ood - d0) / tau)
  mu_final  = (1-w_expert)*mu_main + w_expert*mu_expert
  另: has_struct=False 的样本强制 w_expert=0（expert 没有结构则与 main 无异）。

用法:
  python main_infer_ensemble.py \
      --test_csv  data/my_ood.csv \
      --main_ckpt runs/main_predictor/best.pt \
      --expert_ckpt runs/kcat_expert_v7/best.pt \
      --train_emb_npy runs/moe_index/train_emb.npy \
      --out_csv   result/moe_preds.csv \
      --router_d0 0.15 --router_tau 0.05 --router_k 1 \
      --mc_samples 5

输出 CSV 列:
  y_kcat, mu_main, s2_main, mu_expert, s2_expert, ood_score,
  w_main, w_expert, has_struct, mu_ensemble, s2_ensemble
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import pearsonr, spearmanr
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from models.encoders import ProteinEncoder, LigandEncoder
from models.kcat_expert import KcatExpertConfig
from models.kcat_expert_dataset import (
    KcatExpertDataset, collate_kcat_expert, LIG_ATOM_FEAT_DIM,
)
from models.moe_kcat import KinEAGER, build_router_from_npy
from main_train_kcat_expert import KcatExpertWithText, _ProtWrap, _LigWrap, DEFAULT_RENAME



def load_expert(ckpt_path: str, esm_path: str, molt5_path: str, device: str):
    prot_enc = ProteinEncoder(esm_model=esm_path, d_model=1280, device=device, train_backbone=False)
    lig_enc = LigandEncoder(molt5_model=molt5_path, d_model=768, device=device, train_backbone=False)
    prot_enc.proj = nn.Identity().to(device)
    lig_enc.proj = nn.Identity().to(device)

    ckpt = torch.load(ckpt_path, map_location=device)
    cfg_args = ckpt.get("cfg", {}) or {}
    lora_rank = int(cfg_args.get("esm_lora_rank", 0) or 0)
    if lora_rank > 0:
        from peft import LoraConfig, get_peft_model
        lcfg = LoraConfig(
            r=lora_rank, lora_alpha=int(cfg_args.get("esm_lora_alpha", 32)),
            target_modules=["query", "key", "value"], lora_dropout=0.05, bias="none",
        )
        prot_enc.backbone = get_peft_model(prot_enc.backbone, lcfg)

    cfg = KcatExpertConfig(lig_atom_feat_dim=LIG_ATOM_FEAT_DIM)
    model = KcatExpertWithText(cfg, _ProtWrap(prot_enc), _LigWrap(lig_enc)).to(device)
    missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
    if missing:
        print(f"[expert] missing keys (first 5): {missing[:5]}")
    if unexpected:
        print(f"[expert] unexpected keys (first 5): {unexpected[:5]}")
    return model, prot_enc  # prot_enc 之后给 router 当 query encoder 用（共享 backbone）


def load_main_model(ckpt_path: str, device: str,
                    esm_model: str = "facebook/esm2_t12_35M_UR50D",
                    molt5_model: str = "laituan245/molt5-base-smiles2caption",
                    disable_source_residual: bool = False):
    from main_infer_predictor import build_model_from_ckpt, enable_mc_dropout
    model, _cfg, _ckpt = build_model_from_ckpt(
        ckpt_path, device=device, esm_model=esm_model, molt5_model=molt5_model,
    )
    model.to(device)
    if disable_source_residual and getattr(model, "kcat_source_residual", None) is not None:
        print("[main] disabling kcat_source_residual for ensemble eval")
        model.kcat_source_residual = None
    enable_mc_dropout(model)
    return model



def compute_metrics(mu: np.ndarray, y: np.ndarray, tag: str) -> dict:
    if mu.shape[0] == 0:
        return {f"{tag}_n": 0}
    resid = mu - y
    mse = float((resid ** 2).mean())
    mae = float(np.abs(resid).mean())
    y_var = float(y.var()) if y.size > 1 else 0.0
    r2_det = 1.0 - mse / y_var if y_var > 1e-8 else 0.0
    r = float(pearsonr(mu, y)[0]) if mu.std() > 1e-6 else 0.0
    rho = float(spearmanr(mu, y)[0]) if mu.std() > 1e-6 else 0.0
    return {f"{tag}_n": int(y.size), f"{tag}_mse": mse, f"{tag}_mae": mae,
            f"{tag}_r": r, f"{tag}_r2": r * r, f"{tag}_R2_det": r2_det, f"{tag}_spearman": rho}


def print_metrics(m: dict, title: str):
    print(f"\n=== {title} ===")
    for k, v in m.items():
        print(f"  {k:30s} {v:.4f}" if isinstance(v, float) else f"  {k:30s} {v}")



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_csv", required=True)
    ap.add_argument("--main_ckpt", required=True)
    ap.add_argument("--expert_ckpt", required=True)
    ap.add_argument("--train_emb_npy", required=True,
                    help="由 tools/precompute_train_embed.py 产出的 .npy")
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--esm_path", default="/root/autodl-fs/models/esm2_t33_650M_UR50D",
                    help="expert / router 共用的 ESM2 路径（与索引一致，建议 650M）")
    ap.add_argument("--molt5_path", default="/root/autodl-fs/models/molt5-base-smiles2caption")
    ap.add_argument("--main_esm", default="facebook/esm2_t12_35M_UR50D")
    ap.add_argument("--main_molt5", default="laituan245/molt5-base-smiles2caption")
    ap.add_argument("--source_id", type=int, default=0)
    ap.add_argument("--disable_source_residual", action="store_true", default=False)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--mc_samples", type=int, default=5)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--base_dir", default=None)
    ap.add_argument("--rename_cols", type=str, default="default")
    ap.add_argument("--router_d0", type=float, default=0.15,
                    help="OOD 阈值；距离 > d0 时偏向 expert")
    ap.add_argument("--router_tau", type=float, default=0.05,
                    help="sigmoid 温度；越小越接近硬阈值")
    ap.add_argument("--router_k", type=int, default=1,
                    help="top-k 最近邻平均；k=1 是纯最近邻")
    ap.add_argument("--no_hard_gate", action="store_true",
                    help="不对无结构样本强制 w_expert=0")
    ap.add_argument("--use_precision_weighting", action="store_true",
                    help="用 1/sigma^2 再对软权重做精度加权")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_path = Path(args.out_csv); out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.rename_cols == "default":
        rename = DEFAULT_RENAME
    elif args.rename_cols == "none":
        rename = None
    else:
        import json
        rename = json.loads(args.rename_cols)

    ds = KcatExpertDataset(args.test_csv, base_dir=args.base_dir,
                           rename_cols=rename, require_struct=False)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                    num_workers=args.num_workers, collate_fn=collate_kcat_expert)

    expert, prot_enc = load_expert(args.expert_ckpt, args.esm_path, args.molt5_path, device)
    expert.eval()
    main_model = load_main_model(
        args.main_ckpt, device,
        esm_model=args.main_esm, molt5_model=args.main_molt5,
        disable_source_residual=args.disable_source_residual,
    )

    router = build_router_from_npy(
        args.train_emb_npy, d0=args.router_d0, tau=args.router_tau, k=args.router_k,
        device=device,
    )
    print(f"[router] index size={router.train_emb.shape}  d0={router.d0} tau={router.tau} k={router.k}")

    moe = KinEAGER(
        main_model=main_model,
        expert_model=expert,
        router=router,
        query_encoder=prot_enc,
        main_mc_samples=args.mc_samples,
        expert_mc_samples=args.mc_samples,
        hard_gate_on_no_struct=(not args.no_hard_gate),
        use_precision_weighting=args.use_precision_weighting,
        source_id=args.source_id,
    ).to(device)

    mu_m_all, s2_m_all = [], []
    mu_e_all, s2_e_all = [], []
    ood_all, w_e_all, mu_ens_all, s2_ens_all = [], [], [], []
    has_all, y_all = [], []

    pbar = tqdm(dl, total=len(dl), desc=f"MoE infer ({Path(args.test_csv).stem})",
                dynamic_ncols=True)
    for batch in pbar:
        for k, v in batch.items():
            if torch.is_tensor(v):
                batch[k] = v.to(device, non_blocking=True)
        out = moe.predict(batch)

        mu_m_all.append(out["mu_main"].cpu().numpy())
        s2_m_all.append(out["s2_main"].cpu().numpy())
        mu_e_all.append(out["mu_expert"].cpu().numpy())
        s2_e_all.append(out["s2_expert"].cpu().numpy())
        ood_all.append(out["ood_score"].cpu().numpy())
        w_e_all.append(out["w_expert"].cpu().numpy())
        mu_ens_all.append(out["mu_ensemble"].cpu().numpy())
        s2_ens_all.append(out["s2_ensemble"].cpu().numpy())
        has_all.append(batch["has_struct"].cpu().numpy().astype(bool))
        y_all.append(batch["y_kcat"].cpu().numpy())
        pbar.set_postfix(w_e=f"{float(out['w_expert'].mean()):.2f}",
                         ood=f"{float(out['ood_score'].mean()):.2f}")

    mu_m = np.concatenate(mu_m_all); s2_m = np.concatenate(s2_m_all)
    mu_e = np.concatenate(mu_e_all); s2_e = np.concatenate(s2_e_all)
    ood = np.concatenate(ood_all); w_e = np.concatenate(w_e_all)
    mu_ens = np.concatenate(mu_ens_all); s2_ens = np.concatenate(s2_ens_all)
    has = np.concatenate(has_all); y = np.concatenate(y_all)
    w_m = 1.0 - w_e

    df = pd.DataFrame({
        "y_kcat": y,
        "mu_main": mu_m, "s2_main": s2_m,
        "mu_expert": mu_e, "s2_expert": s2_e,
        "ood_score": ood,
        "w_main": w_m, "w_expert": w_e,
        "has_struct": has,
        "mu_ensemble": mu_ens, "s2_ensemble": s2_ens,
    })
    df.to_csv(out_path, index=False)
    print(f"\n[saved] {out_path}  rows={len(df)}  "
          f"has_struct_frac={has.mean():.3f}  mean_w_expert={w_e.mean():.3f}  mean_ood={ood.mean():.3f}")

    print_metrics(compute_metrics(mu_m, y, "main_full"), "Main — full set")
    print_metrics(compute_metrics(mu_e, y, "expert_full"), "Expert — full set")
    print_metrics(compute_metrics(mu_ens, y, "moe_full"), "KinEAGER — full set")
    if has.any():
        print_metrics(compute_metrics(mu_m[has], y[has], "main_sub"),
                      "Main — has_struct subset only")
        print_metrics(compute_metrics(mu_e[has], y[has], "expert_sub"),
                      "Expert — has_struct subset only")
        print_metrics(compute_metrics(mu_ens[has], y[has], "moe_sub"),
                      "KinEAGER — has_struct subset only")


if __name__ == "__main__":
    main()
