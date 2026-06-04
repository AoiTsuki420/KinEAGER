"""训练 kcat-OOD 专家模型（单 GPU / 小规模；DDP 版本可后续加）。

示例:
  python main_train_kcat_expert.py \
      --train_csv data/skid_kcat_train.csv \
      --val_csv   data/skid_kcat_val.csv \
      --out_dir   runs/kcat_expert_v1 \
      --batch_size 32 --epochs 40 --lr 3e-4 --lora_lr 3e-5
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import random
import time
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import pearsonr
from torch.utils.data import DataLoader

from models.encoders import ProteinEncoder, LigandEncoder
from models.kcat_expert import (
    KcatExpert, KcatExpertConfig, kcat_expert_loss,
)
from models.kcat_expert_dataset import (
    KcatExpertDataset, collate_kcat_expert, LIG_ATOM_FEAT_DIM,
)

DEFAULT_RENAME = {
    "Sequence": "sequence",
    "Substrate SMILES": "smiles",
    "pkcat_value": "kcat_log10",
    "Protein_path": "prot_pdb_path",
    "Ligand_path": "lig_sdf_path",
}


class _ProtWrap(nn.Module):
    """适配 KcatExpert 期望的 (ids, mask) -> [B,L,d] 签名。
    这里让 wrap.forward 接收原始 seqs 列表（从 batch["seqs"] 传）。
    """
    def __init__(self, encoder: ProteinEncoder):
        super().__init__()
        self.encoder = encoder
    def forward(self, seqs, _mask_unused=None):
        tok, mask = self.encoder(seqs)
        return tok, mask


class _LigWrap(nn.Module):
    def __init__(self, encoder: LigandEncoder):
        super().__init__()
        self.encoder = encoder
    def forward(self, smi, _mask_unused=None):
        tok, mask = self.encoder(smi)
        return tok, mask


class KcatExpertWithText(KcatExpert):
    """覆盖 encode：用 raw seqs/smiles 走 HF encoder，mask 也来自 encoder。"""

    def encode(self, batch):
        h_p_seq, prot_mask = self.esm(batch["seqs"])       # [B,L_p,d], [B,L_p]
        h_l_seq, lig_mask = self.molt5(batch["smiles"])    # [B,L_l,d], [B,L_l]

        h_p_struct_res = self.prot_struct(batch["prot_dist"], batch["prot_struct_mask"])
        L = h_p_seq.size(1)
        if h_p_struct_res.size(1) < L:
            h_p_struct_res = torch.nn.functional.pad(h_p_struct_res, (0, 0, 0, L - h_p_struct_res.size(1)))
        elif h_p_struct_res.size(1) > L:
            h_p_struct_res = h_p_struct_res[:, :L]

        h_l_atom = self.lig_struct(batch["lig_atom_feat"], batch["lig_atom_dist"], batch["lig_atom_mask"])
        lig_struct_vec = (h_l_atom * batch["lig_atom_mask"].unsqueeze(-1).float()).sum(1) / \
                         batch["lig_atom_mask"].float().sum(1, keepdim=True).clamp_min(1.0)
        h_l_struct_seq = lig_struct_vec.unsqueeze(1).expand(-1, h_l_seq.size(1), -1)

        has_struct = batch["has_struct"]
        H_p = self.prot_merge(h_p_seq, h_p_struct_res, has_struct)
        H_l = self.lig_merge(h_l_seq, h_l_struct_seq, has_struct)
        batch["_prot_mask"] = prot_mask
        batch["_lig_mask"] = lig_mask
        return H_p, H_l

    def forward(self, batch):
        H_p, H_l = self.encode(batch)
        mask_p = batch["_prot_mask"]; mask_l = batch["_lig_mask"]
        for blk in self.blocks:
            H_p, H_l = blk(H_p, H_l, mask_p, mask_l)
        pool_p = (H_p * mask_p.unsqueeze(-1).float()).sum(1) / mask_p.float().sum(1, keepdim=True).clamp_min(1.0)
        pool_l = (H_l * mask_l.unsqueeze(-1).float()).sum(1) / mask_l.float().sum(1, keepdim=True).clamp_min(1.0)
        feats = torch.cat([pool_p, pool_l, batch["geom_feats"]], dim=-1)
        out = self.head(self.dropout(feats))
        return out[..., 0], out[..., 1]


def move_batch(batch, device):
    for k, v in batch.items():
        if torch.is_tensor(v):
            batch[k] = v.to(device, non_blocking=True)
    return batch


def build_model(args, device):
    prot_enc = ProteinEncoder(esm_model=args.esm_path, d_model=1280, device=device, train_backbone=False)
    lig_enc = LigandEncoder(molt5_model=args.molt5_path, d_model=768, device=device, train_backbone=False)
    prot_enc.proj = nn.Identity().to(device)
    lig_enc.proj = nn.Identity().to(device)

    if getattr(args, "esm_lora_rank", 0) > 0:
        from peft import LoraConfig, get_peft_model
        lcfg = LoraConfig(
            r=args.esm_lora_rank, lora_alpha=args.esm_lora_alpha,
            target_modules=["query", "key", "value"],
            lora_dropout=0.05, bias="none",
        )
        prot_enc.backbone = get_peft_model(prot_enc.backbone, lcfg)
        prot_enc.backbone.train()
        for n, p in prot_enc.backbone.named_parameters():
            p.requires_grad_("lora_" in n)
        prot_enc.train_backbone = True
        trainable = sum(p.numel() for p in prot_enc.backbone.parameters() if p.requires_grad)
        print(f"[lora] ESM LoRA r={args.esm_lora_rank} trainable={trainable:,}")

    cfg = KcatExpertConfig(lig_atom_feat_dim=LIG_ATOM_FEAT_DIM)
    model = KcatExpertWithText(cfg, _ProtWrap(prot_enc), _LigWrap(lig_enc)).to(device)
    return model


def param_groups(model, lr, lora_lr):
    lora_params, other = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "lora_" in n:
            lora_params.append(p)
        else:
            other.append(p)
    groups = [{"params": other, "lr": lr}]
    if lora_params:
        groups.append({"params": lora_params, "lr": lora_lr})
    return groups


def load_init_checkpoint(model: nn.Module, ckpt_path: str, device: str, strict: bool = False):
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    missing, unexpected = model.load_state_dict(state, strict=strict)
    print(f"[init] loaded: {ckpt_path}")
    print(f"[init] strict={strict}  missing={len(missing)}  unexpected={len(unexpected)}")
    if missing:
        print(f"[init] missing keys (first 10): {missing[:10]}")
    if unexpected:
        print(f"[init] unexpected keys (first 10): {unexpected[:10]}")


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)

    def flush(self):
        for s in self.streams:
            s.flush()

    def isatty(self):
        return any(getattr(s, "isatty", lambda: False)() for s in self.streams)


def setup_logging(out_dir: Path):
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = out_dir / f"train_{ts}.log"
    latest_log = out_dir / "train_latest.log"

    fh = open(log_file, "a", encoding="utf-8", buffering=1)
    lf = open(latest_log, "a", encoding="utf-8", buffering=1)

    sys.stdout = _Tee(sys.__stdout__, fh, lf)
    sys.stderr = _Tee(sys.__stderr__, fh, lf)
    print(f"[log] writing to: {log_file}")
    print(f"[log] appending latest to: {latest_log}")


def append_epoch_metrics(csv_path: Path, row: dict):
    exists = csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    mus, ys = [], []
    for batch in loader:
        batch = move_batch(batch, device)
        mu, _ = model(batch)
        mus.append(mu.cpu().numpy())
        ys.append(batch["y_kcat"].cpu().numpy())
    mu = np.concatenate(mus); y = np.concatenate(ys)
    mse = float(((mu - y) ** 2).mean())
    r = float(pearsonr(mu, y)[0]) if mu.std() > 1e-6 else 0.0
    return {"mse": mse, "pearson_r": r, "pred_std": float(mu.std()), "y_std": float(y.std())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", required=True)
    ap.add_argument("--val_csv", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--esm_path", default="/root/autodl-fs/models/esm2_t33_650M_UR50D")
    ap.add_argument("--molt5_path", default="/root/autodl-fs/models/molt5-base-smiles2caption")
    ap.add_argument("--base_dir", default=None)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--lora_lr", type=float, default=3e-5)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--seed", type=int, default=42,
                    help="随机种子（DataLoader shuffle、numpy、torch）")
    ap.add_argument("--rank_weight", type=float, default=0.3)
    ap.add_argument("--var_reg_weight", type=float, default=0.1)
    ap.add_argument("--nll_weight", type=float, default=0.05)
    ap.add_argument("--ccc_weight", type=float, default=0.0,
                    help="CCC loss 权重：直接优化相关+方差匹配；推荐 0.5-1.0")
    ap.add_argument("--grad_clip", type=float, default=1.0,
                    help="梯度裁剪 max norm（默认 1.0；梯度爆炸严重时提到 5.0）")
    ap.add_argument("--grad_accum", type=int, default=1,
                    help="梯度累积步数；等效 batch = batch_size * grad_accum")
    ap.add_argument("--weight_decay", type=float, default=5e-4,
                    help="AdamW weight decay (旧默认 1e-4)")
    ap.add_argument("--randomize_smiles", action="store_true", default=True,
                    help="训练时随机 SMILES 同分异构排列增广（默认开）")
    ap.add_argument("--no_randomize_smiles", dest="randomize_smiles", action="store_false")
    ap.add_argument("--require_struct", action="store_true", default=False,
                    help="仅保留有 pdb+sdf 的样本（默认关闭，允许混合结构/无结构）")
    ap.add_argument("--seq_crop_min", type=float, default=1.0,
                    help="训练时蛋白序列随机 crop 的下界比例；1.0=关闭，0.8=保留 80-100%")
    ap.add_argument("--seq_mask_prob", type=float, default=0.0,
                    help="训练时残基随机替换为 X 的概率；0 关闭，0.05 较稳")
    ap.add_argument("--esm_lora_rank", type=int, default=16)
    ap.add_argument("--esm_lora_alpha", type=int, default=32)
    ap.add_argument("--log_every", type=int, default=50,
                    help="打印训练进度的 step 间隔")
    ap.add_argument("--rename_cols", type=str, default="default",
                    help='"default" 使用内置 SKiD_kcat 别名；"none" 不重命名；'
                         '或传 JSON 字符串如 \'{"Sequence":"sequence",...}\'')
    ap.add_argument("--eval_ood40_csv", type=str, default=None,
                    help="可选：每个 epoch 额外评估的 OOD40 测试集 CSV")
    ap.add_argument("--eval_ood60_csv", type=str, default=None,
                    help="可选：每个 epoch 额外评估的 OOD60 测试集 CSV")
    ap.add_argument("--init_ckpt", type=str, default=None,
                    help="初始化权重路径；支持 {'model': state_dict} 或纯 state_dict")
    ap.add_argument("--init_strict", action="store_true", default=False,
                    help="严格加载初始化权重（默认 False，允许部分键不匹配）")
    args = ap.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    setup_logging(out)
    set_seed(int(args.seed))
    print(f"[seed] seed={args.seed} cudnn.deterministic=True cudnn.benchmark=False")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.rename_cols == "default":
        rename_map = DEFAULT_RENAME
    elif args.rename_cols == "none":
        rename_map = None
    else:
        import json
        rename_map = json.loads(args.rename_cols)

    ds_tr = KcatExpertDataset(args.train_csv, base_dir=args.base_dir, rename_cols=rename_map,
                              randomize_smiles=bool(getattr(args, "randomize_smiles", True)),
                              require_struct=bool(getattr(args, "require_struct", False)),
                              seq_crop_min=float(getattr(args, "seq_crop_min", 1.0)),
                              seq_mask_prob=float(getattr(args, "seq_mask_prob", 0.0)))
    ds_va = KcatExpertDataset(args.val_csv, base_dir=args.base_dir, rename_cols=rename_map,
                              randomize_smiles=False,
                              require_struct=bool(getattr(args, "require_struct", False)))
    dl_tr = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True,
                       num_workers=args.num_workers, collate_fn=collate_kcat_expert, pin_memory=True)
    dl_va = DataLoader(ds_va, batch_size=args.batch_size, shuffle=False,
                       num_workers=args.num_workers, collate_fn=collate_kcat_expert, pin_memory=True)

    dl_ood40 = None
    if args.eval_ood40_csv:
        ds_ood40 = KcatExpertDataset(args.eval_ood40_csv, base_dir=args.base_dir, rename_cols=rename_map,
                                     randomize_smiles=False,
                                     require_struct=bool(getattr(args, "require_struct", False)))
        dl_ood40 = DataLoader(ds_ood40, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, collate_fn=collate_kcat_expert, pin_memory=True)

    dl_ood60 = None
    if args.eval_ood60_csv:
        ds_ood60 = KcatExpertDataset(args.eval_ood60_csv, base_dir=args.base_dir, rename_cols=rename_map,
                                     randomize_smiles=False,
                                     require_struct=bool(getattr(args, "require_struct", False)))
        dl_ood60 = DataLoader(ds_ood60, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, collate_fn=collate_kcat_expert, pin_memory=True)

    model = build_model(args, device)
    if args.init_ckpt:
        load_init_checkpoint(model, args.init_ckpt, device, strict=bool(args.init_strict))
    opt = torch.optim.AdamW(param_groups(model, args.lr, args.lora_lr),
                            weight_decay=float(args.weight_decay))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"[setup] device={device}  train_rows={len(ds_tr)}  val_rows={len(ds_va)}  "
        f"train_batches={len(dl_tr)}  val_batches={len(dl_va)}"
    )
    if dl_ood40 is not None:
        print(f"[setup] ood40_rows={len(dl_ood40.dataset)}  ood40_batches={len(dl_ood40)}")
    if dl_ood60 is not None:
        print(f"[setup] ood60_rows={len(dl_ood60.dataset)}  ood60_batches={len(dl_ood60)}")
    print(
        f"[setup] params total={total_params:,}  trainable={trainable_params:,}  "
        f"log_every={max(1, args.log_every)}"
    )

    best_r = -1e9
    best_ood40_r = -1e9
    best_ood60_r = -1e9
    bad_epochs = 0
    metrics_csv = out / "metrics.csv"
    for epoch in range(args.epochs):
        t0 = time.time()
        model.train()
        running = 0.0
        n_batches = 0
        running_logs = {"huber": 0.0, "rank": 0.0, "var_reg": 0.0, "nll": 0.0, "ccc": 0.0, "total": 0.0}
        steps_total = max(1, len(dl_tr))
        log_every = max(1, args.log_every)
        accum_steps = max(1, int(getattr(args, "grad_accum", 1)))
        opt.zero_grad(set_to_none=True)
        for step, batch in enumerate(dl_tr, start=1):
            batch = move_batch(batch, device)
            mu, lv = model(batch)
            loss, logs = kcat_expert_loss(
                mu, lv, batch["y_kcat"],
                rank_weight=args.rank_weight,
                var_reg_weight=args.var_reg_weight,
                nll_weight=args.nll_weight,
                ccc_weight=float(getattr(args, "ccc_weight", 0.0)),
            )
            if not torch.isfinite(loss):
                print(f"[health] non-finite loss at epoch={epoch+1} step={step}: {float(loss.detach())}")
                raise RuntimeError("non-finite loss detected")
            (loss / accum_steps).backward()
            if step % accum_steps == 0 or step == steps_total:
                grad_norm = float(torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    float(getattr(args, "grad_clip", 1.0))))
                opt.step()
                opt.zero_grad(set_to_none=True)
            else:
                grad_norm = 0.0  # 累积中，不显示
            running += float(loss.detach())
            n_batches += 1
            for k in running_logs:
                if k in logs:
                    running_logs[k] += float(logs[k])

            if step == 1 or step % log_every == 0 or step == steps_total:
                lr_main = opt.param_groups[0]["lr"]
                lr_lora = opt.param_groups[1]["lr"] if len(opt.param_groups) > 1 else lr_main
                denom = float(step)
                print(
                    f"[train] epoch {epoch+1}/{args.epochs} step {step}/{steps_total} "
                    f"loss={running/denom:.4f} huber={running_logs['huber']/denom:.4f} "
                    f"rank={running_logs['rank']/denom:.4f} var={running_logs['var_reg']/denom:.4f} "
                    f"nll={running_logs['nll']/denom:.4f} ccc={running_logs['ccc']/denom:.4f} grad_norm={grad_norm:.3f} "
                    f"lr={lr_main:.2e} lora_lr={lr_lora:.2e}"
                )
        sched.step()
        metrics = evaluate(model, dl_va, device)
        metrics_ood40 = evaluate(model, dl_ood40, device) if dl_ood40 is not None else None
        metrics_ood60 = evaluate(model, dl_ood60, device) if dl_ood60 is not None else None
        epoch_sec = time.time() - t0
        print(f"[epoch] {epoch+1}/{args.epochs}  train_loss={running / max(1, n_batches):.4f}  "
              f"val_mse={metrics['mse']:.4f}  val_r={metrics['pearson_r']:.4f}  "
              f"pred_std={metrics['pred_std']:.3f} y_std={metrics['y_std']:.3f}  "
              f"time={epoch_sec:.1f}s")
        if metrics_ood40 is not None:
            print(f"[ood40] epoch={epoch+1}  r={metrics_ood40['pearson_r']:.4f}  "
                  f"mse={metrics_ood40['mse']:.4f}  pred_std={metrics_ood40['pred_std']:.3f}  "
                  f"y_std={metrics_ood40['y_std']:.3f}")
        if metrics_ood60 is not None:
            print(f"[ood60] epoch={epoch+1}  r={metrics_ood60['pearson_r']:.4f}  "
                  f"mse={metrics_ood60['mse']:.4f}  pred_std={metrics_ood60['pred_std']:.3f}  "
                  f"y_std={metrics_ood60['y_std']:.3f}")

        row = {
            "epoch": int(epoch + 1),
            "train_loss": float(running / max(1, n_batches)),
            "val_mse": float(metrics["mse"]),
            "val_r": float(metrics["pearson_r"]),
            "val_pred_std": float(metrics["pred_std"]),
            "val_y_std": float(metrics["y_std"]),
            "ood40_mse": float(metrics_ood40["mse"]) if metrics_ood40 is not None else "",
            "ood40_r": float(metrics_ood40["pearson_r"]) if metrics_ood40 is not None else "",
            "ood60_mse": float(metrics_ood60["mse"]) if metrics_ood60 is not None else "",
            "ood60_r": float(metrics_ood60["pearson_r"]) if metrics_ood60 is not None else "",
            "lr_main": float(opt.param_groups[0]["lr"]),
            "lr_lora": float(opt.param_groups[1]["lr"]) if len(opt.param_groups) > 1 else float(opt.param_groups[0]["lr"]),
            "epoch_time_sec": float(epoch_sec),
        }
        append_epoch_metrics(metrics_csv, row)

        if metrics["pearson_r"] > best_r + 1e-4:
            best_r = metrics["pearson_r"]; bad_epochs = 0
            torch.save({"model": model.state_dict(), "cfg": vars(args), "metrics": metrics},
                       out / "best.pt")
            print(f"[checkpoint] updated best.pt with val_r={best_r:.4f}")
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"early stop at epoch {epoch}, best r={best_r:.4f}")
                break

        if metrics_ood40 is not None and metrics_ood40["pearson_r"] > best_ood40_r + 1e-4:
            best_ood40_r = metrics_ood40["pearson_r"]
            torch.save({
                "model": model.state_dict(),
                "cfg": vars(args),
                "metrics": metrics,
                "metrics_ood40": metrics_ood40,
                "metrics_ood60": metrics_ood60,
            }, out / "best_ood40.pt")
            print(f"[checkpoint] updated best_ood40.pt with ood40_r={best_ood40_r:.4f}")

        if metrics_ood60 is not None and metrics_ood60["pearson_r"] > best_ood60_r + 1e-4:
            best_ood60_r = metrics_ood60["pearson_r"]
            torch.save({
                "model": model.state_dict(),
                "cfg": vars(args),
                "metrics": metrics,
                "metrics_ood40": metrics_ood40,
                "metrics_ood60": metrics_ood60,
            }, out / "best_ood60.pt")
            print(f"[checkpoint] updated best_ood60.pt with ood60_r={best_ood60_r:.4f}")


if __name__ == "__main__":
    main()
