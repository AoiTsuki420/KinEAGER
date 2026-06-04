import argparse, torch
from torch.cuda.amp import GradScaler
from torch.optim import AdamW
import numpy as np
import pandas as pd
try:
    from models.predictor import KineticsPredictor, PredictorConfig
    _PREDICTOR_IMPORT_ERROR = None
except Exception as _e:
    KineticsPredictor = None
    PredictorConfig = None
    _PREDICTOR_IMPORT_ERROR = _e
from data import make_dataloader
from utils import set_seed, KineticsScaler, PredictionRecorder, KineticsScalerNoClip,resume_checkpoint_if_available,save_training_checkpoint, _clear_rotary_cache, module_sanity_check,unpack_batch
from tqdm import tqdm
from sklearn.model_selection import train_test_split, KFold, GroupShuffleSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from torch.optim.lr_scheduler import CosineAnnealingLR
import json
from copy import deepcopy
import math
import os
from pathlib import Path
try:
    import swanlab
except Exception:
    class _SwanlabStub:
        @staticmethod
        def init(*args, **kwargs):
            return None

        @staticmethod
        def log(*args, **kwargs):
            return None

    swanlab = _SwanlabStub()
import gc
try:
    from transformers import AutoConfig
except Exception:
    AutoConfig = None
try:
    from peft import LoraConfig, get_peft_model
except Exception:
    LoraConfig = None
    get_peft_model = None
from torch.optim.lr_scheduler import LambdaLR 
from tracing import *
from path_config import resolve_runtime_paths, ensure_paths
import os, math, time, json, random
import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from data import EnzymeDataset
import torch.nn.functional as F
import torch.distributed as dist
from torch.optim.lr_scheduler import OneCycleLR
from torch import amp
from final_contact import load_final_contact_npz,FinalContactAblationConfig
import copy


RUNTIME_PATHS = {}


def require_path(key):
    path = RUNTIME_PATHS.get(key)
    if path is None:
        raise RuntimeError(f"Runtime path '{key}' is not configured")
    return path


DEFAULT_SOURCE_ALIASES = {
    "catapro": 0,
    "cataprodata": 0,
    "catpred-kcat": 0,
    "catpred_kcat": 0,
    "catpredkcat": 0,
    "skid-kcat": 1,
    "skid_kcat": 1,
    "skidkcat": 1,
    "catpred-km": 1,
    "catpred_km": 1,
    "catpredkm": 1,
    "skid-km": 2,
    "skid_km": 2,
    "skidkm": 2,
}


def infer_source_ids(df, source_col: str | None = None, default_source_id: int = 0):
    if source_col and source_col in df.columns:
        src = df[source_col]
    else:
        src = None
        for cand in ["source_id", "source", "dataset", "domain", "data_source"]:
            if cand in df.columns:
                src = df[cand]
                source_col = cand
                break

    if src is None:
        arr = np.full((len(df),), int(default_source_id), dtype=np.int64)
        return arr, None

    if np.issubdtype(src.dtype, np.number):
        arr = src.to_numpy().astype(np.int64)
    else:
        out = []
        for v in src.astype(str).to_list():
            key = v.strip().lower().replace(" ", "").replace("_", "-")
            out.append(int(DEFAULT_SOURCE_ALIASES.get(key, default_source_id)))
        arr = np.asarray(out, dtype=np.int64)
    return arr, source_col


def to_source_tensor(source_ids, batch_size: int, device):
    if source_ids is None:
        return torch.zeros(batch_size, device=device, dtype=torch.long)
    if isinstance(source_ids, torch.Tensor):
        return source_ids.to(device=device, dtype=torch.long).reshape(-1)
    return torch.as_tensor(source_ids, device=device, dtype=torch.long).reshape(-1)


def infer_struct_avail_mask(p_s, l_s, batch_size: int, device, provided_mask=None):
    if provided_mask is not None:
        if isinstance(provided_mask, torch.Tensor):
            return provided_mask.to(device=device, dtype=torch.float32).reshape(-1)
        return torch.as_tensor(provided_mask, device=device, dtype=torch.float32).reshape(-1)

    if (p_s is None) or (l_s is None):
        return torch.zeros(batch_size, device=device, dtype=torch.float32)

    p_ok = (p_s.abs().sum(dim=-1) > 0)
    l_ok = (l_s.abs().sum(dim=-1) > 0)
    return (p_ok & l_ok).to(dtype=torch.float32)


def detect_phys_columns(df):
    return sorted([c for c in df.columns if c.startswith("phys_") and c not in {"phys_mask", "phys_quality"}])


def infer_phys_inputs(phys_feat, phys_mask, phys_quality, batch_size: int, device):
    if phys_feat is None:
        z = torch.zeros(batch_size, device=device, dtype=torch.float32)
        return None, z, z

    if isinstance(phys_feat, torch.Tensor):
        feat = phys_feat.to(device=device, dtype=torch.float32)
    else:
        feat = torch.as_tensor(phys_feat, device=device, dtype=torch.float32)

    if phys_mask is None:
        pm = (feat.abs().sum(dim=-1) > 0).to(dtype=torch.float32)
    elif isinstance(phys_mask, torch.Tensor):
        pm = phys_mask.to(device=device, dtype=torch.float32).reshape(-1)
    else:
        pm = torch.as_tensor(phys_mask, device=device, dtype=torch.float32).reshape(-1)

    if phys_quality is None:
        pq = pm.clone()
    elif isinstance(phys_quality, torch.Tensor):
        pq = phys_quality.to(device=device, dtype=torch.float32).reshape(-1)
    else:
        pq = torch.as_tensor(phys_quality, device=device, dtype=torch.float32).reshape(-1)

    pm = torch.clamp(pm, min=0.0, max=1.0)
    pq = torch.clamp(pq, min=0.0, max=1.0)
    return feat, pm, pq


def apply_phys_ablation(phys_feat, phys_mask, phys_quality, mode: str):
    if phys_feat is None:
        return phys_feat, phys_mask, phys_quality
    mode = (mode or "real").lower()
    if mode == "real":
        return phys_feat, phys_mask, phys_quality
    if mode == "none":
        z = torch.zeros_like(phys_mask)
        return torch.zeros_like(phys_feat), z, z
    if mode == "shuffle":
        idx = torch.randperm(phys_feat.size(0), device=phys_feat.device)
        return phys_feat[idx], phys_mask, phys_quality
    if mode == "random":
        mu = phys_feat.mean(dim=0, keepdim=True)
        std = phys_feat.std(dim=0, keepdim=True).clamp(min=1e-6)
        rnd = torch.randn_like(phys_feat) * std + mu
        return rnd, phys_mask, phys_quality
    raise ValueError(f"Unknown phys_ablation mode: {mode}")


def normalize_training_dataframe(df):
    """
    统一训练所需列名，并容忍单任务数据集。
    产出至少包含：Sequence, Smiles, kcat(s^-1), Km(M)
    """
    out = df.copy()

    if "Sequence" not in out.columns:
        for c in ["sequence", "Protein Sequence"]:
            if c in out.columns:
                out["Sequence"] = out[c]
                break

    if "Smiles" not in out.columns:
        for c in ["SMILES", "Substrate SMILES", "Substrate_SMILES"]:
            if c in out.columns:
                out["Smiles"] = out[c]
                break

    if "kcat(s^-1)" not in out.columns:
        for c in ["kcat", "kcat_value"]:
            if c in out.columns:
                out["kcat(s^-1)"] = out[c]
                break

    if "Km(M)" not in out.columns:
        for c in ["Km", "Km_value"]:
            if c in out.columns:
                out["Km(M)"] = out[c]
                break

    if "kcat(s^-1)" not in out.columns:
        out["kcat(s^-1)"] = np.nan
    if "Km(M)" not in out.columns:
        out["Km(M)"] = np.nan

    out["kcat(s^-1)"] = pd.to_numeric(out["kcat(s^-1)"], errors="coerce")
    out["Km(M)"] = pd.to_numeric(out["Km(M)"], errors="coerce")

    if "Sequence" not in out.columns:
        raise ValueError("Missing sequence column after normalization")
    if "Smiles" not in out.columns:
        raise ValueError("Missing smiles column after normalization")

    return out


def _safe_masked_mean(vec, mask, device, dtype):
    if mask.any():
        return vec[mask].mean()
    return torch.zeros((), device=device, dtype=dtype)


def _barron_loss_per_sample(pred, target, alpha: float = 1.0, scale: float = 1.0):
    """Barron robust loss (CVPR 2019), returns per-sample loss."""
    eps = 1e-8
    c = max(float(scale), eps)
    a = float(alpha)
    r2 = ((pred - target) / c).pow(2)

    if abs(a - 2.0) < 1e-6:
        return 0.5 * r2
    if abs(a) < 1e-6:
        return torch.log1p(0.5 * r2)

    beta = abs(a - 2.0)
    return (beta / a) * (torch.pow(r2 / beta + 1.0, a / 2.0) - 1.0)


def _pearson_r_safe(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.size < 2:
        return float("nan")
    if np.std(y_true) < 1e-12 or np.std(y_pred) < 1e-12:
        return float("nan")
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def _spearman_r_safe(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.size < 2:
        return float("nan")
    yr = pd.Series(y_true).rank(method="average").to_numpy()
    pr = pd.Series(y_pred).rank(method="average").to_numpy()
    return _pearson_r_safe(yr, pr)


def _compute_task_metrics_with_sigma(y_true, y_pred, sigma, task_name):
    valid = np.isfinite(y_true)
    if sigma is not None:
        valid = valid & np.isfinite(sigma) & (sigma > 0)

    if valid.sum() == 0:
        return {
            f"{task_name}_mse": float("nan"),
            f"{task_name}_rmse": float("nan"),
            f"{task_name}_mae": float("nan"),
            f"{task_name}_r2": float("nan"),
            f"{task_name}_pearson_r": float("nan"),
            f"{task_name}_spearman_r": float("nan"),
            f"{task_name}_nll": float("nan"),
            f"{task_name}_cov90": float("nan"),
            f"{task_name}_cov95": float("nan"),
            f"{task_name}_n": 0,
        }

    yt = y_true[valid]
    yp = y_pred[valid]
    mse = float(mean_squared_error(yt, yp))
    rmse = float(np.sqrt(mse))
    mae = float(mean_absolute_error(yt, yp))
    r2 = float(r2_score(yt, yp)) if yt.size > 1 else float("nan")
    pr = _pearson_r_safe(yt, yp)
    sr = _spearman_r_safe(yt, yp)

    out = {
        f"{task_name}_mse": mse,
        f"{task_name}_rmse": rmse,
        f"{task_name}_mae": mae,
        f"{task_name}_r2": r2,
        f"{task_name}_pearson_r": pr,
        f"{task_name}_spearman_r": sr,
        f"{task_name}_n": int(valid.sum()),
    }

    if sigma is not None:
        sg = np.maximum(sigma[valid], 1e-6)
        err = yp - yt
        nll = 0.5 * ((err * err) / (sg * sg) + 2.0 * np.log(sg) + np.log(2.0 * np.pi))
        out[f"{task_name}_nll"] = float(np.mean(nll))
        out[f"{task_name}_cov90"] = float(np.mean(np.abs(err) <= (1.6448536269514722 * sg)))
        out[f"{task_name}_cov95"] = float(np.mean(np.abs(err) <= (1.959963984540054 * sg)))
    else:
        out[f"{task_name}_nll"] = float("nan")
        out[f"{task_name}_cov90"] = float("nan")
        out[f"{task_name}_cov95"] = float("nan")

    return out


def _add_macro_metrics(metric_dict, task_names):
    keys = ["mse", "rmse", "mae", "r2", "pearson_r", "spearman_r", "nll", "cov90", "cov95"]
    for k in keys:
        vals = []
        for t in task_names:
            v = metric_dict.get(f"{t}_{k}", float("nan"))
            if np.isfinite(v):
                vals.append(float(v))
        metric_dict[f"macro_{k}"] = float(np.mean(vals)) if vals else float("nan")


def _source_dist(df, source_col: str):
    if source_col in df.columns:
        return df[source_col].astype(str).value_counts(dropna=False).to_dict()
    return {}


def _canonical_pair_keys(df: pd.DataFrame) -> pd.Series:
    if "Sequence" not in df.columns or "Smiles" not in df.columns:
        raise ValueError("Pair split requires normalized columns: Sequence and Smiles")
    seq = df["Sequence"].astype(str).str.replace(r"\s+", "", regex=True).str.upper()
    smi = df["Smiles"].astype(str).str.strip()
    return seq + "||" + smi


def _count_overlap(a: pd.Series, b: pd.Series) -> int:
    return int(len(set(a.tolist()).intersection(set(b.tolist()))))


def _group_split_indices(n_rows: int, groups: np.ndarray, test_size: float, seed: int):
    n_groups = int(np.unique(groups).size)
    if n_groups < 2:
        raise ValueError(
            f"group_pair split requires at least 2 unique pairs, got {n_groups}. "
            "Check duplicate-heavy input or switch to --split_mode random for smoke runs."
        )
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    all_idx = np.arange(n_rows)
    train_idx, test_idx = next(splitter.split(all_idx, groups=groups))
    return train_idx, test_idx


def split_dataframe_for_experiment(
    df: pd.DataFrame,
    split_mode: str,
    split_seed: int,
    val_ratio: float,
    test_ratio: float,
    source_col: str,
    holdout_source: str | None,
    ood_dedup_pair: bool = True,
):
    if split_mode == "domain_ood":
        if not holdout_source:
            raise ValueError("split_mode=domain_ood requires --holdout_source")
        if source_col not in df.columns:
            raise ValueError(f"domain_ood split requires source column '{source_col}' in csv")

        test_df = df[df[source_col].astype(str) == str(holdout_source)].copy()
        train_pool = df[df[source_col].astype(str) != str(holdout_source)].copy()
        if len(test_df) == 0:
            raise ValueError(f"No samples found for holdout source '{holdout_source}'")

        removed_overlap_rows = 0
        if ood_dedup_pair:
            test_pairs = set(_canonical_pair_keys(test_df).tolist())
            pool_pairs = _canonical_pair_keys(train_pool)
            keep_mask = ~pool_pairs.isin(test_pairs)
            removed_overlap_rows = int((~keep_mask).sum())
            train_pool = train_pool.loc[keep_mask].copy()
            if len(train_pool) == 0:
                raise ValueError("All train_pool rows were removed by ood_dedup_pair; cannot split")

        if val_ratio <= 0 or val_ratio >= 1:
            raise ValueError("For domain_ood, val_ratio must be in (0,1)")

        train_df, val_df = train_test_split(train_pool, test_size=val_ratio, random_state=split_seed)
        meta = {
            "split_mode": split_mode,
            "split_seed": split_seed,
            "holdout_source": holdout_source,
            "source_col": source_col,
            "ood_dedup_pair": bool(ood_dedup_pair),
            "ood_dedup_removed_rows": int(removed_overlap_rows),
            "counts": {
                "train": int(len(train_df)),
                "val": int(len(val_df)),
                "test": int(len(test_df)),
            },
            "source_dist": {
                "train": _source_dist(train_df, source_col),
                "val": _source_dist(val_df, source_col),
                "test": _source_dist(test_df, source_col),
            },
        }
        tr_pairs = _canonical_pair_keys(train_df)
        va_pairs = _canonical_pair_keys(val_df)
        te_pairs = _canonical_pair_keys(test_df)
        meta["pair_overlap"] = {
            "train_test": _count_overlap(tr_pairs, te_pairs),
            "val_test": _count_overlap(va_pairs, te_pairs),
            "train_val": _count_overlap(tr_pairs, va_pairs),
        }
        return train_df, val_df, test_df, meta

    if split_mode not in {"random", "group_pair"}:
        raise ValueError(f"Unknown split_mode: {split_mode}")

    if not (0 < test_ratio < 1):
        raise ValueError("test_ratio must be in (0,1)")
    if not (0 < val_ratio < 1):
        raise ValueError("val_ratio must be in (0,1)")
    if val_ratio + test_ratio >= 1:
        raise ValueError("val_ratio + test_ratio must be < 1")

    if split_mode == "group_pair":
        pair_keys = _canonical_pair_keys(df)
        pair_arr = pair_keys.to_numpy()
        trv_idx, te_idx = _group_split_indices(len(df), pair_arr, test_size=test_ratio, seed=split_seed)
        train_val_df = df.iloc[trv_idx].copy()
        test_df = df.iloc[te_idx].copy()

        rel_val = val_ratio / max(1e-12, (1.0 - test_ratio))
        trv_pairs = pair_keys.iloc[trv_idx].to_numpy()
        tr_idx_local, va_idx_local = _group_split_indices(len(train_val_df), trv_pairs, test_size=rel_val, seed=split_seed)
        train_df = train_val_df.iloc[tr_idx_local].copy()
        val_df = train_val_df.iloc[va_idx_local].copy()
    else:
        train_val_df, test_df = train_test_split(df, test_size=test_ratio, random_state=split_seed)
        rel_val = val_ratio / max(1e-12, (1.0 - test_ratio))
        train_df, val_df = train_test_split(train_val_df, test_size=rel_val, random_state=split_seed)

    meta = {
        "split_mode": split_mode,
        "split_seed": split_seed,
        "val_ratio": val_ratio,
        "test_ratio": test_ratio,
        "source_col": source_col,
        "counts": {
            "train": int(len(train_df)),
            "val": int(len(val_df)),
            "test": int(len(test_df)),
        },
        "source_dist": {
            "train": _source_dist(train_df, source_col),
            "val": _source_dist(val_df, source_col),
            "test": _source_dist(test_df, source_col),
        },
    }
    if split_mode == "group_pair":
        tr_pairs = _canonical_pair_keys(train_df)
        va_pairs = _canonical_pair_keys(val_df)
        te_pairs = _canonical_pair_keys(test_df)
        meta["pair_overlap"] = {
            "train_test": _count_overlap(tr_pairs, te_pairs),
            "val_test": _count_overlap(va_pairs, te_pairs),
            "train_val": _count_overlap(tr_pairs, va_pairs),
        }
        meta["unique_pairs"] = {
            "train": int(tr_pairs.nunique()),
            "val": int(va_pairs.nunique()),
            "test": int(te_pairs.nunique()),
            "all": int(_canonical_pair_keys(df).nunique()),
        }
    return train_df, val_df, test_df, meta





def setup_ddp():
    """Init for torchrun: env vars LOCAL_RANK, RANK, WORLD_SIZE are set by torchrun."""
    if "RANK" not in os.environ:
        raise RuntimeError("This script must be launched with torchrun.")
    dist.init_process_group(backend="nccl", init_method="env://")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank

def cleanup_ddp():
    dist.barrier()
    dist.destroy_process_group()

def is_main_process():
    return int(os.environ.get("RANK", "0")) == 0


    
def cosine_anneal(epoch_idx: int, warm_epochs: int = 1,
                  hi: float = 0.25, lo: float = 0.02, decay_epochs: int = 8) -> float:
    """
    余弦退火的一致性权重：
    - 前 warm_epochs 个 epoch 固定为 hi
    - 之后从 hi 平滑下降到 lo
    epoch_idx 从 0 开始计（注意 train_epoch 里用 ep-1 传入）
    """
    if epoch_idx < warm_epochs:
        return hi
    t = (epoch_idx - warm_epochs + 1) / max(1, decay_epochs)
    t = min(max(t, 0.0), 1.0)  # clamp 到 [0,1]
    return lo + (hi - lo) * 0.5 * (1.0 + math.cos(math.pi * t))


class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {
            n: p.detach().clone()                  # 直接跟着参数在哪就在哪（通常在 CUDA）
            for n, p in model.named_parameters() if p.requires_grad
        }

    @torch.no_grad()
    def update(self, model):
        one_m = 1.0 - self.decay
        for n, p in model.named_parameters():
            if n in self.shadow:
                self.shadow[n].mul_(self.decay).add_(p, alpha=one_m)

    def state_dict_cpu(self):
        return {n: t.detach().cpu() for n, t in self.shadow.items()}
    
    def copy_to(self, model):
        """将 shadow 参数复制到指定模型"""
        for n, p in model.named_parameters():
            key = n
            if key not in self.shadow and not key.startswith("module."):
                key = f"module.{n}"
            if key not in self.shadow and key.startswith("module."):
                key = key[len("module."):]
            if key in self.shadow:
                p.data.copy_(self.shadow[key].data)

                
def _get_backbone_num_layers(backbone, is_esm: bool) -> int:
    """
    直接利用你之前 LoRA 注入里那套逻辑：
    ESM: encoder.layer.*
    MolT5 encoder: encoder.block.*
    """
    L = None
    cfg = getattr(backbone, "config", None)
    if cfg is not None:
        L = getattr(cfg, "num_hidden_layers", None)
        if L is None:
            L = getattr(cfg, "num_layers", None)
    if L is None:
        if is_esm and hasattr(backbone, "encoder") and hasattr(backbone.encoder, "layer"):
            L = len(backbone.encoder.layer)
        elif (not is_esm) and hasattr(backbone, "encoder") and hasattr(backbone.encoder, "block"):
            L = len(backbone.encoder.block)

    assert L is not None, "无法推断 backbone 层数，请检查模型结构"
    return int(L)                


def freeze_backbone(model, keep_protein_lora: bool = True, keep_ligand_lora: bool = True):
    for name, p in model.named_parameters():
        n = name.lower()
        is_prot_bb = ("protein.backbone" in n)
        is_lig_bb  = ("ligand.backbone"  in n)

        is_lora = ("lora" in n)

        if is_prot_bb or is_lig_bb:
            if is_lora:
                if is_prot_bb and not keep_protein_lora:
                    p.requires_grad_(False)
                elif is_lig_bb and not keep_ligand_lora:
                    p.requires_grad_(False)
                else:
                    p.requires_grad_(True)
            else:
                p.requires_grad_(False)
        else:
            p.requires_grad_(True)

    has_trainable_params = any(p.requires_grad for _, p in model.named_parameters())
    model.train() if has_trainable_params else model.eval()

    n_lora_tr = sum(p.numel() for n,p in model.named_parameters() if ("lora" in n.lower()) and p.requires_grad)
    n_nonbb_tr = sum(
        p.numel()
        for n, p in model.named_parameters()
        if ("protein.backbone" not in n.lower()) and ("ligand.backbone" not in n.lower()) and p.requires_grad
    )
    print(f"[INFO] Backbone frozen. LoRA trainable params={n_lora_tr}, Non-backbone trainable params={n_nonbb_tr}")

def freeze_all_except_kcat_head(
    model,
    disable_source_residual: bool = True,
    train_norm_fusion: bool = False,
):
    """
    kcat OOD 修复用：冻结除 kcat head 以外的全部参数，只重训 kcat head。

    - 假设你已经通过 --kcat_init_weights 载入了完整训练好的权重（encoder/head 都已学好）
    - 默认只放开 `multi_head.heads.kcat` 一个塔；其他 head（Km、ratio）一并冻结
    - 若 train_norm_fusion=True，额外放开 norm_p/norm_l 与融合层（name 含 fusion）
    - 默认把 `kcat_source_residual` 直接 set None（置空模块 → state_dict 也不会保留该键）
      这是修复 OOD kcat 正向 bias 的关键一步
    - 全模型进入 eval()，再把 kcat head 单独 .train() → dropout 只在 kcat head 生效
    """
    real_model = getattr(model, "module", model)

    if disable_source_residual and getattr(real_model, "kcat_source_residual", None) is not None:
        print("[KCAT-HO] drop kcat_source_residual (set module to None)")
        real_model.kcat_source_residual = None

    keep_patterns = ("multi_head.heads.kcat",)
    if train_norm_fusion:
        keep_patterns = keep_patterns + ("norm_p", "norm_l", "fusion")
    if not disable_source_residual:
        keep_patterns = keep_patterns + ("kcat_source_residual",)

    n_trainable = 0
    n_total = 0
    for name, p in model.named_parameters():
        n_total += p.numel()
        if any(pat in name for pat in keep_patterns):
            p.requires_grad_(True)
            n_trainable += p.numel()
        else:
            p.requires_grad_(False)

    model.eval()
    for name, m in model.named_modules():
        if "multi_head.heads.kcat" in name:
            m.train()
        elif (not disable_source_residual) and ("kcat_source_residual" in name):
            m.train()

    print(
        f"[KCAT-HO] frozen all except selected modules. "
        f"trainable={n_trainable} / total={n_total} "
        f"({100.0 * n_trainable / max(n_total, 1):.3f}%)"
    )
    if train_norm_fusion:
        print("[KCAT-HO] extra trainable patterns: norm_p, norm_l, *fusion*")


def configure_backbone_trainable(model, args):
    """
    1) 先用你原来的 freeze_backbone 逻辑冻结 backbone，只保留 LoRA + head + adapter + s_*
    2) 再按参数解冻 protein / ligand 的最后 K 层 backbone（非 LoRA 权重）
    """
    freeze_backbone(model, keep_protein_lora=True, keep_ligand_lora=True)

    if args.unfreeze_protein_last_k <= 0 and args.unfreeze_ligand_last_k <= 0:
        print("[Backbone] keep fully frozen (only LoRA/Head/Adapter trainable).")
        return

    real_model = getattr(model, "module", model)

    prot_backbone = getattr(real_model.protein, "backbone", None)
    lig_backbone  = getattr(real_model.ligand,  "backbone", None)

    if args.unfreeze_protein_last_k > 0 and prot_backbone is not None:
        L_prot = _get_backbone_num_layers(prot_backbone, is_esm=True)
        k = min(args.unfreeze_protein_last_k, L_prot)
        start = L_prot - k
        print(f"[Backbone] Unfreeze protein backbone last {k} layers: [{start}..{L_prot - 1}]")

        for name, p in model.named_parameters():
            if "protein.backbone" in name and "encoder.layer." in name and "lora" not in name.lower():
                try:
                    layer_id = int(name.split("encoder.layer.")[1].split(".")[0])
                except Exception:
                    continue
                if layer_id >= start:
                    p.requires_grad_(True)

    if args.unfreeze_ligand_last_k > 0 and lig_backbone is not None:
        L_lig = _get_backbone_num_layers(lig_backbone, is_esm=False)
        k = min(args.unfreeze_ligand_last_k, L_lig)
        start = L_lig - k
        print(f"[Backbone] Unfreeze ligand backbone last {k} layers: [{start}..{L_lig - 1}]")

        for name, p in model.named_parameters():
            if "ligand.backbone" in name and "encoder.block." in name and "lora" not in name.lower():
                try:
                    block_id = int(name.split("encoder.block.")[1].split(".")[0])
                except Exception:
                    continue
                if block_id >= start:
                    p.requires_grad_(True)

    n_bb_tr = sum(
        p.numel()
        for n, p in model.named_parameters()
        if (("protein.backbone" in n) or ("ligand.backbone" in n))
        and p.requires_grad
        and ("lora" not in n.lower())
    )
    print(f"[Backbone] trainable backbone (non-LoRA) params = {n_bb_tr}")


    


def build_optimizer_single_stage(
       model,
    base_lr_head: float = 6e-4,   # 头部默认学习率（原来值）
    kcat_scale: float = 1.2,       # kcat LR 倍率：建议 1.2~2.0
    km_scale: float   = 0.8,       # Km   LR 倍率：略降，防“抢学习”
    ratio_scale: float = 1.0,      # ratio LR 倍率
    lora_lr: float = 2e-5,
    lora_lr_prot: float | None = None,
    lora_lr_lig:  float | None = None,
    s_lr: float = 4e-4,
    wd: float = 3e-3, #3e-3
    backbone_lr: float = 7e-6,      # 新增：backbone 学习率
    backbone_wd: float = 1e-2,      # 新增：backbone weight decay
):
    if lora_lr_prot is None: lora_lr_prot = lora_lr
    if lora_lr_lig  is None: lora_lr_lig  = lora_lr
    
    def is_no_decay(n: str) -> bool:
        n_low = n.lower()
        return (
            n_low.endswith(".bias")
            or "norm" in n_low
            or "layernorm" in n_low
            or "ln" in n_low
            or "bn" in n_low
            or "rmsnorm" in n_low
        )

    head_kcat_decay, head_kcat_no = [], []
    head_km_decay,   head_km_no   = [], []
    head_ratio_decay,head_ratio_no= [], []
    head_other_decay, head_other_no= [], []

    lora_prot, lora_lig, lora_misc = [], [], []
    s_params, frozen_backbone, others = [], [], []
    
    backbone_decay, backbone_no = [], []
    named = sorted(model.named_parameters(), key=lambda x: x[0])
    name_map = {id(p): n for n, p in named}
    
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        
        nlow = name.lower()

        leaf = name.split(".")[-1].lower()
        if (leaf in {"s_kcat", "s_km", "s_ratio"}) or ("s_domain_task" in nlow):
            s_params.append(p)
            continue

        if "lora_" in nlow:
            if "protein.backbone" in nlow:
                lora_prot.append(p)
            elif "ligand.backbone" in nlow:
                lora_lig.append(p)
            else:
                lora_misc.append(p)
            continue
            
        if ("protein.backbone" in nlow) or ("ligand.backbone" in nlow):
            (backbone_no if is_no_decay(name) else backbone_decay).append(p)
            continue

        if ("norm_p" in nlow) or ("norm_l" in nlow):
            (head_other_no if is_no_decay(name) else head_other_decay).append(p)
            continue

        is_head_like = ("head" in nlow) or ("multi_head" in nlow) or ("adapter" in nlow) or ("fusion" in nlow)

        if is_head_like and ("kcat" in nlow):
            (head_kcat_no if is_no_decay(name) else head_kcat_decay).append(p)
        elif is_head_like and (".km" in nlow or "km" in nlow):
            (head_km_no if is_no_decay(name) else head_km_decay).append(p)
        elif is_head_like and ("ratio" in nlow):
            (head_ratio_no if is_no_decay(name) else head_ratio_decay).append(p)
        elif is_head_like:
            (head_other_no if is_no_decay(name) else head_other_decay).append(p)
        else:
            others.append(p)

    param_groups = []

    if head_kcat_decay:
        param_groups.append({"params": head_kcat_decay, "lr": base_lr_head * kcat_scale, "weight_decay": wd,"tag": "head","head_scale": kcat_scale,"base_lr": base_lr_head * kcat_scale,})
    if head_kcat_no:
        param_groups.append({"params": head_kcat_no,   "lr": base_lr_head * kcat_scale, "weight_decay": 0.0,"tag": "head","head_scale": kcat_scale,"base_lr": base_lr_head * kcat_scale, })

    if head_km_decay:
        param_groups.append({"params": head_km_decay,  "lr": base_lr_head * km_scale,   "weight_decay": wd,"tag": "head","head_scale": km_scale,"base_lr": base_lr_head * km_scale, })
    if head_km_no:
        param_groups.append({"params": head_km_no,     "lr": base_lr_head * km_scale,   "weight_decay": 0.0,"tag": "head","head_scale": km_scale,"base_lr": base_lr_head * km_scale,})

    if head_ratio_decay:
        param_groups.append({"params": head_ratio_decay, "lr": base_lr_head * ratio_scale, "weight_decay": wd,"tag": "head","head_scale": ratio_scale, "base_lr": base_lr_head * ratio_scale,})
    if head_ratio_no:
        param_groups.append({"params": head_ratio_no,    "lr": base_lr_head * ratio_scale, "weight_decay": 0.0,"tag": "head","head_scale": ratio_scale, "base_lr": base_lr_head * ratio_scale,})

    if head_other_decay:
        param_groups.append({"params": head_other_decay, "lr": base_lr_head, "weight_decay": wd,"tag": "head","head_scale": 1.0,"base_lr": base_lr_head,  })
    if head_other_no:
        param_groups.append({"params": head_other_no,    "lr": base_lr_head, "weight_decay": 0.0,"tag": "head","head_scale": 1.0, "base_lr": base_lr_head,})

    if lora_prot:
        param_groups.append({"params": lora_prot,"lr": lora_lr_prot,"weight_decay": 0.0,"tag": "lora_prot","base_lr": lora_lr_prot,})
    if lora_lig:
        param_groups.append({"params": lora_lig,"lr": lora_lr_lig,"weight_decay": 0.0,"tag": "lora_lig","base_lr": lora_lr_lig,})
    if lora_misc:
        param_groups.append({"params": lora_misc,"lr": lora_lr,"weight_decay": 0.0,"tag": "lora","base_lr": lora_lr,})   
    if backbone_decay:
        param_groups.append({"params": backbone_decay,"lr": backbone_lr,"weight_decay": backbone_wd,"tag": "backbone","base_lr": backbone_lr,})
    if backbone_no:
        param_groups.append({"params": backbone_no,"lr": backbone_lr,"weight_decay": 0.0,"tag": "backbone","base_lr": backbone_lr,})

    if s_params:
        param_groups.append({"params": s_params,"lr": s_lr,"weight_decay": 0.0,"tag": "s","base_lr": s_lr,})
    if frozen_backbone:
        param_groups.append({"params": frozen_backbone,"lr": 0.0,"weight_decay": 0.0,"tag": "frozen","base_lr": 0.0,})
    if others:
        param_groups.append({"params": others,"lr": 0.0,"weight_decay": 0.0,"tag": "others","base_lr": 0.0,})

    optimizer = AdamW(param_groups, betas=(0.9, 0.999), eps=1e-8)

    try:
        def _cnt(gs): return sum(p.numel() for p in gs)
        print("[OPT] kcat   lr=", base_lr_head * kcat_scale, "decay:", _cnt(head_kcat_decay), "nodecay:", _cnt(head_kcat_no))
        print("[OPT] Km     lr=", base_lr_head * km_scale,   "decay:", _cnt(head_km_decay),   "nodecay:", _cnt(head_km_no))
        print("[OPT] ratio  lr=", base_lr_head * ratio_scale,"decay:", _cnt(head_ratio_decay),"nodecay:", _cnt(head_ratio_no))
        print("[OPT] otherH lr=", base_lr_head,              "decay:", _cnt(head_other_decay),"nodecay:", _cnt(head_other_no))
        print("[OPT] lora_prot lr=", lora_lr_prot, "params:", _cnt(lora_prot))
        print("[OPT] lora_lig  lr=", lora_lr_lig,  "params:", _cnt(lora_lig))
        if lora_misc:
            print("[OPT] lora_misc lr=", lora_lr,  "params:", _cnt(lora_misc))
        print("[OPT] s_*    lr=", s_lr, "params:", _cnt(s_params))
        print("[OPT] frozen lr=0 params:", _cnt(frozen_backbone))
    except Exception as e:
        print(f"[OPT] WARNING: optimizer summary print failed: {e}")

    return optimizer

def build_optimizer_single_stage_stable(model, **kwargs):
    """
    包装原版 build_optimizer_single_stage
    - 不需要传参即可使用（内部自带默认值，与原函数保持一致）
    - 若传入 kwargs，仅覆盖对应默认键
    - 对 optimizer.param_groups 做稳定化处理，便于 ckpt 恢复
    """
    _defaults = dict(
        base_lr_head=6e-4, #6e-4 #5e-5
        kcat_scale=1.2, #1.2
        km_scale=0.8, #0.8
        ratio_scale=1.0, #1.0
        lora_lr=2e-5, #2e-5 #5e-6
        lora_lr_prot=None,
        lora_lr_lig=None,
        s_lr=4e-4, #4e-4 #5e-5
        wd=1e-5,#1e-5 #3e-3 #3e-5 #0
    )
    _overrides = {k: v for k, v in kwargs.items() if k in _defaults}
    _defaults.update(_overrides)

    optimizer = build_optimizer_single_stage(model, **_defaults)

    named_sorted = sorted(model.named_parameters(), key=lambda x: x[0])
    name_map = {id(p): n for n, p in named_sorted}

    for g in optimizer.param_groups:
        g["params"] = sorted(g["params"], key=lambda p: name_map.get(id(p), ""))

    tag_order = {"head": 0, "lora_prot": 1, "lora_lig": 2,"lora": 3,"backbone": 4, "s": 5, "frozen": 6, "others": 7}
    optimizer.param_groups = sorted(
        optimizer.param_groups,
        key=lambda g: (tag_order.get(g.get("tag", ""), 99), float(g.get("lr", 0.0)))
    )

    try:
        print("[STABLE-OPT] param_groups summary:")
        for i, g in enumerate(optimizer.param_groups):
            tag = g.get("tag", "unk")
            print(f"  [{i:02d}] tag={tag:7s} | lr={g['lr']:.2e} | n_params={len(g['params'])}")
    except Exception as e:
        print(f"[STABLE-OPT] WARNING: summary print failed: {e}")

    return optimizer


def compute_multihead_loss(model_out, labels, criterion):
    """
    model_out: KineticsPredictor.forward 的完整输出
        (pred_kcat_log, pred_Km_log, pred_ratio_log, gate_p, gate_s, s_raw, s_eff)
    labels: (B, 3) 张量，列顺序 = [kcat, Km, ratio]
    criterion: 一般为 nn.MSELoss(reduction='mean')
    """
    if isinstance(model_out, (tuple, list)):
        pred_k, pred_m, pred_r = model_out[:3]
    elif isinstance(model_out, dict):
        pred_k = model_out["kcat"]
        pred_m = model_out["Km"]
        pred_r = model_out.get("ratio", model_out.get("kcat_Km"))
    else:
        raise TypeError(f"Unexpected model_out type: {type(model_out)}")

    if pred_k.ndim > 1:
        pred_k = pred_k.squeeze(-1)
    if pred_m.ndim > 1:
        pred_m = pred_m.squeeze(-1)
    if pred_r.ndim > 1:
        pred_r = pred_r.squeeze(-1)

    y_k = labels[:, 0]
    y_m = labels[:, 1]
    y_r = labels[:, 2]

    loss_k = criterion(pred_k, y_k)
    loss_m = criterion(pred_m, y_m)
    loss_r = criterion(pred_r, y_r)

    loss = loss_k + loss_m + loss_r

    loss_detail = {
        "kcat":  loss_k.detach().item(),
        "Km":    loss_m.detach().item(),
        "ratio": loss_r.detach().item(),
        "total": loss.detach().item(),
    }
    return loss, loss_detail


def lr_range_test_for_tag(
    model,
    train_loader,
    device,
    criterion,
    tag_name,
    start_lr,
    end_lr,
    num_iter=50,
):
    """
    对指定 tag 的 param_group 做 LR Range Test（head 或 lora 单独扫描）：
    - 其它 param_group 使用极小 lr 近似冻结
    - 结束后恢复模型权重
    """
    real_model = getattr(model, "module", model)
    real_model.train()

    backup_state = copy.deepcopy(real_model.state_dict())

    optimizer = build_optimizer_single_stage_stable(real_model)

    orig_lrs = [g["lr"] for g in optimizer.param_groups]

    for g in optimizer.param_groups:
        tag = g.get("tag", "")
        if tag == tag_name:
            g["lr"] = start_lr
        else:
            g["lr"] = 1e-8  # 几乎不更新

    num_iter = min(num_iter, len(train_loader))
    lr_mult = (end_lr / start_lr) ** (1.0 / max(num_iter - 1, 1))

    cur_lr = start_lr
    lrs = []
    losses = []

    it = iter(train_loader)
    pbar = tqdm(range(num_iter), desc=f"[LRF-{tag_name}]", disable=not is_main_process())
    
    for i in range(num_iter):
        try:
            batch = next(it)
        except StopIteration:
            break

        prot, lig, labels = batch
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)

        preds = real_model(prot, lig)
        loss,_ = compute_multihead_loss(preds, labels, criterion)

        loss.backward()
        optimizer.step()

        losses.append(float(loss.item()))
        lrs.append(cur_lr)

        cur_lr *= lr_mult
        for g in optimizer.param_groups:
            tag = g.get("tag", "")
            if tag == tag_name:
                g["lr"] = cur_lr
            else:
                g["lr"] = 1e-8

        if math.isfinite(losses[-1]) and losses[-1] > 10 * losses[0]:
            break

    real_model.load_state_dict(backup_state)

    for g, lr0 in zip(optimizer.param_groups, orig_lrs):
        g["lr"] = lr0

    if not losses:
        return start_lr

    skip = max(int(0.1 * len(losses)), 1)
    losses_trim = losses[skip:]
    lrs_trim = lrs[skip:]
    best_idx = min(range(len(losses_trim)), key=lambda i: losses_trim[i])
    best_lr = float(lrs_trim[best_idx])

    print(f"[LRF-{tag_name}] best lr ≈ {best_lr:.3e}")
    return best_lr



def get_scheduler_with_warmup(optimizer, num_warmup_steps, num_training_steps, min_lr=1e-8):
    """
    - optimizer: your optimizer
    - num_warmup_steps: number of steps (batches) to warm up
    - num_training_steps: total number of steps (batches) you plan to run
    - min_lr: absolute lower bound for lr after clipping
    """
    base_lrs = [pg['lr'] for pg in optimizer.param_groups]

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    def step_and_clip(step=None):
        if step is None:
            scheduler.step()
        else:
            scheduler.step(step)
        for pg, base_lr in zip(optimizer.param_groups, base_lrs):
            if pg['lr'] < min_lr:
                pg['lr'] = min_lr

    scheduler.step_and_clip = step_and_clip
    return scheduler



def train_step(model, batch, optimizer, scaler, recorder=None,
               lam_l1=0, lam_tv=0,
               microbatch_size: int | None = None,
               use_mask: bool=True,
               final_contact_store=None,
               epoch_idx: int = 0,
               lambda_gate_off: float = 0.0,
               lambda_gate_on: float = 0.0,
               gate_on_margin: float = 0.2,
               lambda_cons: float = 0.0,
               cons_warmup_epochs: int = 0,
               cons_weights=(1.0, 1.0, 1.0),
               domain_uncertainty: bool = False,
               struct_dropout_prob: float = 0.0,
               phys_dropout_prob: float = 0.0,
               phys_ablation_mode: str = "real",
               kcat_rank_weight: float = 0.0,
               kcat_rank_pairs: int = 256,
               kcat_rank_margin: float = 0.0,
               kcat_var_reg_weight: float = 0.0,
               kcat_var_reg_mode: str = "abs",
               kcat_loss_type: str = "huber",
               kcat_barron_alpha: float = 1.0,
               kcat_barron_scale: float = 1.0,
               kcat_use_group_dro: bool = False,
               kcat_group_dro_eta: float = 0.05,
               kcat_group_dro_num_groups: int = 3,
               kcat_group_dro_mix: float = 1.0,
                 ):
    
    model.train()
    seqs, smiles, row_ids, source_ids, p_s, l_s, labels, struct_avail_mask, phys_feat, phys_mask, phys_quality = unpack_batch(batch)
    if isinstance(row_ids, torch.Tensor):
        row_ids = row_ids.detach().cpu()    
    device = next(model.parameters()).device

    def _to_scalar_tensor(x: torch.Tensor) -> torch.Tensor:
        return x.mean() if torch.is_tensor(x) and x.ndim > 0 else x

    labels = labels.to(device)
    source_ids = to_source_tensor(source_ids, labels.size(0), device=device)
    phys_feat, phys_mask, phys_quality = infer_phys_inputs(
        phys_feat,
        phys_mask,
        phys_quality,
        batch_size=labels.size(0),
        device=device,
    )
    
    
    if p_s is not None:
        p_s = p_s.to(device, non_blocking=True)
        l_s = l_s.to(device, non_blocking=True)

    struct_avail_mask = infer_struct_avail_mask(
        p_s,
        l_s,
        batch_size=labels.size(0),
        device=device,
        provided_mask=struct_avail_mask,
    )

    if (row_ids is not None) and (final_contact_store is not None):
        p_s2, l_s2 = final_contact_store.get_batch_by_rowids_torch(
            row_ids, device=device
        )
        p_s, l_s = p_s2, l_s2
        struct_avail_mask = infer_struct_avail_mask(
            p_s,
            l_s,
            batch_size=labels.size(0),
            device=device,
        )
    
    
    B = labels.size(0)
    mb = microbatch_size or B
    steps = (B + mb - 1) // mb
    
    optimizer.zero_grad(set_to_none=True)

    beta = 1.0
    s_caps   = (0.6, 0.35, 0.4)
    s_floorM = -0.10

    if not hasattr(model, "_mse_ema"):
        with torch.no_grad():
            model._mse_ema = torch.tensor([0.5, 0.5, 0.5], device=device, dtype=torch.float32)
    with torch.no_grad():
        ema_prev = model._mse_ema.detach()
        r = ema_prev / (ema_prev.mean() + 1e-8)
        delta = torch.clamp(r - 1.0, min=-0.3, max=0.6)
        alpha = 0.35
        bias = (-alpha * delta).to(torch.float32)
        bias_tuple = (float(bias[0]), float(bias[1]), float(bias[2]))
    
    total_loss = 0.0
    mse_k_sum = mse_m_sum = mse_r_sum = 0.0   
    preds_chunks = []   # ★★ 新增：收集微批预测
    labels_chunks = []  # ★★ 新增：收集微批标签

    gate_off_sum = 0.0
    gate_on_sum = 0.0
    cons_sum = 0.0
    
    cap_penalty_sum = 0.0
    s_reg_sum       = 0.0
    s_k_raw_sum = s_m_raw_sum = s_r_raw_sum = 0.0
    s_k_eff_sum = s_m_eff_sum = s_r_eff_sum = 0.0
    
    for t in range(steps):
        s = t * mb
        e = min(B, s + mb)
        
        seqs_mb   = seqs[s:e] if isinstance(seqs, list) else seqs[s:e]
        smiles_mb = smiles[s:e] if isinstance(smiles, list) else smiles[s:e]
        labels_mb = labels[s:e]
        source_mb = source_ids[s:e]
        
        p_mb = None if p_s is None else p_s[s:e]
        l_mb = None if l_s is None else l_s[s:e]
        struct_mb = struct_avail_mask[s:e]
        phys_mb = None if phys_feat is None else phys_feat[s:e]
        phys_m_mb = phys_mask[s:e]
        phys_q_mb = phys_quality[s:e]

        if (struct_dropout_prob > 0.0) and (p_mb is not None) and (l_mb is not None):
            keep = (torch.rand(e - s, device=device) > struct_dropout_prob).to(dtype=struct_mb.dtype)
            struct_mb = struct_mb * keep
            p_mb = p_mb * keep.unsqueeze(-1)
            l_mb = l_mb * keep.unsqueeze(-1)

        if phys_mb is not None:
            phys_mb, phys_m_mb, phys_q_mb = apply_phys_ablation(
                phys_mb,
                phys_m_mb,
                phys_q_mb,
                mode=phys_ablation_mode,
            )
            if phys_dropout_prob > 0.0:
                pkeep = (torch.rand(e - s, device=device) > phys_dropout_prob).to(dtype=phys_m_mb.dtype)
                phys_m_mb = phys_m_mb * pkeep
                phys_q_mb = phys_q_mb * pkeep
                phys_mb = phys_mb * pkeep.unsqueeze(-1)
        
        
        from contextlib import nullcontext
        is_ddp = dist.is_available() and dist.is_initialized()
        sync_ctx = (model.no_sync() if (is_ddp and (t + 1) < steps) else nullcontext())
        
        with sync_ctx:
            with amp.autocast('cuda'):
                try:
                    pred_kcat, pred_Km, pred_act_tmp, gate_e, gate_s, s_raw, s_eff = model(
                        seqs_mb, smiles_mb,
                        prot_struct=p_mb,
                        lig_struct=l_mb,
                        struct_avail_mask=struct_mb,
                        phys_feat=phys_mb,
                        phys_mask=phys_m_mb,
                        phys_quality=phys_q_mb,
                        source_ids=source_mb,
                        use_mask=use_mask,
                        bias_tuple=bias_tuple, caps=s_caps, floor_m=s_floorM
                    )
                except TypeError:
                    pred_kcat, pred_Km, pred_act_tmp, gate_e, gate_s, s_raw, s_eff = model(
                        seqs_mb, smiles_mb, use_mask=use_mask,
                        bias_tuple=bias_tuple, caps=s_caps, floor_m=s_floorM
                    )

                _alpha_direct = 0.4  # 直接 head 权重；0=纯派生（原行为），1=纯直接预测
                pred_act = _alpha_direct * pred_act_tmp + (1.0 - _alpha_direct) * (pred_kcat - pred_Km + 3.0)

                mask_k = torch.isfinite(labels_mb[:, 0])
                mask_m = torch.isfinite(labels_mb[:, 1])
                mask_r = torch.isfinite(labels_mb[:, 2]) & mask_k & mask_m

                loss_k_per = torch.zeros_like(pred_kcat)
                huber_m = torch.zeros_like(pred_Km)
                huber_r = torch.zeros_like(pred_act)
                if mask_k.any():
                    if str(kcat_loss_type).lower() == "barron":
                        loss_k_per[mask_k] = _barron_loss_per_sample(
                            pred_kcat[mask_k],
                            labels_mb[mask_k, 0],
                            alpha=float(kcat_barron_alpha),
                            scale=float(kcat_barron_scale),
                        ).to(loss_k_per.dtype)
                    else:
                        loss_k_per[mask_k] = F.smooth_l1_loss(
                            pred_kcat[mask_k], labels_mb[mask_k, 0], beta=beta, reduction='none'
                        ).to(loss_k_per.dtype)
                if mask_m.any():
                    huber_m[mask_m] = F.smooth_l1_loss(pred_Km[mask_m], labels_mb[mask_m, 1], beta=beta, reduction='none').to(huber_m.dtype)
                if mask_r.any():
                    huber_r[mask_r] = F.smooth_l1_loss(pred_act[mask_r], labels_mb[mask_r, 2], beta=beta, reduction='none').to(huber_r.dtype)

                mse_kcat = _safe_masked_mean(loss_k_per, mask_k, device=device, dtype=pred_kcat.dtype)
                mse_Km = _safe_masked_mean(huber_m, mask_m, device=device, dtype=pred_kcat.dtype)
                mse_ratio = _safe_masked_mean(huber_r, mask_r, device=device, dtype=pred_kcat.dtype)

                if bool(kcat_use_group_dro) and mask_k.any():
                    gid = source_mb.long()[mask_k]
                    lk = loss_k_per[mask_k]
                    if gid.numel() > 0:
                        uniq = torch.unique(gid)
                        lg_list = []
                        for g in uniq:
                            mg = (gid == g)
                            if mg.any():
                                lg_list.append(lk[mg].mean())
                        if lg_list:
                            lg = torch.stack(lg_list)
                            real_model = getattr(model, "module", model)
                            n_groups = max(int(kcat_group_dro_num_groups), 1)
                            if (not hasattr(real_model, "_kcat_group_q")) or (real_model._kcat_group_q.numel() != n_groups):
                                real_model._kcat_group_q = torch.ones(
                                    n_groups, device=device, dtype=torch.float32
                                ) / float(n_groups)
                            q = real_model._kcat_group_q.float()
                            valid_uniq = uniq.clamp(min=0, max=n_groups - 1)
                            lg32 = lg.detach().float()
                            with torch.no_grad():
                                q[valid_uniq] = q[valid_uniq] * torch.exp(float(kcat_group_dro_eta) * lg32)
                                q = q / q.sum().clamp_min(1e-8)
                                real_model._kcat_group_q = q
                            dro_loss = (q[valid_uniq] * lg32).sum().to(mse_kcat.dtype)
                            mix = float(kcat_group_dro_mix)
                            mse_kcat = (1.0 - mix) * mse_kcat + mix * dro_loss

                kcat_rank_loss = torch.zeros((), device=device, dtype=pred_kcat.dtype)
                kcat_var_reg   = torch.zeros((), device=device, dtype=pred_kcat.dtype)

                if mask_k.any() and (float(kcat_rank_weight) > 0.0):
                    idx_valid = mask_k.nonzero(as_tuple=False).reshape(-1)
                    n_valid = int(idx_valid.numel())
                    if n_valid >= 2:
                        n_pairs = int(min(int(kcat_rank_pairs), n_valid * (n_valid - 1)))
                        if n_pairs > 0:
                            i_sel = idx_valid[torch.randint(0, n_valid, (n_pairs,), device=device)]
                            j_sel = idx_valid[torch.randint(0, n_valid, (n_pairs,), device=device)]
                            keep_pair = (i_sel != j_sel)
                            if keep_pair.any():
                                i_sel = i_sel[keep_pair]
                                j_sel = j_sel[keep_pair]
                                d_true = labels_mb[i_sel, 0] - labels_mb[j_sel, 0]
                                y_sign = torch.sign(d_true)
                                nz = y_sign != 0
                                if nz.any():
                                    kcat_rank_loss = F.margin_ranking_loss(
                                        pred_kcat[i_sel[nz]],
                                        pred_kcat[j_sel[nz]],
                                        y_sign[nz],
                                        margin=float(kcat_rank_margin),
                                        reduction="mean",
                                    ).to(pred_kcat.dtype)

                if mask_k.any() and (float(kcat_var_reg_weight) > 0.0):
                    pk = pred_kcat[mask_k]
                    yk = labels_mb[mask_k, 0]
                    if pk.numel() >= 2:
                        with torch.cuda.amp.autocast(enabled=False):
                            std_p = pk.float().std(unbiased=False)
                            std_y = yk.float().std(unbiased=False)
                            if str(kcat_var_reg_mode) == "ratio":
                                var_term = torch.relu(std_y - std_p)
                            else:
                                var_term = (std_p - std_y).abs()
                        kcat_var_reg = var_term.to(pred_kcat.dtype)

                l1 = gate_e.abs().mean() + gate_s.abs().mean()
                tv = (gate_e[:, 1:] - gate_e[:, :-1]).abs().mean() + (gate_s[:, 1:] - gate_s[:, :-1]).abs().mean()

                (s_k_raw, s_m_raw, s_r_raw) = s_raw
                (s_k_eff, s_m_eff, s_r_eff) = s_eff
                
                s_min, s_max = -0.5, 2.0
                s_k_eff = s_k_eff.clamp(min=s_min, max=s_max)
                s_m_eff = s_m_eff.clamp(min=s_min, max=s_max)
                s_r_eff = s_r_eff.clamp(min=s_min, max=s_max)

                n_k = float(mask_k.sum()) + 1e-8
                n_m = float(mask_m.sum()) + 1e-8
                n_r = float(mask_r.sum()) + 1e-8
                n_active_tasks = (1 if mask_k.any() else 0) + (1 if mask_m.any() else 0) + (1 if mask_r.any() else 0)
                n_total = n_k + n_m + n_r
                w_k = min(2.0, n_total / (n_active_tasks * n_k)) if mask_k.any() else 0.0
                w_m = min(2.0, n_total / (n_active_tasks * n_m)) if mask_m.any() else 0.0
                w_r = min(2.0, n_total / (n_active_tasks * n_r)) if mask_r.any() else 0.0

                task_terms = []
                if mask_k.any():
                    task_terms.append(w_k * _to_scalar_tensor(torch.exp(-s_k_eff) * mse_kcat + s_k_eff))
                if mask_m.any():
                    task_terms.append(w_m * _to_scalar_tensor(torch.exp(-s_m_eff) * mse_Km + s_m_eff))
                if mask_r.any():
                    task_terms.append(w_r * _to_scalar_tensor(torch.exp(-s_r_eff) * mse_ratio + s_r_eff))
                if task_terms:
                    loss_tasks = sum(task_terms) / float(len(task_terms))
                else:
                    loss_tasks = torch.zeros((), device=device, dtype=pred_kcat.dtype)

                floor_k, floor_m, floor_r = -0.5, -0.5, -0.5  # 与 s_eff 下界一致
                cap_k, cap_m, cap_r = s_caps
                cap_penalty = 1e-3 * (
                    torch.relu(s_k_raw - cap_k)**2 + torch.relu(floor_k - s_k_raw)**2 +
                    torch.relu(s_m_raw - cap_m)**2 + torch.relu(floor_m - s_m_raw)**2 +
                    torch.relu(s_r_raw - cap_r)**2 + torch.relu(floor_r - s_r_raw)**2
                )
                cap_penalty = _to_scalar_tensor(cap_penalty)

                with torch.cuda.amp.autocast(enabled=False):
                    s_reg = (5e-3) * ((s_k_raw.float() - 0.2)**2 + (s_r_raw.float() - 0.2)**2) \
                          + (7e-3) * ((s_m_raw.float() + 0.2)**2)
                s_reg = _to_scalar_tensor(s_reg)
                    
                
                cap_penalty_sum += float(cap_penalty.detach().item())
                s_reg_sum       += float(s_reg.detach().item())

                s_k_raw_sum += float(s_k_raw.detach().mean().item())
                s_m_raw_sum += float(s_m_raw.detach().mean().item())
                s_r_raw_sum += float(s_r_raw.detach().mean().item())

                s_k_eff_sum += float(s_k_eff.detach().mean().item())
                s_m_eff_sum += float(s_m_eff.detach().mean().item())
                s_r_eff_sum += float(s_r_eff.detach().mean().item())    
                

                aux_ratio_align = 0.1 * F.smooth_l1_loss(pred_act_tmp, pred_act.detach(), beta=beta)

                real_model = getattr(model, "module", model)
                gate_struct = getattr(real_model, "last_gate_struct", None)
                avail_struct = getattr(real_model, "last_struct_avail", None)
                if gate_struct is None:
                    gate_struct = torch.zeros_like(struct_mb)
                else:
                    gate_struct = gate_struct.to(device=device, dtype=pred_kcat.dtype)
                if avail_struct is None:
                    avail_struct = struct_mb
                else:
                    avail_struct = avail_struct.to(device=device, dtype=pred_kcat.dtype)

                off_mask = avail_struct <= 0.5
                on_mask = avail_struct > 0.5
                if off_mask.any():
                    l_gate_off = gate_struct[off_mask].pow(2).mean()
                else:
                    l_gate_off = torch.zeros((), device=device, dtype=pred_kcat.dtype)
                if on_mask.any():
                    l_gate_on = torch.relu(gate_on_margin - gate_struct[on_mask]).mean()
                else:
                    l_gate_on = torch.zeros((), device=device, dtype=pred_kcat.dtype)

                cons_loss = torch.zeros((), device=device, dtype=pred_kcat.dtype)
                if (lambda_cons > 0.0) and (epoch_idx >= cons_warmup_epochs) and on_mask.any():
                    try:
                        pred_kcat_wo, pred_Km_wo, pred_act_tmp_wo, _, _, _, _ = model(
                            seqs_mb,
                            smiles_mb,
                            prot_struct=None,
                            lig_struct=None,
                            struct_avail_mask=torch.zeros_like(struct_mb),
                            phys_feat=phys_mb,
                            phys_mask=phys_m_mb,
                            phys_quality=phys_q_mb,
                            source_ids=source_mb,
                            use_mask=use_mask,
                            bias_tuple=bias_tuple,
                            caps=s_caps,
                            floor_m=s_floorM,
                        )
                    except TypeError:
                        pred_kcat_wo, pred_Km_wo, pred_act_tmp_wo, _, _, _, _ = model(
                            seqs_mb,
                            smiles_mb,
                            use_mask=use_mask,
                            bias_tuple=bias_tuple,
                            caps=s_caps,
                            floor_m=s_floorM,
                        )
                    pred_act_wo = pred_kcat_wo - pred_Km_wo + 3.0
                    wk, wm, wr = cons_weights
                    cons_loss = (
                        float(wk) * F.smooth_l1_loss(pred_kcat[on_mask], pred_kcat_wo[on_mask], beta=beta)
                        + float(wm) * F.smooth_l1_loss(pred_Km[on_mask], pred_Km_wo[on_mask], beta=beta)
                        + float(wr) * F.smooth_l1_loss(pred_act[on_mask], pred_act_wo[on_mask], beta=beta)
                    )

                gate_off_sum += float(l_gate_off.detach().item())
                gate_on_sum += float(l_gate_on.detach().item())
                cons_sum += float(cons_loss.detach().item())

                loss_mb = (
                    loss_tasks
                    + cap_penalty
                    + s_reg.to(loss_tasks.dtype)
                    + lam_l1 * l1
                    + lam_tv * tv
                    + aux_ratio_align
                    + float(lambda_gate_off) * l_gate_off
                    + float(lambda_gate_on) * l_gate_on
                    + float(lambda_cons) * cons_loss
                    + float(kcat_rank_weight)   * kcat_rank_loss.to(loss_tasks.dtype)
                    + float(kcat_var_reg_weight) * kcat_var_reg.to(loss_tasks.dtype)
                )

                loss = loss_mb / steps
            
            scaler.scale(loss).backward()
            
        total_loss += float(loss_mb.detach().item())
        mse_k_sum += float(mse_kcat.detach().item())
        mse_m_sum += float(mse_Km.detach().item())
        mse_r_sum += float(mse_ratio.detach().item())
        
        preds_chunks.append(torch.stack([pred_kcat, pred_Km, pred_act], dim=1).detach().cpu().numpy())
        labels_chunks.append(labels_mb.detach().cpu().numpy())
              
            
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    scaler.step(optimizer)
    scaler.update()

    with torch.no_grad():
        cur = torch.stack([mse_kcat.detach(), mse_Km.detach(), mse_ratio.detach()]).to(device)
        beta_ema = 0.9
        model._mse_ema = beta_ema * model._mse_ema + (1.0 - beta_ema) * cur
        
    if recorder is not None:
        preds_np  = np.concatenate(preds_chunks,  axis=0)  # 可能是 (B,3) or (steps*B,3)（异常）
        labels_np = np.concatenate(labels_chunks, axis=0)  # 正常应是 (B,3)

        k = min(len(preds_np), len(labels_np))
        if k == 0:
            return float(total_loss / steps)  # 空就不写

        preds_np  = preds_np[:k]
        labels_np = labels_np[:k]

        remain = recorder.preds.shape[0] - recorder.ptr
        if remain <= 0:
            return float(total_loss / steps)
        if k > remain:
            preds_np  = preds_np[:remain]
            labels_np = labels_np[:remain]

        recorder.update(preds=preds_np, labels=labels_np)
        
        



        


    return float(total_loss / steps)



def train_epoch(model, train_loader, optimizer, scheduler, scaler, ema, ep, args, esm_layers, molt5_layers,global_step, sanity_check=True, sanity_verbose=True, sanity_diff_thresh=1e-6, sanity_interval=500,last_ckpt_time=None, weights_dir=None, use_onecycle=False, onecycle=None,
final_contact_store=None):
    model.train()
    losses = []
    n_batches = len(train_loader)
    n_samples = args.batch_size * n_batches #problem   
    recorder = PredictionRecorder(n_samples, output_dim=3)
    real_model = getattr(model, "module", model)
    use_gate_mask = True
    if ep <= getattr(args, "gate_warmup_epochs", 2):
        use_gate_mask = False
        if hasattr(real_model, "resmask"):
            real_model.resmask.droprate = 0.0
    else:
        if hasattr(real_model, "resmask"):
            t = (ep - getattr(args, "gate_warmup_epochs", 2)) / max(1, args.epochs - getattr(args, "gate_warmup_epochs", 2))
            real_model.resmask.droprate = 0.5 * float(min(1.0, max(0.0, t)))
    
    progress = tqdm(train_loader, desc=f"Epoch {ep}/{args.epochs}", leave=True)
    
    
    
    
    for i, batch in enumerate(progress):
        loss = train_step(
            model,
            batch,
            optimizer,
            scaler,
            recorder=recorder,
            lam_l1=0,
            lam_tv=0,
            microbatch_size=None,
            use_mask=use_gate_mask,
            final_contact_store=final_contact_store,
            epoch_idx=ep,
            lambda_gate_off=float(getattr(args, "lambda_gate_off", 0.0)),
            lambda_gate_on=float(getattr(args, "lambda_gate_on", 0.0)),
            gate_on_margin=float(getattr(args, "gate_on_margin", 0.2)),
            lambda_cons=float(getattr(args, "lambda_cons", 0.0)),
            cons_warmup_epochs=int(getattr(args, "cons_warmup_epochs", 0)),
            cons_weights=(
                float(getattr(args, "cons_w_kcat", 1.0)),
                float(getattr(args, "cons_w_km", 1.0)),
                float(getattr(args, "cons_w_ratio", 1.0)),
            ),
            domain_uncertainty=bool(getattr(args, "domain_uncertainty", False)),
            struct_dropout_prob=float(getattr(args, "struct_dropout_prob", 0.0)),
            phys_dropout_prob=float(getattr(args, "phys_drop_p", 0.0)),
            phys_ablation_mode=str(getattr(args, "phys_ablation", "real")),
            kcat_rank_weight=float(getattr(args, "kcat_rank_weight", 0.0)),
            kcat_rank_pairs=int(getattr(args, "kcat_rank_pairs", 256)),
            kcat_rank_margin=float(getattr(args, "kcat_rank_margin", 0.0)),
            kcat_var_reg_weight=float(getattr(args, "kcat_var_reg_weight", 0.0)),
            kcat_var_reg_mode=str(getattr(args, "kcat_var_reg_mode", "abs")),
            kcat_loss_type=str(getattr(args, "kcat_loss_type", "huber")),
            kcat_barron_alpha=float(getattr(args, "kcat_barron_alpha", 1.0)),
            kcat_barron_scale=float(getattr(args, "kcat_barron_scale", 1.0)),
            kcat_use_group_dro=bool(getattr(args, "kcat_use_group_dro", False)),
            kcat_group_dro_eta=float(getattr(args, "kcat_group_dro_eta", 0.05)),
            kcat_group_dro_num_groups=int(getattr(args, "kcat_group_dro_num_groups", 3)),
            kcat_group_dro_mix=float(getattr(args, "kcat_group_dro_mix", 1.0)),
        )
        ema.update(getattr(model, "module", model))
        losses.append(float(loss))
        avg_loss = float(np.mean(losses))
        
        if use_onecycle and (onecycle is not None):
            if global_step < onecycle.total_steps:
                onecycle.step()
            for g in optimizer.param_groups:
                if g.get("tag","") not in {"head", "lora", "lora_prot", "lora_lig"}:
                    g["lr"] = g.get("base_lr", g["lr"])
        elif scheduler is not None:
            scheduler.step_and_clip()
        

            

        global_step += 1
        
        
        if (i+1) % 10 == 0:  # 每10步打印一次（可调）
            lr_summary = [f"g{gi}:{g.get('tag','?')}={g['lr']:.10e}" 
                          for gi, g in enumerate(optimizer.param_groups)]
            progress.set_postfix({
                "avg_loss": f"{avg_loss:.4f}",
                "lrs": " ".join(lr_summary)
            })
        
    
        if is_main_process():
            swanlab.log({
                "lr_step": optimizer.param_groups[0]['lr'],        # 当前 step 的 lr
                "step": global_step,
            })
            
    if is_main_process():
        swanlab.log({
            "train_loss": avg_loss, 
            "epoch": ep, 
            "lr_epoch": optimizer.param_groups[0]['lr'],


        })
        print(f"{avg_loss:.4f} | lr={optimizer.param_groups[0]['lr']:.6e}")

    all_preds, all_labels = recorder.get()
    
    if is_main_process():
        train_data_dir = require_path("train_data_dir")
        train_data_dir.mkdir(parents=True, exist_ok=True)
        np.save(str(train_data_dir / f"preds_epoch{ep}.npy"), all_preds)
        np.save(str(train_data_dir / f"labels_epoch{ep}.npy"), all_labels)

    return global_step,last_ckpt_time


def val_epoch_with_ema(model, ema, val_loader, ep, args, use_ema=True, final_contact_store=None):
    if (ema is None) or (not use_ema):
        return val_epoch(model, val_loader, ep, args, final_contact_store=final_contact_store)

    real_model = getattr(model, "module", model)
    backup = {k: v.detach().clone() for k, v in real_model.state_dict().items()}
    ema.copy_to(real_model)
    
    try:
        val_loss = val_epoch(model, val_loader, ep, args, final_contact_store=final_contact_store)  # 这里仍然把 wrapper model 传给 val_epoch 没问题
    finally:
        real_model.load_state_dict(backup, strict=True)
    return val_loss
    
    
    
def val_epoch(model, val_loader, ep, args, final_contact_store=None):
    """
    验证一个 epoch：与训练端“口味一致”的损失（派生 ratio + Huber + soft-cap + 非零中心）。
    支持 DDP：仅在 rank0 显示进度条；最终用 all_reduce 聚合加权平均。
    """
    is_ddp = dist.is_available() and dist.is_initialized()
    is_main = (not is_ddp) or (dist.get_rank() == 0)

    model.eval()
    device = next(model.parameters()).device  # 循环外先拿 device，避免空集报错

    def _to_scalar_tensor(x: torch.Tensor) -> torch.Tensor:
        return x.mean() if torch.is_tensor(x) and x.ndim > 0 else x

    beta = 1.0                          # Huber beta
    lam_l1, lam_tv = 0, 0         # gate 正则系数
    s_cap_k, s_cap_m, s_cap_r = 0.5, 0.5, 0.4   # soft-cap 上限
    s_caps = (s_cap_k, s_cap_m, s_cap_r)

    total_loss_sum = 0.0   # 累计: sum(loss_i * batch_size_i)
    total_samples  = 0     # 累计: sum(batch_size_i)
            
    cap_penalty_sum = 0.0
    s_reg_sum       = 0.0

    mse_k_sum = mse_m_sum = mse_r_sum = 0.0

    s_k_raw_sum = s_m_raw_sum = s_r_raw_sum = 0.0
    s_k_eff_sum = s_m_eff_sum = s_r_eff_sum = 0.0

    progress = tqdm(val_loader, desc=f"Val Epoch {ep}/{args.epochs}",
                    leave=False, disable=not is_main)

    with torch.no_grad():
        for batch in progress:
            seqs, smiles, row_ids, source_ids, p_s, l_s, labels, struct_avail_mask, phys_feat, phys_mask, phys_quality = unpack_batch(batch)
            
            labels = labels.to(device)
            bs = labels.size(0)
            source_ids = to_source_tensor(source_ids, bs, device=device)
            phys_feat, phys_mask, phys_quality = infer_phys_inputs(
                phys_feat,
                phys_mask,
                phys_quality,
                batch_size=bs,
                device=device,
            )
            if phys_feat is not None:
                phys_feat, phys_mask, phys_quality = apply_phys_ablation(
                    phys_feat,
                    phys_mask,
                    phys_quality,
                    mode=str(getattr(args, "phys_ablation", "real")),
                )
            
            if p_s is not None:
                p_s = p_s.to(device, non_blocking=True)
                l_s = l_s.to(device, non_blocking=True)

            struct_avail_mask = infer_struct_avail_mask(
                p_s,
                l_s,
                batch_size=bs,
                device=device,
                provided_mask=struct_avail_mask,
            )

            if (row_ids is not None) and (final_contact_store is not None):
                if isinstance(row_ids, torch.Tensor):
                    row_ids = row_ids.detach().cpu()
                p_s2, l_s2 = final_contact_store.get_batch_by_rowids_torch(row_ids, device=device)
                p_s, l_s = p_s2, l_s2
                struct_avail_mask = infer_struct_avail_mask(
                    p_s,
                    l_s,
                    batch_size=bs,
                    device=device,
                )

            with amp.autocast('cuda'):
                try:
                    pred_kcat, pred_Km, pred_act_tmp, gate_e, gate_s, s_raw, s_eff = model(
                        seqs, smiles,
                        prot_struct=p_s, lig_struct=l_s,
                        struct_avail_mask=struct_avail_mask,
                        phys_feat=phys_feat,
                        phys_mask=phys_mask,
                        phys_quality=phys_quality,
                        source_ids=source_ids,
                        use_mask=False,
                        bias_tuple=(0.0, 0.0, 0.0),
                        caps=s_caps,
                    )
                except TypeError:
                    pred_kcat, pred_Km, pred_act_tmp, gate_e, gate_s, s_raw, s_eff = model(
                        seqs, smiles, use_mask=False,
                        bias_tuple=(0.0, 0.0, 0.0),
                        caps=s_caps,
                    )
                log_pred_kcat = pred_kcat
                log_pred_Km   = pred_Km
                log_labels    = labels

                mask_k = torch.isfinite(log_labels[:, 0])
                mask_m = torch.isfinite(log_labels[:, 1])

                pred_act  = pred_kcat - pred_Km + 3.0
                mask_r = torch.isfinite(log_labels[:, 2]) & mask_k & mask_m

                huber_k = torch.zeros_like(log_pred_kcat)
                huber_m = torch.zeros_like(log_pred_Km)
                huber_r = torch.zeros_like(pred_act)
                if mask_k.any():
                    huber_k[mask_k] = F.smooth_l1_loss(log_pred_kcat[mask_k], log_labels[mask_k, 0], beta=beta, reduction='none').to(huber_k.dtype)
                if mask_m.any():
                    huber_m[mask_m] = F.smooth_l1_loss(log_pred_Km[mask_m], log_labels[mask_m, 1], beta=beta, reduction='none').to(huber_m.dtype)
                if mask_r.any():
                    huber_r[mask_r] = F.smooth_l1_loss(pred_act[mask_r], log_labels[mask_r, 2], beta=beta, reduction='none').to(huber_r.dtype)

                mse_kcat = _safe_masked_mean(huber_k, mask_k, device=device, dtype=pred_kcat.dtype)
                mse_Km = _safe_masked_mean(huber_m, mask_m, device=device, dtype=pred_kcat.dtype)
                mse_ratio = _safe_masked_mean(huber_r, mask_r, device=device, dtype=pred_kcat.dtype)

                (s_k_raw, s_m_raw, s_r_raw) = s_raw
                (s_k_eff, s_m_eff, s_r_eff) = s_eff

                task_terms = []
                if mask_k.any():
                    task_terms.append(_to_scalar_tensor(torch.exp(-s_k_eff) * mse_kcat + s_k_eff))
                if mask_m.any():
                    task_terms.append(_to_scalar_tensor(torch.exp(-s_m_eff) * mse_Km + s_m_eff))
                if mask_r.any():
                    task_terms.append(_to_scalar_tensor(torch.exp(-s_r_eff) * mse_ratio + s_r_eff))
                if task_terms:
                    loss_tasks = sum(task_terms) / float(len(task_terms))
                else:
                    loss_tasks = torch.zeros((), device=device, dtype=pred_kcat.dtype)

                cap_penalty = 1e-3 * (
                    F.relu(s_k_raw - s_cap_k)**2 +
                    F.relu(s_m_raw - s_cap_m)**2 +
                    F.relu(s_r_raw - s_cap_r)**2
                )
                cap_penalty = _to_scalar_tensor(cap_penalty)

                with amp.autocast('cuda', enabled=False):
                    s_reg = (5e-3) * ((s_k_raw.float() - 0.2)**2 + (s_r_raw.float() - 0.2)**2) \
                          + (7e-3) * ((s_m_raw.float() + 0.2)**2)
                s_reg = _to_scalar_tensor(s_reg)
                    
                cap_penalty_sum += float(cap_penalty.detach().item()) * bs
                s_reg_sum       += float(s_reg.detach().item())       * bs

                mse_k_sum += float(mse_kcat.detach().item())  * bs
                mse_m_sum += float(mse_Km.detach().item())    * bs
                mse_r_sum += float(mse_ratio.detach().item()) * bs

                s_k_raw_sum += float(s_k_raw.detach().mean().item()) * bs
                s_m_raw_sum += float(s_m_raw.detach().mean().item()) * bs
                s_r_raw_sum += float(s_r_raw.detach().mean().item()) * bs

                s_k_eff_sum += float(s_k_eff.detach().mean().item()) * bs
                s_m_eff_sum += float(s_m_eff.detach().mean().item()) * bs
                s_r_eff_sum += float(s_r_eff.detach().mean().item()) * bs
            
                l1 = gate_e.abs().mean() + gate_s.abs().mean()
                tv = (gate_e[:, 1:] - gate_e[:, :-1]).abs().mean() + (gate_s[:, 1:] - gate_s[:, :-1]).abs().mean()

                real_model = getattr(model, "module", model)
                gate_struct = getattr(real_model, "last_gate_struct", None)
                avail_struct = getattr(real_model, "last_struct_avail", None)
                if gate_struct is None:
                    gate_struct = torch.zeros(bs, device=device, dtype=pred_kcat.dtype)
                if avail_struct is None:
                    avail_struct = struct_avail_mask
                else:
                    avail_struct = avail_struct.to(device=device, dtype=pred_kcat.dtype)

                off_mask = avail_struct <= 0.5
                on_mask = avail_struct > 0.5
                l_gate_off = gate_struct[off_mask].pow(2).mean() if off_mask.any() else torch.zeros((), device=device, dtype=pred_kcat.dtype)
                l_gate_on = torch.relu(float(getattr(args, "gate_on_margin", 0.2)) - gate_struct[on_mask]).mean() if on_mask.any() else torch.zeros((), device=device, dtype=pred_kcat.dtype)

                val_loss = (loss_tasks + cap_penalty + s_reg.to(loss_tasks.dtype)
                            + lam_l1 * l1 + lam_tv * tv
                            + float(getattr(args, "lambda_gate_off", 0.0)) * l_gate_off
                            + float(getattr(args, "lambda_gate_on", 0.0)) * l_gate_on)

            total_loss_sum += float(val_loss.item()) * bs
            total_samples  += bs

            if is_main:
                avg_val_loss = total_loss_sum / max(total_samples, 1)
                progress.set_postfix({"avg_val_loss": f"{avg_val_loss:.4f}"})

    if is_ddp:
        t = torch.tensor([total_loss_sum, float(total_samples)],
                         device=device, dtype=torch.float64)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        total_loss_sum = t[0].item()
        total_samples  = int(t[1].item())

    avg_val_loss = total_loss_sum / max(total_samples, 1)
    
    """
    mean_cap_penalty = cap_penalty_sum / max(total_samples, 1)
    mean_s_reg       = s_reg_sum       / max(total_samples, 1)

    mean_s_k_raw = s_k_raw_sum / max(total_samples, 1)
    mean_s_m_raw = s_m_raw_sum / max(total_samples, 1)
    mean_s_r_raw = s_r_raw_sum / max(total_samples, 1)

    mean_s_k_eff = s_k_eff_sum / max(total_samples, 1)
    mean_s_m_eff = s_m_eff_sum / max(total_samples, 1)
    mean_s_r_eff = s_r_eff_sum / max(total_samples, 1)

    if is_main:
        print(
            f"[ValCaps][ep={ep}] cap_penalty={mean_cap_penalty:.4e} | "
            f"s_reg={mean_s_reg:.4e} | "
            f"s_raw(k/m/r)={mean_s_k_raw:.3f}/{mean_s_m_raw:.3f}/{mean_s_r_raw:.3f} | "
            f"s_eff(k/m/r)={mean_s_k_eff:.3f}/{mean_s_m_eff:.3f}/{mean_s_r_eff:.3f}"
        )        
    """
    
    return avg_val_loss



def train_one_stage(model, train_loader, val_loader, args,
                    esm_layers, molt5_layers, criterion,
                    run_dir):
    """
    单阶段训练（全程使用配置）
    - 只训练头部或adapter等（冻结backbone）
    """
    epochs = args.epochs
    total_steps = len(train_loader) * epochs
    warmup_steps = min(500, int(0.1 * total_steps))
    
    is_ddp = dist.is_available() and dist.is_initialized()
    
    patience       = getattr(args, "es_patience", 12)         # 可改
    delta          = getattr(args, "es_delta", 5e-4)      # 改善阈值
    rel_delta      = getattr(args, "es_rel_delta", 0.0) 
    warm_shield    = getattr(args, "es_warm_shield", True)
    no_improve     = 0
    best_val       = float("inf")
    save_dir       = run_dir
    os.makedirs(save_dir, exist_ok=True)
    best_ema_path  = os.path.join(save_dir, "predictor_best_ema.pt")
    best_raw_path  = os.path.join(save_dir, "predictor_best_raw.pt")
    warm_epochs = int(math.ceil(args.epochs * args.onecycle_pct_start)) + 1

    if bool(getattr(args, "kcat_disable_source_residual", False)):
        real_model = getattr(model, "module", model)
        if getattr(real_model, "kcat_source_residual", None) is not None:
            real_model.kcat_source_residual = None
            if is_main_process():
                print("[KCAT-FIX] kcat_source_residual removed (--kcat_disable_source_residual)")

    if bool(getattr(args, "kcat_head_plus_norm_fusion", False)) and (not bool(getattr(args, "kcat_head_only", False))):
        if is_main_process():
            print("[KCAT-HO] --kcat_head_plus_norm_fusion is ignored unless --kcat_head_only is enabled")

    if bool(getattr(args, "kcat_head_only", False)):
        freeze_all_except_kcat_head(
            model,
            disable_source_residual=False,  # 已在上面处理
            train_norm_fusion=bool(getattr(args, "kcat_head_plus_norm_fusion", False)),
        )
    else:
        configure_backbone_trainable(model, args)

    optimizer = build_optimizer_single_stage_stable(model)

    if bool(getattr(args, "kcat_head_only", False)):
        ho_lr = float(getattr(args, "kcat_head_only_lr", 1e-4))
        overridden = 0
        for g in optimizer.param_groups:
            if g.get("tag", "") == "head":
                g["lr"] = ho_lr
                g["base_lr"] = ho_lr
                overridden += 1
        if is_main_process():
            print(f"[KCAT-HO] override {overridden} head param_group(s) lr -> {ho_lr:.2e}")
        try:
            setattr(args, "onecycle_maxlr_head", ho_lr)
        except Exception:
            pass
    
    device = next(model.parameters()).device
    final_contact_store = None
    if getattr(args, "final_contact_npz", None):
        from final_contact import load_final_contact_npz, FinalContactAblationConfig
        final_contact_store = load_final_contact_npz(args.final_contact_npz)
        final_contact_store.set_ablation(
            FinalContactAblationConfig(
                mode=getattr(args, "struct_ablation_mode", "real"),   # none/random/real
                random_seed=getattr(args, "struct_random_seed", 123),
            )
        )
        if is_main_process():
            print(f"[FinalContact] enabled npz={args.final_contact_npz} "
                  f"mode={getattr(args,'struct_ablation_mode','real')} "
                  f"seed={getattr(args,'struct_random_seed',123)}")
    else:
        if is_main_process():
            print("[FinalContact] disabled (args.final_contact_npz is None)")
    
    
    if getattr(args, "lr_find", False) and is_main_process():
        print("[LRF] start lr_range_test_for_tag(head)")
        best_head_lr = lr_range_test_for_tag(
            model=model,
            train_loader=train_loader,
            device=device,
            criterion=criterion,
            tag_name="head",
            start_lr=1e-5,
            end_lr=2e-3,
            num_iter=50,
        )
        args.onecycle_maxlr_head = best_head_lr
        print(f"[LRF] update onecycle_maxlr_head -> {best_head_lr:.3e}")

        print("[LRF] start lr_range_test_for_tag(lora)")
        best_lora_lr = lr_range_test_for_tag(
            model=model,
            train_loader=train_loader,
            device=device,
            criterion=criterion,
            tag_name="lora",
            start_lr=5e-6,
            end_lr=5e-4,
            num_iter=50,
        )
        args.onecycle_maxlr_lora = best_lora_lr
        print(f"[LRF] update onecycle_maxlr_lora -> {best_lora_lr:.3e}")

        optimizer = build_optimizer_single_stage_stable(model)

    if dist.is_available() and dist.is_initialized():
        lr_sync = torch.tensor(
            [float(args.onecycle_maxlr_head), float(args.onecycle_maxlr_lora)],
            device=device,
            dtype=torch.float64,
        )
        dist.broadcast(lr_sync, src=0)
        args.onecycle_maxlr_head = float(lr_sync[0].item())
        args.onecycle_maxlr_lora = float(lr_sync[1].item())

    
    use_onecycle = (args.scheduler == "onecycle")
    use_cosine   = (args.scheduler == "cosine")
    use_constant = (args.scheduler == "none")
    
    onecycle = None
    scheduler = None
    head_group_idx = []
    
    if use_onecycle:
        steps_per_epoch = len(train_loader)
        max_lrs = []
        lora_seen = False
        for g in optimizer.param_groups:
            tag = g.get("tag", "")
            g.setdefault("base_lr", g.get("lr", 0.0))
            
            if tag == "head":
                scale = g.get("head_scale", 1.0)
                max_lrs.append(args.onecycle_maxlr_head * scale)    # ★ kcat/Km/ratio 可用 head_scale 区分
            elif tag in {"lora", "lora_prot", "lora_lig"}:
                lora_scale = g.get("lora_scale", 1.0)
                lora_max = getattr(args, "onecycle_maxlr_lora", 3e-4)
                max_lrs.append(lora_max * lora_scale)
                lora_seen = True
            else:
                max_lrs.append(g["base_lr"])            
        onecycle = OneCycleLR(
            optimizer,
            max_lr=max_lrs,
            epochs=args.epochs, 
            steps_per_epoch=steps_per_epoch,
            pct_start=args.onecycle_pct_start,
            div_factor=args.onecycle_div_factor,
            final_div_factor=args.onecycle_final_div_factor,
            anneal_strategy="cos",
            cycle_momentum=False 
        )
        
        if is_main_process():
            peek = " ".join([
                f"g{gi}:{g.get('tag','?')} max_lr={mlr:.6g} base={g.get('base_lr', g['lr']):.6g}"
                for gi, (g, mlr) in enumerate(zip(optimizer.param_groups, max_lrs))
            ])
            print("[OneCycle] max_lr plan ->", peek)
            if not lora_seen:
                print("[OneCycle][WARN] 未检测到 tag=='lora' 的 param_group；若希望 LoRA 也随 OneCycle 调度，请在构建 optimizer 时给 LoRA 组添加 tag='lora' 与 base_lr。")
    elif use_cosine:
        total_steps  = len(train_loader) * args.epochs
        warmup_steps = min(500, int(0.1 * total_steps))

        scheduler = get_scheduler_with_warmup(
            optimizer, warmup_steps, total_steps, min_lr=1e-6
        )
        
    elif use_constant:
        scheduler = None
        if is_main_process():
            print("[Scheduler] 使用恒定学习率模式 (scheduler = none)，不启用任何 LR 调度器。")
        
    scaler = GradScaler()
    real_model = getattr(model, "module", model)
    ema = EMA(real_model)
    
    print("[INFO] Single-stage  training begins")
    
    
    if getattr(args, "resume_model_only", False):
        start_epoch, global_step = resume_checkpoint_if_available(
            model,
            optimizer=None,
            scheduler=None,
            scaler=None,
            ema=None,
            weights_dir=run_dir,
            device=next(model.parameters()).device,
        )
        start_epoch, global_step = 0, 0
        if is_main_process():
            print("[RESUME] Model-only resume: loaded weights, optimizer/scheduler/EMA/scaler are freshly initialized.")
    else:
        start_epoch, global_step = resume_checkpoint_if_available(
            model, optimizer=optimizer,
            scheduler=(onecycle if use_onecycle else scheduler),
            scaler=scaler, ema=ema,
            weights_dir=run_dir,
            device=next(model.parameters()).device
        )


    device = next(model.parameters()).device

    if getattr(args, "resume", False) or start_epoch > 0 or global_step > 0:
        print("[RESUME] Moving optimizer/scheduler/ema states to current device...")

        for s in optimizer.state.values():
            for k, v in s.items():
                if isinstance(v, torch.Tensor):
                    s[k] = v.to(device, non_blocking=True)

        if use_onecycle and onecycle is not None:
            onecycle._step_count = min(global_step, onecycle.total_steps - 1)
            lrs = onecycle.get_last_lr()
            for pg, lr in zip(optimizer.param_groups, lrs):
                pg["lr"] = lr
            print(f"[MANUAL RESUME] Restored OneCycle step={onecycle._step_count}/{onecycle.total_steps}, "
                  f"lr={[f'{lr:.6g}' for lr in lrs]}")

        if ema is not None and hasattr(ema, "shadow"):
            for k, v in ema.shadow.items():
                if isinstance(v, torch.Tensor):
                    ema.shadow[k] = v.to(device)


    last_ckpt_time = None
    
    for ep in range(start_epoch, epochs):
        
        print(f"[INFO] Epoch {ep}/{epochs} start") 
        
        if is_ddp and isinstance(train_loader.sampler, DistributedSampler):
            train_loader.sampler.set_epoch(ep)
        if is_ddp and isinstance(val_loader.sampler, DistributedSampler):
            val_loader.sampler.set_epoch(ep)
        
        if (not use_onecycle) and (ep >= args.epochs - 1):
            print(f"[Epoch {ep}] 冷却 LoRA，微调 Head 学习率（仅非 OneCycle 模式生效）。")
            for g in optimizer.param_groups:
                names = [
                    n for n, _ in model.named_parameters()
                    if _.requires_grad and id(_) in {id(p) for p in g["params"]}
                ]
                if any("lora_" in n for n in names):
                    g['lr'] = 0.0  # 冷却 LoRA
                if any(
                    ("head" in n or "multi_head" in n or "adapter" in n or "fusion" in n)
                    for n in names
                ):
                    g['lr'] = max(g['lr'], 3.5e-4)  # 头部拉高 lr 做收尾

        
        global_step, last_ckpt_time = train_epoch(
            model, train_loader, optimizer, (onecycle if use_onecycle else scheduler), scaler, ema,
            ep, args, esm_layers, molt5_layers, global_step,
            sanity_check=True, sanity_verbose=True, sanity_diff_thresh=1e-6, sanity_interval=500,
            last_ckpt_time=last_ckpt_time, weights_dir=run_dir,
            use_onecycle=use_onecycle, onecycle=onecycle, 
            final_contact_store=final_contact_store,
        )
        
        if is_main_process() and (((ep + 1) % 5 == 0) or (ep == epochs - 1)):
            save_training_checkpoint(model, optimizer, (onecycle if use_onecycle else scheduler), scaler, ema,
                                     epoch=ep, global_step=global_step,
                                     weights_dir=run_dir, tag="epoch")

        
        val_loss = val_epoch_with_ema(model, ema, val_loader, ep, args,use_ema=True, final_contact_store=final_contact_store)
        if is_main_process():
            swanlab.log({
                "val_loss": val_loss,
                "epoch": ep,
            })
            print(f"[INFO] Epoch {ep} Val Loss: {val_loss:.4f}")
        
        if is_main_process() and (val_loss + delta < best_val):
            best_val = val_loss
            no_improve = 0

            backup = {k: v.detach().clone() for k, v in model.state_dict().items()}
            real_model = getattr(model, "module", model)
            ema.copy_to(real_model)
            to_save = model.module.state_dict() if is_ddp else model.state_dict()
            torch.save(to_save, best_ema_path)
            model.load_state_dict(backup, strict=True)

            to_save = model.module.state_dict() if is_ddp else model.state_dict()
            torch.save(to_save, best_raw_path)  
            
            print(f"[CKPT] New best model saved (val={best_val:.4f})")
        else:
            if (not warm_shield) or (ep > warm_epochs):
                no_improve += 1
                if is_main_process():
                    print(f"[EarlyStop] no_improve={no_improve}/{patience}")
            else:
                if is_main_process():
                    print(f"[EarlyStop] (warmup shield) skip at ep={ep}/{warm_epochs}")

        torch.cuda.empty_cache()
        gc.collect()           
                    
        device = next(model.parameters()).device
        stop = 1 if (is_main_process() and no_improve >= patience) else 0
        if dist.is_available() and dist.is_initialized():
            flag = torch.tensor([stop], device=device, dtype=torch.int32)
            dist.broadcast(flag, src=0)
            stop = int(flag.item())
        if stop:
            if is_main_process():
                print(f"[EarlyStop] Stop training at epoch {ep}, best_val={best_val:.4f}")
            break
        
    ema.copy_to(getattr(model, "module", model))

    print("[INFO] Single-stage training completed")

    


def main():
    global RUNTIME_PATHS
    p = argparse.ArgumentParser()
    p.add_argument("-csv", type=str, required=True)
    p.add_argument("--out_dir", type=str, default=None)
    p.add_argument("--run_name", type=str, default="exp")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("-epochs", type=int, default=40) #40
    p.add_argument("-batch_size", type=int, default=20)
    p.add_argument("-lr", type=float, default=5e-4) #未传入，先不删除，初始化
    p.add_argument("-device", type=str, default="cuda")
    p.add_argument("-esm", type=str, default=None)
    p.add_argument("-molt5", type=str, default=None)
    p.add_argument("-scheduler", type=str, default="cosine", choices=["onecycle", "cosine","none"])
    p.add_argument("-onecycle_maxlr_head", type=float, default=3.545e-4)
    p.add_argument("-onecycle_maxlr_lora", type=float, default=4.143e-4)
    p.add_argument("-onecycle_pct_start", type=float, default=0.3)
    p.add_argument("-onecycle_div_factor", type=float, default=10)        # init lr = max_lr/div
    p.add_argument("-onecycle_final_div_factor", type=float, default=50)  # final lr = max_lr/final_div
    p.add_argument("-es_patience", type=int, default=5)         # from 5 -> 12，更稳
    p.add_argument("-es_delta", type=float, default=5e-4)        # 绝对改进阈值，略放宽
    p.add_argument("-es_rel_delta", type=float, default=0.002)     # 相对改进阈值（例如 0.002 表示 0.2%）
    p.add_argument("-es_warm_shield", action="store_true", default=True,
                   help="仅在超过 onecycle 峰值后才开始累计 no_improve")
    p.add_argument("-resume_model_only",action="store_true",help="只从已有 checkpoint 恢复模型权重，optimizer/scheduler/EMA/scaler 全部重新初始化，用于第二阶段训练",)
    p.add_argument("--kcat_head_only", action="store_true", default=False,
                   help="冻结除 multi_head.heads.kcat 以外的全部参数，只重训 kcat head（修 OOD kcat）")
    p.add_argument("--kcat_head_plus_norm_fusion", action="store_true", default=False,
                   help="与 --kcat_head_only 搭配：额外放开 norm_p/norm_l 与 fusion 层，做小范围适配")
    p.add_argument("--kcat_disable_source_residual", dest="kcat_disable_source_residual", action="store_true",
                   help="训练时把 kcat_source_residual 设为 None（head-only 推荐默认）")
    p.add_argument("--keep_kcat_source_residual", dest="kcat_disable_source_residual", action="store_false",
                   help="保留 kcat_source_residual（用于对照实验）")
    p.set_defaults(kcat_disable_source_residual=True)
    p.add_argument("--kcat_init_weights", type=str, default=None,
                   help="载入已有 checkpoint 作为 head-only 微调的初始权重（strict=False，DDP 包装前加载）")
    p.add_argument("--kcat_head_only_lr", type=float, default=1e-4,
                   help="head-only 模式下 head 组的学习率，覆盖 optimizer 默认值")
    p.add_argument("--kcat_rank_weight", type=float, default=0.0,
                   help="kcat pairwise ranking loss 权重（直接优化 Pearson/Spearman，推荐 0.3）")
    p.add_argument("--kcat_rank_pairs", type=int, default=256,
                   help="每个 microbatch 内采样的 ranking 配对数")
    p.add_argument("--kcat_rank_margin", type=float, default=0.0,
                   help="margin_ranking_loss 的 margin（log10 空间，0 即可）")
    p.add_argument("--kcat_var_reg_weight", type=float, default=0.0,
                   help="kcat 方差保持正则项权重（对抗方差坍塌，推荐 0.1）")
    p.add_argument("--kcat_var_reg_mode", type=str, default="abs", choices=["abs", "ratio"],
                   help="abs=|std(pred)-std(y)|；ratio=max(0, std(y)-std(pred))，只罚过度收缩")
    p.add_argument("--kcat_loss_type", type=str, default="huber", choices=["huber", "barron"],
                   help="kcat 回归损失类型：huber（默认）或 barron（重尾更鲁棒）")
    p.add_argument("--kcat_barron_alpha", type=float, default=1.0,
                   help="Barron robust loss 形状参数 alpha（2=L2, 1=Charbonnier, 0=Cauchy）")
    p.add_argument("--kcat_barron_scale", type=float, default=1.0,
                   help="Barron robust loss 尺度参数 c")
    p.add_argument("--kcat_use_group_dro", action="store_true", default=False,
                   help="仅对 kcat 启用 GroupDRO（按 source_id 分组的最坏组优化）")
    p.add_argument("--kcat_group_dro_eta", type=float, default=0.05,
                   help="GroupDRO 组权重更新速率 eta")
    p.add_argument("--kcat_group_dro_num_groups", type=int, default=3,
                   help="GroupDRO 分组数（通常设为 num_domains）")
    p.add_argument("--kcat_group_dro_mix", type=float, default=1.0,
                   help="GroupDRO 混合权重：0=纯平均, 1=纯DRO")
    p.add_argument("--lr_find",action="store_true",default=False,help="启用 LR Finder，对 head 和 lora 分别扫描推荐 onecycle_maxlr",)
    p.add_argument("--unfreeze_protein_last_k", type=int, default=0,help="解冻 ESM 最后 N 层 encoder block，0=不解冻")
    p.add_argument("--unfreeze_ligand_last_k", type=int, default=0,help="解冻 MolT5 最后 N 层 encoder block，0=不解冻")
    p.add_argument("--lr_backbone", type=float, default=7e-6,help="解冻的 backbone 层使用的学习率")
    p.add_argument("--wd_backbone", type=float, default=1e-2,help="解冻 backbone 层 weight decay")
    p.add_argument("--use_interactions", action="store_true", dest="use_interactions")
    p.add_argument("--no_use_interactions", action="store_false", dest="use_interactions")
    p.set_defaults(use_interactions=True)
    p.add_argument("--use_gate_p", action="store_true", dest="use_gate_p")
    p.add_argument("--no_use_gate_p", action="store_false", dest="use_gate_p")
    p.set_defaults(use_gate_p=True)
    p.add_argument("--use_gate_l", action="store_true", dest="use_gate_l")
    p.add_argument("--no_use_gate_l", action="store_false", dest="use_gate_l")
    p.set_defaults(use_gate_l=True)
    p.add_argument("--use_struct_branch", action="store_true", default=False,
               help="是否启用结构分支（prot_struct/lig_struct）")
    p.add_argument("--struct_in_dim_prot", type=int, default=45,
               help="prot_struct 向量维度（NPZ里 prot_struct 的第二维）")
    p.add_argument("--struct_in_dim_lig", type=int, default=135,
                   help="lig_struct 向量维度（NPZ里 lig_struct 的第二维）")
    p.add_argument("--struct_fusion_mode", type=str, default="concat",
                   choices=["concat", "add"],
                   help="结构向量与主干向量融合方式（按 predictor 实现）")   
    p.add_argument("--final_contact_npz", type=str, default=None,
               help="final_contact结构向量npz路径。None表示不启用store注入。")
    p.add_argument("--struct_ablation_mode", type=str, default="real",
               choices=["none", "random", "real"],
               help="结构分支消融模式：none(不加) / random(同维随机) / real(真实结构)")
    p.add_argument("--struct_random_seed", type=int, default=123,
               help="random结构向量的随机种子（仅struct_ablation_mode=random生效）")
    p.add_argument("--source_col", type=str, default="source_id",
               help="域标签列名（用于 domain id 映射），如 source_id/source/dataset")
    p.add_argument("--split_source_col", type=str, default=None,
               help="仅用于 split_mode=domain_ood 的划分列名；不影响域标签映射")
    p.add_argument("--default_source_id", type=int, default=0,
               help="当找不到域标签列时使用的默认域 id")
    p.add_argument("--enable_avail_gate", action="store_true", default=False,
               help="启用结构可用性感知门控")
    p.add_argument("--avail_gate_hidden", type=int, default=256,
               help="可用性感知门控 MLP 隐层维度")
    p.add_argument("--lambda_gate_off", type=float, default=0.0,
               help="无结构样本门控约束权重")
    p.add_argument("--lambda_gate_on", type=float, default=0.0,
               help="有结构样本门控利用约束权重")
    p.add_argument("--gate_on_margin", type=float, default=0.2,
               help="有结构样本门控下界 margin")
    p.add_argument("--gate_warmup_epochs", type=int, default=2,
               help="前 N 个 epoch 关闭 gate mask")
    p.add_argument("--find_unused_parameters", action="store_true", default=False,
               help="DDP find_unused_parameters（消融实验关闭分支时开启）")
    p.add_argument("--lambda_cons", type=float, default=0.0,
               help="一致性双视图损失权重")
    p.add_argument("--cons_warmup_epochs", type=int, default=0,
               help="一致性损失 warmup epoch")
    p.add_argument("--cons_w_kcat", type=float, default=1.0)
    p.add_argument("--cons_w_km", type=float, default=1.0)
    p.add_argument("--cons_w_ratio", type=float, default=1.0)
    p.add_argument("--struct_dropout_prob", type=float, default=0.0,
               help="训练时结构分支 dropout 比例")
    p.add_argument("--domain_uncertainty", action="store_true", default=False,
               help="启用按域×任务不确定性参数")
    p.add_argument("--num_domains", type=int, default=3,
               help="域数量，用于 domain uncertainty")
    p.add_argument("--split_mode", type=str, default="random", choices=["random", "group_pair", "domain_ood"],
               help="数据集划分策略：random / group_pair / domain_ood")
    p.add_argument("--split_seed", type=int, default=42,
               help="数据划分随机种子")
    p.add_argument("--val_ratio", type=float, default=0.2,
               help="验证集占全量比例（random）或占非 holdout 训练池比例（domain_ood）")
    p.add_argument("--test_ratio", type=float, default=0.2,
               help="测试集占全量比例（仅 random 有效）")
    p.add_argument("--holdout_source", type=str, default=None,
               help="domain_ood 时作为测试域的 source_id 值，例如 catapro/skid-kcat/skid-km")
    p.add_argument("--ood_dedup_pair", action="store_true", dest="ood_dedup_pair",
               help="domain_ood 时去掉 train/val 中与 holdout 测试集 pair 重叠样本")
    p.add_argument("--no_ood_dedup_pair", action="store_false", dest="ood_dedup_pair",
               help="关闭 domain_ood 的 pair 去重")
    p.set_defaults(ood_dedup_pair=True)
    p.add_argument("--save_split_csv", action="store_true", default=True,
               help="是否在 run_dir 保存 train/val/test 划分文件")
    p.add_argument("--test_struct_keep_ratio", type=float, default=1.0,
               help="测试时结构证据保留比例（0~1），用于缺失鲁棒性曲线")
    p.add_argument("--use_phys_evidence", action="store_true", default=False,
               help="启用物理证据分支（phys_*）")
    p.add_argument("--phys_drop_p", type=float, default=0.0,
               help="训练时物理证据 dropout 比例")
    p.add_argument("--phys_ablation", type=str, default="real", choices=["real", "shuffle", "random", "none"],
               help="物理证据反事实模式")
    p.add_argument("--eval_only", action="store_true", default=False,
               help="跳过训练，只跑测试集评估（需配合 --weights 指定权重文件）")
    p.add_argument("--weights", type=str, default=None,
               help="eval_only 模式下加载的权重文件路径（.pt）")
    p.add_argument("--test_csv", type=str, default=None,
               help="eval_only 模式下指定测试集 CSV（覆盖 -csv 的默认划分逻辑）")

    args = p.parse_args()
    if (KineticsPredictor is None) or (PredictorConfig is None):
        raise ImportError(
            "Failed to import models.predictor dependencies. "
            f"Original error: {_PREDICTOR_IMPORT_ERROR}"
        )
    if AutoConfig is None:
        raise ImportError("transformers is required for training. Please install transformers.")

    RUNTIME_PATHS = resolve_runtime_paths()
    ensure_paths(
        RUNTIME_PATHS,
        required_keys=["runs_dir", "train_data_dir", "esm_model_path", "molt5_model_path"],
    )

    if args.out_dir is None:
        args.out_dir = str(require_path("runs_dir"))
    if args.esm is None:
        args.esm = RUNTIME_PATHS.get("esm_model_path")
    if args.molt5 is None:
        args.molt5 = RUNTIME_PATHS.get("molt5_model_path")
    if not args.esm or not args.molt5:
        raise ValueError("ESM/MolT5 model path is not configured. Set -esm/-molt5 or CATA_* env vars.")

    print(f"[PATHS] profile={RUNTIME_PATHS['profile']}")
    print(f"[PATHS] runs_dir={args.out_dir}")
    print(f"[PATHS] train_data_dir={require_path('train_data_dir')}")
    print(f"[PATHS] esm_model={args.esm}")
    print(f"[PATHS] molt5_model={args.molt5}")

    need_row_id = True
    run_dir = os.path.join(args.out_dir, args.run_name, f"seed{args.seed}")
    os.makedirs(run_dir, exist_ok=True)
    globals()["run_dir"] = run_dir

    if is_main_process():
        exp_manifest = {
            "args": vars(args),
            "runtime_paths": {k: (str(v) if isinstance(v, Path) else v) for k, v in RUNTIME_PATHS.items()},
        }
        with open(os.path.join(run_dir, "experiment_manifest.json"), "w", encoding="utf-8") as f:
            json.dump(exp_manifest, f, ensure_ascii=False, indent=2)
    
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    
    use_ddp = int(os.environ.get("WORLD_SIZE", "1")) > 1
    if use_ddp:
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        torch.cuda.set_device(local_rank)          # 先绑定设备
        dist.init_process_group(backend="nccl", init_method="env://")
        is_ddp = True
    else:
        local_rank = 0
        rank = 0
        world_size = 1
        is_ddp = False

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    
    
    run = None
    if is_main_process():
        run = swanlab.init(
            project="enzyme-kinetics-cv", 
            config={
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "device": args.device,
                "esm": args.esm,
                "molt5": args.molt5,
                "csv": args.csv
            }
        )       

    df = pd.read_csv(args.csv)
    df = normalize_training_dataframe(df)
    phys_cols = detect_phys_columns(df)
    use_phys_evidence = bool(getattr(args, "use_phys_evidence", False)) and (len(phys_cols) > 0)
    if is_main_process():
        print(f"[PHYS] use_phys_evidence={use_phys_evidence} phys_dim={len(phys_cols)}")
    if args.split_source_col is not None:
        if args.split_source_col not in df.columns:
            raise ValueError(f"split_source_col '{args.split_source_col}' not found in csv columns")
        split_source_col = args.split_source_col
    else:
        split_source_col = args.source_col if args.source_col in df.columns else "source_id"
    train_df, val_df, test_df, split_meta = split_dataframe_for_experiment(
        df=df,
        split_mode=args.split_mode,
        split_seed=args.split_seed,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        source_col=split_source_col,
        holdout_source=args.holdout_source,
        ood_dedup_pair=bool(getattr(args, "ood_dedup_pair", True)),
    )

    train_source_ids, source_col_used = infer_source_ids(
        train_df,
        source_col=getattr(args, "source_col", None),
        default_source_id=int(getattr(args, "default_source_id", 0)),
    )
    val_source_ids, _ = infer_source_ids(
        val_df,
        source_col=source_col_used,
        default_source_id=int(getattr(args, "default_source_id", 0)),
    )
    test_source_ids, _ = infer_source_ids(
        test_df,
        source_col=source_col_used,
        default_source_id=int(getattr(args, "default_source_id", 0)),
    )
    if is_main_process():
        uniq, cnt = np.unique(train_source_ids, return_counts=True)
        print(f"[SOURCE] col={source_col_used or 'None(default)'} train_dist={dict(zip(uniq.tolist(), cnt.tolist()))}")
    
    if is_main_process() and args.save_split_csv:
        train_df.to_csv(os.path.join(run_dir, "train_set.csv"), index=False)
        val_df.to_csv(os.path.join(run_dir, "val_set.csv"), index=False)
        test_df.to_csv(os.path.join(run_dir, "test_set.csv"), index=False)
        print(f"[Split] train/val/test csv saved to {run_dir}")
    if is_main_process():
        with open(os.path.join(run_dir, "split_meta.json"), "w", encoding="utf-8") as f:
            json.dump(split_meta, f, ensure_ascii=False, indent=2)
        print(f"[Split] mode={split_meta.get('split_mode')} counts={split_meta.get('counts')}")
        if "pair_overlap" in split_meta:
            print(f"[Split] pair_overlap={split_meta.get('pair_overlap')}")
        if split_meta.get("ood_dedup_pair", False):
            print(f"[Split] ood_dedup_removed_rows={split_meta.get('ood_dedup_removed_rows', 0)}")

    kscaler = KineticsScaler()
    kscaler.fit(train_df["kcat(s^-1)"], train_df["Km(M)"])
    if is_main_process():
        kscaler.save(os.path.join(run_dir, "scaler_fold.json"))
        
    

    train_dataset = EnzymeDataset(
        train_df,
        scaler=kscaler,
        supervised=True,
        return_row_id=need_row_id,
        source_ids=train_source_ids,
        default_source_id=int(getattr(args, "default_source_id", 0)),
        phys_cols=phys_cols if use_phys_evidence else [],
    )
    val_dataset = EnzymeDataset(
        val_df,
        scaler=kscaler,
        supervised=True,
        return_row_id=need_row_id,
        source_ids=val_source_ids,
        default_source_id=int(getattr(args, "default_source_id", 0)),
        phys_cols=phys_cols if use_phys_evidence else [],
    )
    
    is_ddp = dist.is_available() and dist.is_initialized()

    if is_ddp:
        train_sampler = DistributedSampler(train_dataset, shuffle=True, drop_last=False)
        val_sampler   = DistributedSampler(val_dataset,   shuffle=False, drop_last=False)
    else:
        train_sampler = None
        val_sampler   = None

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,                      # 自动分区
        shuffle=(train_sampler is None),            # 只有单卡时再 shuffle
        num_workers=8,
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        sampler=val_sampler,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=False
    )

    print("Dataset & Dataloader initialized successfully.")
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    if is_ddp:
        print(f"DistributedSampler enabled. World size = {torch.distributed.get_world_size()}")

    print("dataset division finished")
        
    esm_cfg = AutoConfig.from_pretrained(args.esm)
    esm_layers = esm_cfg.num_hidden_layers
    print(f"[INFO] ESM ({args.esm}) has {esm_layers} layers")
        
    molt5_cfg = AutoConfig.from_pretrained(args.molt5)
    molt5_layers = molt5_cfg.num_hidden_layers
    print(f"[INFO] MolT5 ({args.molt5}) has {molt5_layers} layers")
        
    total_layers = max(esm_layers, molt5_layers)
        
    
    

    cfg = PredictorConfig(
    d_model=768,num_heads=8,num_interaction_layers=4,
    num_mamba_layers=1,rate=0.1,head_hidden=768,
    use_attention_pool=True,use_checkpoint=True,             
    num_stages=2 ,num_interaction_layers_per_stage=3,
        
    use_struct_branch=args.use_struct_branch,         # 新增：总开关
    struct_in_dim_prot=args.struct_in_dim_prot,       # 新增：prot_struct 维度（45）
    struct_in_dim_lig=args.struct_in_dim_lig,         # 新增：lig_struct 维度（135）
    struct_fusion_mode=getattr(args, "struct_fusion_mode", "concat"),  # 若 predictor 支持
    enable_avail_gate=bool(getattr(args, "enable_avail_gate", False)),
    avail_gate_hidden=int(getattr(args, "avail_gate_hidden", 256)),
    use_phys_branch=use_phys_evidence,
    phys_in_dim=len(phys_cols) if use_phys_evidence else 0,
    phys_hidden=128,
    domain_uncertainty=bool(getattr(args, "domain_uncertainty", False)),
    num_domains=int(getattr(args, "num_domains", 3)),

    use_interactions=args.use_interactions,
    use_gate_p=args.use_gate_p,
    use_gate_l=args.use_gate_l,
                              
    active_heads=["kcat", "Km", "ratio"],

    fusion_norm=True,
    fusion_proj_dim=512,  

    head_specs={
        "kcat":  {"type": "mlp", "hidden_dim": 768, "depth": 3, "dropout": 0.10, "out_dim": 1},
        "Km":    {"type": "mlp",  "hidden_dim": 640,  "depth": 3, "dropout": 0.10, "out_dim": 1},
        "ratio": {"type": "mlp",  "out_dim": 1},
    },

    use_moe_in_head=False,
    moe_num_experts=2,
    moe_top_k=1,
                          
    num_heads_in_head=1,
    head_hidden_dim=768,
    head_depth=3)    
    
    model = KineticsPredictor(device=device, esm_model=args.esm, molt5_model=args.molt5, cfg=cfg, use_lora=True, lora_r=32, lora_alpha=64,lora_dropout=0.05)

    model.to(device)

    _kcat_init = getattr(args, "kcat_init_weights", None)
    if _kcat_init:
        if os.path.exists(_kcat_init):
            _ckpt_obj = torch.load(_kcat_init, map_location=next(model.parameters()).device)
            if isinstance(_ckpt_obj, dict) and ("state_dict" in _ckpt_obj) and not any(
                isinstance(v, torch.Tensor) for v in _ckpt_obj.values()
            ):
                _state = _ckpt_obj["state_dict"]
            else:
                _state = _ckpt_obj
            _missing, _unexpected = model.load_state_dict(_state, strict=False)
            if is_main_process():
                print(f"[KCAT-INIT] loaded init weights from {_kcat_init}")
                print(f"[KCAT-INIT] missing={len(_missing)} unexpected={len(_unexpected)}")
                if _missing:
                    print(f"[KCAT-INIT] first missing keys: {list(_missing)[:5]}")
                if _unexpected:
                    print(f"[KCAT-INIT] first unexpected keys: {list(_unexpected)[:5]}")
            del _ckpt_obj, _state
        else:
            if is_main_process():
                print(f"[KCAT-INIT] WARNING file not found: {_kcat_init}")

    criterion = torch.nn.MSELoss()
    if is_ddp:
        find_unused = bool(getattr(args, "find_unused_parameters", False))
        if (not find_unused) and (
            bool(getattr(args, "domain_uncertainty", False))
            or str(getattr(args, "split_mode", "random")) == "domain_ood"
        ):
            find_unused = True
            if is_main_process():
                print("[DDP] Auto-enable find_unused_parameters=True for domain_uncertainty/domain_ood stability")
        if (not find_unused) and bool(getattr(args, "kcat_head_only", False)):
            find_unused = True
            if is_main_process():
                print("[DDP] Auto-enable find_unused_parameters=True for --kcat_head_only (only kcat head receives grads)")

        has_cons = float(getattr(args, "lambda_cons", 0.0)) > 0.0
        if has_cons and not find_unused:
            if is_main_process():
                print("[DDP] Auto-disable static_graph because lambda_cons > 0 (dual forward pass per step)")

        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=find_unused,
            static_graph=(not find_unused) and (not has_cons),
        )
    
    if is_main_process():
        print("Model set")
        print("[ARGS] use_interactions =", args.use_interactions)
        print("[CFG ] use_interactions =", cfg.use_interactions)
        print("[CFG ] use_gate_p =", cfg.use_gate_p, "use_gate_l =", cfg.use_gate_l)
        real_model = getattr(model, "module", model)
        print("Number of interaction blocks:",
          sum(1 for _ in real_model.interactions))
        
    if getattr(args, "eval_only", False):
        weights_path = getattr(args, "weights", None)
        if weights_path and os.path.exists(weights_path):
            state = torch.load(weights_path, map_location=next(model.parameters()).device)
            real_model = getattr(model, "module", model)
            real_model.load_state_dict(state, strict=True)
            if is_main_process():
                print(f"[EVAL_ONLY] Loaded weights from {weights_path}")
        else:
            if is_main_process():
                print(f"[EVAL_ONLY] WARNING: --weights not provided or file not found: {weights_path}")
        test_csv_override = getattr(args, "test_csv", None)
        if test_csv_override and os.path.exists(test_csv_override):
            import pandas as _pd
            test_df = normalize_training_dataframe(_pd.read_csv(test_csv_override))
            test_source_ids, _ = infer_source_ids(
                test_df,
                source_col=getattr(args, "source_col", None),
                default_source_id=int(getattr(args, "default_source_id", 0)),
            )
            if is_main_process():
                print(f"[EVAL_ONLY] Using test CSV: {test_csv_override} ({len(test_df)} rows)")
    else:
        print("Training begin")
        train_one_stage(model, train_loader, val_loader, args, esm_layers, molt5_layers,criterion,run_dir)

    test_loader = make_dataloader(
        test_df,
        scaler=kscaler,
        batch_size=args.batch_size,
        shuffle=False,
        supervised=True,
        return_row_id=need_row_id,
        source_ids=test_source_ids,
        default_source_id=int(getattr(args, "default_source_id", 0)),
        phys_cols=phys_cols if use_phys_evidence else [],
    ) # 非十折版本，使用十折时请修改
    
    device = next(model.parameters()).device
    
    if is_main_process():
        real_model = getattr(model, "module", model)

        test_dataset = test_loader.dataset
        test_loader_rank0 = torch.utils.data.DataLoader(
            test_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=0, pin_memory=True, drop_last=False
        )

    
    
        if not getattr(args, "eval_only", False):
            save_dir = run_dir
            best_ema_path = os.path.join(save_dir, "predictor_best_ema.pt")
            if os.path.exists(best_ema_path):
                state = torch.load(best_ema_path, map_location=device)
                real_model = getattr(model, "module", model)
                real_model.load_state_dict(state, strict=True)
                print(f"[TEST] Loaded best EMA weights from {best_ema_path}")
            else:
                print(f"[TEST] WARNING: best EMA weights not found, using current weights.")   
    
        real_model.eval()
        
        final_contact_store = None
        if args.final_contact_npz is not None:
            from final_contact import load_final_contact_npz, FinalContactAblationConfig
            final_contact_store = load_final_contact_npz(args.final_contact_npz)
            final_contact_store.set_ablation(
                FinalContactAblationConfig(
                    mode=args.struct_ablation_mode,
                    random_seed=args.struct_random_seed,
                )
            )
            print(f"[FinalContact][TEST] mode={args.struct_ablation_mode} npz={args.final_contact_npz}")
        else:
            print("[FinalContact][TEST] disabled (final_contact_npz=None)")
        
        y_true, y_pred = [], []
        y_source = []
        sigma_pred = []
        y_struct_avail = []  # 每行结构证据可用性（1=结构已注入），用于结构可用子集口径
        eval_rng = np.random.default_rng(args.seed + 2026)
        
        with torch.no_grad():
            for batch in tqdm(test_loader_rank0, desc="Test (Rank0)"):
                seqs, smiles, row_ids, source_ids, p_s, l_s, labels, struct_avail_mask, phys_feat, phys_mask, phys_quality = unpack_batch(batch)
                labels = labels.to(device, non_blocking=True)
                phys_feat, phys_mask, phys_quality = infer_phys_inputs(
                    phys_feat,
                    phys_mask,
                    phys_quality,
                    batch_size=labels.size(0),
                    device=device,
                )
                if phys_feat is not None:
                    phys_feat, phys_mask, phys_quality = apply_phys_ablation(
                        phys_feat,
                        phys_mask,
                        phys_quality,
                        mode=str(getattr(args, "phys_ablation", "real")),
                    )
                struct_avail_mask = infer_struct_avail_mask(
                    p_s,
                    l_s,
                    batch_size=labels.size(0),
                    device=device,
                    provided_mask=struct_avail_mask,
                )
                
                if isinstance(row_ids, torch.Tensor):
                    row_ids = row_ids.detach().cpu()
                    
                if p_s is not None:
                    p_s = p_s.to(device, non_blocking=True)
                    l_s = l_s.to(device, non_blocking=True)
                    struct_avail_mask = infer_struct_avail_mask(
                        p_s,
                        l_s,
                        batch_size=labels.size(0),
                        device=device,
                    )

                if (row_ids is not None) and (final_contact_store is not None):
                    p_s2, l_s2 = final_contact_store.get_batch_by_rowids_torch(row_ids, device=device)
                    p_s, l_s = p_s2, l_s2
                    struct_avail_mask = infer_struct_avail_mask(
                        p_s,
                        l_s,
                        batch_size=labels.size(0),
                        device=device,
                    )

                keep_ratio = float(getattr(args, "test_struct_keep_ratio", 1.0))
                if keep_ratio < 1.0:
                    keep_ratio = max(0.0, min(1.0, keep_ratio))
                    keep_np = (eval_rng.random(labels.size(0)) < keep_ratio).astype(np.float32)
                    keep = torch.as_tensor(keep_np, device=device, dtype=struct_avail_mask.dtype)
                    struct_avail_mask = struct_avail_mask * keep
                    if p_s is not None:
                        p_s = p_s * keep.unsqueeze(-1)
                        l_s = l_s * keep.unsqueeze(-1)

                source_ids_eval = to_source_tensor(source_ids, labels.size(0), device=device)

                try:
                    p1, p2, p3_tmp, gate_e, gate_s, s_raw, s_eff = real_model(
                        seqs, smiles,
                        prot_struct=p_s,
                        lig_struct=l_s,
                        struct_avail_mask=struct_avail_mask,
                        phys_feat=phys_feat,
                        phys_mask=phys_mask,
                        phys_quality=phys_quality,
                        source_ids=source_ids_eval,
                        use_mask=False
                    )
                except TypeError:
                    try:
                        p1, p2, p3_tmp, gate_e, gate_s, s_raw, s_eff = real_model(
                            seqs, smiles,
                            prot_struct=p_s,
                            lig_struct=l_s,
                            struct_avail_mask=struct_avail_mask,
                            phys_feat=phys_feat,
                            phys_mask=phys_mask,
                            phys_quality=phys_quality,
                            use_mask=False,
                        )
                    except TypeError:
                        p1, p2, p3_tmp, gate_e, gate_s, s_raw, s_eff = real_model(
                            seqs, smiles, use_mask=False
                        )
                p3 = p1 - p2 + 3.0  # 派生 ratio

                preds = torch.stack([p1, p2, p3], dim=-1).cpu().numpy()
                y_true.append(labels.cpu().numpy())
                y_pred.append(preds)
                y_source.append(source_ids_eval.detach().cpu().numpy().reshape(-1))
                y_struct_avail.append(struct_avail_mask.detach().cpu().numpy().reshape(-1))

                s_k_eff, s_m_eff, s_r_eff = s_eff

                def _expand_s(x, bsz):
                    if isinstance(x, torch.Tensor):
                        t = x.detach().to(device=labels.device, dtype=labels.dtype).reshape(-1)
                    else:
                        t = torch.as_tensor(x, device=labels.device, dtype=labels.dtype).reshape(-1)
                    if t.numel() == 1:
                        t = t.expand(bsz)
                    return t

                bsz = labels.size(0)
                sk = _expand_s(s_k_eff, bsz)
                sm = _expand_s(s_m_eff, bsz)
                sr = _expand_s(s_r_eff, bsz)
                sig = torch.stack([
                    torch.exp(0.5 * sk),
                    torch.exp(0.5 * sm),
                    torch.exp(0.5 * sr),
                ], dim=-1).cpu().numpy()
                sigma_pred.append(sig)

        y_true = np.concatenate(y_true, axis=0)  # (N, 3)
        y_pred = np.concatenate(y_pred, axis=0)  # (N, 3)
        y_source = np.concatenate(y_source, axis=0)
        sigma_pred = np.concatenate(sigma_pred, axis=0)
        y_struct_avail = np.concatenate(y_struct_avail, axis=0)

        def _temperature_scale_sigma(y_true_col, y_pred_col, sigma_col):
            valid = np.isfinite(y_true_col) & np.isfinite(sigma_col) & (sigma_col > 0)
            if valid.sum() < 10:
                return sigma_col  # 样本太少，跳过校准
            n_cal = max(10, int(valid.sum() * 0.2))
            idx = np.where(valid)[0][:n_cal]
            err_cal = np.abs(y_true_col[idx] - y_pred_col[idx])
            sig_cal = sigma_col[idx]
            from scipy.optimize import minimize_scalar
            def nll(log_T):
                T = np.exp(log_T)
                sg = sig_cal * T
                return float(np.mean(0.5 * (err_cal**2 / sg**2 + 2.0 * np.log(sg))))
            try:
                res = minimize_scalar(nll, bounds=(-2.0, 2.0), method='bounded')
                T_opt = float(np.exp(res.x))
            except Exception:
                T_opt = 1.0
            sigma_out = sigma_col.copy()
            sigma_out[valid] = sigma_col[valid] * T_opt
            return sigma_out

        sigma_calibrated = sigma_pred.copy()
        for _ti in range(sigma_pred.shape[1]):
            sigma_calibrated[:, _ti] = _temperature_scale_sigma(
                y_true[:, _ti], y_pred[:, _ti], sigma_pred[:, _ti]
            )

        fold_metrics = {}
        task_names = ["kcat", "Km", "kcat/Km"]
        for i, name in enumerate(task_names):
            fold_metrics.update(_compute_task_metrics_with_sigma(y_true[:, i], y_pred[:, i], sigma_calibrated[:, i], name))
        _add_macro_metrics(fold_metrics, task_names)

        unique_sources = sorted({str(x) for x in y_source.tolist()})
        fold_metrics["by_source"] = {}
        for src in unique_sources:
            m = (y_source.astype(str) == src)
            src_metrics = {}
            for i, name in enumerate(task_names):
                src_metrics.update(_compute_task_metrics_with_sigma(y_true[m, i], y_pred[m, i], sigma_calibrated[m, i], name))
            _add_macro_metrics(src_metrics, task_names)
            fold_metrics["by_source"][src] = src_metrics

        if is_main_process():
            swanlab_log = {}
            for name in task_names:
                for key in ["mse", "rmse", "mae", "r2", "pearson_r", "spearman_r", "nll", "cov90", "cov95"]:
                    k = f"{name}_{key}"
                    if k in fold_metrics:
                        swanlab_log[k] = fold_metrics[k]
            swanlab.log(swanlab_log)

        np.savez_compressed(
            os.path.join(run_dir, "test_predictions.npz"),
            y_true=y_true,
            y_pred=y_pred,
            sigma=sigma_calibrated,
            sigma_raw=sigma_pred,
            source_id=y_source,
            struct_avail=y_struct_avail,
        )

        struct_sub_mask = (y_struct_avail > 0.5)
        n_struct_sub = int(struct_sub_mask.sum())
        print(f"[STRUCT-SUBSET] struct-available rows: {n_struct_sub} / {y_struct_avail.shape[0]}")
        if n_struct_sub > 0:
            yt_sub = y_true[struct_sub_mask]
            yp_sub = y_pred[struct_sub_mask]
            sg_raw_sub = sigma_pred[struct_sub_mask]
            ys_sub = y_source[struct_sub_mask]

            sg_cal_sub = sg_raw_sub.copy()
            for _ti in range(sg_raw_sub.shape[1]):
                sg_cal_sub[:, _ti] = _temperature_scale_sigma(
                    yt_sub[:, _ti], yp_sub[:, _ti], sg_raw_sub[:, _ti]
                )

            sub_metrics = {}
            for i, name in enumerate(task_names):
                sub_metrics.update(_compute_task_metrics_with_sigma(yt_sub[:, i], yp_sub[:, i], sg_cal_sub[:, i], name))
            _add_macro_metrics(sub_metrics, task_names)

            sub_unique_sources = sorted({str(x) for x in ys_sub.tolist()})
            sub_metrics["by_source"] = {}
            for src in sub_unique_sources:
                m = (ys_sub.astype(str) == src)
                src_metrics = {}
                for i, name in enumerate(task_names):
                    src_metrics.update(_compute_task_metrics_with_sigma(yt_sub[m, i], yp_sub[m, i], sg_cal_sub[m, i], name))
                _add_macro_metrics(src_metrics, task_names)
                sub_metrics["by_source"][src] = src_metrics

            sub_path = os.path.join(run_dir, "metrics_struct_available.json")
            with open(sub_path, "w") as f:
                json.dump(sub_metrics, f, indent=2)
            print(
                f"[STRUCT-SUBSET] macro_mae={sub_metrics.get('macro_mae'):.6f} "
                f"macro_r2={sub_metrics.get('macro_r2'):.6f} "
                f"macro_nll={sub_metrics.get('macro_nll'):.6f} "
                f"kcat_n={sub_metrics.get('kcat_n')} Km_n={sub_metrics.get('Km_n')} "
                f"ratio_n={sub_metrics.get('kcat/Km_n')} -> {sub_path}"
            )

    if is_main_process():
        print("测试集指标：")
        print(json.dumps(fold_metrics, indent=2))
        metrics_path = os.path.join(run_dir, "metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(fold_metrics, f, indent=2)
        print("指标已保存为 metrics.json in {metrics_path}")

    if is_main_process():
        real_model = getattr(model, "module", model)
        to_save = model.module.state_dict() if is_ddp else model.state_dict()
        model_path = os.path.join(run_dir, "predictor.pt")
        torch.save(to_save, model_path)
        print("Saved predictor.pt in {model_path}")

if __name__ == "__main__":
    start_tracing()
    try:
        main()
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
