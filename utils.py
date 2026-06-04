import torch
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
try:
    import swanlab
except Exception:
    class _SwanlabStub:
        @staticmethod
        def log(*args, **kwargs):
            return None
    swanlab = _SwanlabStub()
import numpy as np
import json
from sklearn.preprocessing import StandardScaler
import os, torch
import torch.distributed as dist


def masked_mean(x, mask):
    if mask is None:
        return x.mean(dim=1)
    mask = mask.unsqueeze(-1)  # [B, T, 1]
    s = (x * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp_min(1)
    return s / denom

def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class EMA:
    def __init__(self, params, decay=0.999):
        self.shadow = [p.detach().clone() for p in params if p.requires_grad]
        self.decay = decay
    @torch.no_grad()
    def update(self, params):
        i = 0
        for p in params:
            if not p.requires_grad: 
                continue
            self.shadow[i].mul_(self.decay).add_(p.detach(), alpha=1-self.decay)
            i += 1

            
class KineticsScaler:
    def __init__(self, eps=1e-12, quantile_range=(0.5, 99.5)):
        self.eps = eps
        self.quantile_range = quantile_range
        self.ratio_min = None
        self.ratio_max = None

        self.scaler_kcat = StandardScaler()
        self.scaler_Km = StandardScaler()
        self.scaler_ratio = StandardScaler()

    def fit(self, kcat, Km):
        kcat = np.array(kcat, dtype=np.float32)
        Km = np.array(Km, dtype=np.float32)
        mask_k = np.isfinite(kcat) & (kcat > 0)
        mask_m = np.isfinite(Km) & (Km > 0)
        mask_r = mask_k & mask_m

        log_kcat = np.log10(kcat[mask_k] + self.eps)
        log_Km = np.log10(Km[mask_m] * 1000 + self.eps)  # 转 mM 再 log

        if mask_r.any():
            ratio = kcat[mask_r] / (Km[mask_r] + self.eps)
            ratio = ratio[np.isfinite(ratio) & (ratio > 0)]
        else:
            ratio = np.array([], dtype=np.float32)

        if ratio.size == 0:
            self.ratio_min = 1e-12
            self.ratio_max = 1.0
            log_ratio = np.log10(np.array([1.0], dtype=np.float32) + self.eps)
        else:
            self.ratio_min = np.percentile(ratio, self.quantile_range[0])
            self.ratio_max = np.percentile(ratio, self.quantile_range[1])
            log_ratio = np.log10(ratio + self.eps)

        print(f"[Scaler] ratio clip range: {self.ratio_min:.3e} ~ {self.ratio_max:.3e}")

        if log_kcat.size == 0:
            log_kcat = np.array([0.0], dtype=np.float32)
        if log_Km.size == 0:
            log_Km = np.array([0.0], dtype=np.float32)

        self.scaler_kcat.fit(log_kcat.reshape(-1, 1))
        self.scaler_Km.fit(log_Km.reshape(-1, 1))
        self.scaler_ratio.fit(log_ratio.reshape(-1, 1))

    def transform(self, kcat, Km):
        kcat = np.array(kcat, dtype=np.float32)
        Km = np.array(Km, dtype=np.float32)
        ratio = kcat / (Km + self.eps)

        log_kcat = np.full_like(kcat, np.nan, dtype=np.float32)
        log_Km = np.full_like(Km, np.nan, dtype=np.float32)
        log_ratio = np.full_like(kcat, np.nan, dtype=np.float32)

        mask_k = np.isfinite(kcat) & (kcat > 0)
        mask_m = np.isfinite(Km) & (Km > 0)
        mask_r = mask_k & mask_m & np.isfinite(ratio) & (ratio > 0)

        log_kcat[mask_k] = np.log10(kcat[mask_k] + self.eps)
        log_Km[mask_m] = np.log10(Km[mask_m] * 1000 + self.eps)
        if mask_r.any():
            ratio_clipped = np.clip(ratio[mask_r], self.ratio_min, self.ratio_max)
            log_ratio[mask_r] = np.log10(ratio_clipped + self.eps)

        X = np.stack([log_kcat, log_Km, log_ratio], axis=1)

        return X

    def inverse_transform(self, log_kcat, log_Km, log_ratio):

        kcat_linear = np.power(10, log_kcat) - self.eps
        Km_linear = (np.power(10, log_Km) - self.eps) / 1000.0  # mM → M
        ratio_linear = np.power(10, log_ratio) - self.eps

        return kcat_linear, Km_linear, ratio_linear

    def save(self, path: str):
        params = {
            "eps": self.eps,
            "ratio_min": float(self.ratio_min),
            "ratio_max": float(self.ratio_max),
        }
        with open(path, "w") as f:
            json.dump(params, f, indent=2)

    @classmethod
    def load(cls, path: str):
        with open(path, "r") as f:
            params = json.load(f)

        obj = cls(eps=params.get("eps", 1e-12))
        obj.ratio_min = params["ratio_min"]
        obj.ratio_max = params["ratio_max"]
        return obj
      
class PredictionRecorder:
    def __init__(self, n_samples, output_dim=3):
        self.preds = np.zeros((n_samples, output_dim), dtype=np.float32)
        self.labels = np.zeros((n_samples, output_dim), dtype=np.float32)
        self.ptr = 0  # 写入位置指针

    def update(self, preds, labels):
        bs = len(preds)
        self.preds[self.ptr:self.ptr+bs] = preds
        self.labels[self.ptr:self.ptr+bs] = labels
        self.ptr += bs

    def get(self):
        return self.preds[:self.ptr], self.labels[:self.ptr]  
    
    
class KineticsScalerNoClip:
    """
    与原来功能接口兼容，但**不再对 ratio 做任何裁剪 (np.clip)**，
    只做 log 转换 + StandardScaler 标准化。
    """
    def __init__(self, eps=1e-12):
        self.eps = eps
        self.scaler_kcat = StandardScaler()
        self.scaler_Km = StandardScaler()
        self.scaler_ratio = StandardScaler()

    def fit(self, kcat, Km):
        kcat = np.array(kcat, dtype=np.float64)
        Km = np.array(Km, dtype=np.float64)
        ratio = kcat / (Km + self.eps)

        log_kcat = np.log10(kcat + self.eps).reshape(-1, 1)
        log_Km = np.log10(Km * 1000.0 + self.eps).reshape(-1, 1)  # 保持 mM 转换一致
        log_ratio = np.log10(ratio + self.eps).reshape(-1, 1)

        self.scaler_kcat.fit(log_kcat)
        self.scaler_Km.fit(log_Km)
        self.scaler_ratio.fit(log_ratio)

    def transform(self, kcat, Km):
        kcat = np.array(kcat, dtype=np.float64)
        Km = np.array(Km, dtype=np.float64)
        ratio = kcat / (Km + self.eps)

        log_kcat = np.log10(kcat + self.eps).reshape(-1, 1)
        log_Km = np.log10(Km * 1000.0 + self.eps).reshape(-1, 1)
        log_ratio = np.log10(ratio + self.eps).reshape(-1, 1)

        z_kcat = self.scaler_kcat.transform(log_kcat).flatten()
        z_Km = self.scaler_Km.transform(log_Km).flatten()
        z_ratio = self.scaler_ratio.transform(log_ratio).flatten()

        X = np.stack([z_kcat, z_Km, z_ratio], axis=1)
        return X

    def inverse_transform(self, z_kcat, z_Km=None, z_ratio=None):
        """
        inverse_transform 有两种调用方式以兼容旧脚本：
        1) inverse_transform(Z) -> 传入 shape (N,3) 的 Z
        2) inverse_transform(z_kcat, z_Km, z_ratio) -> 逐个数组
        """
        if z_Km is None and z_ratio is None:
            Z = np.array(z_kcat)
            z_kcat = Z[:, 0]
            z_Km = Z[:, 1]
            z_ratio = Z[:, 2]

        z_kcat = np.array(z_kcat).reshape(-1, 1)
        z_Km = np.array(z_Km).reshape(-1, 1)
        z_ratio = np.array(z_ratio).reshape(-1, 1)

        log_kcat = self.scaler_kcat.inverse_transform(z_kcat).flatten()
        log_Km = self.scaler_Km.inverse_transform(z_Km).flatten()
        log_ratio = self.scaler_ratio.inverse_transform(z_ratio).flatten()

        kcat_linear = np.power(10.0, log_kcat) - self.eps
        Km_linear = (np.power(10.0, log_Km) - self.eps) / 1000.0  # mM -> M
        ratio_linear = np.power(10.0, log_ratio) - self.eps

        return kcat_linear, Km_linear, ratio_linear

    def save(self, path: str):
        params = {
            "eps": self.eps,
            "scaler_kcat": {"mean": self.scaler_kcat.mean_.tolist(), "scale": self.scaler_kcat.scale_.tolist()},
            "scaler_Km":   {"mean": self.scaler_Km.mean_.tolist(),   "scale": self.scaler_Km.scale_.tolist()},
            "scaler_ratio":{"mean": self.scaler_ratio.mean_.tolist(),"scale": self.scaler_ratio.scale_.tolist()},
        }
        with open(path, "w") as f:
            json.dump(params, f, indent=2)

    @classmethod
    def load(cls, path: str):
        with open(path, "r") as f:
            params = json.load(f)
        obj = cls(eps=params.get("eps", 1e-12))

        def restore_scaler(mean, scale):
            scaler = StandardScaler()
            scaler.mean_ = np.array(mean, dtype=np.float64)
            scaler.scale_ = np.array(scale, dtype=np.float64)
            scaler.var_ = scaler.scale_ ** 2
            scaler.n_features_in_ = 1
            scaler.n_samples_seen_ = np.array([1e9])
            return scaler

        obj.scaler_kcat = restore_scaler(params["scaler_kcat"]["mean"], params["scaler_kcat"]["scale"])
        obj.scaler_Km = restore_scaler(params["scaler_Km"]["mean"], params["scaler_Km"]["scale"])
        obj.scaler_ratio = restore_scaler(params["scaler_ratio"]["mean"], params["scaler_ratio"]["scale"])
        return obj

import os, torch
import torch.distributed as dist

def _is_main_process():
    return (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0

def _print0(msg: str):
    if _is_main_process():
        print(msg, flush=True)

def resume_checkpoint_if_available(
    model,
    optimizer=None,
    scheduler=None,
    scaler=None,
    ema=None,
    weights_dir="/root/autodl-tmp/weights",
    device="cuda",
    verbose=True,   # 新增：控制是否打印
):
    """
    统一返回: (start_epoch, global_step)
    - 优先 predictor_checkpoint.pt（完整断点，含 epoch/global_step）
    - 其次 predictor.pt / predictor_best_raw.pt / predictor_best_ema.pt（仅权重）
    - 兼容 EMA 的两种保存方式（state_dict / shadow）。
    - 对 OneCycleLR 恢复做兜底（_step_count/total_steps 提示）
    - 若均不存在，返回 (0, 0)
    同时在主进程打印实际加载的文件与已恢复的组件。
    """
    start_epoch = 0
    global_step = 0

    ckpt_full     = os.path.join(weights_dir, "predictor_checkpoint.pt")
    ckpt_normal   = os.path.join(weights_dir, "predictor.pt")
    ckpt_best_raw = os.path.join(weights_dir, "predictor_best_raw.pt")
    ckpt_best_ema = os.path.join(weights_dir, "predictor_best_ema.pt")

    def _safe_load_state(state):
        loaded = {"model": False, "optimizer": False, "scheduler": False, "scaler": False, "ema": False}
        real_model = getattr(model, "module", model)
        if isinstance(state, dict) and "model" in state:
            real_model.load_state_dict(state["model"], strict=False)
            loaded["model"] = True
        elif isinstance(state, dict) and "state_dict" in state:
            real_model.load_state_dict(state["state_dict"], strict=False)
            loaded["model"] = True
        else:
            try:
                real_model.load_state_dict(state, strict=False)
                loaded["model"] = True
            except Exception as e:
                _print0(f"[RESUME] WARNING: model load_state_dict failed: {e}")
            
            
        if optimizer is not None and isinstance(state, dict) and "optimizer" in state:
            try:
                optimizer.load_state_dict(state["optimizer"])
                loaded["optimizer"] = True
            except Exception as e:
                _print0(f"[RESUME] WARNING: optimizer state not restored: {e}")
                try:
                    ckpt_pgs = [len(pg.get("params", [])) for pg in state["optimizer"]["param_groups"]]
                    cur_pgs  = [len(pg.get("params", [])) for pg in optimizer.param_groups]
                    _print0(f"[RESUME] ckpt param_groups lens={ckpt_pgs}, current={cur_pgs}")
                    ckpt_tags = [pg.get("tag","?") for pg in state["optimizer"]["param_groups"]]
                    cur_tags  = [pg.get("tag","?") for pg in optimizer.param_groups]
                    _print0(f"[RESUME] ckpt tags={ckpt_tags}, current tags={cur_tags}")
                except Exception as e2:
                    _print0(f"[RESUME] WARNING: could not inspect optimizer mismatch: {e2}")
                
                
        if scheduler is not None and isinstance(state, dict) and state.get("scheduler") is not None:
            try:
                scheduler.load_state_dict(state["scheduler"])
                loaded["scheduler"] = True
                sd = state["scheduler"]
                for key in ("_step_count", "step_num"):
                    if hasattr(scheduler, key) and isinstance(sd, dict) and key in sd:
                        setattr(scheduler, key, sd[key])
                if isinstance(sd, dict) and "total_steps" in sd and hasattr(scheduler, "total_steps"):
                    if getattr(scheduler, "total_steps", None) != sd["total_steps"] and verbose:
                        _print0(f"[RESUME] Warning: OneCycle total_steps mismatch: current={getattr(scheduler,'total_steps',None)} vs ckpt={sd['total_steps']}.")
                                
            except Exception as e:
                if verbose: _print0(f"[RESUME] Warning: scheduler state not fully restored: {e}")
                
                
        if scaler is not None and isinstance(state, dict) and "scaler" in state:
            try:
                scaler.load_state_dict(state["scaler"])
                loaded["scaler"] = True
            except Exception as e:
                _print0(f"[RESUME] WARNING: scaler state not restored: {e}")

        if ema is not None and isinstance(state, dict) and "ema" in state:
            try:
                if hasattr(ema, "load_state_dict") and isinstance(state["ema"], dict):
                    ema.load_state_dict(state["ema"])
                    loaded["ema"] = True
                elif hasattr(ema, "shadow"):   # 直接 shadow dict
                    ema.shadow = state["ema"]
                    loaded["ema"] = True
                if hasattr(ema, "shadow") and isinstance(ema.shadow, dict):
                    for k, v in ema.shadow.items():
                        if isinstance(v, torch.Tensor):
                            ema.shadow[k] = v.to(device)
            except Exception as e:
                _print0(f"[RESUME] WARNING: EMA state not restored: {e}")

        return loaded

    def _report(path, loaded_flags, se, gs):
        if not verbose:
            return
        bits = [k for k, v in loaded_flags.items() if v]
        miss = [k for k, v in loaded_flags.items() if not v]
        _print0(f"[RESUME] Loaded from: {path}")
        _print0(f"[RESUME] Components loaded: {', '.join(bits) if bits else 'none'}"
                f"{' | missing: ' + ', '.join(miss) if miss else ''}")
        _print0(f"[RESUME] start_epoch={se}, global_step={gs}")

    map_location = "cpu"

    if os.path.exists(ckpt_full):
        try:
            state = torch.load(ckpt_full, map_location=map_location)
            loaded = _safe_load_state(state)
            start_epoch = int(state.get("epoch", 0)) if isinstance(state, dict) else 0
            global_step = int(state.get("global_step", 0)) if isinstance(state, dict) else 0
            _report(ckpt_full, loaded, start_epoch, global_step)
        except Exception as e:
            _print0(f"[RESUME] Failed to load {ckpt_full}: {e}")
        finally:
            if dist.is_available() and dist.is_initialized():
                dist.barrier()
        return start_epoch, global_step

    if os.path.exists(ckpt_normal):
        try:
            state = torch.load(ckpt_normal, map_location=map_location)
            loaded = _safe_load_state(state)
            _report(ckpt_normal, loaded, start_epoch, global_step)  # 仅权重，没有 se/gs
        except Exception as e:
            _print0(f"[RESUME] Failed to load {ckpt_normal}: {e}")
        finally:
            if dist.is_available() and dist.is_initialized():
                dist.barrier()
        return start_epoch, global_step

    path = ckpt_best_ema if os.path.exists(ckpt_best_ema) else (ckpt_best_raw if os.path.exists(ckpt_best_raw) else None)
    if path is not None:
        try:
            state = torch.load(path, map_location=map_location)
            loaded = _safe_load_state(state)
            _report(path, loaded, start_epoch, global_step)
        except Exception as e:
            _print0(f"[RESUME] Failed to load {path}: {e}")
        finally:
            if dist.is_available() and dist.is_initialized():
                dist.barrier()
        return start_epoch, global_step

    if verbose:
        _print0(f"[RESUME] No checkpoint found in {weights_dir}. Starting from scratch.")
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
    return start_epoch, global_step


def save_training_checkpoint(model, optimizer, scheduler, scaler, ema, epoch, global_step,
                             weights_dir="/root/autodl-tmp/weights", tag=None):
    """
    仅在 rank0 调用；会覆盖主文件，并可选写出带时间戳的副本。
    tag: 例如 "hourly" / "epoch" 用于区分副本文件。
    """
    import os, time, torch
    os.makedirs(weights_dir, exist_ok=True)
    is_ddp = torch.distributed.is_available() and torch.distributed.is_initialized()
    real_model = getattr(model, "module", model)
    
    if ema is not None:
        if hasattr(ema, "state_dict"):
            try:
                ema_payload = ema.state_dict()
            except Exception:
                ema_payload = getattr(ema, "shadow", None)
        else:
            ema_payload = getattr(ema, "shadow", None)
    else:
        ema_payload = None
    
    
    
    try:
        scheduler_state = scheduler.state_dict() if scheduler else None
    except Exception as e:
        scheduler_state = None
        print(f"[CKPT] skip scheduler state_dict (unserializable): {e}")

    payload = {
        "epoch": epoch,
        "global_step": global_step,
        "model": real_model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer else None,
        "scheduler": scheduler_state,
        "scaler": scaler.state_dict() if scaler else None,
        "ema": ema_payload,
        "time": time.time(),
    }

    tmp_path = os.path.join(weights_dir, "predictor_checkpoint.pt.tmp")
    fin_path = os.path.join(weights_dir, "predictor_checkpoint.pt")
    try:
        torch.save(payload, tmp_path)
    except Exception as e:
        print(f"[CKPT] retry without scheduler state due to serialization error: {e}")
        payload["scheduler"] = None
        torch.save(payload, tmp_path)
    os.replace(tmp_path, fin_path)
    print(f"[CKPT] checkpoint saved -> {fin_path} (epoch={epoch}, step={global_step})")






    
def _clear_rotary_cache(module):
    """
    安全清理 ESM 中的旋转位置编码缓存：
    - 如果 _cos_cached / _sin_cached 是 inference tensor，则清理；
    - 清理后在必要时尝试重建；
    - 兼容 transformers 不同版本的 RotaryEmbedding 实现。
    """
    import torch

    for m in module.modules():
        if m.__class__.__name__.lower().startswith("rotary"):
            cos_bad = hasattr(m, "_cos_cached") and isinstance(m._cos_cached, torch.Tensor) and not m._cos_cached.requires_grad
            sin_bad = hasattr(m, "_sin_cached") and isinstance(m._sin_cached, torch.Tensor) and not m._sin_cached.requires_grad

            if cos_bad or sin_bad:
                m._cos_cached = None
                m._sin_cached = None
                m._seq_len_cached = None

                try:
                    rotary_dim = getattr(m, "rotary_dim", None) or getattr(m, "_rotary_dim", None)
                    if rotary_dim is None:
                        rotary_dim = 64  # fallback

                    dummy = torch.zeros(1, 1, rotary_dim,
                                        device="cuda" if torch.cuda.is_available() else "cpu")
                    if hasattr(m, "_update_cos_sin_tables"):
                        m._cos_cached, m._sin_cached = m._update_cos_sin_tables(dummy, seq_dimension=-2)
                        m._seq_len_cached = 1
                        print(f"[RotaryCache] Rebuilt cache for {m.__class__.__name__} (dim={rotary_dim})")
                except Exception as e:
                    print(f"[WARN] Rotary cache rebuild failed ({m.__class__.__name__}): {e}")  
                    

                    
                    
def module_sanity_check(model, batch, use_mask: bool = False, verbose: bool = True,
                        diff_threshold: float = 1e-6, head_names=None,optimizer=None, show_lr: bool = False, rank0_only: bool = True):
    """
    模块级 sanity check（推理态），逐层打印 |out - in| 的均值。仅 rank0 执行。
    - use_mask: 是否应用 gate 的擦除（与训练开关对齐）
    - diff_threshold: 只打印超过该阈值的模块
    - head_names: 仅检查这些 head 名称（None=全部）
    """
    import torch
    import torch.distributed as dist

    is_ddp = dist.is_available() and dist.is_initialized()
    if is_ddp and dist.get_rank() != 0:
        return

    real_model = getattr(model, "module", model)
    real_model.eval()

    seqs, smiles, labels = batch
    device = next(real_model.parameters()).device
    labels = labels.to(device, non_blocking=True)

    seqs  = list(seqs)
    smiles= list(smiles)
    
    def _M(x): return x / 1e6

    def _param_overview(mod):
        ligand_ids = set()
        lig_mod = getattr(mod, "ligand", None)
        if lig_mod is not None:
            for _, p in lig_mod.named_parameters(recurse=True):
                ligand_ids.add(id(p))
                
        total = trainable = head_p = lora_p = 0
        lig_train = lig_frozen = 0
        lora_names = []
        
        for n, p in mod.named_parameters():
            num = p.numel()
            total += num
            nlow = n.lower()
            
            is_lora = ("lora" in nlow)
            is_lig = (id(p) in ligand_ids)
            
            if p.requires_grad:
                trainable += num
                if is_lora:
                    lora_p += num
                    if len(lora_names) < 6:
                        lora_names.append(n)
                if ("head" in nlow) or ("multi_head" in nlow) or ("adapter" in nlow) or ("fusion" in nlow):
                    head_p += num
                if is_lig:
                    lig_train += num
            else:
                if is_lig:
                    lig_frozen += num

        return {
            "total": total, "trainable": trainable,
            "head": head_p, "lora": lora_p,
            "lig_train": lig_train, "lig_frozen": lig_frozen,
            "lora_names": lora_names,
        }

    def _r2_per_col(y_true: torch.Tensor, y_pred: torch.Tensor):
        assert y_true.shape == y_pred.shape
        outs = []
        for i in range(y_true.shape[1]):
            yt = y_true[:, i].float()
            yp = y_pred[:, i].float()
            if yt.numel() < 2:
                outs.append(float("nan")); continue
            mu = yt.mean()
            sst = ((yt - mu) ** 2).sum()
            sse = ((yt - yp) ** 2).sum()
            if sst.abs() < 1e-12:
                outs.append(float("nan"))
            else:
                outs.append((1.0 - (sse / sst)).item())
        return outs

    def _lr_snapshot(opt):
        try:
            def pick(tag):
                return [pg["lr"] for pg in opt.param_groups if pg.get("tag","") == tag]
            head_lrs = pick("head")
            lora_lrs = pick("lora")
            other = [ (gi, pg.get("tag","?"), pg["lr"]) for gi, pg in enumerate(opt.param_groups)
                      if pg.get("tag","") not in ("head","lora") ]
            s = "[LR]"
            if head_lrs: s += " head=" + ",".join(f"{x:.2e}" for x in head_lrs)
            if lora_lrs: s += " lora=" + ",".join(f"{x:.2e}" for x in lora_lrs)
            if other:    s += " | " + " ".join(f"g{gi}:{tg}={lr:.2e}" for gi,tg,lr in other[:4])
            print(s)
        except Exception as e:
            print("[LR][WARN]", e)


    ctx = torch.inference_mode if hasattr(torch, "inference_mode") else torch.no_grad
    with ctx():
        if verbose:
            print("\n=== Module Sanity Check ===")
        
        ov = _param_overview(real_model)
        print(f"[DBG][params] total={_M(ov['total']):.2f}M, trainable={_M(ov['trainable']):.2f}M "
              f"({ov['trainable']/max(1,ov['total']):.2%})")
        print(f"[DBG][params]   head={_M(ov['head']):.2f}M, lora={_M(ov['lora']):.2f}M")
        print(f"[DBG][ligand]   trainable={_M(ov['lig_train']):.2f}M, frozen={_M(ov['lig_frozen']):.2f}M")
        if ov["lora_names"]:
            print("[DBG] sample lora params:")
            for n in ov["lora_names"]:
                print("   -", n)
        else:
            print("[DBG][WARN] 未找到任何 'lora_' 参数名（检查 LoRA 注入与命名匹配）")

        if optimizer is not None:
            _lr_snapshot(optimizer)

        if len(seqs) > 0:
            raw_lens = [len(s) for s in seqs]
            srt = sorted(raw_lens)
            p50 = srt[len(srt)//2]
            print(f"[DBG][raw_seq_len] min={min(raw_lens)} p50={p50} max={max(raw_lens)} (n={len(raw_lens)})")

        p_tok, p_mask = real_model.protein(seqs)
        l_tok, l_mask = real_model.ligand(smiles)
        if p_mask is None:
            p_mask = torch.ones(p_tok.shape[:2], device=device, dtype=torch.long)
        if l_mask is None:
            l_mask = torch.ones(l_tok.shape[:2], device=device, dtype=torch.long)
            
        if len(seqs) > 0:
            toks_len = int(p_tok.shape[1])
            n_trunc = sum(1 for L in raw_lens if L > toks_len)
            print(f"[DBG][protein_len] toks_len={toks_len} truncated={n_trunc}/{len(raw_lens)}  "
                  f"(≈{(n_trunc/max(1,len(raw_lens))):.1%})")

        for idx, m in enumerate(real_model.mamba_layers):
            x_in = p_tok.clone()
            p_tok = m(p_tok)
            diff = (p_tok - x_in).abs().mean().item()
            if verbose and diff > diff_threshold:
                print(f"[Mamba {idx}] mean(|out-in|)={diff:.6f}")

        x_in = p_tok.clone()
        gate_e, erase_e = real_model.resmask(p_tok, train_mask=use_mask)
        try:
            gmean_e = gate_e.float().mean().item() if torch.is_tensor(gate_e) else float(gate_e)
        except Exception:
            gmean_e = float("nan")
        
        p_tok = p_tok * (1.0 - erase_e)
        diff = (p_tok - x_in).abs().mean().item()
        if verbose and diff > diff_threshold:
            print(f"[ResidueMask] |Δ|={(p_tok - x_in).abs().mean().item():.6f}, "
                  f"gate mean={gmean_e:.4f}, use_mask={use_mask}, droprate={getattr(real_model.resmask,'droprate',None)}")

        x_in = l_tok.clone()
        gate_s, erase_s = real_model.ligmask(l_tok, train_mask=use_mask)
        try:
            gmean_s = gate_s.float().mean().item() if torch.is_tensor(gate_s) else float(gate_s)
        except Exception:
            gmean_s = float("nan")
            
        l_tok = l_tok * (1.0 - erase_s)
        diff = (l_tok - x_in).abs().mean().item()
        if verbose and diff > diff_threshold:
            print(f"[LigandMask]  |Δ|={(l_tok - x_in).abs().mean().item():.6f}, "
                  f"gate mean={gmean_s:.4f}, use_mask={use_mask}, droprate={getattr(real_model.ligmask,'droprate',None)}")


        for si, stage in enumerate(real_model.interactions):
            for bi, blk in enumerate(stage):
                p_in, l_in = p_tok.clone(), l_tok.clone()
                p_tok, l_tok = blk(p_tok, l_tok, p_mask, l_mask)
                dp = (p_tok - p_in).abs().mean().item()
                dl = (l_tok - l_in).abs().mean().item()
                if verbose and (dp > diff_threshold or dl > diff_threshold):
                    print(f"[XAttn s{si} b{bi}] |Δp|={dp:.6f}, |Δl|={dl:.6f}")

        p_vec = real_model._pool(p_tok, p_mask, which='p')
        l_vec = real_model._pool(l_tok, l_mask, which='l')

        x = torch.cat([p_vec, l_vec], dim=-1)
        if getattr(real_model.cfg, "fusion_norm", False):
            x = real_model.fusion_ln(x)
        x = real_model.fusion_proj(x)
        if verbose:
            print(f"[Fusion] fused_dim={x.shape[-1]}, mean={x.mean().item():.6f}, std={x.std().item():.6f}")

        outs_list, outs_dict = real_model.multi_head(x, cond=x)
        names = list(real_model.cfg.active_heads) if head_names is None else list(head_names)
        names = [nm for nm in names if nm in outs_dict]  # 只保留存在的头
        
        for name in names:
            y = outs_dict[name]
            print(f"[Head:{name}] shape={tuple(y.shape)}, mean={y.mean().item():.6f}, std={y.std().item():.6f}")
        try:
            if labels.dim() == 2 and labels.size(1) >= 3:
                col_map = {"kcat": 0, "Km": 1, "ratio": 2, "kcat_per_Km": 2}
                preds = []
                cols  = []
                for nm in ["kcat", "Km", "ratio"]:
                    if nm in outs_dict:
                        yp = outs_dict[nm].squeeze(-1)
                        preds.append(yp)
                        cols.append(col_map[nm])
                if preds:
                    y_pred = torch.stack(preds, dim=1)                         # [B, C']
                    y_true = labels[:, cols].to(y_pred.dtype)                  # [B, C']
                    r2s = _r2_per_col(y_true, y_pred)
                    r2_str = " ".join(f"{n}={r:.3f}" for n, r in zip(["kcat","Km","ratio"][:len(r2s)], r2s))
                    print(f"[Sanity R2] {r2_str}")
        except Exception as e:
            print("[Sanity R2][WARN]", e)

        real_model = getattr(model, "module", model)
        try:
            _clear_rotary_cache(real_model)
        except Exception as e:
            print(f"[Sanity][WARN] rotary cache clear failed: {e}")

        print("=== End of Sanity Check ===\n")




def unpack_batch(batch):
    """
    统一解包为 11 项：
      seqs, smis, row_ids, source_ids, p_s, l_s, y, struct_avail_mask, phys_feat, phys_mask, phys_quality

    兼容历史格式：
      3: (seq, smi, y)
      4: (seq, smi, row_id, y) 或 (seq, smi, source_id, y)
      5: (seq, smi, p_s, l_s, y) 或 (seq, smi, row_id, source_id, y)
      6: (seq, smi, row_id, p_s, l_s, y) 或 (seq, smi, source_id, p_s, l_s, y)
      7: (seq, smi, row_id, source_id, p_s, l_s, y)
    """
    def _is_struct(x):
        if torch.is_tensor(x):
            return x.dim() >= 2
        if isinstance(x, np.ndarray):
            return x.ndim >= 2
        return False

    if len(batch) == 3:
        seqs, smis, y = batch
        return seqs, smis, None, None, None, None, y, None, None, None, None
    if len(batch) == 4:
        seqs, smis, a, y = batch
        return seqs, smis, a, None, None, None, y, None, None, None, None
    if len(batch) == 5:
        seqs, smis, a, b, y = batch
        if _is_struct(a) and _is_struct(b):
            return seqs, smis, None, None, a, b, y, None, None, None, None
        return seqs, smis, a, b, None, None, y, None, None, None, None
    if len(batch) == 6:
        seqs, smis, a, b, c, y = batch
        if _is_struct(b) and _is_struct(c):
            return seqs, smis, a, None, b, c, y, None, None, None, None
        return seqs, smis, None, a, b, c, y, None, None, None, None
    if len(batch) == 7:
        seqs, smis, a, b, c, d, y = batch
        if _is_struct(c) and _is_struct(d):
            return seqs, smis, a, b, c, d, y, None, None, None, None
        return seqs, smis, None, a, None, None, y, None, b, c, d
    if len(batch) == 8:
        seqs, smis, a, b, c, d, e, y = batch
        if _is_struct(c) and _is_struct(d):
            return seqs, smis, a, b, c, d, e, y, None, None, None
        return seqs, smis, a, b, None, None, y, None, c, d, e
    if len(batch) == 9:
        seqs, smis, source_ids, p_s, l_s, phys_feat, phys_mask, phys_quality, y = batch
        return seqs, smis, None, source_ids, p_s, l_s, y, None, phys_feat, phys_mask, phys_quality
    if len(batch) == 10:
        seqs, smis, row_ids, source_ids, p_s, l_s, phys_feat, phys_mask, phys_quality, y = batch
        return seqs, smis, row_ids, source_ids, p_s, l_s, y, None, phys_feat, phys_mask, phys_quality
    if len(batch) == 11:
        seqs, smis, row_ids, source_ids, p_s, l_s, y, struct_avail_mask, phys_feat, phys_mask, phys_quality = batch
        return seqs, smis, row_ids, source_ids, p_s, l_s, y, struct_avail_mask, phys_feat, phys_mask, phys_quality
    raise ValueError(f"Unexpected batch len={len(batch)}")
