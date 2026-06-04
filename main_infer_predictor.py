import argparse, json, os, torch
import pandas as pd
import numpy as np
import re
from models.predictor import KineticsPredictor, PredictorConfig
from utils import set_seed
from tqdm import tqdm


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


def _first_smiles_component(smi: str) -> str:
    """从多组分 SMILES 中取第一个组分（按顶层 '.' 切分，忽略括号内的 '.'）。"""
    depth = 0
    for i, c in enumerate(smi):
        if c == '(':
            depth += 1
        elif c == ')':
            depth -= 1
        elif c == '.' and depth == 0:
            return smi[:i]
    return smi


def _infer_lora_rank_from_ckpt(ckpt: dict) -> int | None:
    ranks = []
    for k, v in ckpt.items():
        if ("lora_A" in k) and hasattr(v, "shape") and len(v.shape) == 2:
            ranks.append(int(v.shape[0]))
    if not ranks:
        return None
    return max(ranks)


def _infer_predictor_cfg_from_ckpt(ckpt: dict) -> dict:
    cfg_updates = {}

    if "norm_p.weight" in ckpt and hasattr(ckpt["norm_p.weight"], "shape"):
        cfg_updates["d_model"] = int(ckpt["norm_p.weight"].shape[0])

    stg_ids, lyr_ids, flat_ids = set(), set(), set()
    for k in ckpt.keys():
        m_nested = re.match(r"^interactions\.(\d+)\.(\d+)\.", k)
        if m_nested:
            stg_ids.add(int(m_nested.group(1)))
            lyr_ids.add(int(m_nested.group(2)))
            continue
        m_flat = re.match(r"^interactions\.(\d+)\.", k)
        if m_flat:
            flat_ids.add(int(m_flat.group(1)))

    if stg_ids and lyr_ids:
        cfg_updates["num_stages"] = max(stg_ids) + 1
        cfg_updates["num_interaction_layers_per_stage"] = max(lyr_ids) + 1
        cfg_updates["num_interaction_layers"] = cfg_updates["num_stages"] * cfg_updates["num_interaction_layers_per_stage"]
    elif flat_ids:
        cfg_updates["num_stages"] = 1
        cfg_updates["num_interaction_layers_per_stage"] = max(flat_ids) + 1
        cfg_updates["num_interaction_layers"] = cfg_updates["num_interaction_layers_per_stage"]

    two_c = int(cfg_updates.get("d_model", 0)) * 2 if "d_model" in cfg_updates else None
    if "fusion_proj.weight" in ckpt and hasattr(ckpt["fusion_proj.weight"], "shape"):
        cfg_updates["fusion_proj_dim"] = int(ckpt["fusion_proj.weight"].shape[0])
    else:
        head_in_dim = None
        for head_name in ["kcat", "Km", "ratio"]:
            k = f"multi_head.heads.{head_name}.net.0.weight"
            if k in ckpt and hasattr(ckpt[k], "shape") and len(ckpt[k].shape) == 2:
                head_in_dim = int(ckpt[k].shape[1])
                break
        if (head_in_dim is not None) and (two_c is not None):
            cfg_updates["fusion_proj_dim"] = None if head_in_dim == two_c else head_in_dim

    head_specs = {}
    for head_name in ["kcat", "Km", "ratio"]:
        k = f"multi_head.heads.{head_name}.net.0.weight"
        if k in ckpt and hasattr(ckpt[k], "shape") and len(ckpt[k].shape) == 2:
            head_specs[head_name] = {
                "type": "mlp",
                "hidden_dim": int(ckpt[k].shape[0]),
                "depth": 3,
                "dropout": 0.10,
                "out_dim": 1,
            }
    if head_specs:
        cfg_updates["head_specs"] = head_specs

    source_head_indices = set()
    for k in ckpt.keys():
        m = re.match(r"^kcat_source_residual\.heads\.(\d+)\.", k)
        if m:
            source_head_indices.add(int(m.group(1)))
    if source_head_indices:
        cfg_updates["use_source_residual_head"] = True
        cfg_updates["source_head_tasks"] = ["kcat"]
        cfg_updates["num_domains"] = max(source_head_indices) + 1
        k0 = "kcat_source_residual.heads.0.net.0.weight"
        if k0 in ckpt and hasattr(ckpt[k0], "shape") and len(ckpt[k0].shape) == 2:
            cfg_updates["source_head_hidden"] = int(ckpt[k0].shape[0])

    return cfg_updates


def _map_source_id(row, source_col, default_source_id):
    if (source_col is None) or (source_col not in row.index):
        return int(default_source_id)
    v = row[source_col]
    if pd.isna(v):
        return int(default_source_id)
    if isinstance(v, (int, np.integer)):
        return int(v)
    if isinstance(v, (float, np.floating)) and np.isfinite(v):
        return int(v)
    key = str(v).strip().lower().replace(" ", "").replace("_", "-")
    return int(DEFAULT_SOURCE_ALIASES.get(key, default_source_id))


def build_model_from_ckpt(ckpt_path, device="cuda", esm_model=None, molt5_model=None, lora_r=None, lora_alpha=None):
    """
    自动解析 checkpoint，调整 config，返回匹配的模型
    """
    ckpt = torch.load(ckpt_path, map_location="cpu")
    ckpt_keys = ckpt.keys()

    cfg_updates = _infer_predictor_cfg_from_ckpt(ckpt)
    config = PredictorConfig(**cfg_updates)
    if cfg_updates:
        print(f"[AUTO-CONFIG] inferred cfg keys -> {sorted(cfg_updates.keys())}")

    if any(k.startswith("att_pool_p") or k.startswith("att_pool_l") for k in ckpt_keys):
        config.use_attention_pool = True
        print("[AUTO-CONFIG] use_attention_pool -> True")
    else:
        config.use_attention_pool = False
        print("[AUTO-CONFIG] use_attention_pool -> False")

    mamba_layers = [k for k in ckpt_keys if k.startswith("mamba_layers.")]
    config.num_mamba_layers = max([int(k.split(".")[1]) for k in mamba_layers]) + 1 if mamba_layers else 0
    print(f"[AUTO-CONFIG] num_mamba_layers -> {config.num_mamba_layers}")

    inferred_r = _infer_lora_rank_from_ckpt(ckpt)
    use_lora = inferred_r is not None
    eff_r = int(lora_r) if lora_r is not None else (int(inferred_r) if inferred_r is not None else 8)
    eff_alpha = int(lora_alpha) if lora_alpha is not None else int(2 * eff_r)
    print(f"[AUTO-CONFIG] use_lora -> {use_lora}, lora_r -> {eff_r}, lora_alpha -> {eff_alpha}")

    model = KineticsPredictor(
        device=device,
        esm_model=esm_model,
        molt5_model=molt5_model,
        cfg=config,
        use_lora=use_lora,
        lora_r=eff_r,
        lora_alpha=eff_alpha,
        lora_dropout=0.05,
    )

    missing, unexpected = model.load_state_dict(ckpt, strict=False)
    print(f"[INFO] Missing keys: {len(missing)} | Unexpected keys: {len(unexpected)}")

    return model, config, ckpt


def enable_mc_dropout(model):
    """保持 BatchNorm/LayerNorm 在 eval 模式，仅开启 Dropout 层，用于 MC-Dropout 不确定性估计。"""
    model.eval()
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.train()


def check_model_weights(model, ckpt):
    """ckpt: 已加载的 state_dict（避免重复 torch.load）。"""
    mismatched = []
    for name, param in model.state_dict().items():
        if name in ckpt:
            if not torch.allclose(param.cpu(), ckpt[name].cpu(), atol=1e-6):
                mismatched.append(name)
        else:
            mismatched.append(name)
    if mismatched:
        print("以下层权重与 checkpoint 不匹配或未加载:")
        for n in mismatched:
            print("  -", n)
    else:
        print("✅ 所有模型层权重已正确加载。")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("-csv", type=str, required=True, help="输入CSV文件")
    p.add_argument("-weights", type=str, required=True, help="模型权重文件（.pt）")
    p.add_argument("-device", type=str, default="cuda")
    p.add_argument("-esm", type=str, default="facebook/esm2_t12_35M_UR50D")
    p.add_argument("-molt5", type=str, default="laituan245/molt5-base-smiles2caption")
    p.add_argument("-scaler", type=str, default=None, help="可选：当前推理不会使用 scaler；训练标签已在 log10 空间")
    p.add_argument("--lora_r", type=int, default=None, help="可选：手动指定 LoRA rank；不传则从 ckpt 自动推断")
    p.add_argument("--lora_alpha", type=int, default=None, help="可选：手动指定 LoRA alpha；不传则默认 2*lora_r")
    p.add_argument("--task", type=str, choices=["kcat", "km"], default="kcat",
                   help="任务类型：kcat（默认）或 km，决定 SMILES 列名和真实值字段")
    p.add_argument("--smiles-col", type=str, default=None,
                   help="手动指定 SMILES 列名；不传则根据 --task 自动推断")
    p.add_argument("--use_first_component", action="store_true", default=False,
                   help="仅对 kcat 生效：将多组分 SMILES 截断为第一个组分（默认关闭，保留完整 reactant_smiles）")
    p.add_argument("--source_col", type=str, default="source_id",
                   help="推理时 source 列名，用于 source-specific residual head")
    p.add_argument("--default_source_id", type=int, default=0,
                   help="source 列缺失或未知时的默认 source id")
    p.add_argument("--fixed_source_id", type=int, default=None,
                   help="若设置则覆盖每行 source，全部样本使用该 source id")
    p.add_argument("--diag_disable_kcat_source_residual", action="store_true", default=False,
                   help="诊断开关：将 model.kcat_source_residual 置为 None（仅推理时生效）")
    p.add_argument("--disable_mc_dropout", action="store_true", default=False,
                   help="诊断开关：关闭 MC-Dropout（model.eval() + 单次前向）")
    p.add_argument("--mc_dropout_samples", type=int, default=5,
                   help="MC-Dropout 采样次数（默认 5；--disable_mc_dropout 时会强制为 1）")
    args = p.parse_args()
    
    set_seed(42)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    model, cfg, ckpt = build_model_from_ckpt(
        args.weights,
        device=device,
        esm_model=args.esm,
        molt5_model=args.molt5,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
    )
    check_model_weights(model, ckpt)
    del ckpt  # 释放内存
    model.to(device)
    if args.diag_disable_kcat_source_residual and getattr(model, "kcat_source_residual", None) is not None:
        print("[DIAG-A] disabling kcat_source_residual")
        model.kcat_source_residual = None

    if args.disable_mc_dropout:
        model.eval()
        mc_samples = 1
        print("[DIAG-B] MC-Dropout disabled (eval + single forward)")
    else:
        enable_mc_dropout(model)
        mc_samples = max(1, int(args.mc_dropout_samples))
        print(f"[INFO] MC-Dropout enabled, samples={mc_samples}")

    df = pd.read_csv(args.csv)
    df.columns = [c.lower() for c in df.columns]
    print(f"✅ 加载 {args.csv}，共 {len(df)} 条数据")

    if args.smiles_col:
        smiles_col = args.smiles_col.lower()
    else:
        candidates = (
            ["reactant_smiles", "smiles", "substrate_smiles"]
            if args.task == "kcat"
            else ["substrate_smiles", "smiles", "reactant_smiles"]
        )
        smiles_col = next((c for c in candidates if c in df.columns), None)
    if smiles_col is None or smiles_col not in df.columns:
        raise ValueError(f"找不到 SMILES 列，可用列: {list(df.columns)}")
    print(f"[INFO] task={args.task}, SMILES 列 -> '{smiles_col}'")
    smiles_mode = "first_component" if (args.task == "kcat" and args.use_first_component) else "full_reactant"
    print(f"[INFO] kcat smiles mode -> {smiles_mode}")
    if args.fixed_source_id is not None:
        print(f"[INFO] source mode -> fixed={int(args.fixed_source_id)}")
    else:
        print(f"[INFO] source mode -> from column '{args.source_col}' (default={int(args.default_source_id)})")

    if args.scaler and os.path.exists(args.scaler):
        print(f"[INFO] Scaler file provided but ignored: {args.scaler}")
    print("[INFO] Predictor outputs are interpreted in log10 space, consistent with training labels.")

    results = []

    with torch.no_grad():
        for i, row in tqdm(df.iterrows(), total=len(df), desc="推理中"):
            seq = row["sequence"]
            smi = row[smiles_col]
            if args.task == "kcat" and args.use_first_component:
                smi = _first_smiles_component(str(smi))

            source_id = int(args.fixed_source_id) if args.fixed_source_id is not None else _map_source_id(row, args.source_col.lower(), args.default_source_id)
            source_tensor = torch.tensor([source_id], device=device, dtype=torch.long)

            preds_list = []
            for _ in range(mc_samples):
                out = model([seq], [smi], source_ids=source_tensor, use_mask=False)
                pred_kcat, pred_km, _pred_ratio = out[:3]
                pred_ratio = pred_kcat - pred_km + 3.0
                preds_list.append(torch.stack([pred_kcat, pred_km, pred_ratio], dim=-1).cpu().numpy())

            preds_arr = np.stack(preds_list, axis=0)   # (5, 1, 3)
            preds_mean = preds_arr.mean(axis=0)         # (1, 3) — MC-Dropout 均值
            preds_std  = preds_arr.std(axis=0)          # (1, 3) — MC-Dropout 不确定性

            pred_kcat_log  = preds_mean[:, 0]
            pred_Km_log    = preds_mean[:, 1]
            pred_ratio_log = preds_mean[:, 2]           # 直接使用 MC 平均的 ratio（log10 空间）

            kcat_pred_linear  = np.power(10.0, pred_kcat_log)
            Km_pred_linear    = np.power(10.0, pred_Km_log) / 1000.0
            ratio_pred_linear = np.power(10.0, pred_ratio_log)

            res = {
                "sequence": seq,
                "smiles": smi,
                "source_id": source_id,
                "pred_kcat_log": pred_kcat_log[0],
                "pred_Km_log": pred_Km_log[0],
                "pred_ratio_log": pred_ratio_log[0],
                "pred_kcat": kcat_pred_linear[0],
                "pred_Km": Km_pred_linear[0],
                "pred_kcat_div_Km": ratio_pred_linear[0],
                "unc_kcat_log": preds_std[0, 0],
                "unc_Km_log": preds_std[0, 1],
                "unc_ratio_log": preds_std[0, 2],
            }

            true_log = np.nan
            if args.task == "kcat":
                linear_candidates = ["kcat(s^-1)", "kcat", "kcat_value", "value"]
            else:
                linear_candidates = ["km(m)", "km", "km_value", "value"]

            if "log10_value" in row.index:
                true_log = pd.to_numeric(pd.Series([row["log10_value"]]), errors="coerce").iloc[0]
                if args.task == "km" and np.isfinite(true_log):
                    true_log = true_log + 3.0   # log10(M) → log10(mM)
            else:
                for col in linear_candidates:
                    if col in row.index:
                        v = pd.to_numeric(pd.Series([row[col]]), errors="coerce").iloc[0]
                        if np.isfinite(v) and v > 0:
                            if args.task == "km":
                                true_log = np.log10(v * 1000.0)  # M → mM → log10
                            else:
                                true_log = np.log10(v)
                        break

            if np.isfinite(true_log):
                if args.task == "kcat":
                    res["true_kcat_log"] = true_log   # log10(s^-1)
                    res["true_kcat"] = float(np.power(10.0, true_log))
                else:
                    res["true_Km_log"] = true_log     # log10(mM)，已在读取时统一转换
                    res["true_Km"] = float(np.power(10.0, true_log)) / 1000.0  # 还原为 M

            print(
                f"Predicted values: kcat={kcat_pred_linear[0]:.4g} s^-1 (±{preds_std[0,0]:.3f} log), "
                f"Km={Km_pred_linear[0]:.4g} M (±{preds_std[0,1]:.3f} log), "
                f"kcat/Km={ratio_pred_linear[0]:.4g} M⁻¹s⁻¹ (±{preds_std[0,2]:.3f} log)"
            )
            results.append(res)

    out_csv = f"predictions_{args.task}.csv"
    pd.DataFrame(results).to_csv(out_csv, index=False)
    print(f"✅ 推理完成，结果已保存到 {out_csv}")
    print("⚠️ 当前脚本仅做推理与结果导出，不输出正式评测指标。")

if __name__ == "__main__":
    main()
