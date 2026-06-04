"""
tools/insilico_mut_scan.py
────────────────────────────────────────────────────────────────
In silico 单点突变扫描：对给定蛋白序列的指定位点枚举全部替换，
通过 KineticsPredictor 打分，输出兼容 insilico_mut_scan.csv 格式的结果。

与 run_directed_evo.py 的区别：
  - 本脚本是一次性独立扫描，无迭代循环
  - 输出列格式与原始 insilico_mut_scan.csv 一致（包含 group 标注）
  - score_km 约定：正值=Km升高（与原始 CSV 一致），在 pareto_filter.py 中会自动取反

SIGMA 值来源（实测推导）：
  SIGMA_KCAT = 0.4794  (score_kcat = d_logkcat / SIGMA_KCAT)
  SIGMA_KM   = 0.8288  (score_km   = d_logKm   / SIGMA_KM)

用法示例
────────
  python tools/insilico_mut_scan.py \
      --seq MAAKVLFTS... \
      --smi "CC(=O)OC1=CC=CC=C1C(=O)O" \
      --weights checkpoints/best.pt \
      --out insilico_mut_scan.csv \
      --top-k 15

  python tools/insilico_mut_scan.py \
      --seq MAAKVLFTS... \
      --smi "CC(=O)..." \
      --weights checkpoints/best.pt \
      --out scan_custom.csv \
      --mutations-csv custom_muts.csv

  python tools/insilico_mut_scan.py \
      --seq MAAKVLFTS... \
      --smi "CC(=O)..." \
      --weights checkpoints/best.pt \
      --out full_scan.csv \
      --scan-all
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from main_infer_predictor import build_model_from_ckpt, enable_mc_dropout


AA20 = list("ACDEFGHIKLMNPQRSTVWY")

CONSERVATIVE_SUBS = {
    "A": ["G", "S", "V", "T"],
    "R": ["K", "H", "Q", "N"],
    "N": ["D", "S", "Q", "H"],
    "D": ["E", "N", "S", "G"],
    "C": ["S", "A", "T", "M"],
    "Q": ["E", "N", "K", "H"],
    "E": ["D", "Q", "K", "N"],
    "G": ["A", "S", "N", "D"],
    "H": ["R", "K", "N", "Q"],
    "I": ["L", "V", "M", "F"],
    "L": ["I", "V", "M", "F"],
    "K": ["R", "Q", "H", "E"],
    "M": ["I", "L", "V", "F"],
    "F": ["Y", "W", "L", "I"],
    "P": ["A", "G", "S", "V"],
    "S": ["T", "A", "N", "G"],
    "T": ["S", "A", "N", "V"],
    "W": ["F", "Y", "L", "H"],
    "Y": ["F", "W", "H", "S"],
    "V": ["I", "L", "A", "T"],
}
CONSERVATIVE_N = 4

SIGMA_KCAT = 0.4794
SIGMA_KM   = 0.8288



def get_gate_probs(model, seq: str, smi: str) -> np.ndarray:
    """
    KineticsPredictor.forward() 返回第 4 个值 gate_p: (B, Lp)
    对应每个残基的结构响应权重（由 ResidueMask 在交互层前计算）。
    """
    model.eval()
    with torch.no_grad():
        out = model([seq], [smi], use_mask=False)

    gate_p_tensor = out[3]  # (1, Lp)
    gp = gate_p_tensor.squeeze(0).cpu().numpy()

    if len(gp) == len(seq):
        return gp
    elif len(gp) == len(seq) + 2:
        return gp[1:-1]
    else:
        L = len(seq)
        return gp[:L] if len(gp) >= L else np.pad(gp, (0, L - len(gp)))



def infer_wt(model, seq: str, smi: str, mc_samples: int = 5) -> tuple[float, float]:
    """返回野生型的 (wt_logkcat, wt_logKm)，MC-Dropout 均值。"""
    preds = []
    enable_mc_dropout(model)
    with torch.no_grad():
        for _ in range(mc_samples):
            out = model([seq], [smi], use_mask=False)
            preds.append([out[0].item(), out[1].item()])
    preds = np.array(preds)
    return float(preds[:, 0].mean()), float(preds[:, 1].mean())



def scan(
    model,
    seq: str,
    smi: str,
    positions: list[tuple[int, list[str]]],   # [(pos, [mut_aas]), ...]
    wt_logkcat: float,
    wt_logKm: float,
    gate_probs: np.ndarray,
    group_labels: dict[int, str] | None = None,
    mc_samples: int = 5,
) -> pd.DataFrame:
    """
    Parameters
    ----------
    positions    : list of (0-indexed position, list of mut amino acids)
    group_labels : {pos: group_name}，默认全部为 "scan"

    Returns
    -------
    df 列格式与 insilico_mut_scan.csv 一致
    """
    if group_labels is None:
        group_labels = {}

    enable_mc_dropout(model)
    s_km   = wt_logKm   / SIGMA_KM        # wt 归一化（常量）
    s_kcat = wt_logkcat / SIGMA_KCAT      # wt 归一化（常量）

    records = []
    for pos, mut_aas in positions:
        wt_aa = seq[pos]
        gp    = float(gate_probs[pos]) if pos < len(gate_probs) else 0.0
        group = group_labels.get(pos, "scan")

        for mut_aa in mut_aas:
            if mut_aa == wt_aa:
                continue

            mut_seq = list(seq)
            mut_seq[pos] = mut_aa
            mut_seq = "".join(mut_seq)

            preds = []
            with torch.no_grad():
                for _ in range(mc_samples):
                    out = model([mut_seq], [smi], use_mask=False)
                    preds.append([out[0].item(), out[1].item()])

            preds    = np.array(preds)
            mut_kcat = float(preds[:, 0].mean())
            mut_km   = float(preds[:, 1].mean())
            unc_kcat = float(preds[:, 0].std())
            unc_km   = float(preds[:, 1].std())

            d_logkcat = mut_kcat - wt_logkcat
            d_logKm   = mut_km   - wt_logKm

            score_kcat = d_logkcat / SIGMA_KCAT
            score_km   = d_logKm   / SIGMA_KM

            records.append({
                "pos0":         pos,
                "pos1":         pos + 1,
                "group":        group,
                "wt_aa":        wt_aa,
                "mut_aa":       mut_aa,
                "gate_p":       gp,
                "wt_logKm":     wt_logKm,
                "mut_logKm":    mut_km,
                "d_logKm":      d_logKm,
                "wt_logkcat":   wt_logkcat,
                "mut_logkcat":  mut_kcat,
                "d_logkcat":    d_logkcat,
                "s_km":         s_km,
                "s_kcat":       s_kcat,
                "score_km":     score_km,
                "score_kcat":   score_kcat,
                "unc_kcat":     unc_kcat,
                "unc_km":       unc_km,
            })

    return pd.DataFrame(records)



def select_positions_topk(
    gate_probs: np.ndarray,
    seq: str,
    top_k: int,
    control_k: int = 0,
    scan_all_aa: bool = False,
) -> tuple[list[tuple[int, list[str]]], dict[int, str]]:
    """
    Returns
    -------
    positions    : [(pos, [mut_aas]), ...]
    group_labels : {pos: "gate_top" | "control_low"}
    """
    n = len(seq)
    sorted_idx = np.argsort(gate_probs)[::-1]

    gate_top_idx   = sorted_idx[:top_k].tolist()
    control_low_idx = sorted_idx[n - control_k:].tolist() if control_k > 0 else []

    group_labels = {}
    for p in gate_top_idx:
        group_labels[p] = "gate_top"
    for p in control_low_idx:
        group_labels[p] = "control_low"

    all_pos = list(dict.fromkeys(gate_top_idx + control_low_idx))  # 保持顺序去重

    positions = []
    for pos in all_pos:
        wt_aa = seq[pos]
        if scan_all_aa:
            mut_aas = [a for a in AA20 if a != wt_aa]
        else:
            mut_aas = [a for a in CONSERVATIVE_SUBS.get(wt_aa, AA20[:4]) if a != wt_aa]
        positions.append((pos, mut_aas))

    return positions, group_labels



def main():
    p = argparse.ArgumentParser(description="In silico 单点突变扫描")
    p.add_argument("--seq",          required=True, help="蛋白质序列（单行字符串）")
    p.add_argument("--smi",          required=True, help="底物 SMILES")
    p.add_argument("--weights",      required=True, help="模型权重 .pt")
    p.add_argument("--out",          required=True, help="输出 CSV 路径")
    p.add_argument("--device",       default="cuda")
    p.add_argument("--mc-samples",   type=int, default=5, help="MC-Dropout 采样次数")

    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--top-k",      type=int, default=15,
                      help="按 gate_p 选 top-K 位点（默认 15）")
    mode.add_argument("--scan-all",   action="store_true",
                      help="全序列扫描所有位点（慢，适合短序列）")
    mode.add_argument("--mutations-csv", type=str, default=None,
                      help="手动指定突变列表 CSV（列：pos,wt_aa,mut_aa）")

    p.add_argument("--control-k",    type=int, default=0,
                   help="同时扫描 gate_p 最低的 K 个对照位点（用于生成 control_low 组）")
    p.add_argument("--all-aa",       action="store_true",
                   help="每位点枚举全部 19 种替换（默认仅保守替换集合，4种）")

    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    seq = args.seq.strip()
    if seq.startswith(">"):
        seq = "".join(seq.split("\n")[1:])

    print(f"[insilico_mut_scan] 加载模型：{args.weights}")
    model, _, ckpt = build_model_from_ckpt(args.weights, device=str(device))
    del ckpt
    model.to(device)
    enable_mc_dropout(model)

    print("[insilico_mut_scan] 推理野生型基准…")
    wt_logkcat, wt_logKm = infer_wt(model, seq, args.smi, mc_samples=args.mc_samples)
    print(f"  wt log(kcat) = {wt_logkcat:.4f}  wt log(Km) = {wt_logKm:.4f}")

    print("[insilico_mut_scan] 提取 gate_p…")
    gate_probs = get_gate_probs(model, seq, args.smi)
    print(f"  gate_p: mean={gate_probs.mean():.4f}, max={gate_probs.max():.4f}")

    group_labels: dict[int, str] = {}

    if args.mutations_csv:
        mut_df = pd.read_csv(args.mutations_csv)
        pos_dict: dict[int, list[str]] = {}
        for _, row in mut_df.iterrows():
            pos = int(row["pos"])
            if seq[pos] != row["wt_aa"]:
                print(f"  警告：位点 {pos} WT 氨基酸不匹配（期望 {row['wt_aa']}，实际 {seq[pos]}），跳过")
                continue
            pos_dict.setdefault(pos, []).append(str(row["mut_aa"]))
        positions = list(pos_dict.items())
        group_labels = {pos: "custom" for pos in pos_dict}

    elif args.scan_all:
        scan_all_aa = True
        positions   = [(i, [a for a in AA20 if a != seq[i]]) for i in range(len(seq))]
        group_labels = {i: "all_scan" for i in range(len(seq))}

    else:
        positions, group_labels = select_positions_topk(
            gate_probs, seq,
            top_k=args.top_k,
            control_k=args.control_k,
            scan_all_aa=args.all_aa,
        )

    total_muts = sum(len(muts) for _, muts in positions)
    print(f"[insilico_mut_scan] 扫描 {len(positions)} 个位点，{total_muts} 个突变体…")

    df = scan(
        model=model,
        seq=seq,
        smi=args.smi,
        positions=positions,
        wt_logkcat=wt_logkcat,
        wt_logKm=wt_logKm,
        gate_probs=gate_probs,
        group_labels=group_labels,
        mc_samples=args.mc_samples,
    )

    df = df.sort_values("gate_p", ascending=False).reset_index(drop=True)

    df.to_csv(args.out, index=False)
    print(f"\n[insilico_mut_scan] 完成 → {args.out}  ({len(df)} 条)")

    n_kcat_good = int((df["d_logkcat"] > 0).sum())
    n_km_good   = int((df["d_logKm"]   < 0).sum())
    n_both      = int(((df["d_logkcat"] > 0) & (df["d_logKm"] < 0)).sum())
    print(f"  kcat↑：{n_kcat_good}/{len(df)}  Km↓：{n_km_good}/{len(df)}  双重有益：{n_both}/{len(df)}")

    doubly = df[(df["d_logkcat"] > 0) & (df["d_logKm"] < 0)].copy()
    doubly["combined"] = doubly["d_logkcat"] - doubly["d_logKm"]
    top5 = doubly.nlargest(5, "combined")
    if len(top5):
        print("  Top-5 双重有益：")
        for _, r in top5.iterrows():
            print(f"    pos={int(r.pos0)} {r.wt_aa}>{r.mut_aa} "
                  f"d_kcat={r.d_logkcat:+.4f} d_km={r.d_logKm:+.4f} gate_p={r.gate_p:.3f}")


if __name__ == "__main__":
    main()
