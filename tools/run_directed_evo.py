"""
tools/run_directed_evo.py
────────────────────────────────────────────────────────────────
计算辅助定向进化主循环（In Silico Directed Evolution）。

流程（每轮迭代）
─────────────
  1. Gate 位点选择器：提取当前序列的 gate_p，选 top-K 候选位点
  2. 突变评分器    ：枚举候选位点全部 19 种替换，批量模型推理
  3. Pareto 过滤器 ：提取双目标 Pareto 前沿（score_kcat ↑, score_km ↑）
  4. 组合提案器    ：生成双/三突变体加性预测，选出 top 组合
  5. 收敛判断      ：当最优增益 < ε 或达到 max_iter 时停止

用法示例
────────
  python tools/run_directed_evo.py \
      --wt-seq MAAKVLFTS... \
      --wt-smi "CC(=O)OC1=CC=CC=C1C(=O)O" \
      --weights checkpoints/best.pt \
      --out-dir results/directed_evo \
      --top-k 15 \
      --max-iter 5 \
      --max-combo 2 \
      --eps 0.01

输出文件（每轮一份，保存在 --out-dir）
──────────────────────────────────────
  iter{i}_mut_scan.csv     全量单点突变扫描
  iter{i}_pareto.csv       Pareto 前沿
  iter{i}_combos.csv       组合突变及加性预测分数
  convergence.csv          每轮最优 combo 得分汇总（用于收敛曲线）
  best_mutant.txt          最终推荐突变序列与突变列表
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tools.pareto_filter import run as pareto_run
from tools.combine_mutations import (
    additive_combinations,
    apply_mutations,
    model_verify,
)
from main_infer_predictor import build_model_from_ckpt, enable_mc_dropout



AA20 = list("ACDEFGHIKLMNPQRSTVWY")

SIGMA_KCAT = 0.4794
SIGMA_KM   = 0.8288



def get_gate_probs(model, seq: str, smi: str, device) -> np.ndarray:
    """
    从模型 forward() 的第 4 个返回值提取残基门控概率 gate_p。

    KineticsPredictor.forward() 返回：
        pred_kcat_log, pred_Km_log, pred_ratio_log, gate_p, gate_s, s_raw, s_eff
    其中 gate_p: (B, Lp) 由 ResidueMask 在 cross-attention 之前计算，
    代表每个残基对模型输出的结构响应权重，范围 [0,1]。

    Returns
    -------
    gate_probs : ndarray of shape (L,)，L = 序列长度（含特殊 token，取 [1:-1]）
    """
    model.eval()
    with torch.no_grad():
        out = model([seq], [smi], use_mask=False)

    gate_p_tensor = out[3]   # (B=1, Lp)
    gp = gate_p_tensor.squeeze(0).cpu().numpy()   # (Lp,)

    if len(gp) == len(seq):
        return gp
    elif len(gp) == len(seq) + 2:
        return gp[1:-1]
    else:
        L = len(seq)
        return gp[:L] if len(gp) >= L else np.pad(gp, (0, L - len(gp)))



def scan_mutations(
    model,
    seq: str,
    smi: str,
    device,
    top_k_positions: list[int],
    wt_kcat_log: float,
    wt_km_log: float,
    mc_samples: int = 5,
) -> pd.DataFrame:
    """
    对 top_k_positions 中的每个位点枚举 19 种替换，推理并计算 Δ 与 score。
    """
    records = []
    model.eval()
    enable_mc_dropout(model)

    gate_probs = get_gate_probs(model, seq, smi, device)

    for pos in top_k_positions:
        wt_aa = seq[pos]
        gp = float(gate_probs[pos]) if pos < len(gate_probs) else 0.0

        for mut_aa in AA20:
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

            preds = np.array(preds)
            mut_kcat = float(preds[:, 0].mean())
            mut_km   = float(preds[:, 1].mean())
            unc_kcat = float(preds[:, 0].std())
            unc_km   = float(preds[:, 1].std())

            d_kcat = mut_kcat - wt_kcat_log
            d_km   = mut_km   - wt_km_log

            s_km   = wt_km_log   / SIGMA_KM
            s_kcat = wt_kcat_log / SIGMA_KCAT

            score_kcat = d_kcat / SIGMA_KCAT
            score_km   = -d_km  / SIGMA_KM     # Km 减少为正

            records.append({
                "pos0":         pos,
                "pos1":         pos + 1,
                "wt_aa":        wt_aa,
                "mut_aa":       mut_aa,
                "gate_p":       gp,
                "wt_logKm":     wt_km_log,
                "mut_logKm":    mut_km,
                "d_logKm":      d_km,
                "wt_logkcat":   wt_kcat_log,
                "mut_logkcat":  mut_kcat,
                "d_logkcat":    d_kcat,
                "s_km":         s_km,
                "s_kcat":       s_kcat,
                "score_km":     score_km,
                "score_kcat":   score_kcat,
                "unc_kcat":     unc_kcat,
                "unc_km":       unc_km,
            })

    return pd.DataFrame(records)



def run_directed_evo(
    wt_seq: str,
    wt_smi: str,
    weights: str,
    out_dir: str,
    top_k: int = 15,
    max_iter: int = 5,
    max_combo: int = 2,
    top_singles: int = 10,
    eps: float = 0.01,
    device: str = "cuda",
    mc_samples: int = 5,
    verify_combos: bool = False,
    top_verify: int = 20,
):
    os.makedirs(out_dir, exist_ok=True)
    device_ = torch.device(device if torch.cuda.is_available() else "cpu")

    print(f"[directed_evo] 加载模型：{weights}")
    model, cfg, ckpt = build_model_from_ckpt(weights, device=str(device_))
    del ckpt
    model.to(device_)
    enable_mc_dropout(model)

    print("[directed_evo] 推理野生型基准…")
    preds_wt = []
    with torch.no_grad():
        for _ in range(mc_samples):
            out = model([wt_seq], [wt_smi], use_mask=False)
            preds_wt.append([out[0].item(), out[1].item()])
    preds_wt = np.array(preds_wt)
    wt_kcat_log = float(preds_wt[:, 0].mean())
    wt_km_log   = float(preds_wt[:, 1].mean())
    print(f"  wt log(kcat) = {wt_kcat_log:.4f}, wt log(Km) = {wt_km_log:.4f}")

    current_seq      = wt_seq
    current_kcat_log = wt_kcat_log
    current_km_log   = wt_km_log
    applied_mutations: list[str] = []

    convergence_rows = []
    best_score_prev  = -np.inf

    for it in range(1, max_iter + 1):
        print(f"\n{'='*60}")
        print(f"  迭代 {it}/{max_iter}")
        print(f"{'='*60}")

        gate_probs = get_gate_probs(model, current_seq, wt_smi, device_)
        mutated_positions = set()
        for m in applied_mutations:
            for part in m.split(","):
                mutated_positions.add(int(part[1:-1]))

        candidates = sorted(
            [i for i in range(len(current_seq)) if i not in mutated_positions],
            key=lambda i: gate_probs[i] if i < len(gate_probs) else 0.0,
            reverse=True,
        )[:top_k]
        print(f"  Step 1 | 选出 {len(candidates)} 个候选位点（gate_p 排名前 {top_k}）")

        print(f"  Step 2 | 扫描 {len(candidates) * 19} 个候选突变…")
        scan_df = scan_mutations(
            model, current_seq, wt_smi, device_,
            top_k_positions=candidates,
            wt_kcat_log=current_kcat_log,
            wt_km_log=current_km_log,
            mc_samples=mc_samples,
        )
        scan_csv = os.path.join(out_dir, f"iter{it}_mut_scan.csv")
        scan_df.to_csv(scan_csv, index=False)
        print(f"  Step 2 | 保存至 {scan_csv}")

        pareto_csv = os.path.join(out_dir, f"iter{it}_pareto.csv")
        pareto_df = pareto_run(
            input_csv=scan_csv,
            out_csv=pareto_csv,
        )
        print(f"  Step 3 | Pareto 前沿：{len(pareto_df)} 条")

        if pareto_df.empty:
            print("  Step 3 | 无 Pareto 候选，提前终止")
            break

        combos_df = additive_combinations(
            pareto_df,
            max_combo=max_combo,
            top_singles=top_singles,
        )
        if verify_combos and len(combos_df) > 0:
            print(f"  Step 4 | 对前 {top_verify} 个组合做模型精确验证…")
            combos_df = model_verify(
                combos_df, current_seq, wt_smi,
                weights=weights,
                device=device,
                top_n=top_verify,
                mc_samples=mc_samples,
            )
        combos_csv = os.path.join(out_dir, f"iter{it}_combos.csv")
        combos_df.to_csv(combos_csv, index=False)
        print(f"  Step 4 | {len(combos_df)} 个组合 → {combos_csv}")

        if combos_df.empty:
            best_score = float(pareto_df["score_kcat"].max() + pareto_df["score_km"].max())
            best_mut   = f"{pareto_df.iloc[0]['wt_aa']}{pareto_df.iloc[0]['pos0']}{pareto_df.iloc[0]['mut_aa']}"
        else:
            best_row   = combos_df.iloc[0]
            best_score = float(best_row["additive_score"])
            best_mut   = best_row["mut_list"]

        delta_score = best_score - best_score_prev
        print(f"  Step 5 | best_score={best_score:.4f}  Δ={delta_score:+.4f}  (ε={eps})")

        convergence_rows.append({
            "iter":        it,
            "best_score":  best_score,
            "delta_score": delta_score,
            "best_mut":    best_mut,
            "n_applied":   len(applied_mutations),
        })

        if delta_score < eps and it > 1:
            print(f"  收敛：Δscore < ε={eps}，停止迭代")
            break

        best_single = pareto_df.iloc[0]
        pos_apply = int(best_single["pos0"])
        if pos_apply not in mutated_positions:
            apply_str = f"{best_single['wt_aa']}{pos_apply}{best_single['mut_aa']}"
            current_seq = apply_mutations(current_seq, apply_str)
            applied_mutations.append(apply_str)
            current_kcat_log += float(best_single["d_logkcat"])
            current_km_log   += float(best_single["d_logKm"])
            print(f"  应用突变：{apply_str}  "
                  f"cum_kcat={current_kcat_log:.4f}  cum_km={current_km_log:.4f}")

        best_score_prev = best_score

    conv_df = pd.DataFrame(convergence_rows)
    conv_csv = os.path.join(out_dir, "convergence.csv")
    conv_df.to_csv(conv_csv, index=False)
    print(f"\n[directed_evo] 收敛曲线 → {conv_csv}")

    best_txt = os.path.join(out_dir, "best_mutant.txt")
    with open(best_txt, "w", encoding="utf-8") as f:
        f.write("=== In Silico Directed Evolution: Best Mutant ===\n\n")
        f.write(f"Applied mutations : {', '.join(applied_mutations)}\n")
        f.write(f"Cumulative Δlog(kcat) : {current_kcat_log - wt_kcat_log:+.4f}\n")
        f.write(f"Cumulative Δlog(Km)   : {current_km_log - wt_km_log:+.4f}\n")
        f.write(f"Final sequence        :\n{current_seq}\n")
    print(f"[directed_evo] 最优突变体 → {best_txt}")

    return conv_df, current_seq, applied_mutations



def main():
    p = argparse.ArgumentParser(description="In Silico 定向进化主循环")
    p.add_argument("--wt-seq",      required=True, help="野生型蛋白序列")
    p.add_argument("--wt-smi",      required=True, help="底物 SMILES")
    p.add_argument("--weights",     required=True, help="模型权重 .pt")
    p.add_argument("--out-dir",     required=True, help="输出目录")
    p.add_argument("--top-k",       type=int, default=15, help="每轮选取 top-K gate 位点（默认 15）")
    p.add_argument("--max-iter",    type=int, default=5,  help="最大迭代轮数（默认 5）")
    p.add_argument("--max-combo",   type=int, default=2,  help="最大组合阶数（默认 2）")
    p.add_argument("--top-singles", type=int, default=10, help="参与组合的最大单点突变数（默认 10）")
    p.add_argument("--eps",         type=float, default=0.01, help="收敛阈值（默认 0.01）")
    p.add_argument("--device",      type=str,  default="cuda")
    p.add_argument("--mc-samples",  type=int,  default=5, help="MC-Dropout 采样次数")
    p.add_argument("--verify",      action="store_true", help="对 top 组合做完整模型精确验证")
    p.add_argument("--top-verify",  type=int, default=20, help="精确验证前 N 个组合（默认 20）")
    args = p.parse_args()

    wt_seq = args.wt_seq.strip()
    if wt_seq.startswith(">"):
        wt_seq = "".join(wt_seq.split("\n")[1:])

    run_directed_evo(
        wt_seq=wt_seq,
        wt_smi=args.wt_smi,
        weights=args.weights,
        out_dir=args.out_dir,
        top_k=args.top_k,
        max_iter=args.max_iter,
        max_combo=args.max_combo,
        top_singles=args.top_singles,
        eps=args.eps,
        device=args.device,
        mc_samples=args.mc_samples,
        verify_combos=args.verify,
        top_verify=args.top_verify,
    )


if __name__ == "__main__":
    main()
