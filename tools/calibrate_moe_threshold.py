"""在一组混合的 ID+OOD 预测缓存上网格搜索 OOD router 的 (d0, tau)。

这不是训练，只是在已缓存的 main/expert/ood_score 上做纯 numpy 网格搜索：
  w_expert = sigmoid((ood_score - d0) / tau)
  mu_ens   = (1-w_expert)*mu_main + w_expert*mu_expert
挑使全局（或指定切分）加权 MAE 最低的 (d0, tau)。

典型流程:
  1) 先对 val_csv (ID) 和 eval_ood40_csv / eval_ood60_csv 分别跑
     main_infer_ensemble.py，得到各自的 predictions CSV（含 ood_score / mu_main /
     mu_expert / has_struct / y_kcat）。
  2) 把它们喂给本脚本:
       python tools/calibrate_moe_threshold.py \\
         --inputs id=result/id_preds.csv ood40=result/ood40_preds.csv ood60=result/ood60_preds.csv \\
         --out_csv result/calib_grid.csv \\
         --select_by overall --metric mae

输出:
  - 一个 CSV，行 = (d0, tau) 组合，列 = 各切分的 MAE/R²，以及 best 标记。
  - stdout 打印 top-K 组合。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_COLS = ("mu_main", "mu_expert", "ood_score", "y_kcat")


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _compute_metrics(mu: np.ndarray, y: np.ndarray) -> dict:
    valid = np.isfinite(mu) & np.isfinite(y)
    if valid.sum() < 2:
        return {"n": int(valid.sum()), "mae": np.nan, "mse": np.nan, "r": np.nan, "r2": np.nan}
    mu, y = mu[valid], y[valid]
    resid = mu - y
    mae = float(np.abs(resid).mean())
    mse = float((resid ** 2).mean())
    y_var = float(y.var())
    r2_det = 1.0 - mse / y_var if y_var > 1e-12 else np.nan
    if mu.std() > 1e-8 and y.std() > 1e-8:
        r = float(np.corrcoef(mu, y)[0, 1])
    else:
        r = np.nan
    return {"n": int(valid.sum()), "mae": mae, "mse": mse, "r": r, "r2": r2_det}


def _parse_inputs(pairs):
    out = {}
    for p in pairs:
        if "=" not in p:
            raise ValueError(f"--inputs expects TAG=PATH pairs, got: {p}")
        tag, path = p.split("=", 1)
        out[tag] = path
    return out


def _parse_grid(s: str) -> list[float]:
    return [float(x) for x in s.split(",") if x.strip()]


def _ensure_cols(df: pd.DataFrame, tag: str):
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"[{tag}] missing required columns: {missing}. "
                         f"run main_infer_ensemble.py first to produce them.")


def _fuse(mu_m, mu_e, s2_m, s2_e, ood, has_struct, d0, tau,
          hard_gate: bool, precision: bool) -> np.ndarray:
    w_e = _sigmoid((ood - d0) / max(tau, 1e-6))
    if hard_gate and has_struct is not None:
        w_e = w_e * has_struct.astype(np.float64)
    if precision and s2_m is not None and s2_e is not None:
        p_m = 1.0 / np.clip(s2_m, 1e-6, None)
        p_e = 1.0 / np.clip(s2_e, 1e-6, None)
        w_m_raw = (1.0 - w_e) * p_m
        w_e_raw = w_e * p_e
        z = np.clip(w_m_raw + w_e_raw, 1e-8, None)
        w_e = w_e_raw / z
    return (1.0 - w_e) * mu_m + w_e * mu_e


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True,
                    help="一个或多个 TAG=PATH（如 id=a.csv ood40=b.csv）")
    ap.add_argument("--d0_grid", type=str, default="0.02,0.05,0.08,0.10,0.12,0.15,0.18,0.20,0.25,0.30")
    ap.add_argument("--tau_grid", type=str, default="0.01,0.02,0.05,0.08,0.12")
    ap.add_argument("--metric", choices=("mae", "mse", "r2"), default="mae",
                    help="用哪个指标选最优（越小越好：mae/mse；越大越好：r2）")
    ap.add_argument("--select_by", default="overall",
                    help="按哪个切分选优；'overall' 表示所有 tag concat 后计算")
    ap.add_argument("--out_csv", default=None)
    ap.add_argument("--hard_gate", action="store_true", default=True,
                    help="无结构样本强制 w_expert=0（默认开启）")
    ap.add_argument("--no_hard_gate", dest="hard_gate", action="store_false")
    ap.add_argument("--use_precision_weighting", action="store_true", default=False)
    ap.add_argument("--top_k", type=int, default=10)
    args = ap.parse_args()

    inputs = _parse_inputs(args.inputs)
    d0_grid = _parse_grid(args.d0_grid)
    tau_grid = _parse_grid(args.tau_grid)

    splits = {}
    for tag, path in inputs.items():
        df = pd.read_csv(path)
        _ensure_cols(df, tag)
        splits[tag] = {
            "mu_m": df["mu_main"].to_numpy(dtype=np.float64),
            "mu_e": df["mu_expert"].to_numpy(dtype=np.float64),
            "s2_m": df["s2_main"].to_numpy(dtype=np.float64) if "s2_main" in df.columns else None,
            "s2_e": df["s2_expert"].to_numpy(dtype=np.float64) if "s2_expert" in df.columns else None,
            "ood":  df["ood_score"].to_numpy(dtype=np.float64),
            "y":    df["y_kcat"].to_numpy(dtype=np.float64),
            "has":  df["has_struct"].to_numpy() if "has_struct" in df.columns else None,
            "n":    len(df),
        }
        has_str = "-"
        if splits[tag]["has"] is not None:
            has_str = f"{splits[tag]['has'].astype(bool).mean():.2f}"
        print(f"[load] {tag}  rows={splits[tag]['n']}  has_struct_frac={has_str}  "
              f"ood range=[{splits[tag]['ood'].min():.3f}, {splits[tag]['ood'].max():.3f}]")

    rows = []
    for d0 in d0_grid:
        for tau in tau_grid:
            row = {"d0": d0, "tau": tau}
            fused_all, y_all = [], []
            for tag, s in splits.items():
                has = s["has"].astype(bool) if s["has"] is not None else None
                mu_ens = _fuse(
                    s["mu_m"], s["mu_e"], s["s2_m"], s["s2_e"],
                    s["ood"], has, d0, tau,
                    hard_gate=args.hard_gate, precision=args.use_precision_weighting,
                )
                m = _compute_metrics(mu_ens, s["y"])
                for k, v in m.items():
                    row[f"{tag}_{k}"] = v
                fused_all.append(mu_ens)
                y_all.append(s["y"])
            fused = np.concatenate(fused_all)
            ya = np.concatenate(y_all)
            m_all = _compute_metrics(fused, ya)
            for k, v in m_all.items():
                row[f"overall_{k}"] = v
            rows.append(row)

    res = pd.DataFrame(rows)

    baselines = {}
    for tag, s in splits.items():
        baselines[f"{tag}_main_only_mae"] = _compute_metrics(s["mu_m"], s["y"])["mae"]
        baselines[f"{tag}_expert_only_mae"] = _compute_metrics(s["mu_e"], s["y"])["mae"]
    all_m_mu = np.concatenate([s["mu_m"] for s in splits.values()])
    all_e_mu = np.concatenate([s["mu_e"] for s in splits.values()])
    all_y = np.concatenate([s["y"] for s in splits.values()])
    baselines["overall_main_only_mae"] = _compute_metrics(all_m_mu, all_y)["mae"]
    baselines["overall_expert_only_mae"] = _compute_metrics(all_e_mu, all_y)["mae"]

    sel_col = f"{args.select_by}_{args.metric}"
    if sel_col not in res.columns:
        raise ValueError(f"select_by={args.select_by} metric={args.metric} 对应列 {sel_col} 不存在。"
                         f"候选列: {[c for c in res.columns if c.endswith(args.metric)]}")
    ascending = args.metric in ("mae", "mse")
    res_sorted = res.sort_values(sel_col, ascending=ascending).reset_index(drop=True)
    res_sorted["_rank"] = np.arange(len(res_sorted))

    print(f"\n=== Baselines ===")
    for k, v in baselines.items():
        print(f"  {k:40s} {v:.4f}")

    print(f"\n=== Top-{args.top_k} by {sel_col} ({'↓' if ascending else '↑'}) ===")
    show_cols = ["_rank", "d0", "tau"] + [c for c in res_sorted.columns if c.endswith("_mae") or c == sel_col]
    show_cols = list(dict.fromkeys(show_cols))  # dedupe 保持顺序
    print(res_sorted.head(args.top_k)[show_cols].to_string(index=False))

    if args.out_csv:
        Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
        res_sorted.to_csv(args.out_csv, index=False)
        print(f"\n[saved] {args.out_csv}")

    best = res_sorted.iloc[0]
    print(f"\n[best] d0={best['d0']:.3f}  tau={best['tau']:.3f}  "
          f"{sel_col}={best[sel_col]:.4f}")


if __name__ == "__main__":
    main()
