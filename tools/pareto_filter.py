"""
tools/pareto_filter.py
────────────────────────────────────────────────────────────────
从突变扫描 CSV 中提取 Pareto 最优集合。

──────────────────────────────────────────────
重要：score_km 方向约定
──────────────────────────────────────────────
insilico_mut_scan.csv（以及 main_infer_predictor 原始输出）中：
  score_km = d_logKm / SIGMA_KM
  即 score_km > 0 表示 Km **升高**（底物亲和力下降，有害）。

本脚本默认对 score_km 取反，使 "正值=有益" 在 Pareto 优化中方向一致：
  score_km_beneficial = -score_km

若你的输入 CSV 中 score_km 已经是 "正值=有益"（如 run_directed_evo.py 输出），
请加 --km-already-beneficial 跳过取反步骤。

目标空间（取反后，均为越大越好）：
  - score_kcat          : Δlog(kcat) / σ_kcat   （kcat 增益，越大越好）
  - score_km_beneficial : -Δlog(Km) / σ_Km       （Km 降低，越大越好）

用法示例
────────
  python tools/pareto_filter.py \
      --input insilico_mut_scan.csv \
      --out pareto_front.csv

  python tools/pareto_filter.py \
      --input insilico_mut_scan.csv \
      --out pareto_front.csv \
      --group gate_top

  python tools/pareto_filter.py \
      --input insilico_mut_scan.csv \
      --out doubly_beneficial.csv \
      --filter-positive

  python tools/pareto_filter.py \
      --input iter1_mut_scan.csv \
      --out pareto_front.csv \
      --km-already-beneficial
"""

import argparse
import numpy as np
import pandas as pd



def is_pareto_efficient(costs: np.ndarray) -> np.ndarray:
    """
    计算 Pareto 最优前沿（最大化问题）。

    Parameters
    ----------
    costs : ndarray of shape (N, M)
        每行是一个候选方案，每列是一个目标（越大越好）。

    Returns
    -------
    mask : bool ndarray of shape (N,)
        True 表示该方案位于 Pareto 前沿。
    """
    n = costs.shape[0]
    is_efficient = np.ones(n, dtype=bool)
    for i in range(n):
        if not is_efficient[i]:
            continue
        dominated = (
            np.all(costs >= costs[i], axis=1) &
            np.any(costs > costs[i], axis=1)
        )
        dominated[i] = False
        if dominated.any():
            is_efficient[i] = False
    return is_efficient



def _add_score_ratio(df: pd.DataFrame) -> pd.DataFrame:
    """score_ratio = score_kcat + score_km（线性加和代理效率增益）。"""
    if "score_ratio" not in df.columns:
        df = df.copy()
        df["score_ratio"] = df["score_kcat"] + df["score_km"]
    return df



def run(
    input_csv: str,
    out_csv: str,
    group: str | None = None,
    objectives: list[str] | None = None,
    filter_positive: bool = False,
    min_gate_p: float | None = None,
    km_already_beneficial: bool = False,
) -> pd.DataFrame:
    """
    Parameters
    ----------
    input_csv            : 突变扫描 CSV 路径
    out_csv              : 输出路径（Pareto 前沿 CSV）
    group                : 若指定则只处理该 group（gate_top / control_low / None=全量）
    objectives           : Pareto 优化目标列名列表，默认 ["score_kcat", "score_km_beneficial"]
    filter_positive      : 为 True 时，仅保留所有目标 > 0 的行（双/多重有益子集，
                           即 kcat↑ AND Km↓）
    min_gate_p           : 可选：过滤 gate_p < 阈值的行
    km_already_beneficial: 若 CSV 中的 score_km 已是正值=有益，设为 True 跳过取反

    Returns
    -------
    pareto_df : Pareto 前沿 DataFrame（同时保存到 out_csv）
    """
    df = pd.read_csv(input_csv)
    print(f"[pareto_filter] 读入 {len(df)} 行，来自 {input_csv}")

    if "score_km" in df.columns and not km_already_beneficial:
        df["score_km_beneficial"] = -df["score_km"]
    elif "score_km" in df.columns:
        df["score_km_beneficial"] = df["score_km"]

    df = _add_score_ratio(df)

    if objectives is None:
        objectives = ["score_kcat", "score_km_beneficial"]

    if group is not None:
        df = df[df["group"] == group].copy()
        print(f"[pareto_filter] 按 group={group} 过滤后：{len(df)} 行")

    if min_gate_p is not None:
        df = df[df["gate_p"] >= min_gate_p].copy()
        print(f"[pareto_filter] gate_p >= {min_gate_p} 过滤后：{len(df)} 行")

    if filter_positive:
        mask = np.ones(len(df), dtype=bool)
        for obj in objectives:
            if obj in df.columns:
                mask &= df[obj].values > 0
        df = df[mask].copy()
        print(f"[pareto_filter] 双重有益（所有目标 > 0）过滤后：{len(df)} 行")

    if df.empty:
        print("[pareto_filter] 警告：过滤后无数据，输出空文件")
        df.to_csv(out_csv, index=False)
        return df

    missing = [o for o in objectives if o not in df.columns]
    if missing:
        raise ValueError(f"目标列不存在：{missing}，可用列：{list(df.columns)}")

    costs = df[objectives].values.astype(float)
    pareto_mask = is_pareto_efficient(costs)
    pareto_df = df[pareto_mask].copy()

    pareto_df = pareto_df.sort_values(
        by=["score_kcat", "score_km_beneficial"],
        ascending=False
    ).reset_index(drop=True)

    pareto_df.to_csv(out_csv, index=False)
    print(f"[pareto_filter] Pareto 前沿：{len(pareto_df)} 行 → 已写入 {out_csv}")

    print("\n  Top-10 Pareto 候选（kcat↑ Km↓ 均正向）：")
    cols_show = ["pos0", "wt_aa", "mut_aa", "group", "gate_p",
                 "score_kcat", "score_km_beneficial", "d_logkcat", "d_logKm"]
    cols_show = [c for c in cols_show if c in pareto_df.columns]
    print(pareto_df[cols_show].head(10).to_string(index=False))

    return pareto_df



def main():
    p = argparse.ArgumentParser(description="Pareto 过滤：从突变扫描结果中提取最优前沿")
    p.add_argument("--input",   required=True, help="突变扫描 CSV（insilico_mut_scan.csv）")
    p.add_argument("--out",     required=True, help="输出 CSV 路径")
    p.add_argument("--group",   default=None,  help="只处理该 group（gate_top/control_low），默认全量")
    p.add_argument("--objectives", nargs="+", default=None,
                   help="Pareto 目标列名（均为越大越好），默认 score_kcat score_km_beneficial")
    p.add_argument("--filter-positive", action="store_true",
                   help="只保留双重有益子集（kcat↑ AND Km↓，即 score_kcat>0 且 score_km_beneficial>0）")
    p.add_argument("--min-gate-p", type=float, default=None,
                   help="过滤 gate_p 低于此阈值的位点")
    p.add_argument("--km-already-beneficial", action="store_true",
                   help="若 CSV 中 score_km 已是正值=Km降低=有益，则跳过取反步骤")
    args = p.parse_args()

    run(
        input_csv=args.input,
        out_csv=args.out,
        group=args.group,
        objectives=args.objectives,
        filter_positive=args.filter_positive,
        min_gate_p=args.min_gate_p,
        km_already_beneficial=args.km_already_beneficial,
    )


if __name__ == "__main__":
    main()
