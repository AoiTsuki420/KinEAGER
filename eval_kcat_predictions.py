import argparse
import json

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def _safe_pearsonr(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return float("nan")
    if np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(pearsonr(x, y)[0])


def _fit_linear_calibration(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float, np.ndarray]:
    if len(y_true) < 2:
        return float("nan"), float("nan"), np.full_like(y_pred, np.nan, dtype=float)
    if np.std(y_pred) == 0:
        a = 0.0
        b = float(np.mean(y_true))
    else:
        a, b = np.polyfit(y_pred, y_true, deg=1)
        a = float(a)
        b = float(b)
    y_pred_cal = a * y_pred + b
    return a, b, y_pred_cal


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-pred", type=str, required=True, help="predictions.csv 文件路径")
    p.add_argument(
        "--pred-col",
        type=str,
        default="pred_kcat_log",
        help="预测列名，默认 pred_kcat_log（log10 空间）；Km 评估改为 pred_Km_log",
    )
    p.add_argument(
        "--true-col",
        type=str,
        default="true_kcat_log",
        help="真实值列名，默认 true_kcat_log（log10 空间）；Km 评估改为 true_Km_log",
    )
    p.add_argument(
        "--log10",
        action="store_true",
        help="将输入列从线性空间转换到 log10 后再计算指标；若列本身已是 log10（*_log 列），请勿使用此选项",
    )
    args = p.parse_args()

    df = pd.read_csv(args.pred)

    if args.pred_col not in df.columns:
        raise ValueError(f"预测列不存在: {args.pred_col}")
    if args.true_col not in df.columns:
        raise ValueError(f"真实值列不存在: {args.true_col}")

    y_pred = pd.to_numeric(df[args.pred_col], errors="coerce").to_numpy(dtype=float)
    y_true = pd.to_numeric(df[args.true_col], errors="coerce").to_numpy(dtype=float)

    if args.log10:
        mask = np.isfinite(y_true) & np.isfinite(y_pred) & (y_true > 0) & (y_pred > 0)
    else:
        mask = np.isfinite(y_true) & np.isfinite(y_pred)
    n = int(mask.sum())
    if n == 0:
        raise ValueError("没有可用样本：请确认 predictions.csv 中包含正数的 true_kcat 和 pred_kcat")

    y_true = y_true[mask]
    y_pred = y_pred[mask]

    space = "linear"
    if args.log10:
        y_true = np.log10(y_true)
        y_pred = np.log10(y_pred)
        space = "log10"

    pearson_r = _safe_pearsonr(y_true, y_pred)
    spearman_r = float(spearmanr(y_true, y_pred)[0]) if n >= 2 else float("nan")
    pearson_r2 = float(pearson_r * pearson_r) if np.isfinite(pearson_r) else float("nan")

    cal_a, cal_b, y_pred_cal = _fit_linear_calibration(y_true, y_pred)
    calibrated_r2 = float(r2_score(y_true, y_pred_cal)) if n >= 2 else float("nan")
    pearson_r_calibrated = _safe_pearsonr(y_true, y_pred_cal)
    pearson_r2_calibrated = (
        float(pearson_r_calibrated * pearson_r_calibrated)
        if np.isfinite(pearson_r_calibrated)
        else float("nan")
    )

    metrics = {
        "n": n,
        "space": space,
        "mse": float(mean_squared_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)) if n >= 2 else float("nan"),
        "calibration_a": cal_a,
        "calibration_b": cal_b,
        "calibrated_r2": calibrated_r2,
        "pearson_r": pearson_r,
        "pearson_r2": pearson_r2,
        "pearson_r_calibrated": pearson_r_calibrated,
        "pearson_r2_calibrated": pearson_r2_calibrated,
        "spearman_r": spearman_r,
    }

    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
