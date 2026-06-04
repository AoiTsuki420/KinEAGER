"""
对 holdout（leave-one-source-out）结果做 post-hoc 处理：
  1. Isotonic regression 尺度校准 -> 把负 R² 拉回正区间
  2. Temperature scaling 不确定性校准 -> 把 Cov90 拉回 0.7+
两者共用同一个小校准子集（默认 200 条），用 5-fold CV 分层。

输入:
  predictions.csv,  含 [y_true, y_pred, sigma]  (任选其一字段名见 ALIASES)
  --task kcat | km | ratio
输出:
  predictions_calibrated.csv  原列 + y_pred_calibrated + sigma_calibrated
  metrics_before.json / metrics_after.json
  fit_summary.json            (isotonic 节点、最优 τ)
用法示例:
  python posthoc_calibrate_loso.py \
      --pred /root/autodl-fs/runs/.../ood_skidkcat/predictions.csv \
      --task kcat \
      --calib_n 200 --seed 42 \
      --out_dir ./posthoc_skidkcat
"""
import argparse, json, os, sys
import numpy as np, pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr, spearmanr, norm
from sklearn.model_selection import KFold

ALIASES = {
    "kcat":  dict(t=["y_true_kcat_log10","log10kcat_max","y_true"],
                  p=["y_pred_kcat_log10","pred_kcat_log10","y_pred"],
                  s=["sigma_kcat_log10","sigma_kcat","sigma"]),
    "km":    dict(t=["y_true_km_log10","log10km_mean","y_true"],
                  p=["y_pred_km_log10","pred_km_log10","y_pred"],
                  s=["sigma_km_log10","sigma_km","sigma"]),
    "ratio": dict(t=["y_true_ratio_log10","log10ratio","y_true"],
                  p=["y_pred_ratio_log10","pred_ratio_log10","y_pred"],
                  s=["sigma_ratio_log10","sigma_ratio","sigma"]),
}

def pick(df, aliases):
    for c in aliases:
        if c in df.columns: return c
    raise KeyError(f"no column among {aliases} in {list(df.columns)}")

def metrics_block(yt, yp, sg):
    out = dict(n=int(len(yt)),
               mae=float(mean_absolute_error(yt, yp)),
               rmse=float(np.sqrt(mean_squared_error(yt, yp))),
               r2=float(r2_score(yt, yp)),
               pearson_r=float(pearsonr(yt, yp)[0]) if len(yt)>1 else np.nan,
               spearman_r=float(spearmanr(yt, yp).correlation) if len(yt)>1 else np.nan)
    if sg is not None and np.all(np.isfinite(sg)) and np.all(sg > 0):
        z   = (yt - yp) / sg
        nll = 0.5*np.log(2*np.pi*sg**2) + 0.5*z**2
        out.update(nll=float(nll.mean()),
                   cov90=float(np.mean(np.abs(z) <= 1.6448536)),
                   cov95=float(np.mean(np.abs(z) <= 1.959964)),
                   sigma_mean=float(sg.mean()))
    return out

def fit_temperature(z):
    """ argmin_tau 0.5 log(2 pi tau^2) + 0.5 z^2/tau^2 """
    def loss(tau):
        if tau <= 0: return 1e9
        return float(np.mean(0.5*np.log(2*np.pi*tau**2) + 0.5*(z/tau)**2))
    res = minimize_scalar(loss, bounds=(1e-3, 1e3), method="bounded")
    return float(res.x)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--task", choices=list(ALIASES.keys()), required=True)
    ap.add_argument("--calib_n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--no_temperature", action="store_true",
                    help="skip temperature scaling for sigma")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    df = pd.read_csv(args.pred)
    al = ALIASES[args.task]
    tc, pc = pick(df, al["t"]), pick(df, al["p"])
    has_sigma = any(c in df.columns for c in al["s"])
    sc = pick(df, al["s"]) if has_sigma else None

    yt = pd.to_numeric(df[tc], errors="coerce").to_numpy()
    yp = pd.to_numeric(df[pc], errors="coerce").to_numpy()
    sg = pd.to_numeric(df[sc], errors="coerce").to_numpy() if has_sigma else None
    v  = np.isfinite(yt) & np.isfinite(yp) & (np.isfinite(sg) if has_sigma else True)
    if v.sum() < args.calib_n + 50:
        sys.exit(f"[ERR] not enough valid rows ({v.sum()}) for calib_n={args.calib_n} + ≥50 eval")

    yt, yp = yt[v], yp[v]
    sg = sg[v] if has_sigma else None
    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(len(yt))
    cal = idx[:args.calib_n]; ev = idx[args.calib_n:]

    before = metrics_block(yt[ev], yp[ev], sg[ev] if has_sigma else None)

    iso = IsotonicRegression(out_of_bounds="clip").fit(yp[cal], yt[cal])
    yp_cal_full = iso.predict(yp)
    after_pt = metrics_block(yt[ev], yp_cal_full[ev], None)

    if has_sigma and not args.no_temperature:
        z_cal  = (yt[cal] - yp_cal_full[cal]) / sg[cal]
        tau    = fit_temperature(z_cal)
        sg_cal = sg * tau
        after_full = metrics_block(yt[ev], yp_cal_full[ev], sg_cal[ev])
    else:
        tau = None; sg_cal = sg
        after_full = after_pt

    out = df.copy()
    full_idx_map = np.where(v)[0]
    yp_col_full = np.full(len(df), np.nan)
    sg_col_full = np.full(len(df), np.nan)
    yp_col_full[full_idx_map] = yp_cal_full
    if has_sigma and tau is not None:
        sg_col_full[full_idx_map] = sg_cal
    out[f"{args.task}_pred_calibrated"]  = yp_col_full
    if has_sigma and tau is not None:
        out[f"{args.task}_sigma_calibrated"] = sg_col_full
    out.to_csv(f"{args.out_dir}/predictions_calibrated.csv", index=False)

    json.dump(before,    open(f"{args.out_dir}/metrics_before.json", "w"), indent=2)
    json.dump(after_full,open(f"{args.out_dir}/metrics_after.json",  "w"), indent=2)
    json.dump(dict(
        task=args.task, calib_n=int(args.calib_n), seed=int(args.seed),
        eval_n=int(len(ev)),
        isotonic_nodes=list(map(float, np.unique(iso.X_thresholds_))),
        temperature_tau=tau,
    ), open(f"{args.out_dir}/fit_summary.json","w"), indent=2)

    print("===== BEFORE =====")
    print(json.dumps(before, indent=2))
    print("===== AFTER  =====")
    print(json.dumps(after_full, indent=2))
    print(f"\n[OK] {args.out_dir}/predictions_calibrated.csv")

if __name__ == "__main__":
    main()
