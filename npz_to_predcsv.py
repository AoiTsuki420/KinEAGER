"""
把训练时落盘的 {test_predictions.npz, test_set.csv, scaler_fold.json}
拼成 posthoc_calibrate_loso.py 可直接消费的 predictions.csv。

约定:
  - npz 中至少包含一个均值键 (pred_log10/pred/mu/y_pred/yhat/mean)
    和可选的不确定性键 (sigma_log10/sigma/std/log_sigma/...)
  - 若 npz 中是「已 scaler 归一化」后的值, 通过 --scaler 反变换回 log10 空间
  - test_set.csv 至少含真值列 (log10kcat_max / log10km_mean / log10ratio)

用法:
  python npz_to_predcsv.py \
      --npz   /root/autodl-fs/runs/ood_catapro/test_predictions.npz \
      --csv   /root/autodl-fs/runs/ood_catapro/test_set.csv \
      --scaler /root/autodl-fs/runs/ood_catapro/scaler_fold.json \
      --task  kcat \
      --out   /root/autodl-fs/runs/ood_catapro/predictions.csv
"""
import argparse, json, sys
import numpy as np
import pandas as pd

TASK_TRUE_COLS = {
    "kcat":  ["log10kcat_max", "y_true_kcat_log10", "log10kcat", "kcat_log10"],
    "km":    ["log10km_mean",  "y_true_km_log10",   "log10km",   "km_log10"],
    "ratio": ["log10ratio",    "y_true_ratio_log10","ratio_log10"],
}
TASK_RAW_COLS = {
    "kcat":  [("kcat(s^-1)", 0.0)],
    "km":    [("Km(M)",      3.0), ("Km(mM)", 0.0)],   # log10(M)+3 -> log10(mM)
    "ratio": [],
}
TRUE_KEYS  = ["y_true", "y_true_log10", "true_log10", "target", "label"]
PRED_KEYS  = ["pred_log10", "pred", "mu", "y_pred", "yhat", "mean"]
SIGMA_KEYS = ["sigma_log10", "sigma", "std", "log_sigma", "y_sigma", "stddev"]


def first_in(d, keys):
    for k in keys:
        if k in d:
            return k
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz",    required=True)
    ap.add_argument("--csv",    required=True)
    ap.add_argument("--task",   choices=list(TASK_TRUE_COLS), required=True)
    ap.add_argument("--out",    required=True)
    ap.add_argument("--scaler", default=None,
                    help="optional scaler_fold.json (mean/std) for inverse-transform")
    args = ap.parse_args()

    z = np.load(args.npz, allow_pickle=True)
    keys = list(z.keys())
    print(f"[npz] keys = {keys}")
    p_key = first_in(z, PRED_KEYS)
    if p_key is None:
        sys.exit(f"[ERR] no prediction key among {PRED_KEYS} in npz")
    s_key = first_in(z, SIGMA_KEYS)
    t_key_npz = first_in(z, TRUE_KEYS)
    pred  = np.asarray(z[p_key]).reshape(-1).astype(float)
    sigma = np.asarray(z[s_key]).reshape(-1).astype(float) if s_key else None
    if s_key == "log_sigma" and sigma is not None:
        sigma = np.exp(sigma)
    print(f"[npz] picked pred='{p_key}'  sigma='{s_key}'  truth='{t_key_npz}'")

    if args.scaler:
        sc = json.load(open(args.scaler))
        mean = float(sc.get("mean", sc.get("y_mean", sc.get("target_mean", 0.0))))
        std  = float(sc.get("std",  sc.get("y_std",  sc.get("target_std",  1.0))))
        if not (mean == 0.0 and std == 1.0):
            pred = pred * std + mean
            if sigma is not None:
                sigma = sigma * std
            print(f"[scaler] applied mean={mean:.6f} std={std:.6f}")
        else:
            print(f"[scaler] mean=0/std=1 -> skip (npz already in log10 space)")

    df = pd.read_csv(args.csv)
    if t_key_npz is not None:
        y_true = np.asarray(z[t_key_npz]).reshape(-1).astype(float)
        truth_src = f"npz['{t_key_npz}']"
    else:
        col = next((c for c in TASK_TRUE_COLS[args.task] if c in df.columns), None)
        if col is not None:
            y_true = pd.to_numeric(df[col], errors="coerce").to_numpy()
            truth_src = f"csv['{col}']"
        else:
            raw = next(((c, off) for c, off in TASK_RAW_COLS[args.task] if c in df.columns), None)
            if raw is None:
                sys.exit(f"[ERR] no truth col in npz {TRUE_KEYS} nor csv {TASK_TRUE_COLS[args.task]}/{[c for c,_ in TASK_RAW_COLS[args.task]]}; csv has: {list(df.columns)}")
            col, off = raw
            raw_v = pd.to_numeric(df[col], errors="coerce").to_numpy()
            y_true = np.log10(raw_v) + off
            truth_src = f"log10(csv['{col}'])+{off}"
    print(f"[truth] source={truth_src}  rows={len(y_true)}")

    n_csv, n_npz = len(df), len(pred)
    if n_npz != n_csv:
        if n_npz % n_csv != 0:
            sys.exit(f"[ERR] length mismatch and not a multiple: csv n={n_csv} npz n={n_npz}")
        k = n_npz // n_csv
        expected = None
        for col, off in TASK_RAW_COLS.get(args.task, []):
            if col in df.columns:
                rv = pd.to_numeric(df[col], errors="coerce").to_numpy()
                ok = np.isfinite(rv) & (rv > 0)
                if ok.sum() >= 50:
                    e = np.full(n_csv, np.nan)
                    e[ok] = np.log10(rv[ok]) + off
                    expected = e
                    print(f"[slice] expected truth from csv['{col}']+{off}  finite={ok.sum()}")
                    break
        if expected is None:
            sys.exit(f"[ERR] cannot derive expected truth from csv to auto-detect slice; "
                     f"add --task_idx/--task_layout manually")

        def take(arr, layout, idx):
            return arr[idx * n_csv:(idx + 1) * n_csv] if layout == "stacked" else arr[idx::k]

        best = None
        for layout in ("stacked", "interleaved"):
            for idx in range(k):
                cand = take(y_true, layout, idx)
                m = np.isfinite(cand) & np.isfinite(expected)
                if m.sum() < 20:
                    continue
                mae = float(np.mean(np.abs(cand[m] - expected[m])))
                if best is None or mae < best[0]:
                    best = (mae, layout, idx)
        if best is None:
            sys.exit(f"[ERR] all slice candidates have <20 finite overlap with expected truth")
        mae, layout, idx = best
        print(f"[slice] picked layout={layout} idx={idx}  truth-vs-csv MAE={mae:.4f}")
        y_true = take(y_true, layout, idx)
        pred   = take(pred,   layout, idx)
        if sigma is not None:
            sigma = take(sigma, layout, idx)

    out = df.copy()
    out[f"y_true_{args.task}_log10"] = y_true
    out[f"y_pred_{args.task}_log10"] = pred
    if sigma is not None:
        out[f"sigma_{args.task}_log10"] = sigma
    out.to_csv(args.out, index=False)
    print(f"[OK] wrote {args.out}  n={len(out)}")


if __name__ == "__main__":
    main()
