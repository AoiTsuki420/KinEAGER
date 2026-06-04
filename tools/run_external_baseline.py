import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


TASK_NAMES = ["kcat", "Km", "kcat/Km"]


def _pearson_r_safe(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.size < 2:
        return float("nan")
    yt = y_true - y_true.mean()
    yp = y_pred - y_pred.mean()
    denom = np.sqrt((yt * yt).sum() * (yp * yp).sum())
    if denom <= 0:
        return float("nan")
    return float((yt * yp).sum() / denom)


def _spearman_r_safe(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.size < 2:
        return float("nan")
    yr = pd.Series(y_true).rank(method="average").to_numpy()
    pr = pd.Series(y_pred).rank(method="average").to_numpy()
    return _pearson_r_safe(yr, pr)


def _compute_task_metrics_with_sigma(y_true, y_pred, sigma, task_name):
    valid = np.isfinite(y_true) & np.isfinite(y_pred)
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


def evaluate_and_save(df, out_dir, model_name, source_col=None):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    y_true = np.full((len(df), 3), np.nan, dtype=np.float32)
    y_pred = np.full((len(df), 3), np.nan, dtype=np.float32)
    sigma = np.full((len(df), 3), np.nan, dtype=np.float32)

    y_true[:, 0] = pd.to_numeric(df["y_true_kcat_log10"], errors="coerce").to_numpy(dtype=np.float32)
    y_pred[:, 0] = pd.to_numeric(df["y_pred_kcat_log10"], errors="coerce").to_numpy(dtype=np.float32)
    source = (
        df[source_col].astype(str).to_numpy()
        if source_col and source_col in df.columns
        else np.asarray(["unknown"] * len(df))
    )

    metrics = {}
    for i, t in enumerate(TASK_NAMES):
        metrics.update(_compute_task_metrics_with_sigma(y_true[:, i], y_pred[:, i], sigma[:, i], t))
    _add_macro_metrics(metrics, TASK_NAMES)

    by_source = {}
    for s in sorted(set(source.tolist())):
        m = source == s
        src_metrics = {}
        for i, t in enumerate(TASK_NAMES):
            src_metrics.update(
                _compute_task_metrics_with_sigma(y_true[m, i], y_pred[m, i], sigma[m, i], t)
            )
        _add_macro_metrics(src_metrics, TASK_NAMES)
        by_source[s] = src_metrics
    metrics["by_source"] = by_source

    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    np.savez_compressed(
        out_dir / "test_predictions.npz",
        y_true=y_true,
        y_pred=y_pred,
        sigma=sigma,
        source_id=source,
    )

    manifest = {
        "runner": "tools/run_external_baseline.py",
        "model": model_name,
        "n_samples": int(len(df)),
        "columns": list(df.columns),
    }
    with open(out_dir / "experiment_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    split_meta = {
        "split_mode": "external_baseline_eval",
        "counts": {"test": int(len(df))},
    }
    with open(out_dir / "split_meta.json", "w", encoding="utf-8") as f:
        json.dump(split_meta, f, indent=2)

    print(json.dumps(metrics, indent=2))
    print(f"[OK] Wrote metrics to {out_dir / 'metrics.json'}")


def _ensure_cols(df, cols, name):
    miss = [c for c in cols if c not in df.columns]
    if miss:
        raise ValueError(f"{name} missing required columns: {miss}")


def run_dlkcat(args):
    repo = Path(args.repo).resolve()
    work = repo / "DeeplearningApproach" / "Code" / "example"
    if not work.exists():
        raise FileNotFoundError(f"DLKcat example directory not found: {work}")

    df = pd.read_csv(args.input_csv)
    _ensure_cols(df, [args.seq_col, args.smiles_col, args.kcat_col], "input_csv")

    substrate_name = (
        df[args.substrate_name_col].astype(str)
        if args.substrate_name_col and args.substrate_name_col in df.columns
        else df[args.smiles_col].astype(str)
    )

    in_tsv = Path(args.out_dir) / "dlkcat_input.tsv"
    out_eval_csv = Path(args.out_dir) / "dlkcat_eval_input.csv"
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    pred_in = pd.DataFrame(
        {
            "Substrate Name": substrate_name,
            "Substrate SMILES": df[args.smiles_col].astype(str),
            "Protein Sequence": df[args.seq_col].astype(str),
        }
    )
    pred_in.to_csv(in_tsv, sep="\t", index=False)

    cmd = ["python", "prediction_for_input.py", str(in_tsv)]
    subprocess.run(cmd, check=True, cwd=str(work))

    produced = work / "output.tsv"
    if not produced.exists():
        raise FileNotFoundError("DLKcat did not produce output.tsv")
    shutil.copy2(produced, Path(args.out_dir) / "dlkcat_raw_output.tsv")

    pred_out = pd.read_csv(produced, sep="\t")
    pred_vals = pd.to_numeric(pred_out.get("Kcat value (1/s)"), errors="coerce")
    y_pred = np.log10(np.clip(pred_vals.to_numpy(dtype=float), 1e-12, None))

    eval_df = pd.DataFrame(
        {
            "y_true_kcat_log10": np.log10(
                np.clip(pd.to_numeric(df[args.kcat_col], errors="coerce").to_numpy(dtype=float), 1e-12, None)
            ),
            "y_pred_kcat_log10": y_pred,
        }
    )
    if args.source_col and args.source_col in df.columns:
        eval_df[args.source_col] = df[args.source_col].astype(str)
    eval_df.to_csv(out_eval_csv, index=False)
    evaluate_and_save(eval_df, args.out_dir, model_name="DLKcat", source_col=args.source_col)


def run_dltkcat(args):
    repo = Path(args.repo).resolve()
    code_dir = repo / "code"
    if not code_dir.exists():
        raise FileNotFoundError(f"DLTKcat code directory not found: {code_dir}")

    df = pd.read_csv(args.input_csv)
    _ensure_cols(df, [args.seq_col, args.smiles_col, args.kcat_col], "input_csv")

    temp_k = None
    if args.temp_k_col and args.temp_k_col in df.columns:
        temp_k = pd.to_numeric(df[args.temp_k_col], errors="coerce").to_numpy(dtype=float)
    elif args.temp_c_col and args.temp_c_col in df.columns:
        temp_k = pd.to_numeric(df[args.temp_c_col], errors="coerce").to_numpy(dtype=float) + 273.15
    else:
        temp_k = np.full((len(df),), float(args.default_temp_k), dtype=float)
    inv_temp = 1.0 / np.clip(temp_k, 1e-6, None)

    pred_input = pd.DataFrame(
        {
            "smiles": df[args.smiles_col].astype(str),
            "seq": df[args.seq_col].astype(str),
            "Temp_K_norm": temp_k,
            "Inv_Temp_norm": inv_temp,
            "kcat": pd.to_numeric(df[args.kcat_col], errors="coerce"),
        }
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    in_csv = out_dir / "dltkcat_input.csv"
    pred_input.to_csv(in_csv, index=False)

    output_prefix = out_dir / "dltkcat_pred"
    model_path = Path(args.model_path) if args.model_path else repo / "data" / "performances" / "model_latentdim=40_outlayer=4_rmsetest=0.8854_rmsedev=0.908.pth"
    param_path = Path(args.param_dict_pkl) if args.param_dict_pkl else repo / "data" / "hyparams" / "param_2.pkl"

    cmd = [
        "python",
        "predict.py",
        "--model_path",
        str(model_path),
        "--param_dict_pkl",
        str(param_path),
        "--input",
        str(in_csv),
        "--output",
        str(output_prefix),
        "--has_label",
        "True",
    ]
    subprocess.run(cmd, check=True, cwd=str(code_dir))

    pred_csv = Path(str(output_prefix) + ".csv")
    out_pred = pd.read_csv(pred_csv)
    if "pred_log10kcat" not in out_pred.columns:
        raise ValueError("DLTKcat output missing pred_log10kcat")

    eval_df = pd.DataFrame(
        {
            "y_true_kcat_log10": np.log10(
                np.clip(pd.to_numeric(df[args.kcat_col], errors="coerce").to_numpy(dtype=float), 1e-12, None)
            ),
            "y_pred_kcat_log10": pd.to_numeric(out_pred["pred_log10kcat"], errors="coerce").to_numpy(dtype=float),
        }
    )
    if args.source_col and args.source_col in df.columns:
        eval_df[args.source_col] = df[args.source_col].astype(str)
    eval_df.to_csv(out_dir / "dltkcat_eval_input.csv", index=False)
    evaluate_and_save(eval_df, out_dir, model_name="DLTKcat", source_col=args.source_col)


def run_csv_eval(args):
    df = pd.read_csv(args.pred_csv)
    _ensure_cols(df, [args.true_col, args.pred_col], "pred_csv")
    out = pd.DataFrame(
        {
            "y_true_kcat_log10": pd.to_numeric(df[args.true_col], errors="coerce"),
            "y_pred_kcat_log10": pd.to_numeric(df[args.pred_col], errors="coerce"),
        }
    )
    if args.source_col and args.source_col in df.columns:
        out[args.source_col] = df[args.source_col].astype(str)
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    out.to_csv(Path(args.out_dir) / "external_eval_input.csv", index=False)
    evaluate_and_save(out, args.out_dir, model_name=args.model, source_col=args.source_col)


def build_parser():
    p = argparse.ArgumentParser(description="Unified runner/evaluator for external baseline models")
    p.add_argument("--model", required=True, choices=["dlkcat", "dltkcat", "pmak", "gelkcat", "csv"])
    p.add_argument("--out_dir", required=True)
    p.add_argument("--source_col", default="source_id")

    p.add_argument("--input_csv")
    p.add_argument("--seq_col", default="Sequence")
    p.add_argument("--smiles_col", default="Smiles")
    p.add_argument("--kcat_col", default="kcat(s^-1)")
    p.add_argument("--substrate_name_col", default=None)

    p.add_argument("--repo", default=None)
    p.add_argument("--model_path", default=None)
    p.add_argument("--param_dict_pkl", default=None)
    p.add_argument("--temp_k_col", default=None)
    p.add_argument("--temp_c_col", default=None)
    p.add_argument("--default_temp_k", type=float, default=298.15)

    p.add_argument("--pred_csv", default=None)
    p.add_argument("--true_col", default="y_true_kcat_log10")
    p.add_argument("--pred_col", default="y_pred_kcat_log10")
    return p


def main():
    args = build_parser().parse_args()
    if args.model == "dlkcat":
        if not args.input_csv:
            raise ValueError("--input_csv is required for dlkcat")
        if args.repo is None:
            args.repo = "external_baselines/DLKcat"
        run_dlkcat(args)
        return
    if args.model == "dltkcat":
        if not args.input_csv:
            raise ValueError("--input_csv is required for dltkcat")
        if args.repo is None:
            args.repo = "external_baselines/DLTKcat"
        run_dltkcat(args)
        return

    if args.model in {"pmak", "gelkcat", "csv"}:
        if not args.pred_csv:
            raise ValueError(f"--pred_csv is required for model={args.model}")
        run_csv_eval(args)
        return


if __name__ == "__main__":
    main()
