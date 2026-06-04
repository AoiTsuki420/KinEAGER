"""
将 splits_revision 某个 iteration 的 kcat_train + km_train
转换为统一训练格式（Sequence / Smiles / kcat(s^-1) / Km(M)）。

用法：
  python tools/build_catpred_iter_csv.py \
      --iter_dir /path/to/splits_revision/iteration_1 \
      --out_csv  /path/to/output/iter1_train.csv

  for i in $(seq 1 10); do
      python tools/build_catpred_iter_csv.py \
          --iter_dir /path/to/splits_revision/iteration_${i} \
          --out_csv  /path/to/output/iter${i}_train.csv
  done
"""
import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd


def _first_smiles_component(smi: str) -> str:
    """从多组分 SMILES 中取第一个顶层组分（处理 kcat reactant_smiles）。"""
    depth = 0
    for i, c in enumerate(smi):
        if c == '(':
            depth += 1
        elif c == ')':
            depth -= 1
        elif c == '.' and depth == 0:
            return smi[:i]
    return smi


def load_kcat_train(iter_dir: Path) -> pd.DataFrame:
    """加载 kcat_train_split_*.csv，转为统一格式。"""
    pattern = str(iter_dir / "kcat_train_split_*.csv")
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"找不到 kcat train split: {pattern}")
    df = pd.read_csv(files[0])
    df.columns = [c.lower() for c in df.columns]

    smiles = df["reactant_smiles"].astype(str).apply(_first_smiles_component)
    kcat_val = pd.to_numeric(df["value"], errors="coerce")

    out = pd.DataFrame({
        "Sequence":    df["sequence"].astype(str),
        "Smiles":      smiles,
        "kcat(s^-1)":  kcat_val,          # 线性值，单位 s^-1
        "Km(M)":       np.nan,
        "source_id":   "catpred-kcat",
    })
    print(f"  [kcat] {files[0]}: {len(out)} 行，有效 kcat={int(kcat_val.notna().sum())}")
    return out


def load_km_train(iter_dir: Path) -> pd.DataFrame:
    """加载 km_train_split_*.csv，转为统一格式。"""
    pattern = str(iter_dir / "km_train_split_*.csv")
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"找不到 km train split: {pattern}")
    df = pd.read_csv(files[0])
    df.columns = [c.lower() for c in df.columns]

    smiles = df["substrate_smiles"].astype(str)
    km_val = pd.to_numeric(df["value"], errors="coerce")   # 单位 M

    out = pd.DataFrame({
        "Sequence":    df["sequence"].astype(str),
        "Smiles":      smiles,
        "kcat(s^-1)":  np.nan,
        "Km(M)":       km_val,             # 线性值，单位 M
        "source_id":   "catpred-km",
    })
    print(f"  [km]   {files[0]}: {len(out)} 行，有效 Km={int(km_val.notna().sum())}")
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--iter_dir", required=True,
                   help="splits_revision/iteration_N 目录路径")
    p.add_argument("--out_csv", required=True,
                   help="输出的统一训练 CSV 路径")
    p.add_argument("--drop_invalid", action="store_true",
                   help="过滤掉序列或 SMILES 为空的行")
    args = p.parse_args()

    iter_dir = Path(args.iter_dir)
    print(f"[INFO] 处理 {iter_dir.name}")

    kcat_df = load_kcat_train(iter_dir)
    km_df   = load_km_train(iter_dir)

    merged = pd.concat([kcat_df, km_df], axis=0, ignore_index=True)

    if args.drop_invalid:
        before = len(merged)
        merged = merged[
            merged["Sequence"].str.strip().str.len() > 0 &
            merged["Smiles"].str.strip().str.len() > 0
        ]
        print(f"  [filter] {before} → {len(merged)} 行（过滤空序列/SMILES）")

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_path, index=False)

    print(f"[done] 输出: {out_path}")
    print(f"[done] 总行数: {len(merged)}")
    print(f"[done] source 分布:\n{merged['source_id'].value_counts().to_string()}")
    kcat_valid = np.isfinite(merged["kcat(s^-1)"].to_numpy(dtype=float))
    km_valid   = np.isfinite(merged["Km(M)"].to_numpy(dtype=float))
    print(f"[done] 有效标签: kcat={kcat_valid.sum()}, Km={km_valid.sum()}")


if __name__ == "__main__":
    main()
