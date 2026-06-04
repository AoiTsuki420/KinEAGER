"""从 reactant_smiles 生成 3D ligand SDF, 并把 lig_sdf_path 写回 CSV。

为 splits_revision 的 csv 准备 KcatExpert / KcatMoE 推理时需要的 ligand SDF。
RDKit ETKDG embed → MMFF94 优化 → 写 SDF 文件。

用法:
  python tools/extract_lig_sdf_from_smiles.py \
      --csvs /autodl-fs/data/runs/splits/splits_revision/iteration_*/kcat_*.csv \
      --sdf_out_dir /root/autodl-tmp/lig_sdfs \
      --smiles_col reactant_smiles \
      --inplace

去重: 同一 canonical SMILES 只生成一次 SDF, 文件名 = sha1(canon_smiles)[:16].sdf
失败处理: embed 失败的 SMILES, lig_sdf_path 写 NaN (推理时该样本 has_struct=False)
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

RDLogger.DisableLog("rdApp.warning")
RDLogger.DisableLog("rdApp.error")


def canonicalize(smi: str) -> str | None:
    if not smi or pd.isna(smi):
        return None
    mol = Chem.MolFromSmiles(str(smi))
    if mol is None:
        return None
    try:
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return None


def smiles_to_sdf(smi: str, out_path: Path, max_attempts: int = 10) -> bool:
    """ETKDG embed + MMFF94 optimize, 写 SDF。返回 True 表示成功。"""
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return False
        mol = Chem.AddHs(mol)
        params = AllChem.ETKDGv3()
        params.randomSeed = 42
        params.maxAttempts = max_attempts
        if AllChem.EmbedMolecule(mol, params) != 0:
            if AllChem.EmbedMolecule(mol, randomSeed=42, maxAttempts=max_attempts) != 0:
                return False
        try:
            AllChem.MMFFOptimizeMolecule(mol, maxIters=200)
        except Exception:
            pass  # 非致命, 至少有 embed 后的坐标
        out_path.parent.mkdir(parents=True, exist_ok=True)
        writer = Chem.SDWriter(str(out_path))
        writer.write(mol)
        writer.close()
        return True
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csvs", nargs="+", required=True, help="一个或多个 CSV 文件 (支持 glob 后展开)")
    ap.add_argument("--sdf_out_dir", required=True, help="输出 SDF 的目录")
    ap.add_argument("--smiles_col", default="reactant_smiles",
                    help="SMILES 列名 (splits_revision 用 reactant_smiles)")
    ap.add_argument("--lig_path_col", default="lig_sdf_path",
                    help="写回 CSV 的列名")
    ap.add_argument("--inplace", action="store_true",
                    help="覆盖原 CSV; 否则另存为 <名>.with_lig.csv")
    ap.add_argument("--max_atoms", type=int, default=200,
                    help="超过这个原子数的分子直接跳过 (避免 embed 卡住)")
    args = ap.parse_args()

    out_dir = Path(args.sdf_out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[scan] {len(args.csvs)} csvs ...")
    all_canon: dict[str, set[str]] = {}  # canon -> set of original SMILES
    for csv_path in args.csvs:
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"[WARN] cannot read {csv_path}: {e}")
            continue
        if args.smiles_col not in df.columns:
            print(f"[WARN] {csv_path}: missing column {args.smiles_col}")
            continue
        for s in df[args.smiles_col].astype(str).unique():
            c = canonicalize(s)
            if c is None:
                continue
            all_canon.setdefault(c, set()).add(s)
    print(f"[scan] unique canonical SMILES: {len(all_canon)}")

    canon_to_path: dict[str, str] = {}
    n_new, n_skip, n_fail = 0, 0, 0
    for i, canon in enumerate(sorted(all_canon.keys())):
        digest = hashlib.sha1(canon.encode("utf-8")).hexdigest()[:16]
        sdf_path = out_dir / f"{digest}.sdf"
        if sdf_path.exists() and sdf_path.stat().st_size > 100:
            canon_to_path[canon] = str(sdf_path)
            n_skip += 1
            continue
        m = Chem.MolFromSmiles(canon)
        if m is None or m.GetNumHeavyAtoms() > args.max_atoms:
            n_fail += 1
            continue
        ok = smiles_to_sdf(canon, sdf_path)
        if ok:
            canon_to_path[canon] = str(sdf_path)
            n_new += 1
        else:
            n_fail += 1
        if (i + 1) % 200 == 0:
            print(f"  [{i+1}/{len(all_canon)}] new={n_new} skip={n_skip} fail={n_fail}")

    print(f"[generate] new={n_new}  skip(reused)={n_skip}  fail={n_fail}  "
          f"total_resolved={len(canon_to_path)}/{len(all_canon)}")

    print(f"[fill] writing {args.lig_path_col} into {len(args.csvs)} csvs ...")
    for csv_path in args.csvs:
        try:
            df = pd.read_csv(csv_path)
        except Exception:
            continue
        if args.smiles_col not in df.columns:
            continue
        canons = df[args.smiles_col].astype(str).map(canonicalize)
        paths = canons.map(lambda c: canon_to_path.get(c) if isinstance(c, str) else None)
        df[args.lig_path_col] = paths
        n_filled = int(paths.notna().sum())

        out_csv = csv_path if args.inplace else str(Path(csv_path).with_suffix(".with_lig.csv"))
        df.to_csv(out_csv, index=False)
        print(f"  {csv_path} -> {out_csv}  ({n_filled}/{len(df)} filled)")


if __name__ == "__main__":
    main()
