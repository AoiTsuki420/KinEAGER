"""仅从 SMILES 抽 ligand 理化特征（无需 PDB/SDF）。

用于 catpred train 这类无结构样本：填满 16 维 geom_feats 的前 8 位
（6 理化 + 2 几何 proxy），后 8 位（口袋相关）保留 0。

用法:
  python tools/extract_geom_feats_smiles.py \
      --csv   /autodl-fs/data/catpred_kcat_train.csv \
      --out_dir /autodl-fs/data/catpred_geom \
      --update_csv /autodl-fs/data/catpred_kcat_train_with_geom.csv \
      --smiles_col smiles \
      --id_col entry_id
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Lipinski


def smiles_to_geom16(smi: str) -> np.ndarray:
    """返回 16 维向量（后 8 位全 0）。"""
    feats = np.zeros(16, dtype=np.float32)
    if not smi or pd.isna(smi):
        return feats
    mol = Chem.MolFromSmiles(str(smi))
    if mol is None:
        return feats
    try:
        feats[0] = Descriptors.MolWt(mol)
        feats[1] = Descriptors.MolLogP(mol)
        feats[2] = Descriptors.TPSA(mol)
        feats[3] = Lipinski.NumHDonors(mol)
        feats[4] = Lipinski.NumHAcceptors(mol)
        feats[5] = Lipinski.NumRotatableBonds(mol)
    except Exception:
        pass
    feats[6] = float(mol.GetNumHeavyAtoms())
    try:
        m2 = Chem.AddHs(mol)
        if AllChem.EmbedMolecule(m2, randomSeed=42, maxAttempts=10) == 0:
            conf = m2.GetConformer()
            coords = np.asarray([[conf.GetAtomPosition(i).x,
                                  conf.GetAtomPosition(i).y,
                                  conf.GetAtomPosition(i).z]
                                 for i in range(m2.GetNumAtoms())], dtype=np.float32)
            if coords.shape[0] >= 2:
                diff = coords[:, None, :] - coords[None, :, :]
                d = np.sqrt((diff ** 2).sum(-1))
                feats[7] = float(d.max() / 2.0)
    except Exception:
        pass
    return np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--update_csv", default=None)
    ap.add_argument("--smiles_col", default="smiles")
    ap.add_argument("--id_col", default=None, help="用作输出文件名；默认用行号")
    ap.add_argument("--skip_if_has_geom", action="store_true", default=True,
                    help="若已有 geom_feat_path 且文件存在则跳过；默认开")
    ap.add_argument("--geom_col", default="geom_feat_path")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    paths, n_new, n_skip, n_fail = [], 0, 0, 0
    for i, row in df.iterrows():
        name = str(row[args.id_col]) if args.id_col and args.id_col in df.columns else f"row_{i:07d}"
        fn = out_dir / f"{name}.npy"

        existing = row.get(args.geom_col) if args.geom_col in df.columns else None
        if args.skip_if_has_geom and existing and not pd.isna(existing):
            p = str(existing)
            if Path(p).exists():
                paths.append(p); n_skip += 1
                continue

        smi = row.get(args.smiles_col)
        feats = smiles_to_geom16(smi)
        if float(np.abs(feats).sum()) == 0.0:
            n_fail += 1
        np.save(fn, feats)
        paths.append(str(fn))
        n_new += 1

        if (i + 1) % 1000 == 0:
            print(f"[{i+1}/{len(df)}] new={n_new} skip={n_skip} fail={n_fail}")

    print(f"done  new={n_new} skip={n_skip} fail={n_fail} total={len(df)}")

    if args.update_csv:
        df[args.geom_col] = paths
        df.to_csv(args.update_csv, index=False)
        print(f"wrote {args.update_csv}")


if __name__ == "__main__":
    main()
