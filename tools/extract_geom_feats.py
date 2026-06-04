"""为 skid-kcat 每条样本抽 16 维几何/理化特征。

用法:
  python tools/extract_geom_feats.py \
      --csv data/skid_kcat.csv \
      --out_dir data/skid_kcat_geom \
      --update_csv data/skid_kcat_with_geom.csv

特征（共 16 维）:
  [0..5]  ligand 理化: MW, logP, TPSA, HBD, HBA, RotB
  [6..8]  ligand 几何: 原子数, 半径(max pairwise dist)/2, 质心到口袋质心距离
  [9..12] 口袋: 口袋原子数, 口袋体积近似(凸包 / 4Å 内残基数), SASA 接触面积近似, 疏水残基比例
  [13..15] 交互: 静电 proxy (带电残基数), H-bond donor/acceptor 残基对数, 口袋芳香残基数

缺失数据位填 0。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors, Lipinski, rdMolDescriptors
except Exception:
    Chem = None


HYDROPHOBIC = set("AVILMFWYC")
CHARGED = set("DEKRH")
AROMATIC = set("FWYH")
HBOND_DONOR = set("STNQKRHWY")
HBOND_ACCEPTOR = set("DENQSTYH")

AA3_TO_1 = {"ALA":"A","ARG":"R","ASN":"N","ASP":"D","CYS":"C","GLN":"Q","GLU":"E",
            "GLY":"G","HIS":"H","ILE":"I","LEU":"L","LYS":"K","MET":"M","PHE":"F",
            "PRO":"P","SER":"S","THR":"T","TRP":"W","TYR":"Y","VAL":"V"}


def _ligand_phys(mol) -> np.ndarray:
    if mol is None:
        return np.zeros(6, dtype=np.float32)
    return np.asarray([
        Descriptors.MolWt(mol),
        Descriptors.MolLogP(mol),
        Descriptors.TPSA(mol),
        Lipinski.NumHDonors(mol),
        Lipinski.NumHAcceptors(mol),
        Lipinski.NumRotatableBonds(mol),
    ], dtype=np.float32)


def _ligand_geom(mol) -> tuple[np.ndarray, np.ndarray]:
    """返回 ligand 坐标 和 (n_atoms, radius) 的 2 维数组（质心到口袋距离在后面算）。"""
    if mol is None or mol.GetNumConformers() == 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros(2, dtype=np.float32)
    conf = mol.GetConformer()
    coords = np.asarray([[conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y, conf.GetAtomPosition(i).z]
                         for i in range(mol.GetNumAtoms())], dtype=np.float32)
    if coords.shape[0] < 2:
        return coords, np.asarray([coords.shape[0], 0.0], dtype=np.float32)
    diff = coords[:, None, :] - coords[None, :, :]
    d = np.sqrt((diff ** 2).sum(-1))
    return coords, np.asarray([coords.shape[0], float(d.max() / 2.0)], dtype=np.float32)


def _parse_protein(pdb_path: str):
    coords, aas, heavy_coords = [], [], []
    with open(pdb_path, "r") as f:
        for line in f:
            if not line.startswith("ATOM"):
                continue
            try:
                x = float(line[30:38]); y = float(line[38:46]); z = float(line[46:54])
            except ValueError:
                continue
            atom = line[12:16].strip()
            res = line[17:20].strip()
            aa = AA3_TO_1.get(res, "X")
            heavy_coords.append(([x, y, z], aa))
            if atom == "CA":
                coords.append([x, y, z]); aas.append(aa)
    return (np.asarray(coords, dtype=np.float32) if coords else np.zeros((0, 3), np.float32),
            aas,
            heavy_coords)


def _pocket_features(lig_coords: np.ndarray, ca_coords: np.ndarray, aas: list, heavy: list,
                     cutoff: float = 6.0) -> np.ndarray:
    """返回 [pocket_n, pocket_vol_proxy, sasa_proxy, hydrophobic_frac,
             charged_n, hbond_pairs, aromatic_n, centroid_dist]（8 维 + centroid_dist）。"""
    out = np.zeros(8, dtype=np.float32)
    if lig_coords.shape[0] == 0 or ca_coords.shape[0] == 0:
        return out
    lig_centroid = lig_coords.mean(axis=0)
    lig_set = lig_coords
    pocket_idx = set()
    d_ca = np.linalg.norm(ca_coords - lig_centroid, axis=1)
    pocket_mask = d_ca < 10.0
    pocket_aas = [aa for aa, m in zip(aas, pocket_mask) if m]
    if not pocket_aas:
        return out
    n_pocket = len(pocket_aas)
    out[0] = n_pocket
    pc = ca_coords[pocket_mask]
    bbox = pc.max(0) - pc.min(0)
    out[1] = float(bbox.prod())
    out[2] = float(n_pocket)  # 占位
    out[3] = sum(1 for a in pocket_aas if a in HYDROPHOBIC) / n_pocket
    out[4] = sum(1 for a in pocket_aas if a in CHARGED)
    out[5] = sum(1 for a in pocket_aas if a in HBOND_DONOR) + sum(1 for a in pocket_aas if a in HBOND_ACCEPTOR)
    out[6] = sum(1 for a in pocket_aas if a in AROMATIC)
    out[7] = float(np.linalg.norm(lig_centroid - pc.mean(0)))
    return out


def extract(pdb_path: str | None, sdf_path: str | None) -> np.ndarray:
    feats = np.zeros(16, dtype=np.float32)
    mol = None
    if sdf_path and Chem is not None:
        try:
            suppl = Chem.SDMolSupplier(sdf_path, removeHs=True)
            for m in suppl:
                if m is not None:
                    mol = m; break
        except Exception:
            mol = None
    feats[0:6] = _ligand_phys(mol)
    lig_coords, lig_geom = _ligand_geom(mol)
    feats[6:8] = lig_geom if lig_geom.shape[0] == 2 else 0.0

    if pdb_path:
        try:
            ca, aas, heavy = _parse_protein(pdb_path)
            pocket = _pocket_features(lig_coords, ca, aas, heavy)
            feats[8] = pocket[7]        # centroid_dist
            feats[9] = pocket[0]        # pocket_n
            feats[10] = pocket[1]       # vol
            feats[11] = pocket[2]       # sasa proxy
            feats[12] = pocket[3]       # hydrophobic frac
            feats[13] = pocket[4]       # charged
            feats[14] = pocket[5]       # hbond pairs
            feats[15] = pocket[6]       # aromatic
        except Exception:
            pass
    return feats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--update_csv", default=None, help="写回带 geom_feat_path 列的新 csv")
    ap.add_argument("--pdb_col", default="prot_pdb_path")
    ap.add_argument("--sdf_col", default="lig_sdf_path")
    ap.add_argument("--id_col", default=None, help="用作输出文件名；默认用行号")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for i, row in df.iterrows():
        name = str(row[args.id_col]) if args.id_col else f"row_{i:06d}"
        fn = out_dir / f"{name}.npy"
        pdb = row.get(args.pdb_col); sdf = row.get(args.sdf_col)
        pdb = None if pd.isna(pdb) else str(pdb)
        sdf = None if pd.isna(sdf) else str(sdf)
        f = extract(pdb, sdf)
        np.save(fn, f)
        paths.append(str(fn))
        if (i + 1) % 500 == 0:
            print(f"[{i+1}/{len(df)}]")

    if args.update_csv:
        df["geom_feat_path"] = paths
        df.to_csv(args.update_csv, index=False)
        print(f"wrote {args.update_csv}")


if __name__ == "__main__":
    main()
