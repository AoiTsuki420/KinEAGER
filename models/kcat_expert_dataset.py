"""skid-kcat 专家模型 Dataset。

CSV 列约定（可通过 col_map 覆盖）:
  protein_id, sequence, smiles, kcat_log10,
  prot_pdb_path, lig_sdf_path, geom_feat_path (可选，缺失则实时算)

结构文件：
  prot_pdb_path:  标准 PDB / mmCIF
  lig_sdf_path :  带 3D 坐标的 SDF（skid-kcat 自带）
  geom_feat_path: npy 文件，16 维 float

Batch collate 输出 dict，字段见 models/kcat_expert.py 顶部注释。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.warning")


ATOM_ELEMENTS = ["C", "N", "O", "S", "P", "F", "Cl", "Br", "I", "H", "B", "Si", "Se", "*"]
HYBRID = ["SP", "SP2", "SP3", "SP3D", "SP3D2", "UNSPECIFIED"]
LIG_ATOM_FEAT_DIM = len(ATOM_ELEMENTS) + len(HYBRID) + 6 + 8  # 元素+杂化+[charge,arom,inring,degree,Hs,chirality]+padding=44


def _onehot(val, vocab):
    v = [0.0] * len(vocab)
    try:
        i = vocab.index(val)
        v[i] = 1.0
    except ValueError:
        v[-1] = 1.0
    return v



def parse_pdb_ca(pdb_path: str, max_len: int = 1024) -> tuple[np.ndarray, np.ndarray, str]:
    """返回 (dist[L,L], mask[L], seq)。缺失残基用 0/False 填充。"""
    AA3_TO_1 = {"ALA": "A","ARG":"R","ASN":"N","ASP":"D","CYS":"C","GLN":"Q","GLU":"E",
                "GLY":"G","HIS":"H","ILE":"I","LEU":"L","LYS":"K","MET":"M","PHE":"F",
                "PRO":"P","SER":"S","THR":"T","TRP":"W","TYR":"Y","VAL":"V"}
    coords, seq = [], []
    with open(pdb_path, "r") as f:
        for line in f:
            if not line.startswith("ATOM"):
                continue
            if line[12:16].strip() != "CA":
                continue
            res = line[17:20].strip()
            aa = AA3_TO_1.get(res, "X")
            try:
                x = float(line[30:38]); y = float(line[38:46]); z = float(line[46:54])
            except ValueError:
                continue
            coords.append([x, y, z]); seq.append(aa)
            if len(coords) >= max_len:
                break
    if not coords:
        return np.zeros((1, 1), dtype=np.float32), np.zeros((1,), dtype=bool), ""
    coords = np.asarray(coords, dtype=np.float32)
    L = coords.shape[0]
    diff = coords[:, None, :] - coords[None, :, :]
    dist = np.sqrt((diff ** 2).sum(-1)).astype(np.float32)
    mask = np.ones((L,), dtype=bool)
    return dist, mask, "".join(seq)



def parse_ligand_sdf(sdf_path: str, max_atoms: int = 128) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """返回 (atom_feat[N, F], atom_dist[N, N], atom_mask[N])。"""
    from rdkit import Chem
    suppl = Chem.SDMolSupplier(sdf_path, removeHs=True)
    mol = None
    for m in suppl:
        if m is not None:
            mol = m; break
    if mol is None or mol.GetNumConformers() == 0:
        return (np.zeros((1, LIG_ATOM_FEAT_DIM), np.float32),
                np.zeros((1, 1), np.float32),
                np.zeros((1,), dtype=bool))
    conf = mol.GetConformer()
    feats, coords = [], []
    for a in mol.GetAtoms():
        if len(feats) >= max_atoms:
            break
        row = []
        row += _onehot(a.GetSymbol(), ATOM_ELEMENTS)
        row += _onehot(str(a.GetHybridization()), HYBRID)
        row += [float(a.GetFormalCharge()),
                1.0 if a.GetIsAromatic() else 0.0,
                1.0 if a.IsInRing() else 0.0,
                float(a.GetDegree()),
                float(a.GetTotalNumHs()),
                1.0 if a.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED else 0.0]
        row += [0.0] * (LIG_ATOM_FEAT_DIM - len(row))
        feats.append(row[:LIG_ATOM_FEAT_DIM])
        p = conf.GetAtomPosition(a.GetIdx())
        coords.append([p.x, p.y, p.z])
    feats = np.asarray(feats, dtype=np.float32)
    coords = np.asarray(coords, dtype=np.float32)
    N = coords.shape[0]
    diff = coords[:, None, :] - coords[None, :, :]
    dist = np.sqrt((diff ** 2).sum(-1)).astype(np.float32)
    mask = np.ones((N,), dtype=bool)
    return feats, dist, mask



class KcatExpertDataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        *,
        max_prot_len: int = 1024,
        max_lig_atoms: int = 128,
        col_map: Optional[dict] = None,
        require_struct: bool = True,
        base_dir: Optional[str] = None,
        rename_cols: Optional[dict] = None,
        randomize_smiles: bool = False,
        seq_crop_min: float = 1.0,
        seq_mask_prob: float = 0.0,
    ):
        self.randomize_smiles = bool(randomize_smiles)
        self.seq_crop_min = float(seq_crop_min)  # 1.0 = 不裁剪；0.8 = 随机保留 80-100%
        self.seq_mask_prob = float(seq_mask_prob)  # 残基随机替换为 X 的概率
        self.df = pd.read_csv(csv_path)
        if rename_cols:
            self.df.rename(columns={k: v for k, v in rename_cols.items() if k in self.df.columns},
                           inplace=True)
        self.col = {
            "seq": "sequence", "smiles": "smiles", "y": "kcat_log10",
            "prot_pdb": "prot_pdb_path", "lig_sdf": "lig_sdf_path",
            "geom": "geom_feat_path",
        }
        if col_map:
            self.col.update(col_map)
        self.max_prot_len = max_prot_len
        self.max_lig_atoms = max_lig_atoms
        self.require_struct = require_struct
        self.base_dir = Path(base_dir) if base_dir else None

        if require_struct:
            before = len(self.df)
            mask = self.df[self.col["prot_pdb"]].notna() & self.df[self.col["lig_sdf"]].notna()
            self.df = self.df[mask].reset_index(drop=True)
            print(f"[KcatExpertDataset] kept {len(self.df)}/{before} rows with structure")

        before_clean = len(self.df)
        self.df = self.df.dropna(subset=[self.col["seq"], self.col["smiles"], self.col["y"]]).reset_index(drop=True)
        self.df[self.col["y"]] = pd.to_numeric(self.df[self.col["y"]], errors="coerce")
        self.df = self.df[np.isfinite(self.df[self.col["y"]].to_numpy())].reset_index(drop=True)
        dropped = before_clean - len(self.df)
        if dropped > 0:
            print(f"[KcatExpertDataset] dropped {dropped} rows with invalid seq/smiles/y")

    def __len__(self):
        return len(self.df)

    def _resolve(self, p):
        if p is None or (isinstance(p, float) and np.isnan(p)):
            return None
        p = str(p)
        if self.base_dir and not os.path.isabs(p):
            return str(self.base_dir / p)
        return p

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        seq = str(row[self.col["seq"]])[: self.max_prot_len]
        if self.seq_crop_min < 1.0 and len(seq) > 50:
            import random as _rnd
            L = len(seq)
            keep_frac = _rnd.uniform(self.seq_crop_min, 1.0)
            keep_len = max(50, int(L * keep_frac))
            start = _rnd.randint(0, max(0, L - keep_len))
            seq = seq[start:start + keep_len]
        if self.seq_mask_prob > 0 and len(seq) > 0:
            import random as _rnd
            seq = "".join("X" if _rnd.random() < self.seq_mask_prob else c for c in seq)
        smi = str(row[self.col["smiles"]])
        if self.randomize_smiles:
            try:
                from rdkit import Chem
                mol = Chem.MolFromSmiles(smi)
                if mol is not None:
                    smi = Chem.MolToSmiles(mol, doRandom=True, canonical=False)
            except Exception:
                pass
        y = float(row[self.col["y"]])

        pdb_path = self._resolve(row.get(self.col["prot_pdb"]))
        sdf_path = self._resolve(row.get(self.col["lig_sdf"]))
        geom_path = self._resolve(row.get(self.col["geom"])) if self.col["geom"] in row else None

        has_struct = pdb_path is not None and sdf_path is not None
        if has_struct:
            try:
                prot_dist, prot_struct_mask, _ = parse_pdb_ca(pdb_path, self.max_prot_len)
            except Exception:
                prot_dist = np.zeros((1, 1), np.float32); prot_struct_mask = np.zeros((1,), bool)
                has_struct = False
            try:
                lig_feat, lig_dist, lig_mask = parse_ligand_sdf(sdf_path, self.max_lig_atoms)
            except Exception:
                lig_feat = np.zeros((1, LIG_ATOM_FEAT_DIM), np.float32)
                lig_dist = np.zeros((1, 1), np.float32); lig_mask = np.zeros((1,), bool)
                has_struct = False
        else:
            prot_dist = np.zeros((1, 1), np.float32); prot_struct_mask = np.zeros((1,), bool)
            lig_feat = np.zeros((1, LIG_ATOM_FEAT_DIM), np.float32)
            lig_dist = np.zeros((1, 1), np.float32); lig_mask = np.zeros((1,), bool)

        if geom_path and os.path.exists(geom_path):
            geom = np.load(geom_path).astype(np.float32).reshape(-1)[:16]
            if geom.shape[0] < 16:
                geom = np.pad(geom, (0, 16 - geom.shape[0]))
            geom = np.nan_to_num(geom, nan=0.0, posinf=0.0, neginf=0.0)
        else:
            geom = np.zeros((16,), dtype=np.float32)

        return {
            "seq": seq,
            "smiles": smi,
            "prot_dist": torch.from_numpy(prot_dist),
            "prot_struct_mask": torch.from_numpy(prot_struct_mask),
            "lig_atom_feat": torch.from_numpy(lig_feat),
            "lig_atom_dist": torch.from_numpy(lig_dist),
            "lig_atom_mask": torch.from_numpy(lig_mask),
            "geom_feats": torch.from_numpy(geom),
            "has_struct": torch.tensor(has_struct, dtype=torch.bool),
            "y_kcat": torch.tensor(y, dtype=torch.float32),
        }



def _pad_stack_2d(mats, pad_val=0.0):
    L = max(m.shape[0] for m in mats)
    out = torch.full((len(mats), L, L), pad_val, dtype=mats[0].dtype)
    for i, m in enumerate(mats):
        n = m.shape[0]
        out[i, :n, :n] = m
    return out


def _pad_stack_feat(mats, pad_val=0.0):
    L = max(m.shape[0] for m in mats)
    F = mats[0].shape[1]
    out = torch.full((len(mats), L, F), pad_val, dtype=mats[0].dtype)
    for i, m in enumerate(mats):
        out[i, :m.shape[0]] = m
    return out


def _pad_mask(masks):
    L = max(m.shape[0] for m in masks)
    out = torch.zeros((len(masks), L), dtype=torch.bool)
    for i, m in enumerate(masks):
        out[i, :m.shape[0]] = m
    return out


def collate_kcat_expert(batch):
    return {
        "seqs": [b["seq"] for b in batch],
        "smiles": [b["smiles"] for b in batch],
        "prot_dist": _pad_stack_2d([b["prot_dist"] for b in batch]),
        "prot_struct_mask": _pad_mask([b["prot_struct_mask"] for b in batch]),
        "lig_atom_feat": _pad_stack_feat([b["lig_atom_feat"] for b in batch]),
        "lig_atom_dist": _pad_stack_2d([b["lig_atom_dist"] for b in batch]),
        "lig_atom_mask": _pad_mask([b["lig_atom_mask"] for b in batch]),
        "geom_feats": torch.stack([b["geom_feats"] for b in batch]),
        "has_struct": torch.stack([b["has_struct"] for b in batch]),
        "y_kcat": torch.stack([b["y_kcat"] for b in batch]),
    }
