import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors, Lipinski, rdMolDescriptors
except Exception:
    Chem = None
    AllChem = None
    Descriptors = None
    Lipinski = None
    rdMolDescriptors = None


AA3_TO_1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}

AA_ORDER = list("ARNDCQEGHILKMFPSTWYV")
AA_SET = set(AA_ORDER)

HYDROPHOBIC = set("AVILMFWY")
POLAR = set("STNQCHG")
POSITIVE = set("KRH")
NEGATIVE = set("DE")
AROMATIC = set("FWYH")

ELEM_ORDER = ["C", "N", "O", "S", "P", "H", "FE", "ZN", "MG", "CA"]

BACKBONE_ATOMS = {"N", "CA", "C", "O"}


def _safe_float(x: str, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _normalize(v: np.ndarray) -> np.ndarray:
    s = float(v.sum())
    if s <= 0:
        return np.zeros_like(v, dtype=np.float32)
    return (v / s).astype(np.float32)


def _resolve_path(raw_value, base_dir: Path | None, default_suffix: str) -> Path | None:
    if raw_value is None:
        return None
    ptxt = str(raw_value).strip()
    if not ptxt or ptxt.lower() in {"nan", "none"}:
        return None

    p = Path(ptxt)
    candidates = []
    if p.is_absolute():
        candidates.append(p)
    else:
        if base_dir is not None:
            candidates.append(base_dir / p)
        candidates.append(p)

    if p.suffix == "":
        extra = []
        for c in candidates:
            extra.append(c.with_suffix(default_suffix))
        candidates.extend(extra)

    for c in candidates:
        if c.exists():
            return c
    return candidates[0] if candidates else None


def pdb_to_prot45(pdb_path: Path) -> np.ndarray:
    aa_counts = {aa: 0 for aa in AA_ORDER}
    elem_counts = {e: 0 for e in ELEM_ORDER}

    residues_seen = set()
    seq = []
    atom_count = 0
    chain_ids = set()
    b_factors = []
    occupancies = []
    backbone_atom_count = 0

    with pdb_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line.startswith("ATOM"):
                continue
            atom_count += 1

            atom_name = line[12:16].strip().upper()
            resname = line[17:20].strip().upper()
            chain_id = line[21:22].strip() or "_"
            resseq = line[22:26].strip()
            icode = line[26:27].strip()
            element = line[76:78].strip().upper()
            if not element:
                element = "".join([c for c in atom_name if c.isalpha()])[:2].upper()

            chain_ids.add(chain_id)

            if atom_name in BACKBONE_ATOMS:
                backbone_atom_count += 1

            b_factors.append(_safe_float(line[60:66], 0.0))
            occupancies.append(_safe_float(line[54:60], 0.0))

            res_key = (chain_id, resseq, icode, resname)
            if res_key not in residues_seen:
                residues_seen.add(res_key)
                aa = AA3_TO_1.get(resname)
                if aa in AA_SET:
                    seq.append(aa)
                    aa_counts[aa] += 1

            if element in elem_counts:
                elem_counts[element] += 1

    n_res = len(seq)
    n_atoms = atom_count

    aa_vec = np.array([aa_counts[aa] for aa in AA_ORDER], dtype=np.float32)
    aa_freq = _normalize(aa_vec)

    if n_res > 0:
        hydrophobic_frac = sum(1 for a in seq if a in HYDROPHOBIC) / n_res
        polar_frac = sum(1 for a in seq if a in POLAR) / n_res
        positive_frac = sum(1 for a in seq if a in POSITIVE) / n_res
        negative_frac = sum(1 for a in seq if a in NEGATIVE) / n_res
        aromatic_frac = sum(1 for a in seq if a in AROMATIC) / n_res
        gly_frac = seq.count("G") / n_res
        pro_frac = seq.count("P") / n_res
    else:
        hydrophobic_frac = polar_frac = positive_frac = negative_frac = aromatic_frac = 0.0
        gly_frac = pro_frac = 0.0

    elem_vec = np.array([elem_counts[e] for e in ELEM_ORDER], dtype=np.float32)
    elem_freq = _normalize(elem_vec)

    mean_b = float(np.mean(b_factors)) if b_factors else 0.0
    std_b = float(np.std(b_factors)) if b_factors else 0.0
    mean_occ = float(np.mean(occupancies)) if occupancies else 0.0
    n_chains = len(chain_ids)
    frac_backbone = (backbone_atom_count / n_atoms) if n_atoms > 0 else 0.0
    frac_sidechain = 1.0 - frac_backbone if n_atoms > 0 else 0.0

    meta10 = np.array(
        [
            math.log1p(n_res) / 10.0,
            math.log1p(n_atoms) / 12.0,
            min(n_chains / 10.0, 1.0),
            mean_b / 100.0,
            std_b / 100.0,
            mean_occ,
            frac_backbone,
            frac_sidechain,
            gly_frac,
            pro_frac,
        ],
        dtype=np.float32,
    )

    group5 = np.array(
        [hydrophobic_frac, polar_frac, positive_frac, negative_frac, aromatic_frac],
        dtype=np.float32,
    )

    feat = np.concatenate([aa_freq, group5, elem_freq, meta10], axis=0).astype(np.float32)
    if feat.shape[0] != 45:
        raise RuntimeError(f"Protein feature dim mismatch: {feat.shape[0]} != 45")
    return feat


def sdf_to_lig135(sdf_path: Path) -> np.ndarray:
    if Chem is None:
        raise ImportError("rdkit is required to build ligand features from SDF")

    supplier = Chem.SDMolSupplier(str(sdf_path), removeHs=False)
    mol = None
    for m in supplier:
        if m is not None:
            mol = m
            break
    if mol is None:
        raise ValueError(f"No valid molecule found in SDF: {sdf_path}")

    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=128)
    fp_arr = np.zeros((128,), dtype=np.float32)
    for i in range(128):
        fp_arr[i] = float(fp.GetBit(i))

    desc7 = np.array(
        [
            Descriptors.MolWt(mol),
            Descriptors.MolLogP(mol),
            Descriptors.TPSA(mol),
            float(Lipinski.NumHAcceptors(mol)),
            float(Lipinski.NumHDonors(mol)),
            float(Lipinski.NumRotatableBonds(mol)),
            float(rdMolDescriptors.CalcNumRings(mol)),
        ],
        dtype=np.float32,
    )

    feat = np.concatenate([fp_arr, desc7], axis=0).astype(np.float32)
    if feat.shape[0] != 135:
        raise RuntimeError(f"Ligand feature dim mismatch: {feat.shape[0]} != 135")
    return feat


def main():
    p = argparse.ArgumentParser(description="Build final_contact npz from PDB/SDF paths")
    p.add_argument("--csv", required=True, help="Input csv used for training")
    p.add_argument("--out_npz", required=True, help="Output npz path")
    p.add_argument("--pdb_col", default="pdb_path", help="CSV column for pdb file path/name")
    p.add_argument("--sdf_col", default="sdf_path", help="CSV column for sdf file path/name")
    p.add_argument("--pdb_dir", default=None, help="Base dir for pdb files")
    p.add_argument("--sdf_dir", default=None, help="Base dir for sdf files")
    p.add_argument("--allow_missing", action="store_true", help="Fill zero vectors on missing/parse error")
    p.add_argument("--verbose_every", type=int, default=500)
    args = p.parse_args()

    df = pd.read_csv(args.csv)
    if args.pdb_col not in df.columns:
        raise ValueError(f"Missing pdb column: {args.pdb_col}")
    if args.sdf_col not in df.columns:
        raise ValueError(f"Missing sdf column: {args.sdf_col}")

    pdb_dir = Path(args.pdb_dir) if args.pdb_dir else None
    sdf_dir = Path(args.sdf_dir) if args.sdf_dir else None

    n = len(df)
    prot = np.zeros((n, 45), dtype=np.float32)
    lig = np.zeros((n, 135), dtype=np.float32)
    row_ids = np.arange(n, dtype=np.int64)

    ok_prot = 0
    ok_lig = 0
    fail_rows = []

    for i, row in enumerate(df.itertuples(index=False), start=0):
        pdb_raw = getattr(row, args.pdb_col)
        sdf_raw = getattr(row, args.sdf_col)

        pdb_path = _resolve_path(pdb_raw, pdb_dir, ".pdb")
        sdf_path = _resolve_path(sdf_raw, sdf_dir, ".sdf")

        try:
            if (pdb_path is None) or (not pdb_path.exists()):
                raise FileNotFoundError(f"PDB not found: {pdb_raw}")
            prot[i] = pdb_to_prot45(pdb_path)
            ok_prot += 1
        except Exception as e:
            if not args.allow_missing:
                raise RuntimeError(f"row={i} protein parse failed: {e}") from e
            fail_rows.append((i, "prot", str(e)))

        try:
            if (sdf_path is None) or (not sdf_path.exists()):
                raise FileNotFoundError(f"SDF not found: {sdf_raw}")
            lig[i] = sdf_to_lig135(sdf_path)
            ok_lig += 1
        except Exception as e:
            if not args.allow_missing:
                raise RuntimeError(f"row={i} ligand parse failed: {e}") from e
            fail_rows.append((i, "lig", str(e)))

        if args.verbose_every > 0 and ((i + 1) % args.verbose_every == 0):
            print(f"[build] {i+1}/{n} rows done")

    out_path = Path(args.out_npz)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, prot_struct=prot, lig_struct=lig, row_id=row_ids)

    print(f"[done] npz saved: {out_path}")
    print(f"[done] prot ok={ok_prot}/{n}, lig ok={ok_lig}/{n}, failures={len(fail_rows)}")
    if fail_rows:
        print("[warn] first failures:")
        for x in fail_rows[:10]:
            print("  ", x)


if __name__ == "__main__":
    main()
