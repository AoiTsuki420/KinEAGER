import argparse

import numpy as np
import pandas as pd

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors
except Exception:
    Chem = None
    Descriptors = None
    Lipinski = None
    rdMolDescriptors = None


def _pick_smiles_col(df):
    for c in ["Smiles", "SMILES", "Substrate SMILES", "Substrate_SMILES"]:
        if c in df.columns:
            return c
    raise ValueError("No SMILES column found")


def _feat_from_smiles(smiles: str):
    if Chem is None:
        raise ImportError("rdkit is required for build_phys_features.py")
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None

    mw = float(Descriptors.MolWt(mol))
    tpsa = float(Descriptors.TPSA(mol))
    logp = float(Descriptors.MolLogP(mol))
    hbd = float(Lipinski.NumHDonors(mol))
    hba = float(Lipinski.NumHAcceptors(mol))
    rotb = float(Lipinski.NumRotatableBonds(mol))
    ring = float(rdMolDescriptors.CalcNumRings(mol))
    aromatic_atoms = sum(1 for a in mol.GetAtoms() if a.GetIsAromatic())
    n_atoms = max(1, mol.GetNumAtoms())
    arom_ratio = float(aromatic_atoms) / float(n_atoms)
    charge = float(sum(a.GetFormalCharge() for a in mol.GetAtoms()))

    return {
        "phys_mw": mw,
        "phys_tpsa": tpsa,
        "phys_logp": logp,
        "phys_hbd": hbd,
        "phys_hba": hba,
        "phys_rotb": rotb,
        "phys_ring": ring,
        "phys_arom_ratio": arom_ratio,
        "phys_charge": charge,
    }


def main():
    p = argparse.ArgumentParser(description="Build RDKit 2D physics features into CSV")
    p.add_argument("--csv", required=True)
    p.add_argument("--out_csv", required=True)
    p.add_argument("--smiles_col", default=None)
    args = p.parse_args()

    df = pd.read_csv(args.csv)
    smiles_col = args.smiles_col or _pick_smiles_col(df)

    feats = []
    mask = []
    quality = []
    for s in df[smiles_col].astype(str).tolist():
        f = _feat_from_smiles(s)
        if f is None:
            feats.append(
                {
                    "phys_mw": 0.0,
                    "phys_tpsa": 0.0,
                    "phys_logp": 0.0,
                    "phys_hbd": 0.0,
                    "phys_hba": 0.0,
                    "phys_rotb": 0.0,
                    "phys_ring": 0.0,
                    "phys_arom_ratio": 0.0,
                    "phys_charge": 0.0,
                }
            )
            mask.append(0.0)
            quality.append(0.0)
        else:
            feats.append(f)
            mask.append(1.0)
            quality.append(1.0)

    feat_df = pd.DataFrame(feats)
    out = pd.concat([df.reset_index(drop=True), feat_df], axis=1)
    out["phys_mask"] = np.asarray(mask, dtype=np.float32)
    out["phys_quality"] = np.asarray(quality, dtype=np.float32)
    out.to_csv(args.out_csv, index=False)
    print(f"[done] saved: {args.out_csv}")
    print(f"[done] phys_mask==1: {int((out['phys_mask'] > 0.5).sum())}/{len(out)}")


if __name__ == "__main__":
    main()
