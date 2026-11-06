import argparse

import pandas as pd

try:
    from rdkit import Chem, RDLogger
except Exception:
    Chem = None
    RDLogger = None


def _pick_smiles_col(df: pd.DataFrame) -> str:
    for c in ["Smiles", "SMILES", "Substrate SMILES", "Substrate_SMILES"]:
        if c in df.columns:
            return c
    raise ValueError("No SMILES column found")


def _canonicalize_smiles(text: str, isomeric: bool = True) -> str | None:
    if Chem is None:
        raise ImportError("rdkit is required for normalize_smiles_csv.py")

    s = str(text).strip()
    if s == "" or s.lower() in {"nan", "none", "null"}:
        return None

    mol = Chem.MolFromSmiles(s)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=isomeric)


def main():
    p = argparse.ArgumentParser(description="Normalize/canonicalize SMILES in CSV")
    p.add_argument("--csv", required=True, help="Input CSV path")
    p.add_argument("--out_csv", required=True, help="Output CSV path")
    p.add_argument("--smiles_col", default=None, help="SMILES column name")
    p.add_argument("--canonical_col", default="Smiles_canonical", help="Output canonical SMILES column")
    p.add_argument("--valid_col", default="smiles_valid", help="Output validity mask column (0/1)")
    p.add_argument("--overwrite_smiles", action="store_true", help="Overwrite original SMILES column with canonical values")
    p.add_argument("--drop_invalid", action="store_true", help="Drop rows with invalid SMILES")
    p.add_argument("--no_isomeric", action="store_true", help="Drop stereochemistry during canonicalization")
    p.add_argument("--quiet_rdkit", action="store_true", help="Suppress RDKit parse warnings")
    args = p.parse_args()

    if args.quiet_rdkit and RDLogger is not None:
        RDLogger.DisableLog("rdApp.error")
        RDLogger.DisableLog("rdApp.warning")

    df = pd.read_csv(args.csv)
    smiles_col = args.smiles_col or _pick_smiles_col(df)

    can_list = []
    valid = []
    for s in df[smiles_col].tolist():
        can = _canonicalize_smiles(s, isomeric=(not args.no_isomeric))
        if can is None:
            can_list.append("")
            valid.append(0)
        else:
            can_list.append(can)
            valid.append(1)

    out = df.copy()
    out[args.canonical_col] = can_list
    out[args.valid_col] = valid

    if args.overwrite_smiles:
        out[smiles_col] = out[args.canonical_col]

    if args.drop_invalid:
        out = out[out[args.valid_col] == 1].copy()

    out.to_csv(args.out_csv, index=False)

    n_valid = int((pd.Series(valid) == 1).sum())
    print(f"[done] saved: {args.out_csv}")
    print(f"[done] smiles valid: {n_valid}/{len(valid)}")
    if args.drop_invalid:
        print(f"[done] rows after drop_invalid: {len(out)}")


if __name__ == "__main__":
    main()
