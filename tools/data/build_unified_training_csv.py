import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _pick_col(df: pd.DataFrame, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _to_num(series):
    return pd.to_numeric(series, errors="coerce")


def _normalize_base(
    df: pd.DataFrame,
    source_id: str,
    seq_candidates,
    smiles_candidates,
    kcat_candidates,
    km_candidates,
    prot_candidates,
    lig_candidates,
):
    seq_col = _pick_col(df, seq_candidates)
    smi_col = _pick_col(df, smiles_candidates)
    kcat_col = _pick_col(df, kcat_candidates)
    km_col = _pick_col(df, km_candidates)
    prot_col = _pick_col(df, prot_candidates)
    lig_col = _pick_col(df, lig_candidates)

    if seq_col is None:
        raise ValueError(f"[{source_id}] missing sequence column")
    if smi_col is None:
        raise ValueError(f"[{source_id}] missing smiles column")

    out = pd.DataFrame(
        {
            "Sequence": df[seq_col].astype(str),
            "Smiles": df[smi_col].astype(str),
            "kcat(s^-1)": _to_num(df[kcat_col]) if kcat_col is not None else np.nan,
            "Km(M)": _to_num(df[km_col]) if km_col is not None else np.nan,
            "source_id": source_id,
            "Protein_path": df[prot_col].astype(str) if prot_col is not None else np.nan,
            "Ligand_path": df[lig_col].astype(str) if lig_col is not None else np.nan,
        }
    )
    return out


def _convert_km_unit_to_m(km_series: pd.Series, unit: str):
    unit = unit.lower()
    if unit == "m":
        return km_series
    if unit == "mm":
        return km_series * 1e-3
    if unit == "um":
        return km_series * 1e-6
    if unit == "nm":
        return km_series * 1e-9
    raise ValueError(f"Unsupported Km unit: {unit}")


def main():
    p = argparse.ArgumentParser(description="Build unified mixed training CSV")
    p.add_argument("--catapro_csv", required=True)
    p.add_argument("--skid_kcat_csv", required=True)
    p.add_argument("--skid_km_csv", required=True)
    p.add_argument("--out_csv", required=True)
    p.add_argument("--skid_km_unit", default="M", choices=["M", "mM", "uM", "nM"], help="Unit of Km_value in SKiD Km csv")
    p.add_argument("--drop_empty_sequence", action="store_true")
    p.add_argument("--drop_empty_smiles", action="store_true")
    args = p.parse_args()

    catapro = pd.read_csv(args.catapro_csv)
    skid_kcat = pd.read_csv(args.skid_kcat_csv)
    skid_km = pd.read_csv(args.skid_km_csv)

    catapro_n = _normalize_base(
        catapro,
        source_id="catapro",
        seq_candidates=["Sequence", "sequence", "Protein Sequence"],
        smiles_candidates=["Smiles", "SMILES", "Substrate SMILES", "Substrate_SMILES"],
        kcat_candidates=["kcat(s^-1)", "kcat", "kcat_value"],
        km_candidates=["Km(M)", "Km", "Km_value"],
        prot_candidates=["Protein_path", "Protein Path", "pdb_path"],
        lig_candidates=["Ligand_path", "Ligand Path", "sdf_path"],
    )

    skid_kcat_n = _normalize_base(
        skid_kcat,
        source_id="skid-kcat",
        seq_candidates=["Sequence", "sequence", "Protein Sequence"],
        smiles_candidates=["Smiles", "SMILES", "Substrate SMILES", "Substrate_SMILES"],
        kcat_candidates=["kcat(s^-1)", "kcat", "kcat_value"],
        km_candidates=["Km(M)", "Km", "Km_value"],
        prot_candidates=["Protein_path", "Protein Path", "pdb_path"],
        lig_candidates=["Ligand_path", "Ligand Path", "sdf_path"],
    )

    skid_km_n = _normalize_base(
        skid_km,
        source_id="skid-km",
        seq_candidates=["Sequence", "sequence", "Protein Sequence"],
        smiles_candidates=["Smiles", "SMILES", "Substrate SMILES", "Substrate_SMILES"],
        kcat_candidates=["kcat(s^-1)", "kcat", "kcat_value"],
        km_candidates=["Km(M)", "Km", "Km_value"],
        prot_candidates=["Protein_path", "Protein Path", "pdb_path"],
        lig_candidates=["Ligand_path", "Ligand Path", "sdf_path"],
    )

    skid_km_n["Km(M)"] = _convert_km_unit_to_m(skid_km_n["Km(M)"], args.skid_km_unit)

    merged = pd.concat([catapro_n, skid_kcat_n, skid_km_n], axis=0, ignore_index=True)

    if args.drop_empty_sequence:
        merged = merged[merged["Sequence"].astype(str).str.len() > 0]
    if args.drop_empty_smiles:
        merged = merged[merged["Smiles"].astype(str).str.len() > 0]

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_path, index=False)

    print(f"[done] saved unified csv: {out_path}")
    print(f"[done] rows={len(merged)}")
    print("[done] source distribution:")
    print(merged["source_id"].value_counts(dropna=False).to_string())
    print("[done] non-null labels:")
    print("  kcat:", int(np.isfinite(merged["kcat(s^-1)"].to_numpy()).sum()))
    print("  Km  :", int(np.isfinite(merged["Km(M)"].to_numpy()).sum()))
    both = np.isfinite(merged["kcat(s^-1)"].to_numpy()) & np.isfinite(merged["Km(M)"].to_numpy())
    print("  both:", int(both.sum()))


if __name__ == "__main__":
    main()
