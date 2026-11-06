import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

try:
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold
except Exception:
    Chem = None
    MurckoScaffold = None


def _pick_col(df: pd.DataFrame, candidates: list[str], what: str) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(f"Cannot find {what} column. Tried: {candidates}")


def _seq_norm(s: str) -> str:
    return "".join(str(s).split()).upper()


def _smi_norm(s: str) -> str:
    return str(s).strip()


def _pair_key(seq: str, smi: str) -> str:
    return hashlib.sha1(f"{seq}||{smi}".encode("utf-8")).hexdigest()


def _scaffold_id(smiles: str) -> str:
    if Chem is None or MurckoScaffold is None:
        raise ImportError("rdkit is required for scaffold extraction")
    s = _smi_norm(smiles)
    if s == "":
        return "EMPTY"
    mol = Chem.MolFromSmiles(s)
    if mol is None:
        return "INVALID"
    scf = MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
    return scf if scf else "NO_SCAFFOLD"


def _run_mmseqs(
    mmseqs_bin: str,
    fasta_path: Path,
    out_prefix: Path,
    tmp_dir: Path,
    min_seq_id: float,
    threads: int,
):
    cmd = [
        mmseqs_bin,
        "easy-cluster",
        str(fasta_path),
        str(out_prefix),
        str(tmp_dir),
        "--min-seq-id",
        str(min_seq_id),
        "-c",
        "0.8",
        "--cov-mode",
        "0",
        "--cluster-mode",
        "2",
        "--threads",
        str(threads),
    ]
    subprocess.run(cmd, check=True)


def _parse_cluster_tsv(cluster_tsv: Path) -> dict[str, str]:
    if not cluster_tsv.exists():
        raise FileNotFoundError(f"MMseqs output not found: {cluster_tsv}")
    m = {}
    with cluster_tsv.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rep, member = line.split("\t")[:2]
            m[member] = rep
    return m


def _sample_unseen(ids: np.ndarray, ratio: float, rng: np.random.Generator) -> set:
    ids = np.array(sorted(set(ids.tolist())))
    if len(ids) == 0:
        return set()
    k = int(round(len(ids) * ratio))
    k = max(1, min(len(ids), k))
    picked = rng.choice(ids, size=k, replace=False)
    return set(picked.tolist())


def _group_split(df: pd.DataFrame, group_col: str, test_size: float, seed: int):
    if len(df) == 0:
        return df.copy(), df.copy()
    groups = df[group_col].to_numpy()
    if len(np.unique(groups)) < 2:
        return df.copy(), df.iloc[0:0].copy()
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    idx = np.arange(len(df))
    tr_idx, te_idx = next(gss.split(idx, groups=groups))
    return df.iloc[tr_idx].copy(), df.iloc[te_idx].copy()


def _numeric_summary(df: pd.DataFrame, col: str) -> dict:
    if col not in df.columns:
        return {"n": 0}
    s = pd.to_numeric(df[col], errors="coerce")
    s = s[np.isfinite(s)]
    if len(s) == 0:
        return {"n": 0}
    q = s.quantile([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
    return {
        "n": int(len(s)),
        "mean": float(s.mean()),
        "std": float(s.std(ddof=1)) if len(s) > 1 else 0.0,
        "q01": float(q.loc[0.01]),
        "q05": float(q.loc[0.05]),
        "q25": float(q.loc[0.25]),
        "q50": float(q.loc[0.5]),
        "q75": float(q.loc[0.75]),
        "q95": float(q.loc[0.95]),
        "q99": float(q.loc[0.99]),
    }


def main():
    p = argparse.ArgumentParser(description="Build CatPred-style approximate OOD split CSVs")
    p.add_argument("--csv", required=True, help="Input unified CSV path")
    p.add_argument("--out_dir", required=True, help="Output root directory for split CSVs")
    p.add_argument("--seq_col", default="Sequence", help="Sequence column name")
    p.add_argument("--smiles_col", default=None, help="SMILES column name; auto-detect if omitted")
    p.add_argument("--id_thresholds", default="0.6,0.4", help="Comma-separated MMseqs min_seq_id values")
    p.add_argument("--unseen_enzyme_ratio", type=float, default=0.2, help="Fraction of enzyme clusters as unseen")
    p.add_argument("--unseen_scaffold_ratio", type=float, default=0.2, help="Fraction of scaffolds as unseen")
    p.add_argument("--iid_test_ratio", type=float, default=0.2, help="IID test fraction from IID pool")
    p.add_argument("--val_ratio", type=float, default=0.1, help="Validation fraction from IID pool")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--mmseqs_bin", default="mmseqs", help="Path to mmseqs binary")
    p.add_argument("--threads", type=int, default=8, help="MMseqs threads")
    p.add_argument("--keep_tmp", action="store_true", help="Keep temporary mmseqs files")
    args = p.parse_args()

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv)
    seq_col = args.seq_col
    if seq_col not in df.columns:
        seq_col = _pick_col(df, ["Sequence", "sequence", "Protein Sequence"], "sequence")
    smiles_col = args.smiles_col
    if smiles_col is None:
        smiles_col = _pick_col(df, ["Smiles_canonical", "Smiles", "SMILES", "Substrate SMILES", "Substrate_SMILES"], "SMILES")
    if smiles_col not in df.columns:
        raise ValueError(f"SMILES column not found: {smiles_col}")

    df = df.copy()
    df["seq_norm"] = df[seq_col].astype(str).map(_seq_norm)
    df["smiles_norm"] = df[smiles_col].fillna("").astype(str).map(_smi_norm)
    df["pair_key"] = [
        _pair_key(sq, sm)
        for sq, sm in zip(df["seq_norm"].tolist(), df["smiles_norm"].tolist())
    ]
    df["scaffold_id"] = df["smiles_norm"].map(_scaffold_id)

    seq_unique = sorted(set(df["seq_norm"].tolist()))
    seq_ids = {s: f"seq_{i}" for i, s in enumerate(seq_unique)}
    df["seq_id"] = df["seq_norm"].map(seq_ids)

    fasta_path = out_root / "_tmp_sequences.fasta"
    with fasta_path.open("w", encoding="utf-8") as f:
        for s in seq_unique:
            f.write(f">{seq_ids[s]}\n{s}\n")

    thresholds = [float(x.strip()) for x in args.id_thresholds.split(",") if x.strip()]

    for thr in thresholds:
        thr_tag = f"{int(round(thr * 100)):03d}"
        split_dir = out_root / f"catpred_style_{thr_tag}"
        split_dir.mkdir(parents=True, exist_ok=True)

        out_prefix = split_dir / "mmseqs_cluster"
        tmp_dir = split_dir / "_mmseqs_tmp"

        _run_mmseqs(
            mmseqs_bin=args.mmseqs_bin,
            fasta_path=fasta_path,
            out_prefix=out_prefix,
            tmp_dir=tmp_dir,
            min_seq_id=thr,
            threads=args.threads,
        )

        cluster_map = _parse_cluster_tsv(Path(str(out_prefix) + "_cluster.tsv"))
        dft = df.copy()
        dft["enzyme_cluster_id"] = dft["seq_id"].map(cluster_map).fillna("UNCLUSTERED")

        rng = np.random.default_rng(int(args.seed * 1000 + round(thr * 100)))
        unseen_ec = _sample_unseen(dft["enzyme_cluster_id"].to_numpy(), args.unseen_enzyme_ratio, rng)
        unseen_sc = _sample_unseen(dft["scaffold_id"].to_numpy(), args.unseen_scaffold_ratio, rng)

        e_unseen = dft["enzyme_cluster_id"].isin(unseen_ec)
        s_unseen = dft["scaffold_id"].isin(unseen_sc)

        dft["ood_tag"] = "iid"
        dft.loc[e_unseen & (~s_unseen), "ood_tag"] = "enzyme_ood"
        dft.loc[(~e_unseen) & s_unseen, "ood_tag"] = "substrate_ood"
        dft.loc[e_unseen & s_unseen, "ood_tag"] = "both_ood"

        iid_all = dft[dft["ood_tag"] == "iid"].copy()
        iid_trainval, iid_test = _group_split(iid_all, "pair_key", args.iid_test_ratio, args.seed)
        rel_val = args.val_ratio / max(1e-12, 1.0 - args.iid_test_ratio)
        rel_val = min(max(rel_val, 0.0), 0.9)
        iid_train, iid_val = _group_split(iid_trainval, "pair_key", rel_val, args.seed)

        test_enz = dft[dft["ood_tag"] == "enzyme_ood"].copy()
        test_sub = dft[dft["ood_tag"] == "substrate_ood"].copy()
        test_both = dft[dft["ood_tag"] == "both_ood"].copy()

        outs = {
            "train.csv": iid_train,
            "val.csv": iid_val,
            "test_iid.csv": iid_test,
            "test_enzyme_ood.csv": test_enz,
            "test_substrate_ood.csv": test_sub,
            "test_both_ood.csv": test_both,
        }
        for fn, sub in outs.items():
            sub.to_csv(split_dir / fn, index=False)

        pair_sets = {k: set(v["pair_key"].tolist()) for k, v in outs.items()}
        keys = list(pair_sets.keys())
        overlap = {}
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                a, b = keys[i], keys[j]
                overlap[f"{a}__{b}"] = int(len(pair_sets[a].intersection(pair_sets[b])))

        audit = {
            "config": {
                "csv": str(args.csv),
                "seq_col": seq_col,
                "smiles_col": smiles_col,
                "min_seq_id": thr,
                "unseen_enzyme_ratio": args.unseen_enzyme_ratio,
                "unseen_scaffold_ratio": args.unseen_scaffold_ratio,
                "iid_test_ratio": args.iid_test_ratio,
                "val_ratio": args.val_ratio,
                "seed": args.seed,
            },
            "counts": {k.replace(".csv", ""): int(len(v)) for k, v in outs.items()},
            "ood_tag_counts": dft["ood_tag"].value_counts(dropna=False).to_dict(),
            "n_unique_pairs": {k.replace(".csv", ""): int(v["pair_key"].nunique()) for k, v in outs.items()},
            "pair_overlap": overlap,
            "label_stats": {
                k.replace(".csv", ""): {
                    "kcat": _numeric_summary(v, "kcat(s^-1)" if "kcat(s^-1)" in v.columns else "kcat"),
                    "Km": _numeric_summary(v, "Km(M)" if "Km(M)" in v.columns else "Km"),
                }
                for k, v in outs.items()
            },
        }

        with (split_dir / "split_audit.json").open("w", encoding="utf-8") as f:
            json.dump(audit, f, ensure_ascii=False, indent=2)

        print(f"[done] {split_dir}")
        print("[counts]", audit["counts"])
        print("[pair_overlap]", audit["pair_overlap"])

        if not args.keep_tmp:
            for suffix in ["_all_seqs.fasta", "_cluster.tsv", "_rep_seq.fasta"]:
                pth = Path(str(out_prefix) + suffix)
                if pth.exists():
                    pth.unlink()
            if tmp_dir.exists():
                shutil.rmtree(tmp_dir, ignore_errors=True)

    if fasta_path.exists() and (not args.keep_tmp):
        fasta_path.unlink()


if __name__ == "__main__":
    main()
