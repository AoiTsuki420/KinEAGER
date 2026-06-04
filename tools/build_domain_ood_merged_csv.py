import argparse
from pathlib import Path

import pandas as pd


def _read_required_csv(split_dir: Path, name: str) -> pd.DataFrame:
    p = split_dir / name
    if not p.exists():
        raise FileNotFoundError(f"Missing required split file: {p}")
    return pd.read_csv(p)


def build_merged_csv(
    split_dir: Path,
    out_csv: Path,
    split_col: str,
    train_tag: str,
    holdout_tag: str,
    include_iid: bool,
) -> None:
    train = _read_required_csv(split_dir, "train.csv")
    val = _read_required_csv(split_dir, "val.csv")
    test_iid = _read_required_csv(split_dir, "test_iid.csv")
    test_enzyme_ood = _read_required_csv(split_dir, "test_enzyme_ood.csv")
    test_substrate_ood = _read_required_csv(split_dir, "test_substrate_ood.csv")
    test_both_ood = _read_required_csv(split_dir, "test_both_ood.csv")

    train[split_col] = train_tag
    val[split_col] = train_tag
    if include_iid:
        test_iid[split_col] = train_tag
    test_enzyme_ood[split_col] = holdout_tag
    test_substrate_ood[split_col] = holdout_tag
    test_both_ood[split_col] = holdout_tag

    parts = [train, val]
    if include_iid:
        parts.append(test_iid)
    parts.extend([test_enzyme_ood, test_substrate_ood, test_both_ood])

    out = pd.concat(parts, ignore_index=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)

    split_counts = out[split_col].astype(str).value_counts(dropna=False).to_dict()
    src_counts = out["source_id"].astype(str).value_counts(dropna=False).to_dict() if "source_id" in out.columns else {}
    print(f"saved {out_csv}: {len(out)} rows")
    print(f"  split_col={split_col} counts={split_counts}")
    if src_counts:
        print(f"  source_id counts={src_counts}")


def main() -> None:
    p = argparse.ArgumentParser(description="Build merged csv for domain_ood training from CatPred-style splits")
    p.add_argument("--split_dir", required=True, help="Directory containing train/val/test_*.csv")
    p.add_argument("--out_csv", required=True, help="Output merged csv path")
    p.add_argument("--split_col", default="split_source", help="Column used by --split_source_col during training")
    p.add_argument("--train_tag", default="seq_train_pool", help="Tag value for non-holdout pool")
    p.add_argument("--holdout_tag", default="seq_ood_holdout", help="Tag value for OOD holdout pool")
    p.add_argument("--include_iid", action="store_true", help="If set, test_iid is merged into non-holdout pool")
    args = p.parse_args()

    build_merged_csv(
        split_dir=Path(args.split_dir),
        out_csv=Path(args.out_csv),
        split_col=args.split_col,
        train_tag=args.train_tag,
        holdout_tag=args.holdout_tag,
        include_iid=bool(args.include_iid),
    )


if __name__ == "__main__":
    main()
