import pandas as pd

from main_train_predictor_multigpu import normalize_training_dataframe, split_dataframe_for_experiment


def _pair_keys(df: pd.DataFrame) -> pd.Series:
    seq = df["Sequence"].astype(str).str.replace(r"\s+", "", regex=True).str.upper()
    smi = df["Smiles"].astype(str).str.strip()
    return seq + "||" + smi


def _build_df() -> pd.DataFrame:
    rows = []
    for i in range(30):
        src = "catapro" if i < 12 else ("skid-kcat" if i < 22 else "skid-km")
        if src == "catapro":
            pair_id = i % 8
        elif src == "skid-kcat":
            pair_id = 4 + (i % 10)
        else:
            pair_id = 8 + (i % 10)
        rows.append(
            {
                "Sequence": f"SEQ_{pair_id}",
                "Smiles": f"SMI_{pair_id}",
                "kcat(s^-1)": float(i + 1),
                "Km(M)": float(i + 2),
                "source_id": src,
            }
        )
    return pd.DataFrame(rows)


def test_group_pair_split_removes_pair_leakage():
    df = normalize_training_dataframe(_build_df())
    train_df, val_df, test_df, meta = split_dataframe_for_experiment(
        df=df,
        split_mode="group_pair",
        split_seed=42,
        val_ratio=0.2,
        test_ratio=0.2,
        source_col="source_id",
        holdout_source=None,
        ood_dedup_pair=True,
    )

    tr_pairs = set(_pair_keys(train_df).tolist())
    va_pairs = set(_pair_keys(val_df).tolist())
    te_pairs = set(_pair_keys(test_df).tolist())

    assert len(tr_pairs & te_pairs) == 0
    assert len(va_pairs & te_pairs) == 0
    assert len(tr_pairs & va_pairs) == 0
    assert meta["pair_overlap"]["train_test"] == 0
    assert meta["pair_overlap"]["val_test"] == 0


def test_domain_ood_pair_dedup_removes_test_pair_overlap():
    df = normalize_training_dataframe(_build_df())
    train_df, val_df, test_df, meta = split_dataframe_for_experiment(
        df=df,
        split_mode="domain_ood",
        split_seed=42,
        val_ratio=0.2,
        test_ratio=0.2,
        source_col="source_id",
        holdout_source="catapro",
        ood_dedup_pair=True,
    )

    tr_pairs = set(_pair_keys(train_df).tolist())
    va_pairs = set(_pair_keys(val_df).tolist())
    te_pairs = set(_pair_keys(test_df).tolist())

    assert len(tr_pairs & te_pairs) == 0
    assert len(va_pairs & te_pairs) == 0
    assert meta["pair_overlap"]["train_test"] == 0
    assert meta["pair_overlap"]["val_test"] == 0
