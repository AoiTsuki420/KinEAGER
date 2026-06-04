import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from data import EnzymeDataset
from utils import KineticsScaler, unpack_batch


def _build_df(n=6):
    return pd.DataFrame(
        {
            "Sequence": ["MKT" for _ in range(n)],
            "Smiles": ["CCO" for _ in range(n)],
            "kcat(s^-1)": np.linspace(1.0, 2.0, n, dtype=np.float32),
            "Km(M)": np.linspace(0.01, 0.02, n, dtype=np.float32),
            "source_id": [0, 1, 2, 0, 1, 2][:n],
        }
    )


def _build_scaler(df):
    scaler = KineticsScaler()
    scaler.fit(df["kcat(s^-1)"].to_numpy(), df["Km(M)"].to_numpy())
    return scaler


def test_unpack_batch_with_source_and_rowid_no_struct():
    df = _build_df(4)
    ds = EnzymeDataset(
        df,
        scaler=_build_scaler(df),
        supervised=True,
        return_row_id=True,
        source_col="source_id",
    )
    loader = DataLoader(ds, batch_size=4, shuffle=False)
    batch = next(iter(loader))
    seqs, smiles, row_ids, source_ids, p_s, l_s, labels, struct_avail, phys_feat, phys_mask, phys_quality = unpack_batch(batch)

    assert len(seqs) == 4
    assert row_ids is not None
    assert source_ids is not None
    assert p_s is None and l_s is None
    assert struct_avail is None
    assert phys_feat is None
    assert labels.shape[1] == 3


def test_unpack_batch_with_struct_and_source():
    df = _build_df(4)
    prot = np.random.randn(len(df), 45).astype(np.float32)
    lig = np.random.randn(len(df), 135).astype(np.float32)
    ds = EnzymeDataset(
        df,
        scaler=_build_scaler(df),
        supervised=True,
        return_row_id=True,
        source_col="source_id",
        prot_struct=prot,
        lig_struct=lig,
    )
    loader = DataLoader(ds, batch_size=4, shuffle=False)
    batch = next(iter(loader))
    seqs, smiles, row_ids, source_ids, p_s, l_s, labels, struct_avail, phys_feat, phys_mask, phys_quality = unpack_batch(batch)

    assert p_s is not None and l_s is not None
    assert p_s.shape[-1] == 45
    assert l_s.shape[-1] == 135
    assert source_ids is not None
    assert torch.is_tensor(labels)
    assert struct_avail is None
    assert phys_feat is None


def test_unpack_batch_with_phys_feature_columns():
    df = _build_df(4)
    df["phys_mw"] = [20.0, 30.0, 40.0, 50.0]
    df["phys_logp"] = [1.0, 1.1, 1.2, 1.3]
    df["phys_mask"] = [1.0, 1.0, 1.0, 1.0]
    df["phys_quality"] = [0.9, 0.8, 0.95, 1.0]
    ds = EnzymeDataset(
        df,
        scaler=_build_scaler(df),
        supervised=True,
        return_row_id=True,
        source_col="source_id",
    )
    loader = DataLoader(ds, batch_size=4, shuffle=False)
    batch = next(iter(loader))
    seqs, smiles, row_ids, source_ids, p_s, l_s, labels, struct_avail, phys_feat, phys_mask, phys_quality = unpack_batch(batch)
    assert phys_feat is not None
    assert phys_feat.shape[-1] == 2
    assert phys_mask is not None
    assert phys_quality is not None
