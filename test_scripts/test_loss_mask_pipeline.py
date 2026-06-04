import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.optim import AdamW
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader

from data import EnzymeDataset
from main_train_predictor_multigpu import train_step, normalize_training_dataframe, apply_phys_ablation
from utils import KineticsScaler


def test_normalize_training_dataframe_for_skid_columns():
    df = pd.DataFrame(
        {
            "Sequence": ["MKT", "MKT"],
            "Substrate SMILES": ["CCO", "CCN"],
            "kcat(s^-1)": [1.0, 2.0],
            "Km_value": [np.nan, 0.01],
        }
    )
    out = normalize_training_dataframe(df)
    assert "Smiles" in out.columns
    assert "kcat(s^-1)" in out.columns
    assert "Km(M)" in out.columns
    assert np.isfinite(out["kcat(s^-1)"].iloc[0])


def test_train_step_with_partial_labels_runs_without_nan():
    n = 8
    df = pd.DataFrame(
        {
            "Sequence": ["MKT"] * n,
            "Smiles": ["CCO"] * n,
            "kcat(s^-1)": [1.0, 1.5, np.nan, np.nan, 2.0, np.nan, 1.3, np.nan],
            "Km(M)": [np.nan, 0.01, 0.02, np.nan, 0.03, 0.02, np.nan, 0.04],
            "source_id": [0, 1, 2, 0, 1, 2, 0, 1],
        }
    )

    scaler = KineticsScaler()
    scaler.fit(df["kcat(s^-1)"].to_numpy(), df["Km(M)"].to_numpy())

    ds = EnzymeDataset(df, scaler=scaler, supervised=True, return_row_id=True, source_col="source_id")
    batch = next(iter(DataLoader(ds, batch_size=4, shuffle=False)))

    class Dummy(nn.Module):
        def __init__(self):
            super().__init__()
            self.p_k = nn.Parameter(torch.tensor(0.1))
            self.p_m = nn.Parameter(torch.tensor(0.2))
            self.p_r = nn.Parameter(torch.tensor(0.3))
            self.s_kcat = nn.Parameter(torch.tensor(0.0))
            self.s_Km = nn.Parameter(torch.tensor(0.0))
            self.s_ratio = nn.Parameter(torch.tensor(0.0))
            self.s_domain_task = nn.Parameter(torch.zeros(3, 3))
            self.last_gate_struct = None
            self.last_struct_avail = None
            self._mse_ema = torch.zeros(3)

        def forward(self, seqs, smiles, use_mask=True, bias_tuple=(0.0, 0.0, 0.0), caps=(0.5, 0.5, 0.4), floor_m=-1e9, prot_struct=None, lig_struct=None, struct_avail_mask=None, phys_feat=None, phys_mask=None, phys_quality=None):
            bsz = len(seqs)
            device = self.p_k.device
            one = torch.ones(bsz, device=device)
            pred_k = self.p_k * one
            pred_m = self.p_m * one
            pred_r = self.p_r * one
            gate_e = torch.sigmoid(torch.stack([pred_k, pred_k], dim=1))
            gate_s = torch.sigmoid(torch.stack([pred_m, pred_m], dim=1))
            if struct_avail_mask is None:
                avail = torch.zeros(bsz, device=device)
            else:
                avail = struct_avail_mask.to(device=device, dtype=pred_k.dtype)
            self.last_gate_struct = torch.zeros_like(avail)
            self.last_struct_avail = avail
            s_raw = (self.s_kcat, self.s_Km, self.s_ratio)
            s_eff = (self.s_kcat, self.s_Km, self.s_ratio)
            return pred_k, pred_m, pred_r, gate_e, gate_s, s_raw, s_eff

    model = Dummy()
    optimizer = AdamW(model.parameters(), lr=1e-3)
    scaler_amp = GradScaler(enabled=False)
    loss = train_step(model, batch, optimizer, scaler_amp, epoch_idx=0, lambda_cons=0.0)
    assert np.isfinite(float(loss))


def test_phys_ablation_modes_shape_stable():
    feat = torch.randn(6, 9)
    mask = torch.ones(6)
    quality = torch.ones(6)
    for mode in ["real", "shuffle", "random", "none"]:
        f2, m2, q2 = apply_phys_ablation(feat, mask, quality, mode=mode)
        assert f2.shape == feat.shape
        assert m2.shape == mask.shape
        assert q2.shape == quality.shape
