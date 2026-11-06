"""Run the standalone inference CLI through CSV export with a tiny model stub."""

import sys

import numpy as np
import pandas as pd
import pytest
import torch

import infer


@pytest.mark.parametrize('task', ['kcat', 'km'])
def test_inference_exports_predictions_outside_checkout(tmp_path, monkeypatch, task):
    class TinyPredictor(torch.nn.Module):
        def forward(self, seqs, smiles, **kwargs):
            return tuple(torch.full((len(seqs),), value) for value in (1.0, 0.0, 4.0))

    source = tmp_path / 'input with spaces.csv'
    pd.DataFrame({
        'Sequence': ['MKT', 'MKTA'], 'Smiles': ['CCO', 'CCN'],
        'kcat(s^-1)': [10.0, 10.0], 'Km(M)': [0.001, 0.001],
    }).to_csv(source, index=False)
    monkeypatch.setattr(infer, 'build_model_from_ckpt', lambda *args, **kwargs: (TinyPredictor(), None, {}))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, 'argv', [
        'infer.py', '-csv', str(source), '-weights', 'stub.pt',
        '-device', 'cpu', '--disable_mc_dropout', '--task', task,
    ])
    infer.main()
    result = pd.read_csv(tmp_path / f'predictions_{task}.csv')
    assert len(result) == 2
    np.testing.assert_allclose(result['pred_kcat'], 10.0)
    np.testing.assert_allclose(result['pred_Km'], 0.001)
    np.testing.assert_allclose(result['pred_kcat_div_Km'], 10000.0)
    assert ('true_kcat_log' if task == 'kcat' else 'true_Km_log') in result
