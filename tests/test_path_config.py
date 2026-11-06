"""Runtime paths must work outside the checkout and support custom workspaces."""

import os
from pathlib import Path
from unittest.mock import patch

from kineager.paths import DEFAULT_PATHS, resolve_runtime_paths


def test_defaults_are_independent_of_working_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with patch.dict(os.environ, {}, clear=True):
        paths = resolve_runtime_paths()
    checkout = Path(__file__).resolve().parents[1]
    assert paths['runs_dir'] == checkout / 'runs'
    assert paths['unified_train_csv_path'] == checkout / 'data/unified_train_mixed.csv'
    assert paths['esm_model_path'] == DEFAULT_PATHS['esm_model_path']


def test_workspace_and_individual_overrides(tmp_path):
    runs = tmp_path / 'external-runs'
    model = tmp_path / 'local-esm'
    with patch.dict(os.environ, {
        'CATA_PROJECT_ROOT': str(tmp_path),
        'CATA_RUNS_DIR': str(runs),
        'CATA_WEIGHTS_DIR': 'saved',
        'CATA_ESM_MODEL_PATH': str(model),
    }, clear=True):
        paths = resolve_runtime_paths()
    assert paths['train_data_dir'] == tmp_path / 'data'
    assert paths['runs_dir'] == runs
    assert paths['best_ema_path'] == tmp_path / 'saved/predictor_best_ema.pt'
    assert paths['esm_model_path'] == str(model)


def test_relative_workspace_is_resolved_only_once(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with patch.dict(os.environ, {'CATA_PROJECT_ROOT': 'workspace'}, clear=True):
        paths = resolve_runtime_paths()
    assert paths['project_root'] == tmp_path / 'workspace'
    assert paths['runs_dir'] == paths['project_root'] / 'runs'
