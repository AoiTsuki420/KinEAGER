from pathlib import Path
import os
from typing import Any


# Defaults are resolved from the checkout, independent of the working directory.
DEFAULT_PATHS = {
    "runs_dir": "runs",
    "weights_dir": "weights",
    "train_data_dir": "data",
    "train_csv_path": "data/kcat-over-Km-data_0.4simi-10fold.csv",
    "kcat_struct_csv_path": "data/kcat_dataset_cleaned_after_drop.csv",
    "km_struct_csv_path": "data/Km_dataset_cleaned_after_drop.csv",
    "unified_train_csv_path": "data/unified_train_mixed.csv",
    "unified_struct_npz_path": "data/unified_train_mixed_struct.npz",
    "skid_kcat_archive_dir": "data/SKiD_kcat_archive",
    "skid_km_archive_dir": "data/SKiD_Km_archive",
    "val_csv_path": "data/val_set.csv",
    "scaler_json_path": "weights/scaler_fold.json",
    "metrics_json_path": "runs/metrics.json",
    "esm_model_path": "facebook/esm2_t33_650M_UR50D",
    "molt5_model_path": "laituan245/molt5-base-smiles2caption",
}


ENV_OVERRIDES = {
    "project_root": "CATA_PROJECT_ROOT",
    "runs_dir": "CATA_RUNS_DIR",
    "weights_dir": "CATA_WEIGHTS_DIR",
    "train_data_dir": "CATA_TRAIN_DATA_DIR",
    "train_csv_path": "CATA_TRAIN_CSV_PATH",
    "kcat_struct_csv_path": "CATA_KCAT_STRUCT_CSV_PATH",
    "km_struct_csv_path": "CATA_KM_STRUCT_CSV_PATH",
    "unified_train_csv_path": "CATA_UNIFIED_TRAIN_CSV_PATH",
    "unified_struct_npz_path": "CATA_UNIFIED_STRUCT_NPZ_PATH",
    "skid_kcat_archive_dir": "CATA_SKID_KCAT_ARCHIVE_DIR",
    "skid_km_archive_dir": "CATA_SKID_KM_ARCHIVE_DIR",
    "val_csv_path": "CATA_VAL_CSV_PATH",
    "scaler_json_path": "CATA_SCALER_JSON_PATH",
    "metrics_json_path": "CATA_METRICS_JSON_PATH",
    "esm_model_path": "CATA_ESM_MODEL_PATH",
    "molt5_model_path": "CATA_MOLT5_MODEL_PATH",
}


FILESYSTEM_PATH_KEYS = {
    "project_root",
    "runs_dir",
    "weights_dir",
    "train_data_dir",
    "train_csv_path",
    "kcat_struct_csv_path",
    "km_struct_csv_path",
    "unified_train_csv_path",
    "unified_struct_npz_path",
    "skid_kcat_archive_dir",
    "skid_km_archive_dir",
    "val_csv_path",
    "scaler_json_path",
    "metrics_json_path",
}


def resolve_runtime_paths(profile=None):
    selected = profile or os.getenv("CATA_PROFILE") or "default"
    if selected != "default":
        raise ValueError("Only the 'default' path profile is supported; configure CATA_* variables instead.")
    root = Path(os.getenv("CATA_PROJECT_ROOT") or Path(__file__).resolve().parents[1]).expanduser().resolve()
    raw = dict(DEFAULT_PATHS)
    raw["project_root"] = str(root)
    for key, env_name in ENV_OVERRIDES.items():
        env_value = os.getenv(env_name)
        if env_value and key != "project_root":
            raw[key] = env_value

    paths: dict[str, Any] = {}
    for key, value in raw.items():
        if key in FILESYSTEM_PATH_KEYS:
            path = Path(value).expanduser()
            paths[key] = path if path.is_absolute() else root / path
        else:
            # Model identifiers must remain strings, rather than filesystem paths.
            paths[key] = value

    weights_dir = paths.get("weights_dir")
    if weights_dir is not None:
        paths["best_ema_path"] = weights_dir / "predictor_best_ema.pt"
        paths["best_raw_path"] = weights_dir / "predictor_best_raw.pt"
        paths["final_model_path"] = weights_dir / "predictor.pt"
    else:
        paths["best_ema_path"] = None
        paths["best_raw_path"] = None
        paths["final_model_path"] = None

    paths["profile"] = selected
    return paths


def ensure_paths(paths, required_keys):
    missing = [k for k in required_keys if paths.get(k) is None]
    if missing:
        names = ", ".join(missing)
        profile = paths.get("profile", "unknown")
        raise ValueError(
            f"Missing path values for profile '{profile}': {names}. "
            "Please set CATA_PROFILE or corresponding CATA_* environment variables."
        )
