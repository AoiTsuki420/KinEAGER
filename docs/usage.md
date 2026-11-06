# Usage guide

KinEAGER predicts enzyme kinetic parameters from protein sequences and substrate
SMILES. The main model jointly predicts `kcat`, `Km`, and `kcat/Km`, with optional
structural and physicochemical evidence, source-domain conditioning, and uncertainty
heads. A separate kcat specialist and ESM embedding router support main/expert
mixture inference under sequence distribution shift.

This repository contains source code. Training data, pretrained backbones, trained
checkpoints, and experiment outputs must be supplied separately. Command examples
below are templates for prepared data, rather than a bundled reproduction dataset.

## Installation

Use Python 3.10 or later. Install a PyTorch build suitable for your hardware, then:

```bash
pip install -r requirements.txt
```

`peft` provides LoRA adapters; RDKit provides molecular features. `matplotlib` is
used by plotting tools. SwanLab logging is optional (`pip install swanlab`); the
training code uses a no-op logger when it is unavailable. Linux with CUDA/NCCL is
the intended environment for distributed training; Bash is needed for `.sh` scripts.

Run all commands below from the repository root (the directory containing `train.py`). Download the ESM-2 and MolT5
backbones first if working offline. Use the same model architecture and adapter
settings when loading a checkpoint as when training it.

## Paths and model configuration

Main training defaults to `data/`, `runs/`, and `weights/` under the checkout,
independent of the current working directory. Set `CATA_PROJECT_ROOT` to use a
separate workspace, or override individual paths using the `CATA_*` variables in
`kineager/paths.py`. Relative filesystem overrides are resolved under that workspace.
`CATA_PROFILE` may be unset or `default`.

Python model defaults use `facebook/esm2_t33_650M_UR50D` and
`laituan245/molt5-base-smiles2caption`. Main standalone/mixture inference has its
own model options, so pass the backbones explicitly to match the checkpoint.
For local backbones, use paths such as `backbone/esm2_t33_650M_UR50D` and
`backbone/molt5-base-smiles2caption`; these directories are not included.
Main training also accepts `CATA_ESM_MODEL_PATH` / `CATA_MOLT5_MODEL_PATH`.

## Main model data and features

The main training CSV uses the following columns:

| Column | Meaning |
|---|---|
| `Sequence` | Protein amino-acid sequence |
| `Smiles` | Substrate SMILES |
| `kcat(s^-1)` | Linear turnover rate in s^-1; missing labels may be NaN |
| `Km(M)` | Linear Km in mol/L; missing labels may be NaN |
| `source_id` | Domain identifier, such as `catapro`, `skid-kcat`, `skid-km` |
| `Protein_path`, `Ligand_path` | Optional PDB/SDF paths for feature preparation |

Do **not** log-transform main training labels in advance. The scaler converts
kcat to log10(s^-1), Km to log10(mM), and the ratio to log10(M^-1 s^-1).
The expert dataset has a separate schema, described below.

Normalize source tables and generate features:

```bash
python tools/data/build_unified_training_csv.py \
  --catapro_csv data/kcat-over-Km-data_0.4simi-10fold.csv \
  --skid_kcat_csv data/kcat_dataset_cleaned_after_drop.csv \
  --skid_km_csv data/Km_dataset_cleaned_after_drop.csv \
  --skid_km_unit M \
  --out_csv data/unified_train_mixed.csv

python tools/data/build_struct_npz.py \
  --csv data/unified_train_mixed.csv \
  --pdb_col Protein_path --sdf_col Ligand_path \
  --pdb_dir data/protein_structures --sdf_dir data/ligand_structures \
  --out_npz data/unified_train_mixed_struct.npz --allow_missing

python tools/data/build_phys_features.py \
  --csv data/unified_train_mixed.csv \
  --out_csv data/unified_train_mixed_phys.csv
```

Structural output contains `prot_struct` (N x 45), `lig_struct` (N x 135), and
`row_id`. Missing or unparseable structures become zero vectors with
`--allow_missing`; availability is inferred downstream from the vectors.
Physicochemical output appends nine `phys_*` descriptors plus `phys_mask` and
`phys_quality`. Keep the CSV and NPZ in the same row order; regenerating,
filtering, or reordering the CSV requires rebuilding the corresponding NPZ.

## Main training and evaluation

```bash
torchrun --nproc_per_node=1 train.py \
  -csv data/unified_train_mixed_phys.csv \
  -esm backbone/esm2_t33_650M_UR50D \
  -molt5 backbone/molt5-base-smiles2caption \
  --out_dir runs --run_name id_full_seed42 --seed 42 \
  --split_mode group_pair --split_seed 42 --val_ratio 0.2 --test_ratio 0.2 \
  --source_col source_id \
  --use_struct_branch --final_contact_npz data/unified_train_mixed_struct.npz \
  --enable_avail_gate --use_phys_evidence --phys_ablation real \
  --domain_uncertainty --num_domains 3 \
  -epochs 35 -batch_size 24 -device cuda
```

This example enables optional evidence branches; it is not a claim that these
arguments reproduce every manuscript experiment. Runs write artifacts under
`runs/<run_name>/seed<seed>/`, including model weights, metrics, split metadata,
`experiment_manifest.json`, and predictions. Increase `--nproc_per_node` to match
the GPUs available on one training host.

For evaluation with structural/physical evidence, use this same entry point with
`--eval_only --weights runs/id_full_seed42/seed42/predictor_best_ema.pt`, preserve
the original data, split, backbone and model arguments, and use a separate output
run name. The standalone inference script currently does not pass those extra
evidence inputs to the model.

For sequence/SMILES-only export:

```bash
python infer.py \
  -csv data/test.csv \
  -weights runs/id_full_seed42/seed42/predictor_best_ema.pt \
  -esm backbone/esm2_t33_650M_UR50D \
  -molt5 backbone/molt5-base-smiles2caption \
  --task kcat
```

It writes `predictions_kcat.csv` in the current directory (`--task km` selects
`predictions_km.csv`). Outputs include all three predictions and uncertainty;
`--task` selects the filename and truth-label interpretation. The optional
`-scaler` argument is retained for compatibility and is ignored. Compute metrics
separately with `tools/evaluation/eval_kcat.py` (`-pred`, `--pred-col`, `--true-col`).

## kcat specialist and mixture inference

Expert CSVs use `sequence`, `smiles`, and **log10-transformed** `kcat_log10`, with
optional `prot_pdb_path`, `lig_sdf_path`, and `geom_feat_path`. `--rename_cols none`
uses these names directly; `default` supports the aliases defined in the training
script. This schema is different from the main model's linear-label table.
Structure paths are resolved against `--base_dir` when supplied.

```bash
python train_expert.py \
  --train_csv data/kcat_expert_train.csv --val_csv data/kcat_expert_val.csv \
  --rename_cols none \
  --esm_path backbone/esm2_t33_650M_UR50D \
  --molt5_path backbone/molt5-base-smiles2caption \
  --out_dir runs/kcat_expert

python tools/data/precompute_train_embed.py \
  --train_csv data/kcat_expert_train.csv --seq_col sequence \
  --esm_path backbone/esm2_t33_650M_UR50D \
  --out_dir runs/moe_index --batch_size 8 --max_len 1024

python infer_ensemble.py \
  --test_csv data/kcat_expert_test.csv --rename_cols none \
  --main_ckpt runs/id_full_seed42/seed42/predictor_best_ema.pt \
  --expert_ckpt runs/kcat_expert/best.pt \
  --train_emb_npy runs/moe_index/train_emb.npy \
  --esm_path backbone/esm2_t33_650M_UR50D \
  --molt5_path backbone/molt5-base-smiles2caption \
  --main_esm backbone/esm2_t33_650M_UR50D \
  --main_molt5 backbone/molt5-base-smiles2caption \
  --out_csv results/kineager_test.csv
```

The index must be built from the expert's training sequences, using the same ESM
backbone as the query encoder. Mixture inference currently requires finite labels
in the expert-format input and drops invalid labels; it is not an unlabeled
screening interface. By default, missing protein/ligand structures disable the
expert route for that sample. The main branch receives sequence/SMILES inputs.
The output includes main/expert means and variances, OOD scores, mixture weights,
structure availability, and ensemble mean/variance.

## Additional workflows

- `tools/data/build_catpred_style_ood_splits.py`, `tools/data/build_catpred_iter_csv.py`, and
  `tools/data/build_domain_ood_merged_csv.py` prepare OOD evaluation tables.
- `scripts/prepare_structures.sh` prepares structures for iteration-based splits.
  `scripts/infer_kcat.sh` and `scripts/infer_km.sh` run prepared
  kcat/Km iteration workflows. Inspect their environment settings before use:
  they expect existing split tables, backbones, and checkpoints.
- `tools/evaluation/run_external_baseline.py` adapts external baseline repositories or supplied
  prediction CSVs; it does not bundle baseline models. `tools/evaluation/summarize_sota_results.py`
  collects metrics. `tools/evaluation/calibrate_moe_threshold.py` searches routing thresholds
  from cached predictions; calibrate on held-out validation data, not test labels.
- Mutation utilities (`insilico_mut_scan.py`, `pareto_filter.py`,
  `plot_mut_landscape.py` under `tools/mutation/`) support computational
  prioritization; their outputs do not establish experimental validity.

Use each Python tool's `--help` for its input contract and available options.

## Checks

```bash
pip install pytest
python -m pytest -q tests
python tests/smoke_moe_kcat.py
```

These checks cover paths, dataset metadata, missing labels, split leakage and
mocked routing/fusion logic. The smoke script optionally checks an existing index
at `runs/moe_index_smoke/train_emb.npy` and skips that part if it is absent.
None of these checks substitutes for GPU training or benchmark reproduction.

## License

This project is licensed under the [MIT License](../LICENSE).
