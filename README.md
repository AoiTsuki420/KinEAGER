# KinEAGER

KinEAGER is an evidence-aware multi-task Transformer framework for enzyme kinetics prediction under domain shift. It jointly predicts log10-transformed `kcat`, `Km`, and `kcat/Km`, handles missing structural evidence through availability-aware gating, and reports uncertainty through domain-task heteroscedastic training and ensemble/MoE inference.

This repository contains the functional code needed to reproduce the main computational workflow described in the manuscript: data harmonization, structural and physicochemical feature construction, multi-task model training, OOD-aware kcat expert routing, external baseline adaptation, evaluation, and in silico mutation scanning.

## Model Overview

KinEAGER combines:

- ESM-2 protein sequence encoding and MolT5 substrate SMILES encoding.
- Cross-modal protein-substrate interaction blocks.
- Optional protein and ligand structural feature branches.
- Availability-aware structural gating to distinguish valid structural evidence from missing evidence.
- Physicochemical evidence features and counterfactual evidence ablations.
- Domain-task heteroscedastic loss for multi-source kinetic data.
- A kcat OOD expert and nearest-neighbor ESM embedding router for sequence-cluster OOD inference.

The main model is used for IID multi-task results, ablations, missing-modality experiments, and uncertainty calibration. The KinEAGER mixture route is used for kcat sequence-cluster OOD inference.

## Repository Layout

```text
KinEAGER/
  data.py                              Dataset and collate utilities
  main_train_predictor_multigpu.py      Main multi-task DDP/AMP training entry point
  main_train_kcat_expert.py             kcat specialist training entry point
  main_infer_predictor.py               Main-model inference
  main_infer_ensemble.py                Main + expert KinEAGER inference
  eval_kcat_predictions.py              Prediction metric utility
  models/                               Encoders, interaction blocks, predictor, expert, router
  tools/                                Data preparation, feature extraction, OOD split, baseline and plotting tools
  tests/                                Focused smoke tests
  test_scripts/                         Lightweight pipeline and leakage checks
```

Large training tables, checkpoints, model weights, run outputs, notebooks, caches, and raw experimental assets are intentionally excluded. Store those through a data archive or release asset rather than the source repository.

## Installation

```bash
conda create -n kineager python=3.10 -y
conda activate kineager
pip install -r requirements.txt
```

The original experiments used PyTorch, HuggingFace Transformers, ESM-2 (`esm2_t33_650M_UR50D`), MolT5 (`molt5-base-smiles2caption`), RDKit, pandas, numpy, scikit-learn, scipy, tqdm, and optional wandb logging. For offline servers, download ESM-2 and MolT5 locally and pass their local paths to the training and inference scripts.

## Data Preparation

Build the unified multi-task table:

```bash
python tools/build_unified_training_csv.py \
  --catapro_csv data/kcat-over-Km-data_0.4simi-10fold.csv \
  --skid_kcat_csv data/kcat_dataset_cleaned_after_drop.csv \
  --skid_km_csv data/Km_dataset_cleaned_after_drop.csv \
  --out_csv data/unified_train_mixed.csv \
  --skid_km_unit M
```

Build structural and physicochemical evidence:

```bash
python tools/build_struct_npz.py \
  --csv data/unified_train_mixed.csv \
  --pdb_col Protein_path \
  --sdf_col Ligand_path \
  --pdb_dir data/protein_structures \
  --sdf_dir data/ligand_structures \
  --out_npz data/unified_train_mixed_struct.npz \
  --allow_missing

python tools/build_phys_features.py \
  --csv data/unified_train_mixed.csv \
  --out_csv data/unified_train_mixed_phys.csv
```

Missing structural evidence should be represented by zero vectors plus an availability mask, not by dropping samples.

## Main Multi-task Training

```bash
torchrun --nproc_per_node=1 main_train_predictor_multigpu.py \
  -csv data/unified_train_mixed_phys.csv \
  --final_contact_npz data/unified_train_mixed_struct.npz \
  --run_name id_full_seed42 \
  --split_mode group_pair \
  --split_seed 42 \
  --val_ratio 0.2 \
  --test_ratio 0.2 \
  --source_col source_id \
  --use_struct_branch \
  --enable_avail_gate \
  --use_phys_evidence \
  --phys_drop_p 0.2 \
  --phys_ablation real \
  --lambda_gate_off 0.01 \
  --lambda_gate_on 0.01 \
  --lambda_cons 0.1 \
  --cons_warmup_epochs 3 \
  --struct_dropout_prob 0.4 \
  --domain_uncertainty \
  --num_domains 3 \
  -epochs 35 \
  -batch_size 24 \
  -device cuda
```

Each run writes metrics, split metadata, experiment manifest, predictions, and checkpoints under the configured run directory.

## kcat Expert and MoE Inference

Train the kcat specialist:

```bash
python main_train_kcat_expert.py \
  --train_csv data/kcat_expert_train.csv \
  --val_csv data/kcat_expert_val.csv \
  --esm_path /path/to/esm2_t33_650M_UR50D \
  --out_dir runs/kcat_expert
```

Precompute the expert training embedding index:

```bash
python tools/precompute_train_embed.py \
  --train_csv data/kcat_expert_train.csv \
  --esm_path /path/to/esm2_t33_650M_UR50D \
  --out_dir runs/moe_index \
  --batch_size 8 \
  --max_len 1024
```

Run KinEAGER inference:

```bash
python main_infer_ensemble.py \
  --test_csv data/testood40.csv \
  --main_ckpt runs/id_full_seed42/seed42/predictor_best_ema.pt \
  --expert_ckpt runs/kcat_expert/best.pt \
  --train_emb_npy runs/moe_index/train_emb.npy \
  --out_csv results/kineager_testood40.csv \
  --esm_path /path/to/esm2_t33_650M_UR50D \
  --molt5_path /path/to/molt5-base-smiles2caption \
  --main_esm /path/to/esm2_t33_650M_UR50D \
  --main_molt5 /path/to/molt5-base-smiles2caption \
  --router_d0 0.15 \
  --router_tau 0.05 \
  --mc_samples 5 \
  --batch_size 16
```

The output CSV includes `y_kcat`, `mu_main`, `s2_main`, `mu_expert`, `s2_expert`, `ood_score`, `w_main`, `w_expert`, `has_struct`, `mu_ensemble`, and `s2_ensemble`.

## OOD and Baseline Evaluation

Use `tools/build_catpred_style_ood_splits.py` and `prep_splits_revision_struct.sh` to align sequence-cluster OOD splits with the manuscript protocol. Use `tools/run_external_baseline.py` and `tools/summarize_sota_results.py` to adapt external baseline predictions to the same metric scripts.

Router thresholds can be selected from cached inference CSV files:

```bash
python tools/calibrate_moe_threshold.py \
  --inputs id=results/moe_id.csv ood40=results/moe_ood40.csv ood60=results/moe_ood60.csv \
  --out_csv results/calib_grid.csv \
  --select_by overall \
  --metric mae \
  --top_k 10
```

## In Silico Mutation Scanning

The mutation utilities use model outputs and gate-derived signals for hypothesis generation:

```bash
python tools/insilico_mut_scan.py --help
python tools/pareto_filter.py --help
python tools/plot_mut_landscape.py --help
python tools/run_directed_evo.py --help
```

These scripts support computational prioritization only; experimental validation should be reported separately.

## Reproducibility Notes

- Fix `split_seed=42` for the IID group-pair split used in the manuscript.
- Keep `split_meta.json`, `experiment_manifest.json`, `metrics.json`, and prediction files for each run.
- Report sequence-cluster OOD results separately from IID multi-task results.
- Treat CatPred literature numbers and local baseline reruns as different comparison protocols unless they share the same train/test construction.
- Do not commit raw checkpoints, downloaded foundation models, intermediate run directories, cache files, or private wet-lab data.

## Citation

```bibtex
@article{kineager2026,
  title   = {KinEAGER: Evidence-aware Multi-task Modeling and Uncertainty Calibration for Enzyme Kinetics Prediction},
  author  = {TBD},
  journal = {TBD},
  year    = {2026},
  url     = {https://github.com/Rieide/KinEAGER}
}
```

## License

Add the final project license before public release.
