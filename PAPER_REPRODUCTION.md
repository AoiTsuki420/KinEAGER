# Paper Reproduction Guide

This document maps the KcatMoE manuscript workflow to the source files in this repository. It is intended for reviewers and readers who want to inspect or reproduce the computational experiments without local notebook state.

## Manuscript-to-code Mapping

| Manuscript component | Code entry point |
|---|---|
| Unified multi-source kinetics table | `tools/build_unified_training_csv.py` |
| Protein and ligand structural vectors | `tools/build_struct_npz.py`, `final_contact.py` |
| RDKit physicochemical evidence | `tools/build_phys_features.py` |
| Main multi-task KcatMoE predictor | `models/predictor.py`, `main_train_predictor_multigpu.py` |
| ESM-2 and MolT5 encoders | `models/encoders.py` |
| Cross-modal interaction blocks | `models/interactions.py` |
| Domain-task uncertainty loss and metrics | `main_train_predictor_multigpu.py`, `utils.py` |
| kcat specialist model | `models/kcat_expert.py`, `main_train_kcat_expert.py` |
| OOD-aware MoE router | `models/moe_kcat.py`, `main_infer_ensemble.py` |
| Sequence-cluster OOD split adaptation | `tools/build_catpred_style_ood_splits.py`, `prep_splits_revision_struct.sh` |
| External baseline adaptation | `tools/run_external_baseline.py`, `tools/summarize_sota_results.py` |
| In silico mutation scanning | `tools/insilico_mut_scan.py`, `tools/pareto_filter.py`, `tools/run_directed_evo.py`, `tools/plot_mut_landscape.py` |

## Minimal Reproduction Order

1. Prepare the unified training table with source-domain identifiers and log10-transformed labels.
2. Build structural evidence arrays and physicochemical descriptors.
3. Train the main multi-task predictor with fixed `split_seed=42` and `split_mode=group_pair`.
4. Train the kcat expert on the specialist kcat table.
5. Precompute the expert training ESM embedding index.
6. Run main-model IID inference and KcatMoE sequence-cluster OOD inference.
7. Summarize metrics from `metrics.json`, prediction CSV/NPZ files, and split metadata.
8. Run counterfactual evidence ablations by changing structural and physical evidence modes while keeping the same split.

## Required Run Artifacts

For each manuscript run, retain:

- `experiment_manifest.json`
- `split_meta.json`
- `metrics.json`
- prediction files with row identifiers
- command line and random seed
- checkpoint hash or archive location

Large artifacts should be released through a data archive, Zenodo/Figshare record, or GitHub release asset rather than committed to the source tree.

## Reporting Conventions

- Report `kcat`, `Km`, and `kcat/Km` metrics in log10 space.
- Keep IID, sequence-cluster OOD, LOSO/domain-OOD, missing-modality, and counterfactual ablation results separate.
- State whether a result comes from the main multi-task predictor or from the kcat MoE route.
- For CatPred literature values, explicitly note when numbers are taken from the original paper rather than rerun under the local protocol.
- Treat in silico mutation scans as hypothesis generation unless wet-lab validation is available.

## Excluded from Source Control

The source repository intentionally excludes raw data tables, model checkpoints, foundation model downloads, run directories, generated figures, spreadsheet supplements, notebooks, and private wet-lab files. The `.gitignore` file enforces these exclusions.