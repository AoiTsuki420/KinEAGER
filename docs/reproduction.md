# Paper Reproduction Guide

This document maps the KinEAGER manuscript workflow to the source files in this repository. It is intended for reviewers and readers who want to inspect or reproduce the computational experiments without local notebook state.

## Manuscript-to-code Mapping

| Manuscript component | Code entry point |
|---|---|
| Unified multi-source kinetics table | `tools/data/build_unified_training_csv.py` |
| Protein and ligand structural vectors | `tools/data/build_struct_npz.py`, `kineager/structure.py` |
| RDKit physicochemical evidence | `tools/data/build_phys_features.py` |
| Main multi-task KinEAGER predictor | `kineager/models/predictor.py`, `train.py` |
| ESM-2 and MolT5 encoders | `kineager/models/encoders.py` |
| Cross-modal interaction blocks | `kineager/models/interactions.py` |
| Domain-task uncertainty loss and metrics | `train.py`, `kineager/utils.py` |
| kcat specialist model | `kineager/models/kcat_expert.py`, `train_expert.py` |
| OOD-aware MoE router | `kineager/models/moe_kcat.py`, `infer_ensemble.py` |
| Sequence-cluster OOD split adaptation | `tools/data/build_catpred_style_ood_splits.py`, `scripts/prepare_structures.sh` |
| External baseline adaptation | `tools/evaluation/run_external_baseline.py`, `tools/evaluation/summarize_sota_results.py` |
| In silico mutation scanning | `tools/mutation/insilico_mut_scan.py`, `tools/mutation/pareto_filter.py`, `tools/mutation/plot_mut_landscape.py` |

## Minimal Reproduction Order

1. Prepare the unified training table with source-domain identifiers and linear labels: kcat in s^-1 and Km in M. Training converts these to log10(kcat), log10(Km in mM), and log10(kcat/Km in M^-1 s^-1); do not pre-transform the main training labels.
2. Build structural evidence arrays and physicochemical descriptors.
3. Train the main multi-task predictor with fixed `split_seed=42` and `split_mode=group_pair`.
4. Train the kcat expert on the specialist kcat table.
5. Precompute the expert training ESM embedding index.
6. Evaluate the evidence-aware main model with the training entry point's `--eval_only` mode and matching feature/configuration arguments. The standalone `infer.py` currently accepts sequence/SMILES inputs only. Use `infer_ensemble.py` for kcat main/expert routing.
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

Raw data, checkpoints, foundation models, notebooks, and private wet-lab files are not distributed here. The `.gitignore` file excludes common data and run artifacts; keep any additional generated outputs outside source control. See [usage guide](usage.md) for data contracts, runnable command templates, and limitations.
