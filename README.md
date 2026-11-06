# KinEAGER

Evidence-aware prediction of enzyme kinetic parameters (`kcat`, `Km`, and
`kcat/Km`) from protein sequences and substrate SMILES, with optional structural
and physicochemical evidence and a kcat expert/router.

## Layout

```text
KinEAGER/
├── train.py                 # Main model training and evaluation
├── train_expert.py          # kcat specialist training
├── infer.py                 # Sequence/SMILES prediction export
├── infer_ensemble.py        # Main/expert mixture inference
├── kineager/                # Shared Python package
│   ├── models/              # Encoders, predictor, specialist, router
│   ├── data.py              # Main dataset and loaders
│   ├── structure.py         # Structural feature loading
│   ├── paths.py             # Runtime path configuration
│   └── utils.py             # Scaling, metrics, checkpoints
├── tools/
│   ├── data/                # Table preparation and feature extraction
│   ├── evaluation/          # Export, metrics, calibration, baseline adapters
│   └── mutation/            # Mutation scanning, Pareto filtering, plotting
├── scripts/                 # Optional Bash workflows for prepared OOD splits
├── tests/                   # Unit and mocked routing checks
└── docs/                    # Detailed usage and reproduction notes
```

## Setup

Use Python 3.10+ and a PyTorch build appropriate for your hardware:

```bash
pip install -r requirements.txt
```

Run commands from the repository root. Data, ESM-2/MolT5 backbones, trained
checkpoints, and results are supplied separately. Distributed training targets
Linux with CUDA/NCCL. SwanLab logging is optional.

## Entry points

| Task | Command |
|---|---|
| Train or evaluate the main model | `python train.py --help` |
| Train the kcat specialist | `python train_expert.py --help` |
| Export sequence/SMILES predictions | `python infer.py --help` |
| Run kcat mixture inference | `python infer_ensemble.py --help` |
| Prepare unified data | `python tools/data/build_unified_training_csv.py --help` |
| Compute prediction metrics | `python tools/evaluation/eval_kcat.py --help` |

For single-host distributed training, use `torchrun --nproc_per_node=<GPUs>
train.py ...`.

Main training expects **linear** kcat in s^-1 and Km in M; expert CSVs instead
use `kcat_log10`. Keep structural NPZ rows aligned with their training CSV.
Standalone inference currently passes sequence/SMILES only; evaluate additional
evidence branches with `train.py --eval_only` and matching model/data arguments.

See the [usage guide](docs/usage.md) for data schemas, complete command templates,
model paths, outputs, and limitations. The [reproduction notes](docs/reproduction.md)
map the computational workflow to code and describe which run artifacts to keep.

## Checks

```bash
pip install pytest
python -m pytest -q tests
python tests/smoke_moe_kcat.py
```

These checks do not replace full GPU training or benchmark reproduction.

## License

[MIT](LICENSE).
