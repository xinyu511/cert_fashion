# NNCERT Consolidated (`cert_fashion`)

This folder is the canonical, reproducible implementation of the NNCERT workflow.
It preserves the same functional pipeline used in the `Cert_Temp*` folders, with cleaner structure and reproducibility-focused defaults.

Sibling folders (`Cert_Temp*`) are intentionally untouched and can be used as historical references.

## Pipeline

1. Train a base image classifier.
2. Generate an offline HDF5 vote dataset by Monte Carlo noise sampling.
3. Train a certifier model on `(image, sigma) -> class distribution`.
4. Evaluate one or more certifier checkpoints with CE/L1/KL metrics.

## Repository layout

```text
cert_fashion/
├── nncert/
│   ├── cli/                  # Canonical entrypoints
│   ├── data/                 # torchvision + HDF5 data modules
│   ├── models/               # CertNet / FuseNet
│   ├── lightning/            # Lightning wrappers
│   ├── workflows/            # HDF5 vote-generation workflow
│   ├── eval/                 # Evaluation metrics and runner
│   └── utils/                # Reproducibility helpers
├── configs/                  # Hydra configs
├── train_fashion_clf.py      # compatibility wrapper
├── create_certifier_dataset.py # compatibility wrapper
├── train_cert_offline.py     # compatibility wrapper
├── eval_certifiers.py        # compatibility wrapper
└── utils/                    # legacy import shims
```

## Installation

Use Python 3.10+.

```bash
pip install torch torchvision pytorch-lightning hydra-core omegaconf torchmetrics h5py numpy tqdm einops
```

## Quick start

Run all commands from `cert_fashion/`.

### 1) Train classifier

```bash
python train_fashion_clf.py
```

Examples:

```bash
python train_fashion_clf.py dataset=mnist
python train_fashion_clf.py dataset=kmnist training.epochs=30 training.devices=[0]
python train_fashion_clf.py dataset=cifar10 reproducibility.deterministic=true
```

### 2) Build offline vote dataset

```bash
python create_certifier_dataset.py dataset=fashionmnist classifier_ckpt_path=/abs/path/to/classifier.ckpt output_h5_path=./data/fashionmnist_certify_gauss_votes.h5
```

Vote-generation knobs:

```bash
python create_certifier_dataset.py votes.samples_per_sigma=1000 votes.num_sigmas=20 votes.sigma_min=0.01 votes.sigma_max=0.50 votes.normalize_inputs=true
```

### 3) Train certifier

```bash
python train_cert_offline.py offline_data_path=./data/fashionmnist_certify_gauss_votes.h5 classifier_ckpt_path=/abs/path/to/classifier.ckpt model=certifier_fusenet training.loss_type=l1
```

Try CertNet:

```bash
python train_cert_offline.py model=certifier_certnet
```

### 4) Evaluate certifiers

```bash
python eval_certifiers.py h5_path=./data/fashionmnist_certify_gauss_votes.h5 split=val models='[{name:fusenet_l1,ckpt:/abs/path/fusenet.ckpt},{name:certnet_ce,ckpt:/abs/path/certnet.ckpt}]'
```

Optional JSON export:

```bash
python eval_certifiers.py output_json_path=./outputs/eval_metrics.json
```

## Reproducibility guidance

1. Keep `reproducibility.seed` fixed across runs.
2. Set `reproducibility.deterministic=true` when strict determinism is required.
3. Keep `dataset.split_seed` fixed.
4. Record exact checkpoint paths and vote-generation hyperparameters.
5. Preserve HDF5 metadata (`meta` attribute).

Notes:

- Deterministic mode can reduce throughput.
- Some CUDA ops may still be nondeterministic depending on stack/hardware.

## Configuration overview

Hydra config groups:

- `configs/dataset/*.yaml`: dataset identity and split settings.
- `configs/model/*.yaml`: model/lightning wiring.
- `configs/training/*.yaml`: trainer and optimization settings.

Top-level experiment configs:

- `configs/train_classifier.yaml`
- `configs/create_votes_dataset.yaml`
- `configs/train_certifier.yaml`
- `configs/eval_certifiers.yaml`

## Legacy compatibility

Legacy import paths are still supported for old checkpoints/configs:

- `utils.pl_module.ImageClassifier`
- `utils.pl_module.ImageCertifier`
- `utils.cert_model.*`
- `utils.datasets.*`
- `h5_datamodule.py`

New development should import from `nncert.*`.
