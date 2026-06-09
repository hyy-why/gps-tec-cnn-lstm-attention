# CNN-BiLSTM Attention for TEC Prediction

> Reproducible research code for TEC prediction with CNN-BiLSTM and attention variants.

This repository contains the code used for a TEC prediction study related to ionospheric/GNSS time-series modeling. The project compares a baseline CNN-BiLSTM model with several attention-enhanced variants.

## Models

The implemented models include:

| Model | Description |
|---|---|
| `Base` | CNN-BiLSTM baseline |
| `ECA` | CNN-BiLSTM with ECA-style attention |
| `SE` | CNN-BiLSTM with squeeze-excitation attention |
| `HW` | CNN-BiLSTM with temporal/channel weighting |
| `CBAM` | CNN-BiLSTM with combined channel and temporal attention |

## Reported Results

The original experimental results are summarized below.

| Model | R-squared | RMSE | MAE | MAPE |
|---|---:|---:|---:|---:|
| CNN-BiLSTM | 0.9243 | 0.0105 | 0.0062 | 0.8685% |
| CNN-BiLSTM + ECA | 0.9176 | 0.0109 | 0.0064 | 0.8929% |
| CNN-BiLSTM + SE | 0.9334 | 0.0098 | 0.0057 | 0.8070% |
| CNN-BiLSTM + HW | 0.7359 | 0.0196 | 0.0109 | 1.4956% |
| CNN-BiLSTM + CBAM | 0.7938 | 0.0173 | 0.0063 | 0.8742% |

## Repository Structure

```text
gps-tec-cnn-lstm-attention/
|-- README.md
|-- requirements.txt
|-- configs/
|   `-- default.yaml
|-- data/
|   |-- README.md
|   `-- sample_tec.csv
|-- checkpoints/
|   `-- README.md
|-- results/
|   `-- README.md
|-- scripts/
|   |-- create_sample_data.py
|   `-- shap_feature_analysis.py
`-- src/
    |-- dataloader.py
    |-- model.py
    |-- train.py
    `-- test.py
```

## Installation

```bash
pip install -r requirements.txt
```

For PyTorch, installing the version that matches your CUDA environment is recommended. See the official PyTorch installation page if GPU training is required.

## Quick Start

Create a synthetic sample CSV:

```bash
python scripts/create_sample_data.py
```

Train the SE variant for a quick smoke test:

```bash
python src/train.py --model SE --data data/sample_tec.csv --epochs 2
```

Evaluate the saved checkpoint:

```bash
python src/test.py --model SE --data data/sample_tec.csv
```

## Data Format

The input CSV should place feature columns first and the target TEC value in the last column:

```text
feature_1,feature_2,...,feature_n,TEC
```

The real experimental dataset is not included in this public repository. Please place the dataset under `data/` and pass its path with `--data`.

## Notes

- Large data files, trained checkpoints, and generated results are ignored by Git.
- The repository is packaged for reproducibility and code review.
- The original project contained local absolute paths; the public version uses command-line arguments instead.

## Citation

If this code is used after the related paper is formally available, please cite the corresponding paper. A BibTeX entry can be added here later.
