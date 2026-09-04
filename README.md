# drive-failure-predictor

Predicting hard drive failures before they happen, using real production SMART sensor data from [Backblaze](https://www.backblaze.com/cloud-storage/resources/hard-drive-test-data).

## Why this matters

A single hard drive failure in production infrastructure can trigger a chain reaction: the failed drive degrades a RAID array, the rebuild stresses the remaining drives, a second failure during rebuild means **total data loss**. In a data center serving clients, that translates to service downtime ($1K–$100K+/hour depending on the business), SLA breach penalties, and in regulated sectors (finance, healthcare), potential compliance violations with six-figure fines.

The industry standard is reactive: replace the drive after it fails. But modern drives broadcast distress signals — SMART (Self-Monitoring, Analysis and Reporting Technology) sensor readings — that often deteriorate days or weeks before the actual failure. A model that detects these degradation patterns shifts maintenance from reactive to predictive: replace the drive *before* it dies, during a scheduled window, with zero downtime and zero data loss.

The physical replacement cost is ~$30–50 per drive. The cost of an undetected failure — downtime, lost data, cascading array failures, broken SLAs — is orders of magnitude higher.

## Dataset

Backblaze publishes daily snapshots of every hard drive in their data centers — model, serial number, SMART readings, and whether the drive failed that day. This project uses **Q1 2025** (January–March):

| Metric | Value |
|---|---|
| Daily records | 27,799,986 |
| Unique drives | 318,426 |
| Drives that failed | 1,067 |
| SMART attributes tracked | 20 (raw values) |
| Time span | 90 days |

### Why aggregate to drive level

The raw data has one row per drive per day. Training a model on daily rows would mean:

- **28 million rows** that don't fit in memory on a standard laptop.
- A **0.003% failure rate** — because the `failure` column is 1 only on the exact day the drive died, and 0 on the hundreds of healthy days before it.
- A model that answers the wrong question: *"will today be the failure day?"* instead of *"is this drive going to fail?"*.

Aggregating to one row per drive fixes all three:

| | Daily level | Drive level |
|---|---|---|
| Rows | 27,799,986 | 318,426 |
| Failure rate | 0.003% | 0.335% |
| Prediction unit | "will today be the day?" | "will this drive fail?" |
| Memory | ~13 GB | ~32 MB |

For each drive, the aggregation computes five statistics over its full quarter of readings: **last value**, **max**, **mean**, **standard deviation**, and **slope** (linear trend). The slope is the most important engineered feature — a drive whose reallocated sector count is rising fast is more likely to fail than one with a high but stable count.

## Project structure

```
drive-failure-predictor/
├── README.md
├── pyproject.toml
├── data/
│   ├── raw/                         # Backblaze source data (.gitignore'd)
│   └── processed/
│       └── drives.parquet           # drive-level aggregated dataset (32 MB)
├── notebooks/
│   ├── 01_eda.ipynb                 # exploratory data analysis
│   ├── 02_feature_engineering.ipynb # feature selection, scaling, baseline model
│   ├── 03_modeling.ipynb            # RF, AdaBoost, SMOTE, imbalance strategies
│   └── 04_tuning_ensemble.ipynb     # XGBoost tuning, stacking, critical analysis
├── src/
│   ├── data_prep.py                 # dataset loading
│   ├── features.py                  # SMART constants and manufacturer extraction
│   └── model.py                     # training and evaluation utilities
├── models/                          # saved models and scalers (.gitignore'd)
├── scripts/
│   ├── build_drive_dataset.py       # polars pipeline: daily → drive-level
│   ├── raw_to_parquet.py            # raw Backblaze CSVs → parquet
│   ├── export_figures.py            # generate presentation figures from models
│   ├── predict.py                   # CLI inference script for individual drives
│   └── temporal_validation.py       # train Jan-Feb, test March (distribution shift analysis)
└── figures/                         # exported visualizations
```

## Results

| Model | F1 | Precision | Recall | ROC AUC |
|---|---|---|---|---|
| Logistic Regression (baseline) | 0.1347 | 0.08 | 0.87 | 0.9678 |
| Random Forest (class_weight) | 0.9549 | 0.97 | 0.94 | 0.9976 |
| AdaBoost | 0.9140 | 0.89 | 0.94 | 0.9919 |
| XGBoost (baseline) | 0.9577 | 0.96 | 0.96 | 0.9954 |
| **XGBoost (tuned)** | **0.9602** | **0.96** | **0.96** | **0.9954** |
| Stacking (RF + XGB → LR) | 0.9052 | 0.84 | 0.99 | 0.9955 |

The best model is XGBoost tuned via `RandomizedSearchCV` (30 iterations, 5-fold stratified CV). The difference with RF balanced is ~1–2 drives on 213 test failures — not statistically significant, but XGBoost generalises slightly better.

### Class imbalance strategies

| Strategy | F1 | Notes |
|---|---|---|
| class_weight="balanced" | 0.9549 | Built into the loss function — no resampling needed |
| SMOTE (oversampling) | 0.9506 | Generates synthetic minority samples |
| Tomek links (frontier cleaning) | 0.9551 | Removes 58 ambiguous majority samples |
| SMOTETomek (combined) | 0.9484 | Oversample + clean |
| RandomUnderSampler | 0.6055 | Discards 99% of majority — too aggressive |

Reweighting (`class_weight` / `scale_pos_weight`) matches or beats resampling with zero data manipulation.

### Figures

All figures are in `figures/` and generated from the trained models via `scripts/export_figures.py`:

- `model_comparison.png` — F1 scores across all models
- `roc_pr_curves.png` — ROC and Precision-Recall curves overlaid
- `feature_importance_xgb.png` — top 15 features by XGBoost gain (slopes highlighted)
- `confusion_matrix_xgb.png` — confusion matrix of the best model
- `imbalance_strategies.png` — comparison of 5 class imbalance strategies
- `class_distribution.png` — train/test class distribution
- `cost_sensitive_threshold.png` — cost vs threshold curve (optimal at 0.040)
- `temporal_validation.png` — random vs temporal split comparison with distribution shift analysis
- `target_distribution.png`, `feature_correlation.png`, `smart_slopes.png` — from EDA notebooks

## How to run

This project uses [uv](https://docs.astral.sh/uv/).

```bash
uv sync

# the drive-level dataset is already in the repo (data/processed/drives.parquet)
# to rebuild it from the raw Backblaze data:
# 1. download a quarter from https://www.backblaze.com/cloud-storage/resources/hard-drive-test-data
# 2. convert it to parquet in data/processed/q1_2025_selected.parquet
# 3. run:
uv run python scripts/build_drive_dataset.py

# run the notebooks
uv run jupyter notebook
```

## Key SMART attributes

The attributes most predictive of failure, based on both the literature and this dataset:

| SMART ID | Name | What it measures |
|---|---|---|
| 5 | Reallocated Sectors Count | Bad sectors the drive has already remapped |
| 187 | Reported Uncorrectable Errors | Read errors the drive could not recover from |
| 188 | Command Timeout | Commands that took too long to complete |
| 197 | Current Pending Sector Count | Unstable sectors waiting to be remapped |
| 198 | Offline Uncorrectable | Sectors that failed during offline scans |

## Why polars + pandas

The raw daily data is 28 million rows (~13 GB in memory). Loading it with pandas caused an out-of-memory kill on a 16 GB machine. The aggregation script (`scripts/build_drive_dataset.py`) uses **polars** — its lazy evaluation (`scan_parquet`) processes the data without loading it all into RAM.

The output is a drive-level parquet (318k rows, 32 MB) that the notebooks load with **pandas**. At that scale pandas is comfortable, and the ML ecosystem (scikit-learn, XGBoost, imbalanced-learn, matplotlib) expects pandas DataFrames or numpy arrays.

In short: polars where memory efficiency matters (data engineering), pandas where ecosystem integration matters (analysis and modeling). The project also uses **statsmodels** for VIF (Variance Inflation Factor) analysis during feature selection.

## Scope and limitations

- **Single quarter.** The model is trained on Q1 2025 only. Failure patterns may differ across seasons, drive batches, or firmware versions. A production system would train on multiple quarters.
- **Temporal validation reveals distribution shift.** Training on Jan-Feb data and testing on March produces F1 ≈ 0, but AUC-ROC = 0.82 — the model *has* learned discriminative signal, but the decision boundary doesn't transfer. Root cause: aggregating over 59 days (Jan-Feb) vs 31 days (March) shifts `days_observed`, `std`, and slope features. Production fix: use fixed-length rolling windows (e.g. always last 30 days) and multi-quarter training data.
- **Survivorship bias.** Drives that were replaced or decommissioned before Q1 2025 are not in the dataset. The model only sees drives that were active during the quarter.
- **Class imbalance.** Only 0.335% of drives failed. This requires careful handling (SMOTE, class weights, appropriate metrics) and means the model will always trade off between recall (catching failures) and precision (avoiding false alarms).
- **No external factors.** The model uses only SMART data and drive metadata. It does not account for temperature, workload, rack position, or power supply quality, all of which influence failure rates.

## Data source

[Backblaze Hard Drive Data](https://www.backblaze.com/cloud-storage/resources/hard-drive-test-data) — public, freely downloadable, updated quarterly.
