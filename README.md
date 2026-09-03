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
│   ├── 03_modeling.ipynb
│   └── 04_tuning_ensemble.ipynb
├── src/
│   ├── data_prep.py                 # dataset loading
│   ├── features.py                  # SMART constants and manufacturer extraction
│   └── model.py                     # training and evaluation utilities
├── models/                          # saved models and scalers (.gitignore'd)
├── scripts/
│   └── build_drive_dataset.py       # polars pipeline: daily → drive-level
└── figures/                         # exported visualizations
```

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
- **No temporal validation.** Train/test split is random, not time-based. In production, you would train on older data and validate on newer data to avoid data leakage.
- **Survivorship bias.** Drives that were replaced or decommissioned before Q1 2025 are not in the dataset. The model only sees drives that were active during the quarter.
- **Class imbalance.** Only 0.335% of drives failed. This requires careful handling (SMOTE, class weights, appropriate metrics) and means the model will always trade off between recall (catching failures) and precision (avoiding false alarms).
- **No external factors.** The model uses only SMART data and drive metadata. It does not account for temperature, workload, rack position, or power supply quality, all of which influence failure rates.

## Data source

[Backblaze Hard Drive Data](https://www.backblaze.com/cloud-storage/resources/hard-drive-test-data) — public, freely downloadable, updated quarterly.
