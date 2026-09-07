# Raw Data

Backblaze Hard Drive Stats, Q1 2025.

**Download:** https://www.backblaze.com/cloud-storage/resources/hard-drive-test-data

1. Download the Q1 2025 zip (~1 GB)
2. Extract into this directory as `data_Q1_2025/`
3. Run the pipeline:

```bash
uv run python scripts/raw_to_parquet.py
uv run python scripts/build_drive_dataset.py
```

This produces `data/processed/drives.parquet` (318K drives, ~30 MB).
