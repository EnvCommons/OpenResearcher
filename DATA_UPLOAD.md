# Data Upload Requirements for OpenResearcher

## Overview

This environment requires the OpenResearcher dataset to be uploaded to OpenReward cloud storage. The dataset contains 6,102 research questions from seed_42 configuration.

## Dataset Information

- **Source:** HuggingFace Datasets - OpenResearcher/OpenResearcher-Dataset
- **Configuration:** seed_42
- **Split:** train
- **Size:** ~6,102 research questions
- **Format:** Parquet file
- **Estimated file size:** ~5-10 MB (compressed)

## Directory Structure Required

```
/orwd_data/
└── openresearcher/
    └── openresearcher_seed42.parquet
```

## File Description

**openresearcher_seed42.parquet** contains 6,102 research questions with:
- `qid` (str): Question ID
- `question` (str): Research question text requiring web search and multi-hop reasoning
- `answer` (str): Ground truth answer

## Preparing the Dataset Locally

Before uploading, you can prepare the dataset locally using the provided script:

### Step 1: Install Required Dependencies

```bash
pip install datasets pyarrow pandas
```

### Step 2: Run the Preparation Script

```bash
cd openresearcher
python prepare_dataset.py
```

This will:
1. Download the dataset from HuggingFace
2. Extract only the required columns (qid, question, answer)
3. Save as `openresearcher_seed42.parquet` in the current directory

The script will display:
- Number of tasks loaded
- File size
- Sample tasks for verification

## Upload Instructions

### 1. Create Namespace on OpenReward

- Go to https://openreward.ai
- Create or access namespace: `EnvCommons`
- Navigate to namespace storage settings

### 2. Upload Dataset

**Option A: Via Web Interface**
1. Navigate to namespace storage
2. Create directory: `openresearcher/`
3. Upload `openresearcher_seed42.parquet` to this directory
4. Verify final path: `/orwd_data/openresearcher/openresearcher_seed42.parquet`

**Option B: Via CLI (if available)**
```bash
# Upload to OpenReward namespace storage
or-cli storage upload openresearcher_seed42.parquet /orwd_data/openresearcher/
```

### 3. Verify Upload

After uploading, verify:
- File path is exactly: `/orwd_data/openresearcher/openresearcher_seed42.parquet`
- File size matches your local file (~5-10 MB)
- File is accessible from the deployed environment

## Alternative: Download from HuggingFace Directly

If you don't have the local file, you can download directly from HuggingFace:

```python
from datasets import load_dataset
import pandas as pd

# Load dataset
ds = load_dataset(
    "OpenResearcher/OpenResearcher-Dataset",
    name="seed_42",
    split="train"
)

# Convert to pandas and save
df = ds.to_pandas()[["qid", "question", "answer"]]
df["qid"] = df["qid"].astype(str)
df["question"] = df["question"].astype(str)
df["answer"] = df["answer"].astype(str)
df.to_parquet("openresearcher_seed42.parquet", index=False)

print(f"Saved {len(df)} tasks")
```

## Dataset Schema

The parquet file must have exactly these columns:

| Column     | Type   | Description                              |
|------------|--------|------------------------------------------|
| qid        | string | Unique question identifier               |
| question   | string | Research question text                   |
| answer     | string | Ground truth answer                      |

## Troubleshooting

### Issue: Dataset columns don't match expected names

**Solution:** Check the dataset schema and ensure column names are exactly: `qid`, `question`, `answer`

If the dataset has different column names (e.g., `id`, `problem`), rename them:

```python
df = df.rename(columns={"id": "qid", "problem": "question"})
```

### Issue: File too large to upload

**Solution:** Ensure only required columns are included. The `prepare_dataset.py` script already filters to only `qid`, `question`, `answer`, which should keep the file small.

If still too large:
- Check for duplicate rows
- Verify no extra columns are included
- Ensure no large text fields beyond the three required columns

### Issue: Environment can't find data file

**Error message:** `FileNotFoundError: OpenResearcher parquet not found at...`

**Solution:**
1. Verify file path is exactly `/orwd_data/openresearcher/openresearcher_seed42.parquet`
2. Check file permissions (should be readable)
3. Ensure directory `/orwd_data/openresearcher/` exists
4. Verify file was fully uploaded (not corrupted or partial)

### Issue: Data loading fails with ValueError

**Error message:** `Parquet file missing required columns...`

**Solution:**
1. Re-download the dataset using `prepare_dataset.py`
2. Verify the parquet file has columns: `qid`, `question`, `answer`
3. Check data types are all strings
4. Ensure no null values in required columns

## Data Privacy and Security

- This dataset contains publicly available research questions
- No personal information or sensitive data is included
- Dataset is from HuggingFace public repository: OpenResearcher/OpenResearcher-Dataset

## Support

For questions or issues:
- Check the main [README.md](README.md) for general setup instructions
- Open an issue on the GitHub repository: EnvCommons/openresearcher
- Verify you're using the correct dataset configuration (seed_42, train split)

## Additional Notes

- The dataset should NOT be included in the Docker image (per CLAUDE.md guidelines)
- Data is loaded at module import time for efficiency (AIME pattern)
- The environment will fail fast with clear error messages if data is not found
- Both local development and production deployment are supported via path fallback logic in `constants.py`
