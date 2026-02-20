"""
Download and prepare OpenResearcher dataset for local testing.
Run this script once to set up the local data file.

Usage:
    python prepare_dataset.py
"""

from datasets import load_dataset
import pandas as pd
from pathlib import Path


def prepare_data():
    """
    Download OpenResearcher dataset from HuggingFace and prepare for local use.

    This script:
    1. Downloads the dataset from HuggingFace (seed_42 configuration)
    2. Extracts only required columns (qid, question, answer)
    3. Saves as a parquet file for efficient loading
    """
    print("Loading OpenResearcher-Dataset from HuggingFace...")
    print("This may take a few minutes on first run...")

    try:
        # Load dataset with seed_42 configuration, train split only
        # Try with download_mode to force fresh load
        ds = load_dataset(
            "OpenResearcher/OpenResearcher-Dataset",
            "seed_42",  # Specific configuration (positional arg)
            split="train",
            download_mode="force_redownload"
        )

        print(f"Loaded {len(ds)} examples from HuggingFace")

        # Convert to pandas DataFrame
        df = ds.to_pandas()

        print(f"Available columns: {list(df.columns)}")

        # Extract only needed columns
        keep_cols = ["qid", "question", "answer"]

        # Check if columns exist
        available_cols = set(df.columns)
        missing_cols = set(keep_cols) - available_cols

        if missing_cols:
            print(f"Warning: Missing expected columns {missing_cols}")
            print(f"Trying alternative column names...")

            # Try alternative column names
            col_mapping = {}
            if "id" in available_cols and "qid" not in available_cols:
                col_mapping["id"] = "qid"
                print("  Mapped 'id' -> 'qid'")
            if "problem" in available_cols and "question" not in available_cols:
                col_mapping["problem"] = "question"
                print("  Mapped 'problem' -> 'question'")

            if col_mapping:
                df = df.rename(columns=col_mapping)
                print(f"Column mapping applied: {col_mapping}")

        # Select only required columns
        df_subset = df[keep_cols]

        # Ensure types are correct
        df_subset["qid"] = df_subset["qid"].astype(str)
        df_subset["question"] = df_subset["question"].astype(str)
        df_subset["answer"] = df_subset["answer"].astype(str)

        # Save as parquet
        output_path = Path("openresearcher_seed42.parquet")
        df_subset.to_parquet(output_path, index=False)

        print(f"\nSaved {len(df_subset)} tasks to {output_path}")
        print(f"File size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")

        # Show sample
        print("\nSample tasks (first 3):")
        print("=" * 80)
        for i, row in df_subset.head(3).iterrows():
            print(f"\nTask {i+1}:")
            print(f"  QID: {row['qid']}")
            print(f"  Question: {row['question'][:100]}...")
            print(f"  Answer: {row['answer'][:100]}...")
        print("=" * 80)

        print("\nDataset preparation complete!")
        print("You can now run the environment server with: python server.py")

    except Exception as e:
        print(f"\nError preparing dataset: {e}")
        print("\nTroubleshooting:")
        print("1. Ensure you have installed required packages:")
        print("   pip install datasets pyarrow pandas")
        print("2. Check your internet connection")
        print("3. Verify the dataset name and configuration are correct")
        raise


if __name__ == "__main__":
    prepare_data()
