"""
Path constants for OpenResearcher environment.
Handles both production (/orwd_data) and local development paths.
"""

from pathlib import Path
import os

# Check if /orwd_data exists (production), otherwise use local directory (dev)
if os.path.exists("/orwd_data"):
    ENV_PATH = Path("/orwd_data")
else:
    ENV_PATH = Path(__file__).parent

# Parquet file locations with fallback logic
OPENRESEARCHER_PARQUET_PROD = ENV_PATH / "openresearcher_seed42.parquet"
OPENRESEARCHER_PARQUET_LOCAL = Path(__file__).parent / "openresearcher_seed42.parquet"

# Try production path first, fall back to local
if OPENRESEARCHER_PARQUET_PROD.exists():
    OPENRESEARCHER_PARQUET = OPENRESEARCHER_PARQUET_PROD
else:
    OPENRESEARCHER_PARQUET = OPENRESEARCHER_PARQUET_LOCAL
