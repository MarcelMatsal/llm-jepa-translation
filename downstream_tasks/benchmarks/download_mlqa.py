#!/usr/bin/env python3
"""
Download MLQA dataset before running evaluation.
Run this on the login node before submitting SLURM jobs.

Usage:
    python download_mlqa.py
"""
import os
import zipfile
import urllib.request
from pathlib import Path

MLQA_URL = "https://dl.fbaipublicfiles.com/MLQA/MLQA_V1.zip"
MLQA_CACHE_DIR = Path.home() / ".cache" / "mlqa"


def download_mlqa_data():
    """Download and extract MLQA data."""
    MLQA_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    
    mlqa_dir = MLQA_CACHE_DIR / "MLQA_V1"
    if mlqa_dir.exists():
        print(f"✓ MLQA data already exists at {mlqa_dir}")
        # Verify the data is complete
        test_dir = mlqa_dir / "test"
        if test_dir.exists():
            files = list(test_dir.glob("*.json"))
            print(f"  Found {len(files)} test files")
            return mlqa_dir
        else:
            print("  Warning: test directory missing, re-downloading...")
    
    zip_path = MLQA_CACHE_DIR / "MLQA_V1.zip"
    
    if not zip_path.exists():
        print(f"Downloading MLQA data from {MLQA_URL}...")
        print("This may take a few minutes...")
        urllib.request.urlretrieve(MLQA_URL, zip_path)
        print("✓ Download complete.")
    else:
        print(f"✓ Zip file already exists at {zip_path}")
    
    print("Extracting MLQA data...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(MLQA_CACHE_DIR)
    print(f"✓ MLQA data extracted to {mlqa_dir}")
    
    # Verify
    test_dir = mlqa_dir / "test"
    if test_dir.exists():
        files = list(test_dir.glob("*.json"))
        print(f"✓ Found {len(files)} test files")
    
    return mlqa_dir


if __name__ == "__main__":
    print("=" * 50)
    print("MLQA Dataset Downloader")
    print("=" * 50)
    print()
    
    try:
        mlqa_dir = download_mlqa_data()
        print()
        print("=" * 50)
        print("✓ MLQA data is ready!")
        print(f"  Location: {mlqa_dir}")
        print()
        print("You can now submit your evaluation job:")
        print("  sbatch downstream_tasks/benchmarks/run_mlqa_eval_only.sh ./results/mlqa_xlm-roberta-base")
        print("=" * 50)
    except Exception as e:
        print(f"✗ Error downloading MLQA: {e}")
        raise


