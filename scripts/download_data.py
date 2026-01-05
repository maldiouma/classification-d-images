#!/usr/bin/env python3
"""
Download and prepare HAM10000 dataset from Kaggle.
Requires: kaggle CLI installed and API credentials configured.
"""


import os
import subprocess
import sys
from pathlib import Path
from prepare_data import prepare_ham10000


def download_ham10000(output_path="./data"):
    """
    Download HAM10000 dataset from Kaggle.

    Args:
        output_path: Directory to download dataset
    """

    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check if kaggle CLI is installed
    try:
        subprocess.run(['kaggle', '--version'], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Error: Kaggle CLI not installed or not in PATH")
        print("Install with: pip install kaggle")
        print("Then configure credentials: https://github.com/Kaggle/kaggle-api#api-credentials")
        sys.exit(1)

    # Check if API credentials are configured
    kaggle_config = Path.home() / '.kaggle' / 'kaggle.json'
    if not kaggle_config.exists():
        print("Error: Kaggle API credentials not found")
        print("Configure at: https://github.com/Kaggle/kaggle-api#api-credentials")
        sys.exit(1)

    print("Downloading HAM10000 dataset from Kaggle...")
    print("(This may take a few minutes depending on your connection)")

    try:
        # Download from Kaggle
        cmd = [
            'kaggle', 'datasets', 'download',
            '-d', 'kmader/skin-cancer-mnist-ham10000',
            '-p', str(output_dir),
            '--unzip'
        ]
        subprocess.run(cmd, check=True)
        print("✓ Download completed")

        # Verify dataset structure
        ham10000_dir = output_dir / 'HAM10000_images'
        if not ham10000_dir.exists():
            print(f"Error: Expected directory {ham10000_dir} not found")
            sys.exit(1)

        # Prepare dataset
        print("\nOrganizing dataset into train/val/test splits...")
        prepare_ham10000(str(ham10000_dir), str(output_dir))

        print("\n✓ Dataset preparation completed successfully!")
        print(f"\nDataset structure created in {output_dir}:")
        print(f"  ├── train/")
        print(f"  ├── val/")
        print(f"  └── test/")

    except subprocess.CalledProcessError as e:
        print(f"Error downloading dataset: {e}")
        sys.exit(1)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download and prepare HAM10000 dataset")
    parser.add_argument('--output-path', type=str, default='./data',
                       help='Output directory for dataset')

    args = parser.parse_args()

    download_ham10000(args.output_path)
