#!/usr/bin/env python3
"""
Download all files from Zenodo dataset.

This script downloads all posterior distribution files from:
https://doi.org/10.5281/zenodo.17804233

Usage:
    python download_zenodo_data.py [--output-dir OUTPUT_DIR]
"""

import os
import sys
import requests
import argparse
import shutil
from pathlib import Path

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


def get_zenodo_record(record_id):
    """Fetch record metadata from Zenodo API."""
    api_url = f"https://zenodo.org/api/records/{record_id}"
    response = requests.get(api_url)
    response.raise_for_status()
    return response.json()


def download_file(url, output_path, chunk_size=8192):
    """Download a file with progress bar."""
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    
    if HAS_TQDM and total_size > 0:
        # Use tqdm for progress bar
        with open(output_path, 'wb') as f, tqdm(
            desc=output_path.name,
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
    else:
        # Simple download without progress bar
        downloaded = 0
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0 and not HAS_TQDM:
                        percent = (downloaded / total_size) * 100
                        print(f"\r  Progress: {percent:.1f}%", end='', flush=True)
        if total_size > 0 and not HAS_TQDM:
            print()  # New line after progress


def main():
    parser = argparse.ArgumentParser(
        description='Download all files from Zenodo dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
    python download_posterior_data.py
        """
    )
    parser.add_argument(
        '--record-id',
        type=str,
        default='17804233',
        help='Zenodo record ID (default: 17804233)'
    )
    parser.add_argument(
        '--skip-copy',
        action='store_true',
        help='Skip copying files to Baysian03/analysis/products_baysian/posteriors'
    )
    
    args = parser.parse_args()
    
    # Determine script location to set relative paths
    script_dir = Path(__file__).parent.absolute()
    
    # Primary download location
    output_dir = script_dir / 'random_posterior_sampling_analysis' / 'posteriors'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Secondary location for copying
    copy_dir = script_dir / 'Baysian03' / 'analysis' / 'products_baysian' / 'posteriors'
    if not args.skip_copy:
        copy_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Fetching record metadata from Zenodo (record ID: {args.record_id})...")
    try:
        record = get_zenodo_record(args.record_id)
    except requests.exceptions.RequestException as e:
        print(f"Error fetching record: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Extract file information
    files = record.get('files', [])
    if not files:
        print("No files found in the record.", file=sys.stderr)
        sys.exit(1)
    
    print(f"\nFound {len(files)} files to download")
    total_size_gb = sum(f.get('size', 0) for f in files) / (1024**3)
    print(f"Total size: {total_size_gb:.2f} GB")
    print(f"Downloading to: {output_dir.absolute()}")
    if not args.skip_copy:
        print(f"Copying to: {copy_dir.absolute()}")
    if not HAS_TQDM:
        print("Note: Install 'tqdm' for progress bars: pip install tqdm")
    print()
    
    # Download each file
    downloaded = 0
    skipped = 0
    failed = 0
    
    for file_info in files:
        filename = file_info['key']
        file_url = file_info['links']['self']
        file_size = file_info.get('size', 0)
        output_path = output_dir / filename
        
        # Skip if file already exists and has correct size
        file_already_exists = False
        if output_path.exists():
            if file_size > 0 and output_path.stat().st_size == file_size:
                print(f"Skipping {filename} (already exists)")
                skipped += 1
                file_already_exists = True
                # Still need to copy if not skipping copy
                if not args.skip_copy:
                    copy_path = copy_dir / filename
                    if not (copy_path.exists() and file_size > 0 and copy_path.stat().st_size == file_size):
                        shutil.copy2(output_path, copy_path)
                        print(f"  ✓ Copied existing file to secondary location")
                    else:
                        print(f"  File already exists in secondary location")
                continue
        
        try:
            print(f"\nDownloading: {filename} ({file_size / (1024**2):.2f} MB)")
            download_file(file_url, output_path)
            downloaded += 1
            print(f"✓ Successfully downloaded {filename}")
            
            # Copy to secondary location if not skipping
            if not args.skip_copy:
                copy_path = copy_dir / filename
                # Check if copy already exists with correct size
                if copy_path.exists() and file_size > 0 and copy_path.stat().st_size == file_size:
                    print(f"  Skipping copy (already exists in secondary location)")
                else:
                    shutil.copy2(output_path, copy_path)
                    print(f"  ✓ Copied to secondary location")
        except Exception as e:
            print(f"✗ Error downloading {filename}: {e}", file=sys.stderr)
            failed += 1
            # Remove partial file if it exists
            if output_path.exists():
                output_path.unlink()
    
    # Summary
    print("\n" + "="*60)
    print("Download Summary:")
    print(f"  Successfully downloaded: {downloaded}")
    print(f"  Skipped (already exists): {skipped}")
    print(f"  Failed: {failed}")
    print(f"  Total: {len(files)}")
    print(f"\nFiles saved to:")
    print(f"  Primary location: {output_dir.absolute()}")
    if not args.skip_copy:
        print(f"  Secondary location: {copy_dir.absolute()}")
    print("="*60)


if __name__ == '__main__':
    main()

