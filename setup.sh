#!/bin/bash
# Setup script for ComparativeAgingV3 repository
# This script installs SRtools package with requirements and downloads posterior data

set -e  # Exit on error

echo "=========================================="
echo "ComparativeAgingV3 Setup Script"
echo "=========================================="
echo ""

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "Step 1: Installing SRtools package and requirements..."
echo "------------------------------------------------------"
cd SRtools
pip install -r requirements.txt
pip install -e .
cd "$SCRIPT_DIR"
echo "✓ SRtools installed successfully"
echo ""

echo "Step 2: Downloading posterior distribution data..."
echo "------------------------------------------------------"
python download_posterior_data.py
echo ""

echo "=========================================="
echo "Setup completed successfully!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  - Review the README.md for information about the repository structure"
echo "  - Run notebooks in the Figures/ directory to reproduce figures"
echo "  - Check individual folder READMEs for specific analysis instructions"
echo ""

