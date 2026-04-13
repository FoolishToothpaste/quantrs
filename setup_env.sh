#!/bin/bash

echo "==============================="
echo "  quantrs environment setup"
echo "==============================="

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo ""
    echo "ERROR: conda is not installed."
    echo "Please install Miniconda from https://docs.conda.io/en/latest/miniconda.html"
    echo "Then re-run this script."
    exit 1
fi

# Create the environment
echo "Creating conda environment 'quantrs' with Python 3.10..."
conda create -n quantrs python=3.10 -y

# Activate and install
echo "Installing quantrs..."
source activate quantrs 2>/dev/null || conda activate quantrs

pip install -e ".[dev]"

echo ""
echo "==============================="
echo "  Setup complete!"
echo ""
echo "  To use quantrs, run:"
echo "    conda activate quantrs"
echo "    python"
echo "==============================="