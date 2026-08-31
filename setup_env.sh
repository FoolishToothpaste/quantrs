#!/bin/bash

echo "==============================="
echo "  quantrs environment setup"
echo "==============================="

# ── Step 1: Install conda if not present ─────────────────────────────────────
if ! command -v conda &> /dev/null; then
    echo "conda not found. Installing Miniconda..."

    # Download Miniconda installer
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh

    # Run installer silently
    bash /tmp/miniconda.sh -b -p "$HOME/miniconda3"

    # Clean up installer
    rm /tmp/miniconda.sh

    # Add conda to PATH for this session
    export PATH="$HOME/miniconda3/bin:$PATH"

    # Initialise conda so it works in future sessions
    "$HOME/miniconda3/bin/conda" init bash

    echo "Miniconda installed."
else
    echo "conda already installed. Skipping."
fi

# Make sure conda is available in this session
export PATH="$HOME/miniconda3/bin:$PATH"

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