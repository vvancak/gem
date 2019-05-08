#!/bin/bash

echo "[INFO]: Running initial install script"

# Dataset Directory
DS_DIR=${1:-"ds"}

# Output directory
OUT_DIR=${2:-"output"}

# === DS DIRS ===
echo "[INFO]: Creating dataset directories..."
if [ ! -d "${DS_DIR}" ]; then
    mkdir ${DS_DIR}
fi

# === OUT DIRS ===
echo "[INFO]: Creating output directories..."
if [ ! -d "${OUT_DIR}" ]; then
    mkdir ${OUT_DIR}
fi

# === RBM GIT ===
echo "[INFO]: Getting RBM GIT repository"
if [ ! -d "src/tfrbm" ]; then
    cd src
    git clone https://github.com/meownoid/tensorfow-rbm.git

    # The repository name has a typo so this should work both now and after they fix it
    # The main purpose is to have a fixed short directory name...
    ls | grep tensor | xargs -I {} mv {} tfrbm
fi
