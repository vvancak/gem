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
