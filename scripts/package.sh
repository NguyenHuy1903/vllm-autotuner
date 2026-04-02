#!/usr/bin/env bash
# Đóng gói release cho vLLM Auto-Tuner

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
RELEASE_DIR="${DIR}/build/vllm-autotuner"
TAR_FILE="${DIR}/build/vllm-autotuner-release.tar.gz"

echo "==> Creating release directory at $RELEASE_DIR"
rm -rf "$RELEASE_DIR"
mkdir -p "$RELEASE_DIR"

echo "==> Copying tracked files into release directory..."
rsync -avq --progress \
    --exclude='.git' \
    --exclude='.github' \
    --exclude='node_modules' \
    --exclude='.venv' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='frontend/node_modules' \
    --exclude='.lh' \
    --exclude='data/' \
    --exclude='build/' \
    "$DIR/" "$RELEASE_DIR/"

echo "==> Creating empty data dir in release..."
mkdir -p "$RELEASE_DIR/data"

echo "==> Compressing release to $TAR_FILE"
cd "${DIR}/build"
tar -czf "$TAR_FILE" "vllm-autotuner"

echo "==> Done. Release package is ready at:"
echo "    $TAR_FILE"
