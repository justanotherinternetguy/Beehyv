#!/usr/bin/env bash
# Download and extract Tiny-ImageNet-200 into data/.
# The val reorganization into ImageFolder layout happens automatically on first training run.
set -euo pipefail

DATA_DIR="$(dirname "$0")/data"
ZIP="$DATA_DIR/tiny-imagenet-200.zip"
DEST="$DATA_DIR/tiny-imagenet-200"

mkdir -p "$DATA_DIR"

if [ -d "$DEST" ]; then
  echo "Tiny-ImageNet already extracted at $DEST"
  exit 0
fi

if [ ! -f "$ZIP" ]; then
  echo "Downloading Tiny-ImageNet-200 (~236 MB)..."
  wget -q --show-progress -O "$ZIP" http://cs231n.stanford.edu/tiny-imagenet-200.zip
fi

echo "Extracting..."
unzip -q "$ZIP" -d "$DATA_DIR"
rm "$ZIP"
echo "Done. Dataset ready at $DEST"
