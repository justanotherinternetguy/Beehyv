#!/usr/bin/env bash
set -euo pipefail

cp model_baseline_fcnn.py model.py
rm -rf __pycache__
echo "Reset model.py to the weak fully connected starting baseline."
