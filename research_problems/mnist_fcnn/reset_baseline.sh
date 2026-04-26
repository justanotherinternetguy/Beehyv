#!/usr/bin/env bash
set -euo pipefail

cp model_bad_fcnn.py model.py
rm -rf __pycache__
echo "Reset model.py to the intentionally bad fully connected baseline."
