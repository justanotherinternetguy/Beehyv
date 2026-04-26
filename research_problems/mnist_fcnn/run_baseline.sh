#!/usr/bin/env bash
set -euo pipefail

mkdir -p logs
python train.py \
  --download \
  --metrics-out logs/baseline_metrics.json \
  --log-file logs/baseline_train_events.jsonl \
  "$@"
