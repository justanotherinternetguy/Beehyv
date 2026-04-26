#!/usr/bin/env bash
set -euo pipefail

mkdir -p logs
python ../../run.py research . \
  --problem "Improve the Tiny-ImageNet CNN classifier while keeping the evaluation dataset, 64x64 RGB input pipeline, 200-class output space, and run command fixed." \
  --papers \
    ../../data/cleaned_json/introcnn_cleaned.json \
    ../../data/cleaned_json/vision_transformer_cleaned.json \
    ../../data/cleaned_json/attention_is_all_you_need_cleaned.json \
  --run-command "python train.py --dataset tinyimagenet --data-dir data --metrics-out logs/latest_metrics.json --log-file logs/train_events.jsonl" \
  --metrics-file logs/latest_metrics.json \
  --metric test_accuracy \
  --editable model.py \
  --iterations 3 \
  --max-agents 3 \
  --max-cross-ideas 6 \
  --log-file logs/research_swarm_live.log \
  "$@"
