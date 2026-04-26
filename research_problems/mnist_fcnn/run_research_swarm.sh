#!/usr/bin/env bash
set -euo pipefail

mkdir -p logs
python ../../run.py research . \
  --papers \
    ../../data/cleaned_json/attention_is_all_you_need_cleaned.json \
    ../../data/cleaned_json/og_attention_cleaned.json \
    ../../data/cleaned_json/introcnn_cleaned.json \
  --run-command "python train.py --download --metrics-out logs/latest_metrics.json --log-file logs/train_events.jsonl" \
  --metrics-file logs/latest_metrics.json \
  --metric test_accuracy \
  --editable model.py \
  --iterations 3 \
  --max-agents 3 \
  --max-cross-ideas 6 \
  --log-file logs/research_swarm_live.log \
  "$@"
