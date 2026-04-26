#!/usr/bin/env bash
set -euo pipefail

remote_base="${ASUS_GX10_REMOTE_BASE:-/home/asus/Beehyv_remote}"
remote_dir="${ASUS_GX10_REMOTE_DIR:-$remote_base/research_problems/imagenet_cnn}"
host="${ASUS_GX10_HOST:-asus@100.123.34.54}"

mkdir -p logs
python ../../run.py research . \
  --problem "Improve the Tiny-ImageNet CNN classifier while keeping the evaluation dataset, 64x64 RGB input pipeline, 200-class output space, and run command fixed." \
  --papers \
    ../../data/cleaned_json/introcnn_cleaned.json \
    ../../data/cleaned_json/vision_transformer_cleaned.json \
    ../../data/cleaned_json/attention_is_all_you_need_cleaned.json \
  --run-command "../../tools/run_remote_problem.sh --host $host --remote-dir $remote_dir -- python train.py --dataset tinyimagenet --data-dir data --metrics-out logs/latest_metrics.json --log-file logs/train_events.jsonl" \
  --metrics-file logs/latest_metrics.json \
  --metric test_accuracy \
  --editable model.py \
  --iterations 7 \
  --max-agents 3 \
  --max-cross-ideas 6 \
  --log-file logs/research_swarm_live.log \
  "$@"
