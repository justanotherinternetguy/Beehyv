#!/usr/bin/env bash
set -euo pipefail

remote_base="${ASUS_GX10_REMOTE_BASE:-/home/asus/Beehyv_remote}"
remote_dir="${ASUS_GX10_REMOTE_DIR:-$remote_base/research_problems/mnist_fcnn}"
host="${ASUS_GX10_HOST:-asus@100.123.34.54}"

mkdir -p logs
python ../../run.py research . \
  --papers \
    ../../data/cleaned_json/attention_is_all_you_need_cleaned.json \
    ../../data/cleaned_json/og_attention_cleaned.json \
    ../../data/cleaned_json/introcnn_cleaned.json \
  --run-command "../../tools/run_remote_problem.sh --host $host --remote-dir $remote_dir -- python train.py --download --metrics-out logs/latest_metrics.json --log-file logs/train_events.jsonl" \
  --metrics-file logs/latest_metrics.json \
  --metric test_accuracy \
  --editable model.py \
  --iterations 3 \
  --max-agents 3 \
  --max-cross-ideas 6 \
  --log-file logs/research_swarm_live.log \
  "$@"
