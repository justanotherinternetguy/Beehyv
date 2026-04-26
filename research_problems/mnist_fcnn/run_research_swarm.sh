#!/usr/bin/env bash
set -euo pipefail

# Activate the project virtualenv so torchvision and other deps are available
# even when spawned by a non-interactive process (e.g. the Bun web server).
VENV_PYTHON="$HOME/.pyenv/versions/lahacks1-3-12-13/bin/python"
if [ ! -x "$VENV_PYTHON" ]; then
  # Fall back to whatever python is on PATH
  VENV_PYTHON="python"
fi

mkdir -p logs
"$VENV_PYTHON" ../../run.py research . \
  --papers \
    ../../data/cleaned_json/improving_classification_neural_networks_by_using_absolute_activation_function_mnistlenet-5_example_cleaned.json \
    ../../data/cleaned_json/an_introduction_to_convolutional_neural_networks_cleaned.json \
    ../../data/cleaned_json/attention_is_all_you_need_cleaned.json \
    ../../data/cleaned_json/fast_kv_compaction_via_attention_matching_cleaned.json \
  --run-command "$VENV_PYTHON train.py --download --metrics-out logs/latest_metrics.json --log-file logs/train_events.jsonl" \
  --metrics-file logs/latest_metrics.json \
  --metric test_accuracy \
  --editable model.py \
  --iterations 3 \
  --max-agents 4 \
  --max-cross-ideas 6 \
  --log-file logs/research_swarm_live.log \
  "$@"
