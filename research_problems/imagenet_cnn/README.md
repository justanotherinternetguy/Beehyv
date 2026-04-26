# Tiny-ImageNet Baseline CNN Research Problem

This folder is a target for the agent swarm research loop. It starts with
`model.py`, a deliberately weak CNN for Tiny-ImageNet-200 classification:

- two tiny conv layers (8 and 16 channels)
- sigmoid activation throughout
- 85% spatial dropout after every conv block
- no batch normalization
- no momentum, very low learning rate SGD (1e-5)
- one training epoch on a 5000-sample Tiny-ImageNet training subset
- 64x64 RGB inputs and 200 output classes

The primary metric is `test_accuracy` (top-1). The script also tracks
`test_top5_accuracy`. All metrics are written to `logs/latest_metrics.json`.

## Dataset

Tiny-ImageNet requires a one-time download (~236 MB). The training script
defaults to `--dataset tinyimagenet`, expecting the dataset at
`data/tiny-imagenet-200/`.

To prepare the local dataset:

```bash
./setup_data.sh
```

For an offline smoke test, use fake 64x64 RGB tensors with 200 classes:

```bash
./run_baseline.sh --dataset fake
```

## Reset

The research swarm edits `model.py` in place. To restore the weak starting baseline:

```bash
./reset_baseline.sh
```

## Baseline

```bash
./run_baseline.sh
```

For a quick smoke test with fake data (default):

```bash
./run_baseline.sh --limit-train 256 --limit-test 64
```

## Research Swarm

Set `OPENROUTER_API_KEY`, then run:

```bash
./run_research_swarm.sh
```

To run the training/evaluation steps on the ASUS GX10 host while keeping the
agent planning and file edits local:

```bash
./run_research_swarm_asus.sh
```

The ASUS runner syncs this problem folder to
`/home/asus/Beehyv_remote/research_problems/imagenet_cnn`, runs `train.py` over
SSH, then copies `logs/latest_metrics.json` back so the judge can score the
iteration. It uses `asus@100.123.34.54` by default. Override with
`ASUS_GX10_HOST`, `ASUS_GX10_PORT`, `ASUS_GX10_REMOTE_BASE`, or
`ASUS_GX10_REMOTE_DIR`. The wrapper bypasses the local SSH config by default
with `-F /dev/null`; set `ASUS_GX10_SSH_CONFIG` if you need a specific config.

The wrapper deliberately excludes `data/` during sync, so Tiny-ImageNet must
already exist on the ASUS host at:

```text
/home/asus/Beehyv_remote/research_problems/imagenet_cnn/data/tiny-imagenet-200
```

Prefer SSH keys. If password auth is required, install `sshpass` locally and
export `ASUS_GX10_SSH_PASS` in your shell; do not store the password in this
repo.

The default remote Python is
`/home/asus/Beehyv_remote/venv/bin/python`, which should have PyTorch and
torchvision installed. Set `ASUS_GX10_PYTHON` if you want a different remote
environment.

The swarm uses these papers by default:

- `introcnn_cleaned.json` — foundational CNN architectures
- `vision_transformer_cleaned.json` — ViT attention-based vision
- `attention_is_all_you_need_cleaned.json` — transformer architecture

Before each paper-idea round, the orchestration agent inspects the current
`model.py`, `train.py`, run command, metrics, and Tiny-ImageNet setup for
obvious shape or setup bugs. The swarm launchers pass an explicit problem
statement so agents keep the target fixed at 64x64 RGB Tiny-ImageNet with 200
classes. If a proposed model fails during the judge retraining run, a debugging
agent gets a bounded chance to patch the editable files and rerun the same
command; the loop stops instead of advancing if it still cannot produce
`test_accuracy`.

Artifacts are written under `logs/research_swarm/<timestamp>/`:

- `events.jsonl`: structured event stream
- `transcript.md`: readable event transcript
- `iteration_*/`: diagnosis, seed ideas, cross-pollination, plan, coding response, debugging attempts, judge feedback
- `summary.json`: final session summary
