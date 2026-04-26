# MNIST Baseline FCNN Research Problem

This folder is a small target for the agent swarm research loop. It starts with
`model.py`, a deliberately weak fully connected neural network for MNIST:

- one tiny hidden layer
- sigmoid activation
- 90% dropout
- low learning rate SGD
- one training epoch on a fixed subset

The evaluation command always writes metrics to `logs/latest_metrics.json` and
event logs to `logs/train_events.jsonl`, so the judge agent can compare each
iteration on the same dataset and subset.

## Reset

The research swarm edits `model.py` in place. To restore the weak starting
fully connected baseline before another baseline run:

```bash
./reset_baseline.sh
```

## Baseline

```bash
./run_baseline.sh
```

For an offline smoke test that does not download MNIST:

```bash
./run_baseline.sh --dataset fake
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
`/home/asus/Beehyv_remote/research_problems/mnist_fcnn`, runs `train.py` over
SSH, then copies `logs/latest_metrics.json` back so the judge can score the
iteration. It uses `asus@100.123.34.54` by default. Override with
`ASUS_GX10_HOST`, `ASUS_GX10_PORT`, `ASUS_GX10_REMOTE_BASE`, or
`ASUS_GX10_REMOTE_DIR`. The wrapper bypasses the local SSH config by default
with `-F /dev/null`; set `ASUS_GX10_SSH_CONFIG` if you need a specific config.

Prefer SSH keys. If password auth is required, install `sshpass` locally and
export `ASUS_GX10_SSH_PASS` in your shell; do not store the password in this
repo.

The default remote Python is
`/home/asus/Beehyv_remote/venv/bin/python`, which should have PyTorch and
torchvision installed. Set `ASUS_GX10_PYTHON` if you want a different remote
environment.

The swarm uses only these paper experts by default:

- `attention_is_all_you_need_cleaned.json`
- `og_attention_cleaned.json`
- `introcnn_cleaned.json`

The coding agent defaults to `nvidia/nemotron-3-super-120b-a12b:free` and can
replace only `model.py` unless you pass extra `--editable` files manually.

Before each paper-idea round, the orchestration agent also inspects the current
`model.py`, `train.py`, run command, metrics, and dataset setup for obvious
shape or setup bugs. If a proposed model fails during the judge retraining run,
a debugging agent gets a bounded chance to patch the editable files and rerun
the same command; the loop stops instead of advancing if it still cannot
produce `test_accuracy`.

Artifacts are written under `logs/research_swarm/<timestamp>/`:

- `events.jsonl`: structured event stream
- `transcript.md`: readable event transcript
- `iteration_*/`: diagnosis, paper ideas, cross-pollination, plan, coding response, debugging attempts, judge feedback
- stdout/stderr logs for every baseline and judge run
- `summary.json`: final session summary
