# MNIST Bad FCNN Research Problem

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

The research swarm edits `model.py` in place. To restore the intentionally bad
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

The swarm uses only these paper experts by default:

- `attention_is_all_you_need_cleaned.json`
- `og_attention_cleaned.json`
- `introcnn_cleaned.json`

The coding agent defaults to `nvidia/nemotron-3-super-120b-a12b:free` and can
replace only `model.py` unless you pass extra `--editable` files manually.

Artifacts are written under `logs/research_swarm/<timestamp>/`:

- `events.jsonl`: structured event stream
- `transcript.md`: readable event transcript
- `iteration_*/`: paper ideas, cross-pollination, plan, coding response, judge feedback
- stdout/stderr logs for every baseline and judge run
- `summary.json`: final session summary
