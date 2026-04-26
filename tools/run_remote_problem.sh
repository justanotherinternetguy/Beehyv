#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  run_remote_problem.sh [options] -- <training command...>

Runs the current research problem directory on a remote SSH host, then copies
logs back locally so the research swarm can read metrics.

Options:
  --host HOST           SSH host, default: $ASUS_GX10_HOST or asus@100.123.34.54
  --remote-dir DIR     Remote working directory, default: $ASUS_GX10_REMOTE_DIR
  --port PORT          SSH port, default: $ASUS_GX10_PORT or 22
  --python PYTHON      Remote python executable, default: $ASUS_GX10_PYTHON
                      or /home/asus/Beehyv_remote/venv/bin/python
  --ssh-config PATH    SSH config path, default: $ASUS_GX10_SSH_CONFIG or /dev/null
  --help               Show this help.

Authentication:
  Prefer SSH keys. If password auth is required, install sshpass locally and set
  ASUS_GX10_SSH_PASS in your shell. Do not commit passwords into this repo.
EOF
}

host="${ASUS_GX10_HOST:-asus@100.123.34.54}"
remote_dir="${ASUS_GX10_REMOTE_DIR:-}"
port="${ASUS_GX10_PORT:-22}"
remote_python="${ASUS_GX10_PYTHON:-/home/asus/Beehyv_remote/venv/bin/python}"
ssh_config="${ASUS_GX10_SSH_CONFIG:-/dev/null}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --host)
      host="${2:?missing value for --host}"
      shift 2
      ;;
    --remote-dir)
      remote_dir="${2:?missing value for --remote-dir}"
      shift 2
      ;;
    --port)
      port="${2:?missing value for --port}"
      shift 2
      ;;
    --python)
      remote_python="${2:?missing value for --python}"
      shift 2
      ;;
    --ssh-config)
      ssh_config="${2:?missing value for --ssh-config}"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    --)
      shift
      break
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ $# -eq 0 ]]; then
  echo "Missing training command after --" >&2
  usage >&2
  exit 2
fi

if [[ -z "$remote_dir" ]]; then
  remote_dir="/home/asus/Beehyv_remote/$(basename "$PWD")"
fi

training_args=("$@")
metrics_out="logs/latest_metrics.json"
for ((i = 0; i < ${#training_args[@]}; i++)); do
  case "${training_args[$i]}" in
    --metrics-out)
      if ((i + 1 < ${#training_args[@]})); then
        metrics_out="${training_args[$((i + 1))]}"
      fi
      ;;
    --metrics-out=*)
      metrics_out="${training_args[$i]#--metrics-out=}"
      ;;
  esac
done

if [[ -n "${ASUS_GX10_SSH_PASS:-}" ]] && ! command -v sshpass >/dev/null 2>&1; then
  cat >&2 <<'EOF'
ASUS_GX10_SSH_PASS is set, but sshpass is not installed.
Install sshpass, or set up SSH key authentication with:
  ssh-copy-id asus@100.123.34.54
EOF
  exit 127
fi

control_path="${TMPDIR:-/tmp}/beehyv-asus-%r@%h:%p"
ssh_opts=(
  -F "$ssh_config"
  -p "$port"
  -o StrictHostKeyChecking=accept-new
  -o ControlMaster=auto
  -o ControlPersist=10m
  -o ControlPath="$control_path"
)
rsync_ssh=(ssh "${ssh_opts[@]}")

ssh_cmd=(ssh "${ssh_opts[@]}" "$host")
rsync_cmd=(rsync -az --delete)
if [[ -n "${ASUS_GX10_SSH_PASS:-}" ]]; then
  export SSHPASS="$ASUS_GX10_SSH_PASS"
  ssh_cmd=(sshpass -e ssh "${ssh_opts[@]}" "$host")
  rsync_cmd=(sshpass -e rsync -az --delete)
fi

remote_dir_q="$(printf '%q' "$remote_dir")"
metrics_cleanup=":"
if [[ "$metrics_out" != /* && "$metrics_out" != *".."* ]]; then
  metrics_cleanup="rm -f $(printf '%q' "$metrics_out")"
fi

"${ssh_cmd[@]}" "mkdir -p $remote_dir_q/logs"

"${rsync_cmd[@]}" \
  -e "${rsync_ssh[*]}" \
  --exclude 'data/' \
  --exclude 'logs/research_swarm/' \
  --exclude '__pycache__/' \
  --exclude '*.pyc' \
  ./ "$host:$remote_dir/"

remote_command=()
for arg in "$@"; do
  if [[ "$arg" == "python" ]]; then
    remote_command+=("$remote_python")
  else
    remote_command+=("$arg")
  fi
done

remote_command_q=""
for arg in "${remote_command[@]}"; do
  remote_command_q+=" $(printf '%q' "$arg")"
done

set +e
"${ssh_cmd[@]}" "cd $remote_dir_q && $metrics_cleanup &&$remote_command_q"
status=$?
set -e

mkdir -p logs
"${rsync_cmd[@]}" \
  -e "${rsync_ssh[*]}" \
  --exclude 'research_swarm/' \
  "$host:$remote_dir/logs/" ./logs/

exit "$status"
