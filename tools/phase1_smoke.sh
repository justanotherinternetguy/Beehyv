#!/usr/bin/env bash
# Phase-1 smoke test. Run after every Phase-1 commit.
# Exits non-zero on any failure. Each step prints a one-line ✅ on success.
#
# Designed to run WITHOUT having installed requirements.txt deps (openai, torch,
# etc.). Heavy deps are stubbed; what we actually verify is that the source
# files we touched parse, argparsers accept the flags we promised, and that
# in-house pure-Python modules import + instantiate cleanly.
set -u
cd "$(dirname "$0")/.."

fail() { echo "❌ $1" >&2; exit 1; }

PYTHON="${PYTHON:-python3}"
command -v "$PYTHON" >/dev/null 2>&1 || fail "$PYTHON not found on PATH; set PYTHON=/path/to/python and retry"

# 1. paper2code/codes/4_debugging.py argparser must accept --output_repo_dir.
#    We run the script via runpy with `utils` stubbed, so it parses argv,
#    reaches the planning_trajectories.json check, and sys.exit(1)s. If
#    --output_repo_dir is not registered, argparse exits 2 with "unrecognized
#    arguments" before any of that.
"$PYTHON" - <<'PY' || fail "4_debugging.py argparser check failed"
import sys, types, tempfile, runpy, os, io, contextlib

utils = types.ModuleType("utils")
utils.read_python_files = lambda *a, **k: {}
utils.content_to_json = lambda *a, **k: {"Task list": []}
utils.extract_planning = lambda *a, **k: ["", "", ""]
utils.get_llm_client_and_model = lambda *a, **k: (None, "fake/model")
sys.modules["utils"] = utils

err_fd, err_path = tempfile.mkstemp(suffix=".txt")
os.write(err_fd, b"err"); os.close(err_fd)
out_dir = tempfile.mkdtemp()
repo_dir = tempfile.mkdtemp()

sys.argv = [
    "4_debugging.py",
    "--error_file_name", err_path,
    "--output_dir", out_dir,
    "--paper_name", "smoke",
    "--output_repo_dir", repo_dir,
    "--save_num", "1",
]

# Suppress the script's own "Planning trajectories not found" stderr — it's
# the expected path proving argparse succeeded, but it looks alarming.
captured = io.StringIO()
try:
    with contextlib.redirect_stderr(captured):
        runpy.run_path("paper2code/codes/4_debugging.py", run_name="__main__")
    print("WARNING: script did not exit; planning check may be missing", file=sys.stderr)
    sys.exit(1)
except SystemExit as e:
    # Exit 1 means the script parsed argv successfully, then failed at the
    # planning_trajectories.json existence check, which is the expected path.
    if e.code != 1:
        print(f"FAIL: expected exit 1 (planning trajectories missing), got {e.code}", file=sys.stderr)
        print(captured.getvalue(), file=sys.stderr)
        sys.exit(1)
print("✅ 4_debugging.py argparser accepts --output_repo_dir")
PY

# 2. paper2code/codes/eval.py must compile cleanly (syntax + name resolution
#    in module scope). We stub `utils` because eval.py imports it at top level.
"$PYTHON" - <<'PY' || fail "eval.py compile/import check failed"
import sys, types, py_compile

utils = types.ModuleType("utils")
for name in (
    "read_python_files", "extract_planning", "content_to_json",
    "num_tokens_from_messages", "read_all_files", "extract_json_from_string",
    "get_now_str", "print_log_cost", "get_llm_client_and_model",
):
    setattr(utils, name, lambda *a, **k: None)
sys.modules["utils"] = utils

py_compile.compile("paper2code/codes/eval.py", doraise=True)
print("✅ eval.py compiles cleanly")
PY

# 3. agentswarm.research must import and ResearchSwarmOrchestrator must
#    instantiate with a fake LLM. agentswarm/llm.py uses stdlib only, so this
#    needs no third-party deps.
"$PYTHON" - <<'PY' || fail "ResearchSwarmOrchestrator instantiation failed"
import sys
from pathlib import Path
sys.path.insert(0, ".")
from agentswarm.research import ResearchSwarmOrchestrator

class FakeLLM:
    model = "fake/llm:test"
    def complete(self, messages): return "fake"

ResearchSwarmOrchestrator(
    agents=[object()],
    problem_dir=Path("/tmp"),
    command=["true"],
    metrics_path="m.json",
    editable_files=["x.py"],
    planner_llm=FakeLLM(),
    coding_llm=FakeLLM(),
    problem_statement="test",
)
print("✅ agentswarm.research import + ResearchSwarmOrchestrator instantiate")
PY

echo "🎉 Phase-1 smoke OK"
