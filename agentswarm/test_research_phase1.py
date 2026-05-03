"""Phase-1 stop-the-bleeding tests.

Each test maps to an audit ID in docs/ARCHITECTURE_AUDIT.md.

Designed to run two ways:
  - `pytest agentswarm/test_research_phase1.py` (when pytest is installed)
  - `python agentswarm/test_research_phase1.py`   (stdlib-only; exits non-zero
                                                   if any test fails)

Tests stub heavy third-party deps (openai, torch, etc.) so the suite works in
environments where requirements.txt has not been installed.
"""

from __future__ import annotations

import contextlib
import io
import runpy
import sys
import tempfile
import types
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


# ── Helpers ────────────────────────────────────────────────────────────────────


def _stub_paper2code_utils() -> None:
    """Install a stub `utils` module so paper2code scripts import cleanly."""
    utils = types.ModuleType("utils")
    utils.read_python_files = lambda *a, **k: {}
    utils.content_to_json = lambda *a, **k: {"Task list": []}
    utils.extract_planning = lambda *a, **k: ["", "", ""]
    utils.get_llm_client_and_model = lambda *a, **k: (None, "fake/model")
    sys.modules["utils"] = utils


# ── B8: paper2code/codes/eval.py must use the correct "score_lst" key ─────────


def test_b8_eval_uses_score_lst_not_misspelling() -> None:
    """B8 — eval.py emits the correct ``score_lst`` key, not the misspelling.

    The misspelling literal is split below so a repo-wide ``rg`` for the bad
    key only hits the historical audit document, not this test.
    """
    bad_key = "scr" + "oe_lst"
    good_key = '"score_lst"'
    source = (REPO_ROOT / "paper2code" / "codes" / "eval.py").read_text(encoding="utf-8")
    assert bad_key not in source, (
        f"eval.py still contains the misspelled key {bad_key!r}"
    )
    assert good_key in source, (
        f"eval.py does not emit the {good_key} key"
    )


# ── B7: paper2code/codes/4_debugging.py must accept --output_repo_dir ──────────


def test_b7_debugging_accepts_output_repo_dir() -> None:
    """B7 — argparser registers --output_repo_dir; script reaches planning check."""
    _stub_paper2code_utils()
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        err = tmp_path / "err.txt"
        err.write_text("err")
        out = tmp_path / "out"
        out.mkdir()
        repo = tmp_path / "repo"

        old_argv = sys.argv
        sys.argv = [
            "4_debugging.py",
            "--error_file_name", str(err),
            "--output_dir", str(out),
            "--paper_name", "test",
            "--output_repo_dir", str(repo),
            "--save_num", "1",
        ]
        try:
            with contextlib.redirect_stderr(io.StringIO()):
                try:
                    runpy.run_path(
                        str(REPO_ROOT / "paper2code" / "codes" / "4_debugging.py"),
                        run_name="__main__",
                    )
                    raise AssertionError("script should have exited at planning check")
                except SystemExit as exc:
                    # exit 1 = planning_trajectories.json missing (the path the
                    # fix unblocks). exit 2 = argparse rejected --output_repo_dir
                    # (the bug). Anything else is unexpected.
                    assert exc.code == 1, f"expected SystemExit(1), got {exc.code!r}"
        finally:
            sys.argv = old_argv


# ── Direct-run fallback (no pytest required) ──────────────────────────────────

if __name__ == "__main__":
    failed = 0
    tests = sorted(
        (name, obj) for name, obj in globals().items()
        if name.startswith("test_") and callable(obj)
    )
    for name, fn in tests:
        try:
            fn()
            print(f"✅ {name}")
        except Exception as exc:  # noqa: BLE001
            print(f"❌ {name}: {type(exc).__name__}: {exc}")
            failed += 1
    print(f"\n{len(tests) - failed}/{len(tests)} tests passed")
    sys.exit(0 if failed == 0 else 1)
