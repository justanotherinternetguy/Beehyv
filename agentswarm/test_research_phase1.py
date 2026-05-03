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
# Ensure ``import agentswarm`` works whether the file is run via pytest from
# the repo root or directly via ``python agentswarm/test_research_phase1.py``.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# ── Helpers ────────────────────────────────────────────────────────────────────


def _stub_paper2code_utils() -> None:
    """Install a stub `utils` module so paper2code scripts import cleanly."""
    utils = types.ModuleType("utils")
    utils.read_python_files = lambda *a, **k: {}
    utils.content_to_json = lambda *a, **k: {"Task list": []}
    utils.extract_planning = lambda *a, **k: ["", "", ""]
    utils.get_llm_client_and_model = lambda *a, **k: (None, "fake/model")
    sys.modules["utils"] = utils


# ── B1: real consensus extraction in SwarmOrchestrator.synthesize ────────────


def _make_claim(claim_id: str, paper_id: str, text: str):
    """Build a minimal Claim for synthesize tests."""
    from agentswarm.blackboard import Claim, Evidence

    ev = Evidence(
        paper_id=paper_id, paper_title=paper_id, chunk_id="c0",
        section="abstract", sec_num=None, text="ev", score=1.0,
    )
    return Claim(
        claim_id=claim_id, agent_id=f"expert:{paper_id}",
        paper_id=paper_id, text=text, evidence=[ev],
    )


def test_b1_consensus_clusters_overlapping_claims() -> None:
    """Two claims with shared tokens form a consensus point; the third is a singleton."""
    from agentswarm.blackboard import Blackboard
    from agentswarm.orchestrator import SwarmOrchestrator

    bb = Blackboard(question="how do transformers handle long context?")
    bb.add_claim(_make_claim("c1", "P1",
        "Transformers use attention with positional encoding to handle long context."))
    bb.add_claim(_make_claim("c2", "P2",
        "Transformers use attention with positional encoding for long context tasks."))
    bb.add_claim(_make_claim("c3", "P3",
        "Quaternion algebras over number fields parametrize abelian varieties."))

    orch = SwarmOrchestrator(agents=[object()])
    syn = orch.synthesize(bb)

    assert len(syn.consensus) == 1, f"expected 1 consensus point, got {syn.consensus}"
    assert "P1" in syn.consensus[0] and "P2" in syn.consensus[0], (
        f"consensus should name both clustered papers: {syn.consensus[0]!r}"
    )
    assert any("P3" in d for d in syn.disagreements), (
        f"P3 should appear in disagreements: {syn.disagreements}"
    )


def test_b1_no_consensus_when_all_claims_distinct() -> None:
    """When no two claims overlap, consensus list says so explicitly."""
    from agentswarm.blackboard import Blackboard
    from agentswarm.orchestrator import SwarmOrchestrator

    bb = Blackboard(question="q")
    bb.add_claim(_make_claim("c1", "P1", "Quaternion algebras parametrize abelian varieties."))
    bb.add_claim(_make_claim("c2", "P2", "Photonic crystals exhibit topological band gaps."))
    bb.add_claim(_make_claim("c3", "P3", "Insect pollinator decline correlates with neonicotinoids."))

    orch = SwarmOrchestrator(agents=[object()])
    syn = orch.synthesize(bb)
    assert len(syn.consensus) == 1
    assert "no cross-paper consensus" in syn.consensus[0].lower()


def test_b1_synthesis_does_not_emit_old_boilerplate() -> None:
    """The pre-fix hardcoded consensus strings must not appear anymore."""
    source = (REPO_ROOT / "agentswarm" / "orchestrator.py").read_text(encoding="utf-8")
    assert "constrained to claims retrieved from each expert" not in source, (
        "old hardcoded consensus boilerplate still present"
    )
    assert "stronger keyword overlap in the paper text" not in source, (
        "old hardcoded consensus boilerplate still present"
    )


# ── B4: confidence field is gone from Claim ──────────────────────────────────


def test_b4_claim_has_no_confidence_field() -> None:
    """The ``confidence`` field on Claim is removed in Phase 1."""
    import dataclasses
    from agentswarm.blackboard import Claim
    fields = {f.name for f in dataclasses.fields(Claim)}
    assert "confidence" not in fields, (
        f"Claim still exposes confidence field: {sorted(fields)!r}"
    )


def test_b4_confidence_helper_removed_from_expert_module() -> None:
    """The theatrical ``_confidence_from_evidence`` helper is deleted."""
    import agentswarm.expert as expert_mod
    assert not hasattr(expert_mod, "_confidence_from_evidence"), (
        "_confidence_from_evidence should no longer be importable"
    )


def test_b4_expert_answer_does_not_set_confidence_kwarg() -> None:
    """Source-level guard: PaperExpertAgent.answer must not pass confidence=."""
    source = (REPO_ROOT / "agentswarm" / "expert.py").read_text(encoding="utf-8")
    assert "confidence=" not in source, (
        "expert.py still constructs Claim(confidence=...)"
    )


def test_b4_orchestrator_synthesis_does_not_print_confidence() -> None:
    """Source-level guard: orchestrator must not stringify claim.confidence."""
    source = (REPO_ROOT / "agentswarm" / "orchestrator.py").read_text(encoding="utf-8")
    assert "claim.confidence" not in source, (
        "orchestrator.py still references claim.confidence"
    )


# ── B3: stance classifier in PaperExpertAgent.critique ───────────────────────


class _StanceLLM:
    """Minimal LLM stub: returns a queued response per .complete() call."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []
        self.model = "fake/stance-llm"

    def complete(self, messages):
        self.calls.append(messages)
        if not self.responses:
            return ""
        return self.responses.pop(0)


def _make_expert_with_llm(llm):
    """Instantiate PaperExpertAgent against a fake paper and the given LLM."""
    from agentswarm.expert import PaperExpertAgent
    from agentswarm.paper_loader import Paper, PaperChunk
    from agentswarm.retriever import KeywordRetriever

    chunk = PaperChunk(
        chunk_id="c0", paper_id="P1", paper_title="P1",
        section="abstract", sec_num=None,
        text=(
            "Attention scales quadratically with sequence length; sparse "
            "attention reduces this. The paper analyses claim text directly."
        ),
        source="abstract",
    )
    paper = Paper(
        paper_id="P1", title="Test Paper",
        abstract="A short abstract for the test fixture.",
        path="(in-memory)", chunks=[chunk],
    )
    retriever = KeywordRetriever([paper])
    return PaperExpertAgent(paper, retriever, llm=llm)


def test_b3_stance_returns_valid_classification() -> None:
    """When the classifier returns a valid token, Critique.stance reflects it."""
    # Two complete() calls happen: (1) compose_critique, (2) classify_stance.
    llm = _StanceLLM(["My paper supports this claim because ...", "support"])
    agent = _make_expert_with_llm(llm)
    target = _make_claim("c1", "P0", "Attention scales with sequence length squared.")
    critique = agent.critique("how does attention scale?", target)
    assert critique.stance == "support", (
        f"expected 'support', got {critique.stance!r}; calls={len(llm.calls)}"
    )


def test_b3_stance_handles_messy_llm_output() -> None:
    """Trailing punctuation / extra prose is tolerated; only the first token wins."""
    llm = _StanceLLM(["composed critique text", "Rebut. (this directly contradicts the claim)"])
    agent = _make_expert_with_llm(llm)
    target = _make_claim("c1", "P0", "Sparse attention always beats dense.")
    critique = agent.critique("q", target)
    assert critique.stance == "rebut"


def test_b3_stance_falls_back_on_invalid_response() -> None:
    """Unrecognized tokens map to ``cannot_assess`` rather than leaking through."""
    llm = _StanceLLM(["composed critique text", "definitely-maybe-supportive-ish"])
    agent = _make_expert_with_llm(llm)
    target = _make_claim("c1", "P0", "Some claim.")
    critique = agent.critique("q", target)
    assert critique.stance == "cannot_assess"


def test_b3_stance_falls_back_on_llm_error() -> None:
    """LLM exceptions during classification never escalate."""
    class BoomLLM:
        model = "fake/boom"
        def __init__(self):
            self.calls = 0
        def complete(self, messages):  # noqa: ARG002
            self.calls += 1
            if self.calls == 1:
                return "composed critique text"
            raise RuntimeError("503 from upstream")

    agent = _make_expert_with_llm(BoomLLM())
    target = _make_claim("c1", "P0", "Some claim.")
    critique = agent.critique("q", target)
    assert critique.stance == "cannot_assess"


def test_b3_stance_field_no_longer_hardcoded_context() -> None:
    """The pre-fix hardcoded ``stance = "context"`` line must be gone."""
    source = (REPO_ROOT / "agentswarm" / "expert.py").read_text(encoding="utf-8")
    assert 'stance = "context"' not in source, (
        "hardcoded stance='context' still present in expert.py"
    )


# ── §7 row 4: judge LLM cannot override numeric decision ─────────────────────


def test_judge_decision_set_by_numeric_not_llm() -> None:
    """A bogus LLM response cannot flip the numeric keep/revert decision."""
    from agentswarm.research import (
        CodingResult, ExperimentResult, ResearchPlan,
    )

    class TrickyJudgeLLM:
        """Tries to override the decision via its prose output."""
        model = "fake/tricky-judge"
        def complete(self, messages):  # noqa: ARG002
            return "DECISION: revert\nVERDICT: this was awful, undo everything."

    orch = _make_orchestrator(judge_llm=TrickyJudgeLLM(), min_delta=0.001)

    def _result(iteration: int, label: str, value: float) -> ExperimentResult:
        return ExperimentResult(
            iteration=iteration, label=label, command=[], returncode=0,
            elapsed_s=0.0, metrics={"test_accuracy": value},
            stdout_path="", stderr_path="", metrics_path=None,
        )

    previous = _result(0, "baseline", 0.50)
    current = _result(1, "judge", 0.60)  # +0.10 → numeric "keep"
    plan = ResearchPlan(
        iteration=1, summary="t", target_files=["x.py"], steps=["s"],
        expected_effect="up", validation="t", raw="raw",
    )
    coding = CodingResult(
        iteration=1, applied=True, changed_files=["x.py"],
        raw_response_path="", notes="",
    )

    judge = orch._judge_iteration(
        iteration=1, previous=previous, current=current, plan=plan, coding=coding,
    )
    # Despite the LLM yelling "DECISION: revert", numeric wins.
    assert judge.decision == "keep", (
        f"LLM should not override numeric decision; got {judge.decision!r}"
    )
    # Prose feedback IS captured (the LLM call still runs — just for prescription).
    assert judge.raw, "judge prescription text should be captured"
    assert "DECISION:" in judge.raw or "VERDICT:" in judge.raw, (
        "test setup error: tricky LLM should have said its bogus verdict"
    )


def test_judge_system_prompt_forbids_decision_output() -> None:
    """The judge LLM is told explicitly NOT to emit a decision verdict."""
    source = (REPO_ROOT / "agentswarm" / "research.py").read_text(encoding="utf-8")
    # Hardening: prompt must call out that decision is already final / not the LLM's job.
    assert "next-step prescription agent" in source.lower(), (
        "judge system prompt should identify itself as a prescription agent, not a judge"
    )
    assert "DO NOT output" in source, (
        "judge prompt should explicitly forbid the LLM from outputting a verdict"
    )


# ── §7 row 3: --patience plateau circuit breaker + --test-split ──────────────


def _make_orchestrator(**overrides):
    """Construct a ResearchSwarmOrchestrator with fake LLMs for unit tests."""
    from agentswarm.research import ResearchSwarmOrchestrator

    class FakeLLM:
        model = "fake/llm:test"
        def complete(self, messages):  # noqa: ARG002
            return "fake"

    defaults = dict(
        agents=[object()],
        problem_dir=Path("/tmp"),
        command=["true"],
        metrics_path="m.json",
        editable_files=["x.py"],
        planner_llm=FakeLLM(),
        coding_llm=FakeLLM(),
        problem_statement="t",
    )
    defaults.update(overrides)
    return ResearchSwarmOrchestrator(**defaults)


def test_patience_default_is_3() -> None:
    orch = _make_orchestrator()
    assert orch.patience == 3


def test_patience_arg_is_stored() -> None:
    orch = _make_orchestrator(patience=7)
    assert orch.patience == 7


def test_patience_must_be_positive() -> None:
    try:
        _make_orchestrator(patience=0)
    except ValueError:
        return
    raise AssertionError("expected ValueError for patience=0")


def test_improved_against_best_requires_min_delta() -> None:
    """Plateau detection: improvement must clear ``min_delta``."""
    from agentswarm.research import ExperimentResult

    orch = _make_orchestrator(min_delta=0.01)

    def _r(value: float) -> ExperimentResult:
        return ExperimentResult(
            iteration=0, label="t", command=[], returncode=0, elapsed_s=0.0,
            metrics={"test_accuracy": value},
            stdout_path="", stderr_path="", metrics_path=None,
        )

    # +0.005 < min_delta=0.01 → not improvement (plateau tick)
    assert orch._improved_against_best(_r(0.505), _r(0.500)) is False
    # +0.02 >= min_delta → improvement (plateau reset)
    assert orch._improved_against_best(_r(0.520), _r(0.500)) is True
    # incumbent missing, candidate present → improvement
    bad = ExperimentResult(
        iteration=0, label="t", command=[], returncode=0, elapsed_s=0.0,
        metrics={}, stdout_path="", stderr_path="", metrics_path=None,
    )
    assert orch._improved_against_best(_r(0.5), bad) is True
    # candidate missing → no improvement
    assert orch._improved_against_best(bad, _r(0.5)) is False


def test_test_split_flag_in_mnist_argparser() -> None:
    """research_problems/mnist_fcnn/train.py registers --test-split."""
    source = (REPO_ROOT / "research_problems" / "mnist_fcnn" / "train.py").read_text(encoding="utf-8")
    assert '"--test-split"' in source, "MNIST train.py missing --test-split flag"
    assert "_three_way_split" in source, "MNIST train.py missing 80/10/10 split helper"


def test_test_split_flag_in_imagenet_argparser() -> None:
    """research_problems/imagenet_cnn/train.py registers --test-split."""
    source = (REPO_ROOT / "research_problems" / "imagenet_cnn" / "train.py").read_text(encoding="utf-8")
    assert '"--test-split"' in source, "ImageNet train.py missing --test-split flag"
    assert "_three_way_split" in source, "ImageNet train.py missing 80/10/10 split helper"


# ── §7 row 2: distinct planner / coder / judge model resolution ──────────────


def test_resolve_planner_model_uses_env_var() -> None:
    """Planner model resolves OPENROUTER_PLANNER_MODEL when set."""
    import os
    from agentswarm import llm
    old = os.environ.get("OPENROUTER_PLANNER_MODEL")
    try:
        os.environ["OPENROUTER_PLANNER_MODEL"] = "test/planner:v1"
        assert llm.resolve_planner_model() == "test/planner:v1"
    finally:
        if old is None:
            os.environ.pop("OPENROUTER_PLANNER_MODEL", None)
        else:
            os.environ["OPENROUTER_PLANNER_MODEL"] = old


def test_resolve_planner_model_falls_back_to_openrouter_model() -> None:
    """Planner model falls back to OPENROUTER_MODEL when its env var is unset."""
    import os
    from agentswarm import llm
    old = os.environ.pop("OPENROUTER_PLANNER_MODEL", None)
    try:
        assert llm.resolve_planner_model() == llm.OPENROUTER_MODEL
    finally:
        if old is not None:
            os.environ["OPENROUTER_PLANNER_MODEL"] = old


def test_resolve_judge_model_falls_back_to_openrouter_model() -> None:
    """Judge model falls back to OPENROUTER_MODEL when its env var is unset."""
    import os
    from agentswarm import llm
    old = os.environ.pop("OPENROUTER_JUDGE_MODEL", None)
    try:
        assert llm.resolve_judge_model() == llm.OPENROUTER_MODEL
    finally:
        if old is not None:
            os.environ["OPENROUTER_JUDGE_MODEL"] = old


def test_env_example_planner_differs_from_judge() -> None:
    """The example .env keeps planner ≠ judge — the reward-hacking guard."""
    env_example = (REPO_ROOT / ".env.example").read_text(encoding="utf-8")

    def _value(key: str) -> str:
        for line in env_example.splitlines():
            line = line.strip()
            if line.startswith(f"{key}=") and not line.startswith("#"):
                return line.split("=", 1)[1]
        raise AssertionError(f"{key} not found in .env.example")

    planner = _value("OPENROUTER_PLANNER_MODEL")
    judge = _value("OPENROUTER_JUDGE_MODEL")
    assert planner, "OPENROUTER_PLANNER_MODEL has no value"
    assert judge, "OPENROUTER_JUDGE_MODEL has no value"
    assert planner != judge, (
        f".env.example uses identical planner and judge ({planner!r}); the guard "
        "against same-model reward hacking requires them to differ."
    )


def test_research_orchestrator_judge_llm_independent_of_planner() -> None:
    """ResearchSwarmOrchestrator wires judge_llm separately when provided."""
    from pathlib import Path
    from agentswarm.research import ResearchSwarmOrchestrator

    class FakeLLM:
        def __init__(self, name: str) -> None:
            self.model = name

        def complete(self, messages):  # noqa: ARG002
            return self.model

    planner = FakeLLM("planner-model")
    coder = FakeLLM("coder-model")
    judge = FakeLLM("judge-model")
    orch = ResearchSwarmOrchestrator(
        agents=[object()],
        problem_dir=Path("/tmp"),
        command=["true"],
        metrics_path="m.json",
        editable_files=["x.py"],
        planner_llm=planner,
        coding_llm=coder,
        judge_llm=judge,
        problem_statement="t",
    )
    assert orch.judge_llm is judge, "judge_llm should be the supplied instance"
    assert orch.planner_llm is planner
    assert orch.judge_llm is not orch.planner_llm

    # When judge_llm is omitted, falls back to planner_llm.
    orch2 = ResearchSwarmOrchestrator(
        agents=[object()],
        problem_dir=Path("/tmp"),
        command=["true"],
        metrics_path="m.json",
        editable_files=["x.py"],
        planner_llm=planner,
        coding_llm=coder,
        problem_statement="t",
    )
    assert orch2.judge_llm is planner, "judge_llm should fall back to planner_llm"


# ── B9: default model id has a verification-date comment ─────────────────────


def test_b9_llm_module_documents_default_model_verification() -> None:
    """B9 — agentswarm/llm.py module docstring records the catalog-check date."""
    source = (REPO_ROOT / "agentswarm" / "llm.py").read_text(encoding="utf-8")
    assert "nvidia/nemotron-3-super-120b-a12b:free" in source, (
        "default model id missing"
    )
    assert "2026-05-03" in source, (
        "verification date missing from llm.py module docstring"
    )


def test_b9_run_module_documents_default_model_verification() -> None:
    """B9 — run.py module docstring records the catalog-check date."""
    source = (REPO_ROOT / "run.py").read_text(encoding="utf-8")
    assert "2026-05-03" in source, (
        "verification date missing from run.py module docstring"
    )


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
