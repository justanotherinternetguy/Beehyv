import sys
from pathlib import Path

from agentswarm import KeywordRetriever, Paper, PaperChunk, PaperExpertAgent, ResearchSwarmOrchestrator
from agentswarm.research import _extract_file_replacements


class StubLLM:
    def complete(self, messages):
        system = messages[0]["content"]
        user = messages[-1]["content"]
        if system.startswith("You are the orchestration agent"):
            return """---ORCHESTRATION_DIAGNOSIS---
SUMMARY: The toy dataset emits scalar scores from model.py, so proposed changes must preserve SCORE.
DATASET_CONTEXT:
- train.py imports model.py and writes test_accuracy from model.SCORE.
ISSUES:
- No glaring issue beyond preserving the expected SCORE symbol.
SUGGESTIONS:
- Keep model.py importable and define SCORE as a numeric value.
---END---"""
        if system.startswith("You are the planning agent"):
            return """---RESEARCH_PLAN---
SUMMARY: Replace the intentionally bad baseline with a stronger simple model signal.
TARGET_FILES:
- model.py
STEPS:
- Increase the score produced by the toy model.
EXPECTED_EFFECT: test_accuracy should improve.
VALIDATION: Keep the change if test_accuracy increases.
---END---"""
        if system.startswith("You are the coding agent"):
            return """---FILE: model.py---
```python
SCORE = 0.6
```
---END FILE---"""
        if system.startswith("You are the judge agent"):
            return "Keep the change because the metric improved on the same command."
        if "MODEL_CROSS_IDEA" in user:
            return """---MODEL_CROSS_IDEA---
TEXT: Combine the seed idea with this paper's representation bias.
CONNECTION: The papers agree that better feature mixing can help.
CHANGES: Update model.py.
---END---"""
        return """---MODEL_IDEA---
TEXT: Improve the weak classifier with a more expressive representation.
RATIONALE: The paper evidence supports stronger feature transformations.
EXPECTED_EFFECT: Higher test accuracy.
CHANGES: Update model.py.
---END---"""


class LooseFileBlockStubLLM(StubLLM):
    def complete(self, messages):
        system = messages[0]["content"]
        if system.startswith("You are the coding agent"):
            return """--- model.py ---
```python
SCORE = 0.7
```
---END FILE---"""
        return super().complete(messages)


class DebuggingStubLLM(StubLLM):
    def complete(self, messages):
        system = messages[0]["content"]
        if system.startswith("You are the coding agent"):
            return """---FILE: model.py---
```python
raise RuntimeError("broken proposed model")
```
---END FILE---"""
        if system.startswith("You are the debugging agent"):
            return """---FILE: model.py---
```python
SCORE = 0.5
```
---END FILE---"""
        return super().complete(messages)


def test_research_swarm_runs_one_improvement_iteration(tmp_path):
    problem_dir = _write_toy_problem(tmp_path)
    papers = [_paper("attention"), _paper("og_attention"), _paper("introcnn")]
    retriever = KeywordRetriever(papers)
    llm = StubLLM()
    agents = [PaperExpertAgent(paper, retriever, llm=llm) for paper in papers]

    orchestrator = ResearchSwarmOrchestrator(
        agents,
        problem_dir=problem_dir,
        command=[sys.executable, "train.py", "--metrics-out", "metrics.json"],
        metrics_path="metrics.json",
        editable_files=["model.py"],
        planner_llm=llm,
        coding_llm=llm,
        problem_statement="Improve a weak MNIST classifier.",
        metric_name="test_accuracy",
        max_iterations=1,
        max_agents=3,
        max_cross_ideas=2,
        session_dir=tmp_path / "session",
    )

    session = orchestrator.run()

    assert session.baseline.metric_value("test_accuracy") == 0.1
    assert session.best_result.metric_value("test_accuracy") == 0.6
    assert session.iterations[0].judge is not None
    assert session.iterations[0].judge.decision == "keep"
    assert session.iterations[0].diagnosis is not None
    assert "SCORE" in session.iterations[0].diagnosis.raw
    assert "SCORE = 0.6" in (problem_dir / "model.py").read_text()
    assert (tmp_path / "session" / "events.jsonl").exists()
    assert (tmp_path / "session" / "summary.json").exists()


def test_research_swarm_accepts_common_loose_file_block_and_judges(tmp_path):
    problem_dir = _write_toy_problem(tmp_path)
    papers = [_paper("attention"), _paper("og_attention"), _paper("introcnn")]
    retriever = KeywordRetriever(papers)
    llm = LooseFileBlockStubLLM()
    agents = [PaperExpertAgent(paper, retriever, llm=llm) for paper in papers]

    orchestrator = ResearchSwarmOrchestrator(
        agents,
        problem_dir=problem_dir,
        command=[sys.executable, "train.py", "--metrics-out", "metrics.json"],
        metrics_path="metrics.json",
        editable_files=["model.py"],
        planner_llm=llm,
        coding_llm=llm,
        problem_statement="Improve a weak MNIST classifier.",
        metric_name="test_accuracy",
        max_iterations=1,
        max_agents=3,
        max_cross_ideas=2,
        session_dir=tmp_path / "session",
    )

    session = orchestrator.run()

    assert session.best_result.metric_value("test_accuracy") == 0.7
    assert session.iterations[0].coding.applied is True
    assert session.iterations[0].judge is not None
    assert session.iterations[0].judge.decision == "keep"
    assert "SCORE = 0.7" in (problem_dir / "model.py").read_text()


def test_research_swarm_debugs_failed_judge_run_before_finishing_iteration(tmp_path):
    problem_dir = _write_toy_problem(tmp_path)
    papers = [_paper("attention"), _paper("og_attention"), _paper("introcnn")]
    retriever = KeywordRetriever(papers)
    llm = DebuggingStubLLM()
    agents = [PaperExpertAgent(paper, retriever, llm=llm) for paper in papers]

    orchestrator = ResearchSwarmOrchestrator(
        agents,
        problem_dir=problem_dir,
        command=[sys.executable, "train.py", "--metrics-out", "metrics.json"],
        metrics_path="metrics.json",
        editable_files=["model.py"],
        planner_llm=llm,
        coding_llm=llm,
        debugging_llm=llm,
        problem_statement="Improve a weak MNIST classifier.",
        metric_name="test_accuracy",
        max_iterations=1,
        max_agents=3,
        max_cross_ideas=2,
        max_debug_attempts=2,
        session_dir=tmp_path / "session",
    )

    session = orchestrator.run()

    iteration = session.iterations[0]
    assert iteration.debugging
    assert iteration.debugging[0].applied is True
    assert iteration.result is not None
    assert iteration.result.returncode == 0
    assert iteration.result.metric_value("test_accuracy") == 0.5
    assert session.best_result.metric_value("test_accuracy") == 0.5
    assert "SCORE = 0.5" in (problem_dir / "model.py").read_text()


def test_extract_file_replacements_falls_back_to_single_editable_code_fence():
    raw = """Here is the complete replacement:
```python
SCORE = 0.8
```
"""

    assert _extract_file_replacements(raw, fallback_files=["model.py"]) == {"model.py": "SCORE = 0.8"}


def test_extract_file_replacements_does_not_guess_multi_file_fallback():
    raw = """```python
SCORE = 0.8
```"""

    assert _extract_file_replacements(raw, fallback_files=["model.py", "train.py"]) == {}


def _paper(paper_id: str) -> Paper:
    return Paper(
        paper_id=paper_id,
        title=f"{paper_id} paper",
        abstract="",
        path=Path(f"{paper_id}.json"),
        chunks=[
            PaperChunk(
                chunk_id=f"{paper_id}:0",
                paper_id=paper_id,
                paper_title=f"{paper_id} paper",
                section="Method",
                sec_num="1",
                text=(
                    "MNIST classification neural network architecture training "
                    "accuracy attention convolution feature representation"
                ),
                source="body",
            )
        ],
    )


def _write_toy_problem(tmp_path: Path) -> Path:
    problem_dir = tmp_path / "problem"
    problem_dir.mkdir()
    (problem_dir / "model.py").write_text("SCORE = 0.1\n", encoding="utf-8")
    (problem_dir / "train.py").write_text(
        """import argparse
import json
from pathlib import Path

import model

parser = argparse.ArgumentParser()
parser.add_argument("--metrics-out", default="metrics.json")
args = parser.parse_args()
metrics = {"test_accuracy": model.SCORE}
Path(args.metrics_out).write_text(json.dumps(metrics), encoding="utf-8")
print("METRICS_JSON:" + json.dumps(metrics))
""",
        encoding="utf-8",
    )
    return problem_dir
