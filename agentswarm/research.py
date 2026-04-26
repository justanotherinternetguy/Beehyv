"""Autonomous research-loop orchestration for code experiments."""

from __future__ import annotations

import json
import os
import re
import subprocess
import time
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from .blackboard import Evidence
from .llm import OPENROUTER_MODEL, PaperExpertLLM


@dataclass(frozen=True)
class ExperimentResult:
    """One execution of the research problem's evaluation command."""

    iteration: int
    label: str
    command: list[str]
    returncode: int
    elapsed_s: float
    metrics: dict[str, Any]
    stdout_path: str
    stderr_path: str
    metrics_path: str | None

    @property
    def succeeded(self) -> bool:
        return self.returncode == 0

    def metric_value(self, name: str) -> float | None:
        value = self.metrics.get(name)
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None


@dataclass(frozen=True)
class ModelImprovementIdea:
    """A paper-grounded proposal for improving the current model."""

    idea_id: str
    agent_id: str
    paper_id: str
    paper_title: str
    text: str
    rationale: str
    expected_effect: str
    changes: str
    evidence: list[Evidence]

    @property
    def seed_label(self) -> str:
        return f"{self.paper_title} ({self.paper_id})"


@dataclass(frozen=True)
class CrossPollinatedModelIdea:
    """A proposal produced by combining one paper expert's idea with another paper."""

    idea_id: str
    agent_id: str
    paper_id: str
    paper_title: str
    seed_idea_id: str
    seed_paper_id: str
    seed_paper_title: str
    text: str
    connection: str
    changes: str
    evidence: list[Evidence]


@dataclass(frozen=True)
class ResearchPlan:
    """The planning agent's structured implementation plan."""

    iteration: int
    summary: str
    target_files: list[str]
    steps: list[str]
    expected_effect: str
    validation: str
    raw: str


@dataclass(frozen=True)
class OrchestrationDiagnosis:
    """Orchestrator inspection of the current solution and problem setup."""

    iteration: int
    summary: str
    issues: list[str]
    suggestions: list[str]
    dataset_context: list[str]
    raw: str


@dataclass(frozen=True)
class CodingResult:
    """Files changed by the coding agent for one iteration."""

    iteration: int
    applied: bool
    changed_files: list[str]
    raw_response_path: str
    notes: str


@dataclass(frozen=True)
class DebuggingResult:
    """Files changed by the debugging agent after a failed evaluation."""

    iteration: int
    attempt: int
    applied: bool
    changed_files: list[str]
    raw_response_path: str
    notes: str


@dataclass(frozen=True)
class JudgeFeedback:
    """Judge evaluation after rerunning the modified code."""

    iteration: int
    metric_name: str
    previous_value: float | None
    new_value: float | None
    delta: float | None
    decision: str
    feedback: str
    raw: str


@dataclass
class ResearchIteration:
    """Auditable state for one complete improvement attempt."""

    iteration: int
    seed_ideas: list[ModelImprovementIdea] = field(default_factory=list)
    cross_pollinated_ideas: list[CrossPollinatedModelIdea] = field(default_factory=list)
    diagnosis: OrchestrationDiagnosis | None = None
    plan: ResearchPlan | None = None
    coding: CodingResult | None = None
    debugging: list[DebuggingResult] = field(default_factory=list)
    result: ExperimentResult | None = None
    judge: JudgeFeedback | None = None


@dataclass
class ResearchSession:
    """Complete autoresearch-style session state."""

    problem_dir: str
    session_dir: str
    baseline: ExperimentResult
    iterations: list[ResearchIteration]
    best_result: ExperimentResult
    metric_name: str
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class ResearchEventLog:
    """Writes durable research-loop events and artifacts."""

    def __init__(self, session_dir: Path, logger=None) -> None:
        self.session_dir = session_dir
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.events_path = self.session_dir / "events.jsonl"
        self.transcript_path = self.session_dir / "transcript.md"
        self.logger = logger

    def event(self, event_type: str, payload: dict[str, Any]) -> None:
        record = {
            "time": datetime.now(timezone.utc).isoformat(),
            "event": event_type,
            "payload": _json_safe(payload),
        }
        with self.events_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, sort_keys=True) + "\n")
        with self.transcript_path.open("a", encoding="utf-8") as f:
            f.write(f"\n## {event_type}\n\n")
            f.write("```json\n")
            f.write(json.dumps(record["payload"], indent=2, sort_keys=True))
            f.write("\n```\n")

    def artifact(self, relative_path: str, text: str) -> str:
        path = self.session_dir / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        self.event("artifact_written", {"path": str(path)})
        return str(path)


class ExperimentRunner:
    """Runs the problem command and captures stdout, stderr, and metrics."""

    def __init__(
        self,
        *,
        problem_dir: Path,
        command: Sequence[str],
        metrics_path: str | None,
        event_log: ResearchEventLog,
    ) -> None:
        if not command:
            raise ValueError("research run command cannot be empty")
        self.problem_dir = problem_dir
        self.command = list(command)
        self.metrics_path = metrics_path
        self.event_log = event_log

    def run(self, *, iteration: int, label: str) -> ExperimentResult:
        stdout_path = self.event_log.session_dir / f"iteration_{iteration:02d}_{label}.stdout.log"
        stderr_path = self.event_log.session_dir / f"iteration_{iteration:02d}_{label}.stderr.log"
        metrics_abs = self.problem_dir / self.metrics_path if self.metrics_path else None
        if metrics_abs and metrics_abs.exists():
            metrics_abs.unlink()

        env = os.environ.copy()
        env["RESEARCH_SWARM_SESSION_DIR"] = str(self.event_log.session_dir)
        env["RESEARCH_SWARM_ITERATION"] = str(iteration)
        env["RESEARCH_SWARM_LABEL"] = label
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        env["PYTHONPYCACHEPREFIX"] = str(self.event_log.session_dir / "pycache" / f"{iteration:02d}_{label}")

        self.event_log.event(
            "experiment_start",
            {"iteration": iteration, "label": label, "command": self.command},
        )
        start = time.monotonic()
        proc = subprocess.run(
            self.command,
            cwd=self.problem_dir,
            env=env,
            capture_output=True,
            text=True,
        )
        elapsed = round(time.monotonic() - start, 3)
        stdout_path.write_text(proc.stdout, encoding="utf-8")
        stderr_path.write_text(proc.stderr, encoding="utf-8")

        metrics = _read_metrics(metrics_abs, proc.stdout)
        result = ExperimentResult(
            iteration=iteration,
            label=label,
            command=self.command,
            returncode=proc.returncode,
            elapsed_s=elapsed,
            metrics=metrics,
            stdout_path=str(stdout_path),
            stderr_path=str(stderr_path),
            metrics_path=str(metrics_abs) if metrics_abs else None,
        )
        self.event_log.event("experiment_done", asdict(result))
        return result


class FileSnapshot:
    """Restores editable files if an iteration regresses."""

    def __init__(self, problem_dir: Path, files: Sequence[str]) -> None:
        self.problem_dir = problem_dir
        self.contents: dict[str, str | None] = {}
        for rel in files:
            path = _safe_join(problem_dir, rel)
            self.contents[rel] = path.read_text(encoding="utf-8") if path.exists() else None

    def restore(self) -> None:
        for rel, content in self.contents.items():
            path = _safe_join(self.problem_dir, rel)
            if content is None:
                if path.exists():
                    path.unlink()
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")


class OrchestrationDiagnosticAgent:
    """Inspects the current solution against the dataset and run setup."""

    def __init__(
        self,
        *,
        llm: PaperExpertLLM,
        editable_files: Sequence[str],
        model_name: str = OPENROUTER_MODEL,
        logger=None,
        max_file_chars: int = 9000,
        max_context_chars: int = 12000,
    ) -> None:
        self.llm = llm
        self.editable_files = [_normalize_relpath(path) for path in editable_files]
        self.model_name = model_name
        self.logger = logger
        self.max_file_chars = max_file_chars
        self.max_context_chars = max_context_chars

    def diagnose(
        self,
        *,
        problem_dir: Path,
        event_log: ResearchEventLog,
        iteration: int,
        command: Sequence[str],
        metrics_path: str,
        current_result: ExperimentResult,
        problem_statement: str,
        feedback_history: list[JudgeFeedback],
    ) -> OrchestrationDiagnosis:
        messages = [
            {
                "role": "system",
                "content": (
                    f"You are the orchestration agent for an autonomous research loop. "
                    f"You are running as model {self.model_name}. "
                    "Before paper experts propose ideas, inspect the current solution code, "
                    "training/evaluation command, metrics, logs, and dataset setup. "
                    "Your job is to spot glaring bugs, shape mismatches, impossible assumptions, "
                    "or setup issues that would make otherwise good paper-inspired changes fail. "
                    "Be concrete and focus on implementation blockers, not broad research ideas."
                ),
            },
            {
                "role": "user",
                "content": "\n\n".join(
                    [
                        f"Problem: {problem_statement}",
                        f"Run command: {' '.join(command)}",
                        f"Metrics file: {metrics_path}",
                        f"Current metrics: {_format_metrics(current_result.metrics)}",
                        "Recent judge feedback:",
                        _format_feedback(feedback_history),
                        "Current editable solution files:",
                        _format_editable_files(problem_dir, self.editable_files, self.max_file_chars),
                        "Problem setup files and dataset-loading code:",
                        _format_problem_context(problem_dir, self.max_context_chars),
                        "Most recent evaluation logs:",
                        _format_result_logs(current_result, 6000),
                        (
                            "Return exactly this format:\n"
                            "---ORCHESTRATION_DIAGNOSIS---\n"
                            "SUMMARY: one paragraph\n"
                            "DATASET_CONTEXT:\n"
                            "- observed dataset shape/setup fact\n"
                            "ISSUES:\n"
                            "- concrete bug, mismatch, or likely blocker\n"
                            "SUGGESTIONS:\n"
                            "- direct instruction the paper/planning agents should account for\n"
                            "---END---"
                        ),
                    ]
                ),
            },
        ]
        raw = _complete(self.llm, messages, self.logger, "orchestration-agent", "diagnosing")
        event_log.artifact(f"iteration_{iteration:02d}/orchestration_diagnosis.txt", raw)
        diagnosis = _parse_diagnosis(raw, iteration=iteration)
        event_log.event("orchestration_diagnosis_done", {"iteration": iteration, "diagnosis": diagnosis})
        return diagnosis


class ResearchCodingAgent:
    """Coding agent that applies full-file replacements from an LLM response."""

    def __init__(
        self,
        *,
        llm: PaperExpertLLM,
        editable_files: Sequence[str],
        model_name: str = OPENROUTER_MODEL,
        logger=None,
        max_file_chars: int = 14000,
    ) -> None:
        if not editable_files:
            raise ValueError("ResearchCodingAgent needs at least one editable file")
        self.llm = llm
        self.editable_files = [_normalize_relpath(path) for path in editable_files]
        self.model_name = model_name
        self.logger = logger
        self.max_file_chars = max_file_chars

    def apply_plan(
        self,
        *,
        problem_dir: Path,
        event_log: ResearchEventLog,
        iteration: int,
        plan: ResearchPlan,
        current_result: ExperimentResult,
        problem_statement: str,
    ) -> CodingResult:
        file_context = _format_editable_files(problem_dir, self.editable_files, self.max_file_chars)
        editable_list = ", ".join(self.editable_files)
        example_path = self.editable_files[0]
        messages = [
            {
                "role": "system",
                "content": (
                    f"You are the coding agent for an autonomous research loop. "
                    f"You are running as model {self.model_name}. "
                    "Your ultimate goal is to improve the accuracy of the research model in focus. "
                    "Apply the planning agent's requested changes by replacing complete files. "
                    f"Only modify files explicitly listed as editable. Valid editable paths: {editable_list}. "
                    "STRICT OUTPUT CONTRACT: your response is parsed by a program. "
                    "The first non-whitespace characters of every replacement block must be exactly "
                    "---FILE: followed by the editable relative path and then ---. "
                    "Do not write markdown headings, explanations, diffs, JSON, bullets, or labels like "
                    "'--- model.py ---'. Any text outside valid file blocks is ignored. "
                    "Return one or more blocks exactly formatted as:\n"
                    f"---FILE: {example_path}---\n"
                    "```python\n"
                    "# complete file contents\n"
                    "```\n"
                    "---END FILE---\n"
                    "If no safe change is possible, return exactly: NO_SAFE_CHANGE"
                ),
            },
            {
                "role": "user",
                "content": "\n\n".join(
                    [
                        f"Problem: {problem_statement}",
                        f"Current metrics: {_format_metrics(current_result.metrics)}",
                        "Planning agent output:",
                        plan.raw,
                        "Editable files and current contents:",
                        file_context,
                    ]
                ),
            },
        ]
        raw = _complete(self.llm, messages, self.logger, "coding-agent", "implementing plan")
        raw_path = event_log.artifact(f"iteration_{iteration:02d}/coding_response.txt", raw)
        replacements = _extract_file_replacements(raw, fallback_files=self.editable_files)

        changed_files: list[str] = []
        skipped: list[str] = []
        editable = set(self.editable_files)
        for rel, content in replacements.items():
            normalized = _normalize_relpath(rel)
            if normalized not in editable:
                skipped.append(normalized)
                continue
            path = _safe_join(problem_dir, normalized)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(_ensure_trailing_newline(content), encoding="utf-8")
            changed_files.append(normalized)

        notes = "applied full-file replacements"
        if skipped:
            notes += f"; skipped non-editable files: {', '.join(skipped)}"
        if not changed_files:
            notes = "coding agent did not provide an editable file replacement"

        result = CodingResult(
            iteration=iteration,
            applied=bool(changed_files),
            changed_files=changed_files,
            raw_response_path=raw_path,
            notes=notes,
        )
        event_log.event("coding_done", asdict(result))
        return result


class ResearchDebuggingAgent:
    """Applies targeted fixes when the evaluation command fails or omits metrics."""

    def __init__(
        self,
        *,
        llm: PaperExpertLLM,
        editable_files: Sequence[str],
        model_name: str = OPENROUTER_MODEL,
        logger=None,
        max_file_chars: int = 16000,
    ) -> None:
        if not editable_files:
            raise ValueError("ResearchDebuggingAgent needs at least one editable file")
        self.llm = llm
        self.editable_files = [_normalize_relpath(path) for path in editable_files]
        self.model_name = model_name
        self.logger = logger
        self.max_file_chars = max_file_chars

    def apply_fix(
        self,
        *,
        problem_dir: Path,
        event_log: ResearchEventLog,
        iteration: int,
        attempt: int,
        plan: ResearchPlan,
        coding: CodingResult,
        failed_result: ExperimentResult,
        judge: JudgeFeedback,
        diagnosis: OrchestrationDiagnosis | None,
        problem_statement: str,
        metric_name: str,
    ) -> DebuggingResult:
        editable_list = ", ".join(self.editable_files)
        example_path = self.editable_files[0]
        messages = [
            {
                "role": "system",
                "content": (
                    f"You are the debugging agent for an autonomous research loop. "
                    f"You are running as model {self.model_name}. "
                    "The newly proposed model failed during retraining/retesting or did not "
                    "produce the required metric. Fix only the bug that prevents a clean "
                    "evaluation with metrics; preserve the research intent where possible. "
                    f"Only modify files explicitly listed as editable. Valid editable paths: {editable_list}. "
                    "STRICT OUTPUT CONTRACT: your response is parsed by a program. "
                    "Return complete replacement files using this exact format:\n"
                    f"---FILE: {example_path}---\n"
                    "```python\n"
                    "# complete file contents\n"
                    "```\n"
                    "---END FILE---\n"
                    "If no safe debug fix is possible, return exactly: NO_SAFE_CHANGE"
                ),
            },
            {
                "role": "user",
                "content": "\n\n".join(
                    [
                        f"Problem: {problem_statement}",
                        f"Required metric: {metric_name}",
                        f"Failed return code: {failed_result.returncode}",
                        f"Failed metrics: {_format_metrics(failed_result.metrics)}",
                        "Failed evaluation logs:",
                        _format_result_logs(failed_result, 10000),
                        "Judge feedback after the failed run:",
                        judge.raw,
                        "Orchestration diagnosis:",
                        _format_diagnosis(diagnosis),
                        "Planning agent output:",
                        plan.raw,
                        f"Coding agent changed files: {', '.join(coding.changed_files)}",
                        f"Coding agent notes: {coding.notes}",
                        "Current editable files after the failed change:",
                        _format_editable_files(problem_dir, self.editable_files, self.max_file_chars),
                        (
                            "Patch the runtime, import, tensor-shape, loss/label, device, or metrics-output "
                            "bug so the same evaluation command can finish and write the required metric. "
                            "Do not redesign the experiment unless that is necessary to restore a valid run."
                        ),
                    ]
                ),
            },
        ]
        raw = _complete(
            self.llm,
            messages,
            self.logger,
            "debugging-agent",
            f"debugging attempt {attempt}",
        )
        raw_path = event_log.artifact(
            f"iteration_{iteration:02d}/debugging_attempt_{attempt}.txt",
            raw,
        )
        replacements = _extract_file_replacements(raw, fallback_files=self.editable_files)

        changed_files: list[str] = []
        skipped: list[str] = []
        editable = set(self.editable_files)
        for rel, content in replacements.items():
            normalized = _normalize_relpath(rel)
            if normalized not in editable:
                skipped.append(normalized)
                continue
            path = _safe_join(problem_dir, normalized)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(_ensure_trailing_newline(content), encoding="utf-8")
            changed_files.append(normalized)

        notes = f"debugging attempt {attempt} applied full-file replacements"
        if skipped:
            notes += f"; skipped non-editable files: {', '.join(skipped)}"
        if not changed_files:
            notes = f"debugging attempt {attempt} did not provide an editable file replacement"

        result = DebuggingResult(
            iteration=iteration,
            attempt=attempt,
            applied=bool(changed_files),
            changed_files=changed_files,
            raw_response_path=raw_path,
            notes=notes,
        )
        event_log.event("debugging_done", asdict(result))
        return result


class ResearchSwarmOrchestrator:
    """Runs baseline, paper ideation, planning, coding, judging, and iteration."""

    def __init__(
        self,
        agents: list,
        *,
        problem_dir: Path,
        command: Sequence[str],
        metrics_path: str,
        editable_files: Sequence[str],
        planner_llm: PaperExpertLLM,
        coding_llm: PaperExpertLLM,
        problem_statement: str,
        debugging_llm: PaperExpertLLM | None = None,
        metric_name: str = "test_accuracy",
        max_iterations: int = 2,
        max_agents: int = 3,
        max_cross_ideas: int = 6,
        max_debug_attempts: int = 2,
        goal: float | None = None,
        min_delta: float = 0.001,
        revert_on_regression: bool = True,
        session_dir: Path | None = None,
        logger=None,
    ) -> None:
        if not agents:
            raise ValueError("ResearchSwarmOrchestrator requires at least one agent")
        self.agents = agents
        self.problem_dir = problem_dir
        self.command = list(command)
        self.metrics_path = metrics_path
        self.editable_files = [_normalize_relpath(path) for path in editable_files]
        self.planner_llm = planner_llm
        self.metric_name = metric_name
        self.max_iterations = max_iterations
        self.max_agents = max_agents
        self.max_cross_ideas = max_cross_ideas
        self.max_debug_attempts = max_debug_attempts
        self.goal = goal
        self.min_delta = min_delta
        self.revert_on_regression = revert_on_regression
        self.problem_statement = problem_statement
        self.logger = logger
        self.session_dir = session_dir or self._new_session_dir(problem_dir)
        self.event_log = ResearchEventLog(self.session_dir, logger=logger)
        self.runner = ExperimentRunner(
            problem_dir=problem_dir,
            command=self.command,
            metrics_path=metrics_path,
            event_log=self.event_log,
        )
        diagnostic_llm = planner_llm
        self.diagnostic_agent = OrchestrationDiagnosticAgent(
            llm=diagnostic_llm,
            editable_files=self.editable_files,
            model_name=str(getattr(diagnostic_llm, "model", OPENROUTER_MODEL)),
            logger=logger,
        )
        self.coding_agent = ResearchCodingAgent(
            llm=coding_llm,
            editable_files=self.editable_files,
            model_name=str(getattr(coding_llm, "model", OPENROUTER_MODEL)),
            logger=logger,
        )
        debug_llm = debugging_llm or coding_llm
        self.debugging_agent = ResearchDebuggingAgent(
            llm=debug_llm,
            editable_files=self.editable_files,
            model_name=str(getattr(debug_llm, "model", OPENROUTER_MODEL)),
            logger=logger,
        )
        self._idea_counter = 1
        self._cross_counter = 1

    def run(self, *, dry_run: bool = False) -> ResearchSession:
        log = self.logger
        if log:
            log.phase("Starting research loop")
            log.info(f"Problem directory: {self.problem_dir}")
            log.info(f"Session directory: {self.session_dir}")

        self.event_log.event(
            "session_start",
            {
                "problem_dir": str(self.problem_dir),
                "command": self.command,
                "metric_name": self.metric_name,
                "editable_files": self.editable_files,
                "papers": [agent.paper.paper_id for agent in self.agents],
            },
        )

        if log:
            log.phase("Baseline evaluation")
        baseline = self.runner.run(iteration=0, label="baseline")
        if log:
            log.phase_done(f"Baseline metrics: {_format_metrics(baseline.metrics)}")

        best_result = baseline
        current_result = baseline
        iterations: list[ResearchIteration] = []
        feedback_history: list[JudgeFeedback] = []

        for iteration in range(1, self.max_iterations + 1):
            if self._goal_reached(best_result):
                if log:
                    log.info(f"Stopping because {self.metric_name} reached the goal.")
                break

            if log:
                log.phase(f"Research iteration {iteration}")

            diagnosis = self.diagnostic_agent.diagnose(
                problem_dir=self.problem_dir,
                event_log=self.event_log,
                iteration=iteration,
                command=self.command,
                metrics_path=self.metrics_path,
                current_result=current_result,
                problem_statement=self.problem_statement,
                feedback_history=feedback_history,
            )
            if log:
                issue_count = len(diagnosis.issues)
                log.info(f"Orchestration diagnosis found {issue_count} issue(s) or caveat(s)")

            selected = self._select_agents(current_result, diagnosis)
            self.event_log.event(
                "agents_selected",
                {"iteration": iteration, "agents": [a.agent_id for a in selected]},
            )
            if log:
                log.selected([a.agent_id for a in selected])

            seed_ideas = self._collect_seed_ideas(
                iteration=iteration,
                agents=selected,
                current_result=current_result,
                feedback_history=feedback_history,
                diagnosis=diagnosis,
            )
            cross_ideas = self._cross_pollinate_ideas(
                iteration=iteration,
                agents=selected,
                seeds=seed_ideas,
                current_result=current_result,
                feedback_history=feedback_history,
                diagnosis=diagnosis,
            )
            plan = self._plan_iteration(
                iteration=iteration,
                seed_ideas=seed_ideas,
                cross_ideas=cross_ideas,
                current_result=current_result,
                feedback_history=feedback_history,
                diagnosis=diagnosis,
            )

            research_iteration = ResearchIteration(
                iteration=iteration,
                seed_ideas=seed_ideas,
                cross_pollinated_ideas=cross_ideas,
                diagnosis=diagnosis,
                plan=plan,
            )
            iterations.append(research_iteration)

            if dry_run:
                self.event_log.event("dry_run_stop", {"iteration": iteration})
                if log:
                    log.phase_done("Dry run stopped before coding")
                break

            snapshot = FileSnapshot(self.problem_dir, self.editable_files)
            coding = None
            for _coding_attempt in range(2):
                coding = self.coding_agent.apply_plan(
                    problem_dir=self.problem_dir,
                    event_log=self.event_log,
                    iteration=iteration,
                    plan=plan,
                    current_result=current_result,
                    problem_statement=self.problem_statement,
                )
                if coding.applied:
                    break
                self.event_log.event(
                    "coding_retry",
                    {"iteration": iteration, "attempt": _coding_attempt + 1, "notes": coding.notes},
                )

            research_iteration.coding = coding
            if not coding.applied:
                judge = JudgeFeedback(
                    iteration=iteration,
                    metric_name=self.metric_name,
                    previous_value=current_result.metric_value(self.metric_name),
                    new_value=None,
                    delta=None,
                    decision="revise",
                    feedback=coding.notes,
                    raw=coding.notes,
                )
                research_iteration.judge = judge
                feedback_history.append(judge)
                continue

            result = self.runner.run(iteration=iteration, label="judge")
            research_iteration.result = result
            judge = self._judge_iteration(
                iteration=iteration,
                previous=current_result,
                current=result,
                plan=plan,
                coding=coding,
            )
            research_iteration.judge = judge

            if self._needs_debugging(result):
                result, judge, debugging_resolved = self._debug_until_metrics(
                    iteration_state=research_iteration,
                    previous=current_result,
                    failed_result=result,
                    initial_judge=judge,
                    plan=plan,
                    coding=coding,
                    diagnosis=diagnosis,
                )
                research_iteration.result = result
                research_iteration.judge = judge
                if not debugging_resolved:
                    feedback_history.append(judge)
                    if self.revert_on_regression:
                        snapshot.restore()
                        self.event_log.event(
                            "files_restored",
                            {"iteration": iteration, "reason": "debugging did not restore a valid metric run"},
                        )
                    self.event_log.event(
                        "debugging_unresolved_stop",
                        {
                            "iteration": iteration,
                            "returncode": result.returncode,
                            "metrics": result.metrics,
                            "metric_name": self.metric_name,
                        },
                    )
                    if log:
                        log.phase_done(
                            f"Iteration {iteration} stopped: debugging did not produce {self.metric_name}"
                        )
                    break

            feedback_history.append(judge)

            if judge.decision == "revert" and self.revert_on_regression:
                snapshot.restore()
                self.event_log.event(
                    "files_restored",
                    {"iteration": iteration, "reason": judge.feedback},
                )
            else:
                current_result = result
                if self._is_better(result, best_result):
                    best_result = result

            if log:
                log.phase_done(
                    f"Iteration {iteration} decision={judge.decision} "
                    f"{self.metric_name}={judge.new_value}"
                )

        session = ResearchSession(
            problem_dir=str(self.problem_dir),
            session_dir=str(self.session_dir),
            baseline=baseline,
            iterations=iterations,
            best_result=best_result,
            metric_name=self.metric_name,
        )
        summary_path = self.session_dir / "summary.json"
        summary_path.write_text(json.dumps(_json_safe(session), indent=2, sort_keys=True), encoding="utf-8")
        self.event_log.event("session_done", {"summary_path": str(summary_path)})
        return session

    @staticmethod
    def _new_session_dir(problem_dir: Path) -> Path:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        return problem_dir / "logs" / "research_swarm" / stamp

    def _select_agents(
        self,
        result: ExperimentResult,
        diagnosis: OrchestrationDiagnosis | None = None,
    ) -> list:
        query = self._context_query(result, diagnosis)
        ranked = sorted(self.agents, key=lambda agent: agent.relevance(query), reverse=True)
        return ranked[: self.max_agents]

    def _collect_seed_ideas(
        self,
        *,
        iteration: int,
        agents: list,
        current_result: ExperimentResult,
        feedback_history: list[JudgeFeedback],
        diagnosis: OrchestrationDiagnosis | None,
    ) -> list[ModelImprovementIdea]:
        ideas: list[ModelImprovementIdea] = []
        for agent in agents:
            evidence = agent.retriever.search_evidence(
                self._context_query(current_result, diagnosis),
                top_k=getattr(agent, "top_k", 4),
                paper_id=agent.paper.paper_id,
            )
            messages = [
                {
                    "role": "system",
                    "content": (
                        f"You are the expert for '{agent.paper.title}'. "
                        "Your ultimate goal is to improve the accuracy of the research model in focus. "
                        "There is ALWAYS room for improvement — assume the current model has fixable flaws "
                        "and commit to finding them. Use your paper as a lens but draw freely on your "
                        "broad ML/DL knowledge: architecture choices, optimizers, regularization, "
                        "normalization, augmentation, learning rate schedules, and so on. "
                        "Be specific and decisive — vague suggestions waste iterations."
                    ),
                },
                {
                    "role": "user",
                    "content": "\n\n".join(
                        [
                            f"Problem: {self.problem_statement}",
                            f"Current metrics: {_format_metrics(current_result.metrics)}",
                            "Recent judge feedback:",
                            _format_feedback(feedback_history),
                            "Orchestration diagnosis to account for:",
                            _format_diagnosis(diagnosis),
                            "Current editable source:",
                            _format_editable_files(self.problem_dir, self.editable_files, 6000),
                            "Evidence from your paper:",
                            _format_evidence(evidence),
                            (
                                "Propose exactly 1-2 concrete, high-impact modifications. "
                                "Do not hedge — pick changes you believe will move the metric and commit to them. "
                                "Use this exact format:\n"
                                "---MODEL_IDEA---\n"
                                "TEXT: one sentence summary\n"
                                "RATIONALE: paper-grounded or ML-knowledge-grounded reason\n"
                                "EXPECTED_EFFECT: expected metric effect\n"
                                "CHANGES: implementation-level changes\n"
                                "---END---"
                            ),
                        ]
                    ),
                },
            ]
            raw = _complete(agent.llm, messages, self.logger, agent.agent_id, "research proposal")
            self.event_log.artifact(f"iteration_{iteration:02d}/seed_{agent.paper.paper_id}.txt", raw)
            ideas.extend(self._parse_seed_ideas(raw, agent, evidence))

        self.event_log.event("seed_ideas_done", {"iteration": iteration, "ideas": ideas})
        return ideas

    def _cross_pollinate_ideas(
        self,
        *,
        iteration: int,
        agents: list,
        seeds: list[ModelImprovementIdea],
        current_result: ExperimentResult,
        feedback_history: list[JudgeFeedback],
        diagnosis: OrchestrationDiagnosis | None,
    ) -> list[CrossPollinatedModelIdea]:
        cross_ideas: list[CrossPollinatedModelIdea] = []
        if len(agents) < 2:
            return cross_ideas

        for agent in agents:
            for seed in seeds:
                if seed.agent_id == agent.agent_id:
                    continue
                if len(cross_ideas) >= self.max_cross_ideas:
                    self.event_log.event(
                        "cross_pollination_capped",
                        {"iteration": iteration, "max_cross_ideas": self.max_cross_ideas},
                    )
                    return cross_ideas
                evidence = agent.retriever.search_evidence(
                    f"{self.problem_statement} {seed.text} {seed.changes} {_format_diagnosis(diagnosis)}",
                    top_k=3,
                    paper_id=agent.paper.paper_id,
                )
                messages = [
                    {
                        "role": "system",
                        "content": (
                            f"You are the expert for '{agent.paper.title}'. "
                            "Your ultimate goal is to improve the accuracy of the research model in focus. "
                            "Combine your paper's methods with the other expert's idea to produce something "
                            "more powerful than either alone. Draw on your full ML/DL knowledge — "
                            "the combined idea should be bold and concrete, not a watered-down compromise."
                        ),
                    },
                    {
                        "role": "user",
                        "content": "\n\n".join(
                            [
                                f"Problem: {self.problem_statement}",
                                f"Current metrics: {_format_metrics(current_result.metrics)}",
                                "Recent judge feedback:",
                                _format_feedback(feedback_history),
                                "Orchestration diagnosis to account for:",
                                _format_diagnosis(diagnosis),
                                f"Seed idea from {seed.seed_label}: {seed.text}",
                                f"Seed rationale: {seed.rationale}",
                                f"Seed changes: {seed.changes}",
                                "Evidence from your paper:",
                                _format_evidence(evidence),
                                (
                                    "Produce one hybrid implementation idea. Use this exact format:\n"
                                    "---MODEL_CROSS_IDEA---\n"
                                    "TEXT: one sentence hybrid idea\n"
                                    "CONNECTION: how the two papers intersect\n"
                                    "CHANGES: implementation-level changes\n"
                                    "---END---"
                                ),
                            ]
                        ),
                    },
                ]
                raw = _complete(agent.llm, messages, self.logger, agent.agent_id, "cross-pollinating")
                safe_seed_id = seed.idea_id.replace(":", "_")
                self.event_log.artifact(
                    f"iteration_{iteration:02d}/cross_{agent.paper.paper_id}_{safe_seed_id}.txt",
                    raw,
                )
                cross_ideas.append(self._parse_cross_idea(raw, agent, seed, evidence))

        self.event_log.event("cross_ideas_done", {"iteration": iteration, "ideas": cross_ideas})
        return cross_ideas

    def _plan_iteration(
        self,
        *,
        iteration: int,
        seed_ideas: list[ModelImprovementIdea],
        cross_ideas: list[CrossPollinatedModelIdea],
        current_result: ExperimentResult,
        feedback_history: list[JudgeFeedback],
        diagnosis: OrchestrationDiagnosis | None,
    ) -> ResearchPlan:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are the planning agent in an autonomous research swarm. "
                    "Your ultimate goal is to improve the accuracy of the research model in focus. "
                    "There is ALWAYS room to improve — never accept the current model as good enough. "
                    "Pick the most impactful set of changes from the ideas presented: be decisive and specific. "
                    "Prefer high-leverage interventions (fix the worst bottleneck first) over safe micro-tweaks. "
                    "The coding agent can handle real changes — do not water down the plan."
                ),
            },
            {
                "role": "user",
                "content": "\n\n".join(
                    [
                        f"Problem: {self.problem_statement}",
                        f"Current metrics: {_format_metrics(current_result.metrics)}",
                        f"Metric to improve: {self.metric_name}",
                        f"Editable files: {', '.join(self.editable_files)}",
                        "Recent judge feedback:",
                        _format_feedback(feedback_history),
                        "Orchestration diagnosis to account for:",
                        _format_diagnosis(diagnosis),
                        "Seed ideas:",
                        _format_seed_ideas(seed_ideas),
                        "Cross-pollinated ideas:",
                        _format_cross_ideas(cross_ideas),
                        (
                            "Create one implementation plan. Use this exact format:\n"
                            "---RESEARCH_PLAN---\n"
                            "SUMMARY: one paragraph\n"
                            "TARGET_FILES:\n"
                            "- relative/path.py\n"
                            "STEPS:\n"
                            "- step one\n"
                            "- step two\n"
                            "EXPECTED_EFFECT: expected metric impact\n"
                            "VALIDATION: how the judge should interpret the next run\n"
                            "---END---"
                        ),
                    ]
                ),
            },
        ]
        raw = _complete(self.planner_llm, messages, self.logger, "planning-agent", "planning")
        self.event_log.artifact(f"iteration_{iteration:02d}/plan.txt", raw)
        plan = _parse_plan(raw, iteration=iteration, editable_files=self.editable_files)
        self.event_log.event("plan_done", {"iteration": iteration, "plan": plan})
        return plan

    def _judge_iteration(
        self,
        *,
        iteration: int,
        previous: ExperimentResult,
        current: ExperimentResult,
        plan: ResearchPlan,
        coding: CodingResult,
        label: str = "judge",
    ) -> JudgeFeedback:
        previous_value = previous.metric_value(self.metric_name)
        new_value = current.metric_value(self.metric_name)
        delta = (
            round(new_value - previous_value, 6)
            if previous_value is not None and new_value is not None
            else None
        )
        decision = _numeric_decision(
            previous=previous,
            current=current,
            metric_name=self.metric_name,
            min_delta=self.min_delta,
        )
        messages = [
            {
                "role": "system",
                "content": (
                    "You are the judge agent for an autonomous research loop. "
                    "Your ultimate goal is to improve the accuracy of the research model in focus. "
                    "There is ALWAYS more headroom — your job is to give the next round a clear, "
                    "actionable direction regardless of whether this iteration improved or regressed. "
                    "A revert is not a failure — it is information. Extract the lesson and prescribe "
                    "the next concrete change with confidence."
                ),
            },
            {
                "role": "user",
                "content": "\n\n".join(
                    [
                        f"Problem: {self.problem_statement}",
                        f"Metric: {self.metric_name}",
                        f"Previous metrics: {_format_metrics(previous.metrics)}",
                        f"New metrics: {_format_metrics(current.metrics)}",
                        f"Numeric decision: {decision}",
                        f"Changed files: {', '.join(coding.changed_files)}",
                        "Plan:",
                        plan.raw,
                        (
                            "Write concise, actionable feedback for the next paper-agent round. "
                            "Diagnose what worked or why it regressed, then prescribe 2-3 specific "
                            "changes to try next. Be direct — name exact hyperparameters, layer types, "
                            "or training tricks. Do not hedge or say 'it depends'."
                        ),
                    ]
                ),
            },
        ]
        try:
            raw = _complete(self.planner_llm, messages, self.logger, "judge-agent", "judging")
        except RuntimeError as exc:
            raw = f"Judge LLM unavailable: {exc}. Numeric decision: {decision}."
        feedback = JudgeFeedback(
            iteration=iteration,
            metric_name=self.metric_name,
            previous_value=previous_value,
            new_value=new_value,
            delta=delta,
            decision=decision,
            feedback=_shorten(raw, 1200),
            raw=raw,
        )
        artifact_name = "judge_feedback.txt" if label == "judge" else f"judge_feedback_{label}.txt"
        self.event_log.artifact(f"iteration_{iteration:02d}/{artifact_name}", raw)
        self.event_log.event("judge_done", {"iteration": iteration, "judge": feedback})
        return feedback

    def _debug_until_metrics(
        self,
        *,
        iteration_state: ResearchIteration,
        previous: ExperimentResult,
        failed_result: ExperimentResult,
        initial_judge: JudgeFeedback,
        plan: ResearchPlan,
        coding: CodingResult,
        diagnosis: OrchestrationDiagnosis | None,
    ) -> tuple[ExperimentResult, JudgeFeedback, bool]:
        current_failed = failed_result
        current_judge = initial_judge

        for attempt in range(1, self.max_debug_attempts + 1):
            debug = self.debugging_agent.apply_fix(
                problem_dir=self.problem_dir,
                event_log=self.event_log,
                iteration=iteration_state.iteration,
                attempt=attempt,
                plan=plan,
                coding=coding,
                failed_result=current_failed,
                judge=current_judge,
                diagnosis=diagnosis,
                problem_statement=self.problem_statement,
                metric_name=self.metric_name,
            )
            iteration_state.debugging.append(debug)
            if not debug.applied:
                continue

            rerun = self.runner.run(
                iteration=iteration_state.iteration,
                label=f"debug_{attempt}",
            )
            effective_coding = CodingResult(
                iteration=coding.iteration,
                applied=True,
                changed_files=sorted(set(coding.changed_files) | set(debug.changed_files)),
                raw_response_path=debug.raw_response_path,
                notes=f"{coding.notes}; {debug.notes}",
            )
            current_judge = self._judge_iteration(
                iteration=iteration_state.iteration,
                previous=previous,
                current=rerun,
                plan=plan,
                coding=effective_coding,
                label=f"debug_{attempt}",
            )
            if not self._needs_debugging(rerun):
                self.event_log.event(
                    "debugging_resolved",
                    {
                        "iteration": iteration_state.iteration,
                        "attempt": attempt,
                        "metrics": rerun.metrics,
                    },
                )
                return rerun, current_judge, True
            current_failed = rerun

        return current_failed, current_judge, False

    def _parse_seed_ideas(self, raw: str, agent, evidence: list[Evidence]) -> list[ModelImprovementIdea]:
        blocks = _extract_blocks(raw, "MODEL_IDEA")
        if not blocks:
            blocks = [raw]
        ideas = []
        for block in blocks[:2]:
            text = _field(block, "TEXT") or _first_sentence(block)
            ideas.append(
                ModelImprovementIdea(
                    idea_id=f"{agent.paper.paper_id}:model_idea:{self._next_idea_id()}",
                    agent_id=agent.agent_id,
                    paper_id=agent.paper.paper_id,
                    paper_title=agent.paper.title,
                    text=text,
                    rationale=_field(block, "RATIONALE") or "(not specified)",
                    expected_effect=_field(block, "EXPECTED_EFFECT") or "(not specified)",
                    changes=_field(block, "CHANGES") or "(not specified)",
                    evidence=evidence,
                )
            )
        return ideas

    def _parse_cross_idea(
        self,
        raw: str,
        agent,
        seed: ModelImprovementIdea,
        evidence: list[Evidence],
    ) -> CrossPollinatedModelIdea:
        block = (_extract_blocks(raw, "MODEL_CROSS_IDEA") or [raw])[0]
        return CrossPollinatedModelIdea(
            idea_id=f"{agent.paper.paper_id}:cross_idea:{self._next_cross_id()}",
            agent_id=agent.agent_id,
            paper_id=agent.paper.paper_id,
            paper_title=agent.paper.title,
            seed_idea_id=seed.idea_id,
            seed_paper_id=seed.paper_id,
            seed_paper_title=seed.paper_title,
            text=_field(block, "TEXT") or _first_sentence(block),
            connection=_field(block, "CONNECTION") or "(not specified)",
            changes=_field(block, "CHANGES") or "(not specified)",
            evidence=evidence,
        )

    def _context_query(
        self,
        result: ExperimentResult,
        diagnosis: OrchestrationDiagnosis | None = None,
    ) -> str:
        return " ".join(
            [
                self.problem_statement,
                self.metric_name,
                _format_metrics(result.metrics),
                _format_diagnosis(diagnosis),
                "image classification neural network architecture training accuracy improvement",
            ]
        )

    def _next_idea_id(self) -> int:
        current = self._idea_counter
        self._idea_counter += 1
        return current

    def _next_cross_id(self) -> int:
        current = self._cross_counter
        self._cross_counter += 1
        return current

    def _is_better(self, candidate: ExperimentResult, incumbent: ExperimentResult) -> bool:
        candidate_value = candidate.metric_value(self.metric_name)
        incumbent_value = incumbent.metric_value(self.metric_name)
        if candidate_value is None:
            return False
        if incumbent_value is None:
            return True
        return candidate_value > incumbent_value

    def _goal_reached(self, result: ExperimentResult) -> bool:
        if self.goal is None:
            return False
        value = result.metric_value(self.metric_name)
        return value is not None and value >= self.goal

    def _needs_debugging(self, result: ExperimentResult) -> bool:
        return (not result.succeeded) or result.metric_value(self.metric_name) is None


def _complete(
    llm: PaperExpertLLM,
    messages: list[dict[str, str]],
    logger,
    agent_id: str,
    stage: str,
) -> str:
    if logger:
        logger.agent_start(agent_id, stage)
    try:
        if logger and hasattr(llm, "complete_stream"):
            return llm.complete_stream(messages, logger.on_token)  # type: ignore[attr-defined]
        return llm.complete(messages)
    finally:
        if logger:
            logger.agent_done(agent_id)


def _read_metrics(metrics_path: Path | None, stdout: str) -> dict[str, Any]:
    if metrics_path and metrics_path.exists():
        data: Any = None
        try:
            with metrics_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            data = None
        if isinstance(data, dict):
            return data

    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if not line:
            continue
        if line.startswith("METRICS_JSON:"):
            line = line.split(":", 1)[1].strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(data, dict):
                return data
    return {}


def _numeric_decision(
    *,
    previous: ExperimentResult,
    current: ExperimentResult,
    metric_name: str,
    min_delta: float,
) -> str:
    if not current.succeeded:
        return "revert"
    previous_value = previous.metric_value(metric_name)
    new_value = current.metric_value(metric_name)
    if new_value is None:
        return "revise"
    if previous_value is None:
        return "keep"
    delta = new_value - previous_value
    if delta >= min_delta:
        return "keep"
    if delta < -min_delta:
        return "revert"
    return "revise"


def _parse_plan(raw: str, *, iteration: int, editable_files: Sequence[str]) -> ResearchPlan:
    block = (_extract_blocks(raw, "RESEARCH_PLAN") or [raw])[0]
    target_files = [
        _normalize_relpath(item)
        for item in _bullets(_field(block, "TARGET_FILES"))
        if item.strip()
    ]
    target_files = [path for path in target_files if path in set(editable_files)]
    if not target_files:
        target_files = list(editable_files)

    steps = _bullets(_field(block, "STEPS"))
    if not steps:
        steps = [_first_sentence(block)]

    return ResearchPlan(
        iteration=iteration,
        summary=_field(block, "SUMMARY") or _first_sentence(block),
        target_files=target_files,
        steps=steps,
        expected_effect=_field(block, "EXPECTED_EFFECT") or "(not specified)",
        validation=_field(block, "VALIDATION") or "(not specified)",
        raw=raw,
    )


def _parse_diagnosis(raw: str, *, iteration: int) -> OrchestrationDiagnosis:
    block = (_extract_blocks(raw, "ORCHESTRATION_DIAGNOSIS") or [raw])[0]
    issues = _bullets(_field(block, "ISSUES"))
    suggestions = _bullets(_field(block, "SUGGESTIONS"))
    dataset_context = _bullets(_field(block, "DATASET_CONTEXT"))
    return OrchestrationDiagnosis(
        iteration=iteration,
        summary=_field(block, "SUMMARY") or _first_sentence(block),
        issues=issues,
        suggestions=suggestions,
        dataset_context=dataset_context,
        raw=raw,
    )


def _extract_file_replacements(raw: str, fallback_files: Sequence[str] | None = None) -> dict[str, str]:
    pattern = re.compile(
        r"---FILE:\s*(?P<path>.+?)---\s*(?P<body>.*?)---END FILE---",
        re.IGNORECASE | re.DOTALL,
    )
    replacements = {}
    for match in pattern.finditer(raw):
        rel = match.group("path").strip()
        body = _strip_code_fence(match.group("body").strip())
        replacements[rel] = body
    if replacements:
        return replacements

    loose_pattern = re.compile(
        r"^---\s*(?P<path>[A-Za-z0-9_./-]+\.[A-Za-z0-9_+-]+)\s*---\s*(?P<body>.*?)---END FILE---",
        re.IGNORECASE | re.DOTALL | re.MULTILINE,
    )
    for match in loose_pattern.finditer(raw):
        rel = match.group("path").strip()
        body = _strip_code_fence(match.group("body").strip())
        replacements[rel] = body
    if replacements:
        return replacements

    fallback = [_normalize_relpath(path) for path in fallback_files or []]
    if len(fallback) == 1:
        fenced = _extract_first_code_fence(raw)
        if fenced:
            replacements[fallback[0]] = fenced
    return replacements


def _strip_code_fence(text: str) -> str:
    lines = text.splitlines()
    if lines and lines[0].strip().startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip().startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip("\n")


def _extract_first_code_fence(text: str) -> str | None:
    match = re.search(r"```(?:[A-Za-z0-9_+-]+)?\s*\n(?P<body>.*?)```", text, re.DOTALL)
    if not match:
        return None
    return match.group("body").strip("\n")


def _extract_blocks(text: str, name: str) -> list[str]:
    pattern = re.compile(
        rf"---{re.escape(name)}---(?P<body>.*?)---END---",
        re.IGNORECASE | re.DOTALL,
    )
    return [match.group("body").strip() for match in pattern.finditer(text)]


def _field(text: str, name: str) -> str:
    pattern = re.compile(
        rf"^{re.escape(name)}:\s*(?P<value>.*?)(?=^[A-Z_]+:\s*|\Z)",
        re.IGNORECASE | re.MULTILINE | re.DOTALL,
    )
    match = pattern.search(text)
    return match.group("value").strip() if match else ""


def _bullets(text: str) -> list[str]:
    items = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        line = re.sub(r"^[-*]\s+", "", line)
        line = re.sub(r"^\d+[.)]\s+", "", line)
        items.append(line.strip())
    if not items and text.strip():
        items.append(text.strip())
    return items


def _first_sentence(text: str) -> str:
    collapsed = " ".join(text.split())
    if not collapsed:
        return "(empty response)"
    parts = re.split(r"(?<=[.!?])\s+", collapsed, maxsplit=1)
    return parts[0][:400]


def _format_metrics(metrics: dict[str, Any]) -> str:
    if not metrics:
        return "(no metrics found)"
    parts = []
    for key in sorted(metrics):
        value = metrics[key]
        if isinstance(value, float):
            parts.append(f"{key}={value:.6g}")
        else:
            parts.append(f"{key}={value}")
    return ", ".join(parts)


def _format_feedback(feedback_history: list[JudgeFeedback]) -> str:
    if not feedback_history:
        return "(none yet)"
    return "\n".join(
        f"- iteration {item.iteration}: decision={item.decision}, "
        f"delta={item.delta}, feedback={_shorten(item.feedback, 300)}"
        for item in feedback_history[-3:]
    )


def _format_diagnosis(diagnosis: OrchestrationDiagnosis | None) -> str:
    if diagnosis is None:
        return "(no orchestration diagnosis yet)"
    parts = [f"Summary: {diagnosis.summary}"]
    if diagnosis.dataset_context:
        parts.append("Dataset/context:\n" + "\n".join(f"- {item}" for item in diagnosis.dataset_context))
    if diagnosis.issues:
        parts.append("Issues:\n" + "\n".join(f"- {item}" for item in diagnosis.issues))
    if diagnosis.suggestions:
        parts.append("Suggestions:\n" + "\n".join(f"- {item}" for item in diagnosis.suggestions))
    return "\n".join(parts)


def _format_seed_ideas(ideas: list[ModelImprovementIdea]) -> str:
    if not ideas:
        return "(none)"
    return "\n\n".join(
        "\n".join(
            [
                f"[{idea.idea_id}] {idea.paper_title}",
                f"Text: {idea.text}",
                f"Rationale: {idea.rationale}",
                f"Expected effect: {idea.expected_effect}",
                f"Changes: {idea.changes}",
            ]
        )
        for idea in ideas
    )


def _format_cross_ideas(ideas: list[CrossPollinatedModelIdea]) -> str:
    if not ideas:
        return "(none)"
    return "\n\n".join(
        "\n".join(
            [
                f"[{idea.idea_id}] {idea.paper_title} with {idea.seed_paper_title}",
                f"Text: {idea.text}",
                f"Connection: {idea.connection}",
                f"Changes: {idea.changes}",
            ]
        )
        for idea in ideas
    )


def _format_evidence(evidence: list[Evidence]) -> str:
    if not evidence:
        return "(no matching paper evidence retrieved)"
    return "\n\n".join(
        f"[{item.citation}]\n{_shorten(item.text, 1200)}" for item in evidence
    )


def _format_editable_files(problem_dir: Path, editable_files: Sequence[str], max_chars: int) -> str:
    parts = []
    for rel in editable_files:
        path = _safe_join(problem_dir, rel)
        if not path.exists():
            parts.append(f"--- {rel} (missing) ---")
            continue
        text = path.read_text(encoding="utf-8")
        parts.append(f"--- {rel} ---\n{_shorten(text, max_chars)}")
    return "\n\n".join(parts)


def _format_problem_context(problem_dir: Path, max_chars: int) -> str:
    context_files = [
        "train.py",
        "README.md",
        "data/README.md",
        "run_baseline.sh",
        "run_research_swarm.sh",
        "run_research_swarm_asus.sh",
    ]
    parts = []
    budget = max_chars
    for rel in context_files:
        if budget <= 0:
            break
        path = _safe_join(problem_dir, rel)
        if not path.exists() or not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        clipped = _shorten(text, min(budget, 5000))
        parts.append(f"--- {rel} ---\n{clipped}")
        budget -= len(clipped)
    return "\n\n".join(parts) if parts else "(no setup files found)"


def _format_result_logs(result: ExperimentResult, max_chars: int) -> str:
    stdout = _read_text_if_exists(Path(result.stdout_path))
    stderr = _read_text_if_exists(Path(result.stderr_path))
    metrics = _format_metrics(result.metrics)
    text = "\n\n".join(
        [
            f"returncode={result.returncode}",
            f"metrics={metrics}",
            "--- stdout tail ---\n" + _tail(stdout, max_chars // 2),
            "--- stderr tail ---\n" + _tail(stderr, max_chars // 2),
        ]
    )
    return _tail(text, max_chars)


def _read_text_if_exists(path: Path) -> str:
    if not path.exists():
        return "(missing)"
    return path.read_text(encoding="utf-8", errors="replace")


def _tail(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[-limit:]


def _shorten(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 3].rsplit(" ", 1)[0] + "..."


def _safe_join(root: Path, relative_path: str) -> Path:
    rel = _normalize_relpath(relative_path)
    path = (root / rel).resolve()
    root_resolved = root.resolve()
    if root_resolved != path and root_resolved not in path.parents:
        raise ValueError(f"path escapes problem directory: {relative_path}")
    return path


def _normalize_relpath(path: str) -> str:
    normalized = Path(path.strip()).as_posix()
    if normalized.startswith("/") or normalized == "." or normalized.startswith("../"):
        raise ValueError(f"editable path must be relative to the problem directory: {path}")
    return normalized


def _ensure_trailing_newline(text: str) -> str:
    return text if text.endswith("\n") else text + "\n"


def _json_safe(value: Any) -> Any:
    if is_dataclass(value):
        return _json_safe(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value
