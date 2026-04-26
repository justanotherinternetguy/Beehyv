"""Command-line entrypoint for the paper expert swarm."""

from __future__ import annotations

import argparse
import shlex
import sys
from pathlib import Path

from .brainstorm import BrainstormOrchestrator
from .code_retriever import CodeRetriever
from .expert import PaperExpertAgent
from .llm import OPENROUTER_MODEL, OpenRouterLLM
from .log import SwarmLogger
from .orchestrator import SwarmOrchestrator
from .paper_loader import load_papers
from .research import ResearchSwarmOrchestrator
from .retriever import KeywordRetriever

_OUTPUTS_DIR = Path(__file__).parent.parent / "outputs"
_DEFAULT_PAPERS = ["data/cleaned_json/bert_cleaned.json"]
_RESEARCH_DEFAULT_PAPERS = [
    "data/cleaned_json/attention_is_all_you_need_cleaned.json",
    "data/cleaned_json/og_attention_cleaned.json",
    "data/cleaned_json/introcnn_cleaned.json",
]


# ── shared setup ──────────────────────────────────────────────────────────────

def _build_agents(papers, retriever, llm, top_k, logger):
    agents = []
    for paper in papers:
        code_retriever = CodeRetriever.for_paper(paper.paper_id, _OUTPUTS_DIR)
        if code_retriever and logger:
            logger.info(
                f"{paper.paper_id}: code repo found "
                f"({len(code_retriever._snippets)} snippets indexed)"
            )
        agents.append(
            PaperExpertAgent(
                paper, retriever,
                top_k=top_k, llm=llm, logger=logger,
                code_retriever=code_retriever,
            )
        )
    return agents


def _load(paper_args, logger) -> tuple:
    paper_paths = [Path(p) for p in paper_args]
    logger.phase("Loading papers")
    papers = load_papers(paper_paths)
    for paper in papers:
        logger.info(f"{paper.paper_id}: {paper.title} ({len(paper.chunks)} chunks)")
    retriever = KeywordRetriever(papers)
    return papers, retriever


def _common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--papers", nargs="+",
        default=_DEFAULT_PAPERS,
        help="Cleaned paper JSON files.",
    )
    parser.add_argument("--model", default=OPENROUTER_MODEL,
                        help="OpenRouter model (default: %(default)s).")
    parser.add_argument("--top-k", type=int, default=4,
                        help="Evidence chunks retrieved per expert.")
    parser.add_argument("--log-file", default=None,
                        help="Path for a structured log file (tailable with tail -f).")
    parser.add_argument("--no-stream", action="store_true",
                        help="Disable token streaming.")


# ── discuss subcommand ────────────────────────────────────────────────────────

def cmd_discuss(args: argparse.Namespace) -> int:
    logger = SwarmLogger(stream=sys.stderr, log_file=args.log_file)
    papers, retriever = _load(args.papers, logger)
    llm = OpenRouterLLM(model=args.model)
    agents = _build_agents(papers, retriever, llm, args.top_k,
                           None if args.no_stream else logger)
    orchestrator = SwarmOrchestrator(
        agents,
        max_agents=args.max_agents,
        critique_rounds=args.critique_rounds,
        logger=logger,
    )

    blackboard = orchestrator.run(args.question)
    synthesis = blackboard.synthesis
    if synthesis is None:
        raise RuntimeError("orchestrator did not produce a synthesis")

    print("\n" + "=" * 60)
    print(synthesis.answer)
    print("\nConsensus:")
    for item in synthesis.consensus:
        print(f"- {item}")
    if synthesis.disagreements:
        print("\nDisagreements / limits:")
        for item in synthesis.disagreements:
            print(f"- {item}")
    print("\nCitations:")
    for citation in synthesis.citations:
        print(f"- {citation}")
    return 0


# ── brainstorm subcommand ─────────────────────────────────────────────────────

def cmd_brainstorm(args: argparse.Namespace) -> int:
    logger = SwarmLogger(stream=sys.stderr, log_file=args.log_file)
    papers, retriever = _load(args.papers, logger)
    llm = OpenRouterLLM(model=args.model)
    agents = _build_agents(papers, retriever, llm, args.top_k,
                           None if args.no_stream else logger)
    orchestrator = BrainstormOrchestrator(
        agents,
        llm=llm,
        max_agents=args.max_agents,
        cross_pollinate_rounds=args.cp_rounds,
        logger=logger,
    )

    bb = orchestrator.run(args.area)

    # ── print results to stdout ───────────────────────────────────────────────
    print("\n" + "=" * 60)

    if bb.seeds:
        print(f"\nSEED IDEAS ({len(bb.seeds)} total)\n")
        for idea in bb.seeds:
            print(f"  [{idea.paper_title}]")
            print(f"  {idea.text}")
            print(f"  Grounded in: {idea.grounding}")
            print(f"  Gap: {idea.gap}")
            print()

    if bb.cross_pollinations:
        print(f"CROSS-POLLINATED IDEAS ({len(bb.cross_pollinations)} total)\n")
        for cp in bb.cross_pollinations:
            print(f"  [{cp.from_paper_title}] × [{cp.seed_paper_title}]")
            print(f"  {cp.text}")
            print(f"  Connection: {cp.connection}")
            print()

    if bb.agenda:
        print("=" * 60)
        print("\nRESEARCH AGENDA\n")
        print(bb.agenda)

    return 0


# ── research subcommand ───────────────────────────────────────────────────────

def cmd_research(args: argparse.Namespace) -> int:
    logger = SwarmLogger(stream=sys.stderr, log_file=args.log_file)
    papers, retriever = _load(args.papers, logger)
    planner_llm = OpenRouterLLM(
        model=args.model,
        max_tokens=args.planner_max_tokens,
        temperature=args.temperature,
    )
    coding_llm = OpenRouterLLM(
        model=args.coding_model,
        max_tokens=args.coding_max_tokens,
        temperature=args.coding_temperature,
    )
    agents = _build_agents(
        papers,
        retriever,
        planner_llm,
        args.top_k,
        None if args.no_stream else logger,
    )

    problem_dir = Path(args.problem_dir).expanduser().resolve()
    if not problem_dir.exists():
        print(f"Error: problem directory not found: {problem_dir}", file=sys.stderr)
        return 1
    command = shlex.split(args.run_command)
    if not command:
        print("Error: --run-command cannot be empty", file=sys.stderr)
        return 1

    orchestrator = ResearchSwarmOrchestrator(
        agents,
        problem_dir=problem_dir,
        command=command,
        metrics_path=args.metrics_file,
        editable_files=args.editable,
        planner_llm=planner_llm,
        coding_llm=coding_llm,
        problem_statement=args.problem,
        metric_name=args.metric,
        max_iterations=args.iterations,
        max_agents=args.max_agents,
        max_cross_ideas=args.max_cross_ideas,
        goal=args.goal,
        min_delta=args.min_delta,
        revert_on_regression=not args.keep_regressions,
        session_dir=Path(args.session_dir).expanduser().resolve() if args.session_dir else None,
        logger=logger,
    )
    session = orchestrator.run(dry_run=args.dry_run)

    baseline_value = session.baseline.metric_value(args.metric)
    best_value = session.best_result.metric_value(args.metric)
    print("\n" + "=" * 60)
    print("RESEARCH SESSION COMPLETE")
    print(f"Session directory: {session.session_dir}")
    print(f"Baseline {args.metric}: {baseline_value}")
    print(f"Best {args.metric}: {best_value}")
    print(f"Iterations: {len(session.iterations)}")
    for item in session.iterations:
        decision = item.judge.decision if item.judge else "not judged"
        value = item.result.metric_value(args.metric) if item.result else None
        changed = ", ".join(item.coding.changed_files) if item.coding else "(none)"
        print(f"- iteration {item.iteration}: decision={decision}, {args.metric}={value}, changed={changed}")
    return 0


# ── entry point ───────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        prog="agentswarm",
        description="Multi-paper expert agent swarm.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # discuss
    p_discuss = sub.add_parser("discuss", help="Q&A grounded in paper evidence.")
    _common_args(p_discuss)
    p_discuss.add_argument("question", help="Question for the paper experts.")
    p_discuss.add_argument("--max-agents", type=int, default=5)
    p_discuss.add_argument("--critique-rounds", type=int, default=1)

    # brainstorm
    p_brainstorm = sub.add_parser(
        "brainstorm",
        help="Cross-pollinate ideas across papers to generate future research directions.",
    )
    _common_args(p_brainstorm)
    p_brainstorm.add_argument("area", help="Research area to brainstorm (e.g. 'efficient transformers').")
    p_brainstorm.add_argument("--max-agents", type=int, default=6,
                              help="Max experts in the session (default: 6).")
    p_brainstorm.add_argument("--cp-rounds", type=int, default=1,
                              help="Cross-pollination rounds (default: 1).")

    # research
    p_research = sub.add_parser(
        "research",
        help="Autonomous research loop over a code problem folder.",
    )
    p_research.add_argument("problem_dir", help="Folder containing the model problem.")
    p_research.add_argument(
        "--papers",
        nargs="+",
        default=_RESEARCH_DEFAULT_PAPERS,
        help="Paper JSONs for research agents (default: attention, original attention, intro CNN).",
    )
    p_research.add_argument(
        "--problem",
        default=(
            "Improve a poorly performing MNIST classifier while keeping the evaluation dataset "
            "and run command fixed."
        ),
        help="Problem statement sent to the research agents.",
    )
    p_research.add_argument(
        "--run-command",
        default=(
            "python train.py --download --metrics-out logs/latest_metrics.json "
            "--log-file logs/train_events.jsonl"
        ),
        help="Command to run inside problem_dir for baseline and judge evaluations.",
    )
    p_research.add_argument("--metrics-file", default="logs/latest_metrics.json",
                            help="Metrics JSON produced by --run-command.")
    p_research.add_argument("--metric", default="test_accuracy",
                            help="Metric key to maximize from the metrics JSON.")
    p_research.add_argument("--editable", nargs="+", default=["model.py"],
                            help="Relative files the coding agent may replace.")
    p_research.add_argument("--iterations", type=int, default=2,
                            help="Maximum improvement iterations.")
    p_research.add_argument("--max-agents", type=int, default=3)
    p_research.add_argument("--max-cross-ideas", type=int, default=6)
    p_research.add_argument("--top-k", type=int, default=4)
    p_research.add_argument("--goal", type=float, default=None,
                            help="Stop early when the metric reaches this value.")
    p_research.add_argument("--min-delta", type=float, default=0.001,
                            help="Minimum metric improvement counted as a keep decision.")
    p_research.add_argument("--keep-regressions", action="store_true",
                            help="Do not restore editable files when the judge sees a regression.")
    p_research.add_argument("--dry-run", action="store_true",
                            help="Run baseline, ideation, and planning without editing files.")
    p_research.add_argument("--session-dir", default=None,
                            help="Optional explicit artifact directory.")
    p_research.add_argument("--model", default=OPENROUTER_MODEL,
                            help="Planner/judge/paper-agent OpenRouter model.")
    p_research.add_argument("--coding-model", default=OPENROUTER_MODEL,
                            help="Coding-agent OpenRouter model.")
    p_research.add_argument("--planner-max-tokens", type=int, default=1600)
    p_research.add_argument("--coding-max-tokens", type=int, default=6000)
    p_research.add_argument("--temperature", type=float, default=0.2)
    p_research.add_argument("--coding-temperature", type=float, default=0.1)
    p_research.add_argument("--log-file", default=None)
    p_research.add_argument("--no-stream", action="store_true")

    args = parser.parse_args()
    if args.command == "discuss":
        return cmd_discuss(args)
    if args.command == "brainstorm":
        return cmd_brainstorm(args)
    return cmd_research(args)


if __name__ == "__main__":
    raise SystemExit(main())
