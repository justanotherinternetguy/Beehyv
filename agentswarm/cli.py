"""Command-line entrypoint for the paper expert swarm."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .brainstorm import BrainstormOrchestrator
from .code_retriever import CodeRetriever
from .expert import PaperExpertAgent
from .llm import OPENROUTER_MODEL, OpenRouterLLM
from .log import SwarmLogger
from .orchestrator import SwarmOrchestrator
from .paper_loader import load_papers
from .retriever import KeywordRetriever

_OUTPUTS_DIR = Path(__file__).parent.parent / "outputs"


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
        default=["data/cleaned_json/BERT_cleaned.json"],
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

    args = parser.parse_args()
    if args.command == "discuss":
        return cmd_discuss(args)
    return cmd_brainstorm(args)


if __name__ == "__main__":
    raise SystemExit(main())
