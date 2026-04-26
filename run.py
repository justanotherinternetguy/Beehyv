#!/usr/bin/env python3
"""
generalresearch — unified CLI

Subcommands:
  ingest      PDF → S2ORC JSON → cleaned JSON
  discuss     Ask questions across papers (agent swarm)
  brainstorm  Cross-pollinate ideas across papers for future research directions
  research    Autoresearch-style model improvement loop over a problem folder
  codegen     Generate code repository from a cleaned JSON paper

Run `python run.py <subcommand> --help` for options.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent / ".env")
except ImportError:
    pass

ROOT = Path(__file__).resolve().parent
DEFAULT_OPENROUTER_MODEL = os.environ.get("OPENROUTER_MODEL", "nvidia/nemotron-3-super-120b-a12b:free")
DEFAULT_RESEARCH_PAPERS = [
    str(ROOT / "data" / "cleaned_json" / "attention_is_all_you_need_cleaned.json"),
    str(ROOT / "data" / "cleaned_json" / "og_attention_cleaned.json"),
    str(ROOT / "data" / "cleaned_json" / "introcnn_cleaned.json"),
]


def cmd_ingest(args: argparse.Namespace) -> int:
    """Ingest a PDF: convert to S2ORC JSON then strip to cleaned JSON."""
    pdf = Path(args.pdf)
    if not pdf.exists():
        print(f"Error: PDF not found: {pdf}", file=sys.stderr)
        return 1
    out_dir = args.output or str(ROOT / "data" / "cleaned_json")
    script = ROOT / "ingestion" / "scripts" / "ingest_pdf.sh"
    result = subprocess.run(["bash", str(script), str(pdf), str(out_dir)])
    return result.returncode


def _swarm_argv(base: list[str], args: argparse.Namespace) -> list[str]:
    """Build sys.argv for the agentswarm CLI from run.py args."""
    papers = args.papers or [str(ROOT / "data" / "cleaned_json" / "bert_cleaned.json")]
    argv = base + ["--papers", *papers, "--top-k", str(args.top_k)]
    if args.model:
        argv += ["--model", args.model]
    if getattr(args, "log_file", None):
        argv += ["--log-file", args.log_file]
    if getattr(args, "no_stream", False):
        argv += ["--no-stream"]
    return argv


def cmd_discuss(args: argparse.Namespace) -> int:
    """Run the agent swarm Q&A over one or more cleaned JSON papers."""
    sys.path.insert(0, str(ROOT))
    from agentswarm.cli import main as swarm_main

    sys.argv = _swarm_argv(
        ["agentswarm", "discuss", args.question,
         "--max-agents", str(args.max_agents),
         "--critique-rounds", str(args.critique_rounds)],
        args,
    )
    return swarm_main()


def cmd_brainstorm(args: argparse.Namespace) -> int:
    """Cross-pollinate ideas across papers to generate future research directions."""
    sys.path.insert(0, str(ROOT))
    from agentswarm.cli import main as swarm_main

    sys.argv = _swarm_argv(
        ["agentswarm", "brainstorm", args.area,
         "--max-agents", str(args.max_agents),
         "--cp-rounds", str(args.cp_rounds)],
        args,
    )
    return swarm_main()


def cmd_research(args: argparse.Namespace) -> int:
    """Run the autoresearch-style agent swarm over a code problem folder."""
    sys.path.insert(0, str(ROOT))
    from agentswarm.cli import main as swarm_main

    papers = args.papers or DEFAULT_RESEARCH_PAPERS
    argv = [
        "agentswarm",
        "research",
        args.problem_dir,
        "--papers",
        *papers,
        "--problem",
        args.problem,
        "--run-command",
        args.run_command,
        "--metrics-file",
        args.metrics_file,
        "--metric",
        args.metric,
        "--editable",
        *args.editable,
        "--iterations",
        str(args.iterations),
        "--max-agents",
        str(args.max_agents),
        "--max-cross-ideas",
        str(args.max_cross_ideas),
        "--top-k",
        str(args.top_k),
        "--min-delta",
        str(args.min_delta),
        "--model",
        args.model,
        "--coding-model",
        args.coding_model,
        "--debugging-model",
        args.debugging_model,
        "--planner-max-tokens",
        str(args.planner_max_tokens),
        "--coding-max-tokens",
        str(args.coding_max_tokens),
        "--debugging-max-tokens",
        str(args.debugging_max_tokens),
        "--max-debug-attempts",
        str(args.max_debug_attempts),
    ]
    if args.goal is not None:
        argv += ["--goal", str(args.goal)]
    if args.keep_regressions:
        argv.append("--keep-regressions")
    if args.dry_run:
        argv.append("--dry-run")
    if args.session_dir:
        argv += ["--session-dir", args.session_dir]
    if args.log_file:
        argv += ["--log-file", args.log_file]
    if args.no_stream:
        argv.append("--no-stream")

    sys.argv = argv
    return swarm_main()


def cmd_codegen(args: argparse.Namespace) -> int:
    """Run the paper→code generation pipeline (planning → analysis → coding)."""
    cleaned_json = Path(args.cleaned_json).expanduser().resolve()
    if not cleaned_json.exists():
        print(f"Error: cleaned JSON not found: {cleaned_json}", file=sys.stderr)
        return 1

    paper_name = args.name or cleaned_json.stem.replace("_cleaned", "")
    output_dir = (ROOT / "outputs" / paper_name).resolve()
    repo_dir   = (ROOT / "outputs" / f"{paper_name}_repo").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    repo_dir.mkdir(parents=True, exist_ok=True)

    codes = (ROOT / "paper2code" / "codes").resolve()

    model_flag   = "--gpt_version"
    planning_py  = "1_planning.py"
    analyzing_py = "2_analyzing.py"
    coding_py    = "3_coding.py"
    if args.local:
        model_flag   = "--model_name"
        planning_py  = "1_planning_llm.py"
        analyzing_py = "2_analyzing_llm.py"
        coding_py    = "3_coding_llm.py"

    model = args.model or ("deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct" if args.local else "tencent/hy3-preview:free")

    def run(script: str, extra: list[str] | None = None, include_model: bool = True) -> int:
        cmd = [
            sys.executable, str(codes / script),
            "--paper_name", paper_name,
            "--output_dir", str(output_dir),
        ]
        if include_model:
            cmd += [
                model_flag, model,
                "--pdf_json_path", str(cleaned_json),
            ]
        cmd += extra or []
        print(f"\n=== {script} ===")
        r = subprocess.run(cmd, cwd=str(codes))
        return r.returncode

    steps = [
        ("planning",  planning_py,  None, True),
        ("config",    "1.1_extract_config.py", None, False),
        ("analysis",  analyzing_py, None, True),
        ("coding",    coding_py,    ["--output_repo_dir", str(repo_dir)], True),
    ]

    for label, script, extra, include_model in steps:
        print(f"\n{'='*50}")
        print(f"  Stage: {label}")
        print(f"{'='*50}")
        rc = run(script, extra, include_model)
        if rc != 0:
            print(f"Error: {script} failed (exit {rc}).", file=sys.stderr)
            return rc

    # Copy config into repo
    config_src = output_dir / "planning_config.yaml"
    if config_src.exists():
        import shutil
        shutil.copy(config_src, repo_dir / "config.yaml")

    print(f"\nDone. Generated repo: {repo_dir}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="run.py",
        description="generalresearch — PDF papers to code & discussion",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ── ingest ────────────────────────────────────────────────────────────────
    p_ingest = sub.add_parser("ingest", help="PDF → cleaned JSON via Grobid + s2orc-doc2json")
    p_ingest.add_argument("pdf", help="Path to input PDF")
    p_ingest.add_argument("-o", "--output", default=None,
                          help="Output directory (default: data/cleaned_json/)")

    # ── discuss ───────────────────────────────────────────────────────────────
    p_discuss = sub.add_parser("discuss", help="Q&A over papers with the agent swarm")
    p_discuss.add_argument("question", help="Question to ask the paper experts")
    p_discuss.add_argument("--papers", nargs="+", default=None,
                           help="Cleaned JSON paper files (default: data/cleaned_json/BERT_cleaned.json)")
    p_discuss.add_argument("--max-agents", type=int, default=5)
    p_discuss.add_argument("--top-k", type=int, default=4)
    p_discuss.add_argument("--critique-rounds", type=int, default=1)
    p_discuss.add_argument("--model", default=None, help="OpenRouter model override")
    p_discuss.add_argument("--log-file", default=None)
    p_discuss.add_argument("--no-stream", action="store_true")

    # ── brainstorm ────────────────────────────────────────────────────────────
    p_brainstorm = sub.add_parser(
        "brainstorm",
        help="Cross-pollinate ideas across papers for future research directions",
    )
    p_brainstorm.add_argument("area", help="Research area (e.g. 'efficient transformers')")
    p_brainstorm.add_argument("--papers", nargs="+", default=None,
                              help="Cleaned JSON paper files (default: all in data/cleaned_json/)")
    p_brainstorm.add_argument("--max-agents", type=int, default=6)
    p_brainstorm.add_argument("--cp-rounds", type=int, default=1,
                              help="Cross-pollination rounds (default: 1)")
    p_brainstorm.add_argument("--top-k", type=int, default=4)
    p_brainstorm.add_argument("--model", default=None)
    p_brainstorm.add_argument("--log-file", default=None)
    p_brainstorm.add_argument("--no-stream", action="store_true")

    # ── research ─────────────────────────────────────────────────────────────
    p_research = sub.add_parser(
        "research",
        help="Autoresearch-style model improvement loop over a problem folder",
    )
    p_research.add_argument("problem_dir", help="Folder containing the model problem")
    p_research.add_argument("--papers", nargs="+", default=None,
                            help="Cleaned JSON paper files (default: attention, og_attention, introcnn)")
    p_research.add_argument(
        "--problem",
        default=(
            "Improve a poorly performing MNIST classifier while keeping the evaluation dataset "
            "and run command fixed."
        ),
    )
    p_research.add_argument(
        "--run-command",
        default=(
            "python train.py --download --metrics-out logs/latest_metrics.json "
            "--log-file logs/train_events.jsonl"
        ),
        help="Command to run inside problem_dir for baseline and judge evaluations",
    )
    p_research.add_argument("--metrics-file", default="logs/latest_metrics.json")
    p_research.add_argument("--metric", default="test_accuracy")
    p_research.add_argument("--editable", nargs="+", default=["model.py"])
    p_research.add_argument("--iterations", type=int, default=2)
    p_research.add_argument("--max-agents", type=int, default=3)
    p_research.add_argument("--max-cross-ideas", type=int, default=6)
    p_research.add_argument("--top-k", type=int, default=4)
    p_research.add_argument("--goal", type=float, default=None)
    p_research.add_argument("--min-delta", type=float, default=0.001)
    p_research.add_argument("--keep-regressions", action="store_true")
    p_research.add_argument("--dry-run", action="store_true")
    p_research.add_argument("--session-dir", default=None)
    p_research.add_argument("--model", default=DEFAULT_OPENROUTER_MODEL)
    p_research.add_argument("--coding-model", default=DEFAULT_OPENROUTER_MODEL)
    p_research.add_argument("--debugging-model", default=DEFAULT_OPENROUTER_MODEL)
    p_research.add_argument("--planner-max-tokens", type=int, default=1600)
    p_research.add_argument("--coding-max-tokens", type=int, default=6000)
    p_research.add_argument("--debugging-max-tokens", type=int, default=6000)
    p_research.add_argument("--max-debug-attempts", type=int, default=2)
    p_research.add_argument("--log-file", default=None)
    p_research.add_argument("--no-stream", action="store_true")

    # ── codegen ───────────────────────────────────────────────────────────────
    p_codegen = sub.add_parser("codegen", help="Generate code repo from a cleaned JSON paper")
    p_codegen.add_argument("cleaned_json", help="Path to *_cleaned.json paper file")
    p_codegen.add_argument("--name", default=None, help="Paper name (default: stem of JSON file)")
    p_codegen.add_argument("--model", default=None, help="Model name/ID override")
    p_codegen.add_argument("--local", action="store_true",
                           help="Use local vLLM backend instead of OpenRouter/OpenAI")

    args = parser.parse_args()
    dispatch = {
        "ingest": cmd_ingest,
        "discuss": cmd_discuss,
        "brainstorm": cmd_brainstorm,
        "research": cmd_research,
        "codegen": cmd_codegen,
    }
    return dispatch[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
