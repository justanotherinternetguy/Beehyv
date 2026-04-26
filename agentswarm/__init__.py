"""Paper-grounded multi-agent discussion primitives."""

from .blackboard import Blackboard, Claim, Critique, Evidence, Synthesis
from .brainstorm import (
    BrainstormBlackboard,
    BrainstormOrchestrator,
    CrossPollinatedIdea,
    ResearchIdea,
)
from .code_retriever import CodeRetriever, CodeSnippet
from .expert import PaperExpertAgent
from .llm import OPENROUTER_MODEL, OpenRouterLLM, PaperExpertLLM
from .log import SwarmLogger
from .orchestrator import SwarmOrchestrator
from .paper_loader import Paper, PaperChunk, load_paper, load_papers
from .retriever import KeywordRetriever
from .research import (
    CodingResult,
    DebuggingResult,
    ExperimentResult,
    JudgeFeedback,
    ModelImprovementIdea,
    OrchestrationDiagnosis,
    ResearchIteration,
    ResearchPlan,
    ResearchSession,
    ResearchSwarmOrchestrator,
)

__all__ = [
    "Blackboard",
    "BrainstormBlackboard",
    "BrainstormOrchestrator",
    "Claim",
    "CodeRetriever",
    "CodeSnippet",
    "CodingResult",
    "CrossPollinatedIdea",
    "Critique",
    "DebuggingResult",
    "Evidence",
    "ExperimentResult",
    "JudgeFeedback",
    "KeywordRetriever",
    "ModelImprovementIdea",
    "OPENROUTER_MODEL",
    "OpenRouterLLM",
    "OrchestrationDiagnosis",
    "Paper",
    "PaperChunk",
    "PaperExpertAgent",
    "PaperExpertLLM",
    "ResearchIteration",
    "ResearchIdea",
    "ResearchPlan",
    "ResearchSession",
    "ResearchSwarmOrchestrator",
    "SwarmLogger",
    "SwarmOrchestrator",
    "Synthesis",
    "load_paper",
    "load_papers",
]
