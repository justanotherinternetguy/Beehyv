"""Paper-grounded multi-agent discussion primitives."""

from .blackboard import Blackboard, Claim, Critique, Evidence, Synthesis
from .brainstorm import (
    BrainstormBlackboard,
    BrainstormOrchestrator,
    CrossPollinatedIdea,
    ResearchIdea,
)
from .expert import PaperExpertAgent
from .llm import OPENROUTER_MODEL, OpenRouterLLM, PaperExpertLLM
from .log import SwarmLogger
from .orchestrator import SwarmOrchestrator
from .paper_loader import Paper, PaperChunk, load_paper, load_papers
from .retriever import KeywordRetriever

__all__ = [
    "Blackboard",
    "BrainstormBlackboard",
    "BrainstormOrchestrator",
    "Claim",
    "CrossPollinatedIdea",
    "Critique",
    "Evidence",
    "KeywordRetriever",
    "OPENROUTER_MODEL",
    "OpenRouterLLM",
    "Paper",
    "PaperChunk",
    "PaperExpertAgent",
    "PaperExpertLLM",
    "ResearchIdea",
    "SwarmLogger",
    "SwarmOrchestrator",
    "Synthesis",
    "load_paper",
    "load_papers",
]
