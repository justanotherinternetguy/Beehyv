"""Shared discussion state for paper expert agents."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass(frozen=True)
class Evidence:
    """A paper chunk used to ground an agent statement."""

    paper_id: str
    paper_title: str
    chunk_id: str
    section: str
    sec_num: str | None
    text: str
    score: float

    @property
    def citation(self) -> str:
        label = self.section or "paper"
        if self.sec_num:
            label = f"{self.sec_num} {label}"
        return f"{self.paper_id}, {label}"


@dataclass(frozen=True)
class Claim:
    """An evidence-backed position from a paper expert."""

    claim_id: str
    agent_id: str
    paper_id: str
    text: str
    evidence: list[Evidence]
    confidence: float


@dataclass(frozen=True)
class Critique:
    """A response from one expert to another expert's claim."""

    critique_id: str
    agent_id: str
    target_claim_id: str
    stance: str
    text: str
    evidence: list[Evidence]


@dataclass(frozen=True)
class Synthesis:
    """Final moderated answer."""

    answer: str
    consensus: list[str]
    disagreements: list[str]
    citations: list[str]


@dataclass
class Blackboard:
    """Auditable state shared by the orchestrator and paper experts."""

    question: str
    selected_agents: list[str] = field(default_factory=list)
    claims: list[Claim] = field(default_factory=list)
    critiques: list[Critique] = field(default_factory=list)
    synthesis: Synthesis | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def add_claim(self, claim: Claim) -> None:
        self.claims.append(claim)

    def add_critique(self, critique: Critique) -> None:
        self.critiques.append(critique)

    def set_synthesis(self, synthesis: Synthesis) -> None:
        self.synthesis = synthesis
