"""Paper expert agent implementation."""

from __future__ import annotations

from itertools import count
from typing import TYPE_CHECKING

from .blackboard import Claim, Critique
from .llm import OpenRouterLLM, PaperExpertLLM
from .paper_loader import Paper
from .retriever import KeywordRetriever

if TYPE_CHECKING:
    from .brainstorm import CrossPollinatedIdea, ResearchIdea


class PaperExpertAgent:
    """An expert whose knowledge is scoped to one paper."""

    def __init__(
        self,
        paper: Paper,
        retriever: KeywordRetriever,
        *,
        top_k: int = 4,
        llm: PaperExpertLLM | None = None,
        logger=None,
    ) -> None:
        self.paper = paper
        self.agent_id = f"expert:{paper.paper_id}"
        self.retriever = retriever
        self.top_k = top_k
        self.llm = llm or OpenRouterLLM()
        self.logger = logger
        self._claim_counter = count(1)
        self._critique_counter = count(1)

    def relevance(self, question: str) -> float:
        evidence = self.retriever.search_evidence(question, top_k=1, paper_id=self.paper.paper_id)
        return evidence[0].score if evidence else 0.0

    def answer(self, question: str) -> Claim:
        evidence = self.retriever.search_evidence(
            question,
            top_k=self.top_k,
            paper_id=self.paper.paper_id,
        )
        claim_text = self._compose_answer(question, evidence)
        confidence = _confidence_from_evidence(evidence)
        return Claim(
            claim_id=f"{self.paper.paper_id}:claim:{next(self._claim_counter)}",
            agent_id=self.agent_id,
            paper_id=self.paper.paper_id,
            text=claim_text,
            evidence=evidence,
            confidence=confidence,
        )

    def critique(self, question: str, target: Claim) -> Critique:
        query = f"{question} {target.text}"
        evidence = self.retriever.search_evidence(query, top_k=3, paper_id=self.paper.paper_id)
        stance = "context"
        text = self._compose_critique(target, evidence)
        return Critique(
            critique_id=f"{self.paper.paper_id}:critique:{next(self._critique_counter)}",
            agent_id=self.agent_id,
            target_claim_id=target.claim_id,
            stance=stance,
            text=text,
            evidence=evidence,
        )

    # ── Brainstorm methods ────────────────────────────────────────────────────

    def propose_research(self, area: str, id_counter: count) -> list[ResearchIdea]:
        """
        Seed round: propose 2-3 research directions grounded in this paper.
        Returns a list of ResearchIdea objects (parsed from LLM structured output).
        """
        from .brainstorm import _parse_ideas

        evidence = self.retriever.search_evidence(area, top_k=self.top_k, paper_id=self.paper.paper_id)
        if not evidence:
            evidence = self.retriever.search_evidence("research method contribution", top_k=self.top_k,
                                                       paper_id=self.paper.paper_id)

        messages = [
            {
                "role": "system",
                "content": (
                    f"You are the research expert for the paper '{self.paper.title}'. "
                    "Your job is to propose future research directions that are specifically "
                    "enabled or motivated by this paper's contributions. "
                    "Do not propose generic ideas — every direction must be grounded in a "
                    "specific method, finding, or limitation from the paper."
                ),
            },
            {
                "role": "user",
                "content": "\n\n".join([
                    f"Research area: {area}",
                    "Evidence from your paper (most relevant sections):",
                    _format_evidence(evidence),
                    (
                        "Propose exactly 2-3 concrete future research directions. "
                        "Each must extend or apply something specific from this paper.\n\n"
                        "For each direction use EXACTLY this format:\n"
                        "---IDEA---\n"
                        "TEXT: [one sentence — the research direction]\n"
                        "GROUNDING: [which specific method/finding from the paper enables this]\n"
                        "GAP: [what open question or gap this targets]\n"
                        "---END---"
                    ),
                ]),
            },
        ]

        raw = self._llm_call(messages, stage="proposing ideas")
        return _parse_ideas(raw, self.agent_id, self.paper.paper_id,
                            self.paper.title, evidence, id_counter)

    def cross_pollinate(self, area: str, seed: ResearchIdea, id_counter: count) -> CrossPollinatedIdea:
        """
        Cross-pollination round: read another expert's seed idea and generate one
        hybrid research direction that combines both papers' insights.
        """
        from .brainstorm import _parse_cross_pollination

        # Retrieve evidence from OUR paper most relevant to the other expert's seed
        query = f"{area} {seed.text} {seed.grounding}"
        evidence = self.retriever.search_evidence(query, top_k=3, paper_id=self.paper.paper_id)

        messages = [
            {
                "role": "system",
                "content": (
                    f"You are the research expert for '{self.paper.title}'. "
                    "You are participating in a cross-paper brainstorming session. "
                    "Your task is to generate ONE novel research direction that COMBINES "
                    "insights from your paper with insights from another paper. "
                    "The result should be more powerful than either paper alone."
                ),
            },
            {
                "role": "user",
                "content": "\n\n".join([
                    f"Research area: {area}",
                    f"Another expert (for '{seed.paper_title}') proposed this direction:",
                    f"  Direction: {seed.text}",
                    f"  Grounded in: {seed.grounding}",
                    f"  Gap it targets: {seed.gap}",
                    f"Evidence from YOUR paper ('{self.paper.title}') most relevant to this:",
                    _format_evidence(evidence) if evidence else "(limited overlap found — use your general knowledge of the paper)",
                    (
                        "Generate ONE hybrid research direction that combines both papers. "
                        "Be specific about HOW the two papers' contributions intersect.\n\n"
                        "Use EXACTLY this format:\n"
                        "---CROSSPOLLINATE---\n"
                        "TEXT: [one sentence — the combined research direction]\n"
                        "CONNECTION: [specifically how your paper's work enables or extends the seed idea]\n"
                        "---END---"
                    ),
                ]),
            },
        ]

        raw = self._llm_call(messages, stage=f"cross-pollinating with {seed.paper_id}")
        return _parse_cross_pollination(raw, self.agent_id, self.paper.paper_id,
                                        self.paper.title, seed, evidence, id_counter)

    def summarize_position(self, claim: Claim) -> str:
        citation = claim.evidence[0].citation if claim.evidence else self.paper.paper_id
        return f"{self.paper.title}: {claim.text} [{citation}]"

    def _compose_answer(self, question: str, evidence: list) -> str:
        if not evidence:
            return (
                f"The paper '{self.paper.title}' does not contain a strong local match for: "
                f"{question!r}. I would not make a paper-grounded claim from it."
            )

        lead = evidence[0]
        supporting = evidence[1:3]
        messages = [
            {"role": "system", "content": self._system_prompt()},
            {
                "role": "user",
                "content": "\n\n".join(
                    [
                        f"Question: {question}",
                        "Retrieved evidence from your assigned paper:",
                        _format_evidence([lead, *supporting]),
                        (
                            "Answer as the expert for this paper. Use only the retrieved evidence. "
                            "Be direct, cite section labels in brackets, and say when the evidence is limited."
                        ),
                    ]
                ),
            },
        ]
        return self._llm_call(messages, stage="answering")

    def _compose_critique(self, target: Claim, evidence: list) -> str:
        if not evidence:
            return (
                f"I cannot verify or challenge {target.claim_id} from '{self.paper.title}' because "
                "no relevant evidence was retrieved."
            )

        messages = [
            {"role": "system", "content": self._system_prompt()},
            {
                "role": "user",
                "content": "\n\n".join(
                    [
                        f"Original claim from {target.agent_id}: {target.text}",
                        "Relevant evidence from your assigned paper:",
                        _format_evidence(evidence),
                        (
                            "Respond as this paper's expert. State whether your paper supports, "
                            "qualifies, or cannot assess the claim. Use only the evidence above."
                        ),
                    ]
                ),
            },
        ]
        return self._llm_call(messages, stage="critiquing")

    def _llm_call(self, messages: list, stage: str) -> str:
        """Call the LLM with streaming if available, plain otherwise."""
        log = self.logger
        can_stream = hasattr(self.llm, "complete_stream")

        if log is not None:
            log.agent_start(self.agent_id, stage)

        if log is not None and can_stream:
            text = self.llm.complete_stream(messages, log.on_token)  # type: ignore[attr-defined]
        else:
            text = self.llm.complete(messages)

        if log is not None:
            log.agent_done(self.agent_id)

        return text

    def _system_prompt(self) -> str:
        return (
            f"You are the expert agent for the paper '{self.paper.title}' "
            f"({self.paper.paper_id}). Your job is to answer only from this paper. "
            "Do not use outside knowledge. If the provided excerpts are insufficient, say so."
        )


def _confidence_from_evidence(evidence: list) -> float:
    if not evidence:
        return 0.0
    top_score = evidence[0].score
    return round(min(0.95, 0.35 + top_score / 20), 2)


def _shorten(text: str, limit: int = 260) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 3].rsplit(" ", 1)[0] + "..."


def _format_evidence(evidence: list) -> str:
    return "\n\n".join(f"[{item.citation}]\n{_shorten(item.text, 1200)}" for item in evidence)
