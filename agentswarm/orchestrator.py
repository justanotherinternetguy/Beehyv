"""Moderated orchestration for paper expert discussions."""

from __future__ import annotations

from .blackboard import Blackboard, Synthesis
from .expert import PaperExpertAgent


class SwarmOrchestrator:
    """Selects agents, runs discussion rounds, and synthesizes the answer."""

    def __init__(
        self,
        agents: list[PaperExpertAgent],
        *,
        max_agents: int = 5,
        critique_rounds: int = 1,
        logger=None,
    ) -> None:
        if not agents:
            raise ValueError("SwarmOrchestrator requires at least one agent.")
        self.agents = agents
        self.max_agents = max_agents
        self.critique_rounds = critique_rounds
        self.logger = logger

    def run(self, question: str) -> Blackboard:
        log = self.logger
        blackboard = Blackboard(question=question)

        # ── Select agents ──────────────────────────────────────────────────
        if log:
            log.phase("Selecting relevant experts")
        selected = self.select_agents(question)
        blackboard.selected_agents = [agent.agent_id for agent in selected]
        if log:
            log.selected(blackboard.selected_agents)
            log.info(f"{len(selected)} expert(s) selected out of {len(self.agents)}")

        # ── Each agent answers ─────────────────────────────────────────────
        if log:
            log.phase(f"Expert answers ({len(selected)} agent(s))")
        for agent in selected:
            blackboard.add_claim(agent.answer(question))
        if log:
            log.phase_done(f"{len(blackboard.claims)} claim(s) recorded")

        # ── Critique rounds ────────────────────────────────────────────────
        if self.critique_rounds > 0:
            total_critiques = self.critique_rounds * len(selected) * max(0, len(selected) - 1)
            if log:
                log.phase(
                    f"Cross-paper critique ({self.critique_rounds} round(s), "
                    f"up to {total_critiques} critique(s))"
                )
            for round_num in range(self.critique_rounds):
                if log and self.critique_rounds > 1:
                    log.info(f"Round {round_num + 1}/{self.critique_rounds}")
                for agent in selected:
                    for claim in blackboard.claims:
                        if claim.agent_id == agent.agent_id:
                            continue
                        blackboard.add_critique(agent.critique(question, claim))
            if log:
                log.phase_done(f"{len(blackboard.critiques)} critique(s) recorded")

        # ── Synthesize ─────────────────────────────────────────────────────
        if log:
            log.phase("Synthesizing final answer")
        blackboard.set_synthesis(self.synthesize(blackboard))
        if log:
            log.phase_done("Synthesis complete")

        return blackboard

    def select_agents(self, question: str) -> list[PaperExpertAgent]:
        ranked = sorted(
            ((agent.relevance(question), agent) for agent in self.agents),
            key=lambda item: item[0],
            reverse=True,
        )
        relevant = [agent for score, agent in ranked if score > 0]
        if not relevant:
            relevant = [ranked[0][1]]
        return relevant[: self.max_agents]

    def synthesize(self, blackboard: Blackboard) -> Synthesis:
        if not blackboard.claims:
            return Synthesis(
                answer="No paper experts produced an evidence-backed claim.",
                consensus=[],
                disagreements=[],
                citations=[],
            )

        claim_lines = []
        citations = []
        for claim in blackboard.claims:
            citation = claim.evidence[0].citation if claim.evidence else claim.paper_id
            citations.append(citation)
            claim_lines.append(f"- {claim.paper_id}: {claim.text} (confidence {claim.confidence})")

        critique_lines = [
            f"- {critique.agent_id} on {critique.target_claim_id}: {critique.text}"
            for critique in blackboard.critiques
        ]

        answer_parts = [
            f"Question: {blackboard.question}",
            "",
            "Expert positions:",
            *claim_lines,
        ]
        if critique_lines:
            answer_parts.extend(["", "Cross-paper context:", *critique_lines])

        consensus = [
            "The final answer is constrained to claims retrieved from each expert's assigned paper.",
            "Higher-confidence claims are those with stronger keyword overlap in the paper text.",
        ]
        disagreements = []
        if len(blackboard.claims) <= 1:
            disagreements.append("Only one paper expert participated, so no cross-paper disagreement was tested.")

        return Synthesis(
            answer="\n".join(answer_parts),
            consensus=consensus,
            disagreements=disagreements,
            citations=sorted(set(citations)),
        )
