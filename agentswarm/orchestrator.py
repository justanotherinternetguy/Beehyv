"""Moderated orchestration for paper expert discussions."""

from __future__ import annotations

import re

from .blackboard import Blackboard, Claim, Synthesis
from .expert import PaperExpertAgent

# B1 (audit) — derive consensus from claim-text agreement instead of the
# previous hardcoded boilerplate. Single-link clustering by Jaccard over
# alphanumeric tokens (length >= 3) at threshold 0.4.
_JACCARD_TOKEN_RE = re.compile(r"[A-Za-z0-9_-]+")
_JACCARD_THRESHOLD = 0.4


def _tokenize_for_jaccard(text: str) -> set[str]:
    return {tok for tok in _JACCARD_TOKEN_RE.findall(text.lower()) if len(tok) >= 3}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    union = a | b
    return len(a & b) / max(len(union), 1)


def _cluster_claims_by_jaccard(
    claims: list[Claim], threshold: float = _JACCARD_THRESHOLD
) -> list[list[Claim]]:
    """Group claims by transitive Jaccard similarity over tokens.

    Returns a list of clusters; each cluster is a non-empty list of Claim.
    Single-claim clusters represent disagreements / singletons.
    """
    n = len(claims)
    if n == 0:
        return []
    token_sets = [_tokenize_for_jaccard(c.text) for c in claims]
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj

    for i in range(n):
        for j in range(i + 1, n):
            if _jaccard(token_sets[i], token_sets[j]) >= threshold:
                union(i, j)

    groups: dict[int, list[Claim]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(claims[i])
    seen: list[int] = []
    for i in range(n):
        r = find(i)
        if r not in seen:
            seen.append(r)
    return [groups[r] for r in seen]


def _shorten(text: str, limit: int = 220) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 3].rsplit(" ", 1)[0] + "..."


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

        # B1 (audit §3): consensus is derived from claim-text clustering, not
        # boilerplate. Multi-claim clusters become consensus points; singletons
        # surface as disagreements when at least one consensus point exists.
        clusters = _cluster_claims_by_jaccard(blackboard.claims)
        multi_clusters = [cluster for cluster in clusters if len(cluster) > 1]
        singletons = [cluster[0] for cluster in clusters if len(cluster) == 1]

        consensus: list[str] = []
        for cluster in multi_clusters:
            paper_ids = sorted({claim.paper_id for claim in cluster})
            representative = max(cluster, key=lambda c: len(c.text))
            consensus.append(
                f"{len(cluster)} experts agree ({', '.join(paper_ids)}): "
                f"{_shorten(representative.text)}"
            )
        if not consensus:
            consensus.append(
                "No cross-paper consensus emerged — every expert advanced a distinct claim."
            )

        disagreements: list[str] = []
        if multi_clusters and singletons:
            disagreements = [
                f"{claim.paper_id} alone: {_shorten(claim.text)}"
                for claim in singletons
            ]
        if len(blackboard.claims) <= 1:
            disagreements.append(
                "Only one paper expert participated, so no cross-paper disagreement was tested."
            )

        return Synthesis(
            answer="\n".join(answer_parts),
            consensus=consensus,
            disagreements=disagreements,
            citations=sorted(set(citations)),
        )
