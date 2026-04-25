"""Dependency-free keyword retrieval for paper chunks."""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass

from .blackboard import Evidence
from .paper_loader import Paper, PaperChunk

TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_\-]*")


@dataclass(frozen=True)
class RetrievalResult:
    chunk: PaperChunk
    score: float


class KeywordRetriever:
    """Small BM25-style retriever scoped to one or more papers."""

    def __init__(self, papers: list[Paper]) -> None:
        self._chunks = [chunk for paper in papers for chunk in paper.chunks]
        self._term_freqs = [Counter(_tokenize(chunk.text)) for chunk in self._chunks]
        self._doc_freqs: dict[str, int] = defaultdict(int)
        for tf in self._term_freqs:
            for term in tf:
                self._doc_freqs[term] += 1
        self._avg_doc_len = (
            sum(sum(tf.values()) for tf in self._term_freqs) / len(self._term_freqs)
            if self._term_freqs
            else 0.0
        )

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        paper_id: str | None = None,
    ) -> list[RetrievalResult]:
        """Return the most relevant chunks for the query."""

        query_terms = Counter(_tokenize(query))
        if not query_terms:
            return []

        scored: list[RetrievalResult] = []
        for chunk, tf in zip(self._chunks, self._term_freqs, strict=True):
            if paper_id and chunk.paper_id != paper_id:
                continue
            score = self._score(query_terms, tf)
            if score > 0:
                scored.append(RetrievalResult(chunk=chunk, score=score))

        scored.sort(key=lambda result: result.score, reverse=True)
        return scored[:top_k]

    def search_evidence(
        self,
        query: str,
        *,
        top_k: int = 5,
        paper_id: str | None = None,
    ) -> list[Evidence]:
        return [_to_evidence(result) for result in self.search(query, top_k=top_k, paper_id=paper_id)]

    def _score(self, query_terms: Counter[str], tf: Counter[str]) -> float:
        k1 = 1.5
        b = 0.75
        doc_len = sum(tf.values()) or 1
        total_docs = len(self._chunks) or 1
        score = 0.0

        for term, query_weight in query_terms.items():
            freq = tf.get(term, 0)
            if freq == 0:
                continue
            doc_freq = self._doc_freqs.get(term, 0)
            idf = math.log(1 + (total_docs - doc_freq + 0.5) / (doc_freq + 0.5))
            denom = freq + k1 * (1 - b + b * doc_len / (self._avg_doc_len or 1))
            score += query_weight * idf * (freq * (k1 + 1)) / denom

        return score


def _to_evidence(result: RetrievalResult) -> Evidence:
    chunk = result.chunk
    return Evidence(
        paper_id=chunk.paper_id,
        paper_title=chunk.paper_title,
        chunk_id=chunk.chunk_id,
        section=chunk.section,
        sec_num=chunk.sec_num,
        text=chunk.text,
        score=result.score,
    )


def _tokenize(text: str) -> list[str]:
    return [match.group(0).lower() for match in TOKEN_RE.finditer(text)]
