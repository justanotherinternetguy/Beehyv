"""Load S2ORC-style cleaned paper JSON into normalized chunks."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class PaperChunk:
    """A searchable unit of paper text."""

    chunk_id: str
    paper_id: str
    paper_title: str
    section: str
    sec_num: str | None
    text: str
    source: str


@dataclass(frozen=True)
class Paper:
    """A parsed paper and its normalized text chunks."""

    paper_id: str
    title: str
    abstract: str
    path: Path
    chunks: list[PaperChunk]


def load_paper(path: str | Path) -> Paper:
    """Load one cleaned JSON paper file."""

    paper_path = Path(path)
    with paper_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    paper_id = str(data.get("paper_id") or paper_path.stem)
    title = str(data.get("title") or paper_id)
    abstract = str(data.get("abstract") or "")
    pdf_parse = data.get("pdf_parse") or {}

    chunks: list[PaperChunk] = []
    chunks.extend(_chunks_from_blocks(paper_id, title, pdf_parse.get("abstract", []), "abstract"))
    chunks.extend(_chunks_from_blocks(paper_id, title, pdf_parse.get("body_text", []), "body"))
    chunks.extend(_chunks_from_blocks(paper_id, title, pdf_parse.get("back_matter", []), "back_matter"))
    chunks.extend(_chunks_from_refs(paper_id, title, pdf_parse.get("ref_entries", {})))

    if not chunks and abstract:
        chunks.append(
            PaperChunk(
                chunk_id=f"{paper_id}:abstract:0",
                paper_id=paper_id,
                paper_title=title,
                section="Abstract",
                sec_num=None,
                text=abstract,
                source="abstract",
            )
        )

    return Paper(paper_id=paper_id, title=title, abstract=abstract, path=paper_path, chunks=chunks)


def load_papers(paths: Iterable[str | Path]) -> list[Paper]:
    """Load multiple papers, preserving input order."""

    return [load_paper(path) for path in paths]


def _chunks_from_blocks(
    paper_id: str,
    title: str,
    blocks: list[dict[str, Any]],
    source: str,
) -> list[PaperChunk]:
    chunks = []
    for index, block in enumerate(blocks):
        text = _clean_text(str(block.get("text") or ""))
        if not text:
            continue
        section = str(block.get("section") or source.title())
        sec_num = block.get("sec_num")
        chunks.append(
            PaperChunk(
                chunk_id=f"{paper_id}:{source}:{index}",
                paper_id=paper_id,
                paper_title=title,
                section=section,
                sec_num=str(sec_num) if sec_num else None,
                text=text,
                source=source,
            )
        )
    return chunks


def _chunks_from_refs(paper_id: str, title: str, refs: dict[str, Any]) -> list[PaperChunk]:
    chunks = []
    for ref_id, ref in sorted(refs.items()):
        text_parts = [str(ref.get("text") or "")]
        content = ref.get("content")
        if content:
            text_parts.append(str(content))
        text = _clean_text(" ".join(text_parts))
        if not text:
            continue

        type_str = str(ref.get("type_str") or "reference").title()
        num = ref.get("fig_num") or ref.get("num") or ref_id
        section = f"{type_str} {num}".strip()
        chunks.append(
            PaperChunk(
                chunk_id=f"{paper_id}:ref:{ref_id}",
                paper_id=paper_id,
                paper_title=title,
                section=section,
                sec_num=None,
                text=text,
                source="ref_entries",
            )
        )
    return chunks


def _clean_text(text: str) -> str:
    return " ".join(text.split())
