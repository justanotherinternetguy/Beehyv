"""BM25 retriever over AST-extracted snippets from generated code repos."""

from __future__ import annotations

import ast
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_]*")
_MAX_SNIPPET_LINES = 60


@dataclass(frozen=True)
class CodeSnippet:
    file: str
    name: str
    code: str
    score: float = 0.0

    def format(self) -> str:
        return f"[{self.file} — {self.name}]\n```python\n{self.code}\n```"


class CodeRetriever:
    """Keyword retriever over a generated code repository for one paper."""

    def __init__(self, repo_dir: Path) -> None:
        self._repo_dir = repo_dir
        self._snippets: list[tuple[str, str, str]] = []  # (file, name, code)
        self._term_freqs: list[Counter] = []
        self._doc_freqs: dict[str, int] = defaultdict(int)
        self._avg_doc_len = 0.0
        self._load()

    @classmethod
    def for_paper(
        cls,
        paper_id: str,
        outputs_dir: Path | None = None,
    ) -> "CodeRetriever | None":
        """Return a CodeRetriever if a generated repo exists for paper_id."""
        if outputs_dir is None:
            outputs_dir = Path(__file__).parent.parent / "outputs"
        repo_dir = outputs_dir / f"{paper_id}_repo"
        if not repo_dir.is_dir():
            return None
        py_files = list(repo_dir.glob("*.py"))
        if not py_files:
            return None
        return cls(repo_dir)

    def search(self, query: str, top_k: int = 2) -> list[CodeSnippet]:
        if not self._snippets:
            return []
        query_terms = Counter(_tokenize(query))
        if not query_terms:
            return []

        scored: list[tuple[float, str, str, str]] = []
        for i, (file, name, code) in enumerate(self._snippets):
            score = self._bm25(query_terms, self._term_freqs[i])
            if score > 0:
                scored.append((score, file, name, code))

        scored.sort(reverse=True)
        return [
            CodeSnippet(file=f, name=n, code=c, score=s)
            for s, f, n, c in scored[:top_k]
        ]

    # ── internals ─────────────────────────────────────────────────────────────

    def _load(self) -> None:
        raw: list[tuple[str, str, str]] = []
        for py_file in sorted(self._repo_dir.glob("*.py")):
            try:
                source = py_file.read_text(encoding="utf-8", errors="replace")
                raw.extend(_extract_snippets(py_file.name, source))
            except Exception:
                continue

        self._snippets = raw
        self._term_freqs = [Counter(_tokenize(code)) for _, _, code in raw]
        for tf in self._term_freqs:
            for term in tf:
                self._doc_freqs[term] += 1
        if self._term_freqs:
            total = sum(sum(tf.values()) for tf in self._term_freqs)
            self._avg_doc_len = total / len(self._term_freqs)

    def _bm25(self, query_terms: Counter, tf: Counter) -> float:
        k1, b = 1.5, 0.75
        doc_len = sum(tf.values()) or 1
        total = len(self._snippets) or 1
        score = 0.0
        for term, qw in query_terms.items():
            freq = tf.get(term, 0)
            if not freq:
                continue
            df = self._doc_freqs.get(term, 0)
            idf = math.log(1 + (total - df + 0.5) / (df + 0.5))
            denom = freq + k1 * (1 - b + b * doc_len / (self._avg_doc_len or 1))
            score += qw * idf * (freq * (k1 + 1)) / denom
        return score


# ── AST-based snippet extraction ──────────────────────────────────────────────

def _extract_snippets(filename: str, source: str) -> list[tuple[str, str, str]]:
    """Yield (filename, label, code) for top-level defs and class methods."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return [(filename, filename, _truncate(source))]

    lines = source.splitlines()
    results: list[tuple[str, str, str]] = []

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            label = f"def {node.name}()"
            code = _slice_lines(lines, node.lineno - 1, node.end_lineno)
            results.append((filename, label, code))

        elif isinstance(node, ast.ClassDef):
            # Whole class (truncated)
            class_code = _slice_lines(lines, node.lineno - 1, node.end_lineno)
            results.append((filename, f"class {node.name}", class_code))
            # Each method separately for finer-grained retrieval
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    method_label = f"{node.name}.{child.name}()"
                    method_code = _slice_lines(lines, child.lineno - 1, child.end_lineno)
                    results.append((filename, method_label, method_code))

    return results


def _slice_lines(lines: list[str], start: int, end: int) -> str:
    chunk = lines[start:end]
    if len(chunk) > _MAX_SNIPPET_LINES:
        chunk = chunk[:_MAX_SNIPPET_LINES]
        chunk.append("    ...")
    return "\n".join(chunk)


def _truncate(text: str, limit: int = 1500) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "\n..."


def _tokenize(text: str) -> list[str]:
    return [m.group(0).lower() for m in _TOKEN_RE.finditer(text)]
