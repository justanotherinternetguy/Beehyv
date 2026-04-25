"""Progress logging and live token streaming for the paper expert swarm."""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import TextIO

# ANSI codes — disabled automatically when output is not a TTY
_RESET  = "\033[0m"
_BOLD   = "\033[1m"
_DIM    = "\033[2m"
_CYAN   = "\033[36m"
_YELLOW = "\033[33m"
_GREEN  = "\033[32m"
_MAGENTA = "\033[35m"


class SwarmLogger:
    """
    Writes live progress to stderr and optionally to a structured log file.

    stderr receives human-readable colour output (ANSI stripped when not a TTY).
    The log file (if given) receives plain-text timestamped lines via the stdlib
    logging module — safe to tail with `tail -f`.
    """

    def __init__(
        self,
        stream: TextIO = sys.stderr,
        log_file: Path | str | None = None,
    ) -> None:
        self._out   = stream
        self._color = stream.isatty()
        self._t0    = time.monotonic()
        self._agent_t: float = 0.0
        self._file_log: logging.Logger | None = None

        if log_file:
            path = Path(log_file)
            path.parent.mkdir(parents=True, exist_ok=True)
            file_logger = logging.getLogger(f"agentswarm.{path.stem}")
            file_logger.setLevel(logging.DEBUG)
            file_logger.propagate = False
            handler = logging.FileHandler(path, encoding="utf-8")
            handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)-8s %(message)s"))
            file_logger.addHandler(handler)
            self._file_log = file_logger
            self._log_info("LOG_START log_file=%s", str(path))

    # ── internal helpers ──────────────────────────────────────────────────────

    def _c(self, code: str, text: str) -> str:
        return f"{code}{text}{_RESET}" if self._color else text

    def _elapsed(self) -> str:
        return f"{time.monotonic() - self._t0:.1f}s"

    def _agent_elapsed(self) -> str:
        return f"{time.monotonic() - self._agent_t:.1f}s"

    def _write(self, text: str) -> None:
        self._out.write(text)
        self._out.flush()

    def _log_info(self, msg: str, *args: object) -> None:
        if self._file_log:
            self._file_log.info(msg, *args)

    def _log_debug(self, msg: str, *args: object) -> None:
        if self._file_log:
            self._file_log.debug(msg, *args)

    # ── public API ────────────────────────────────────────────────────────────

    def phase(self, text: str) -> None:
        """Print a top-level stage header (Selecting agents, Answering, etc.)."""
        header = self._c(_BOLD + _CYAN, f"▶ {text}") + " " + self._c(_DIM, f"[{self._elapsed()}]")
        self._write(f"\n{header}\n")
        self._log_info("PHASE %s", text)

    def agent_start(self, agent_id: str, stage: str) -> None:
        """Print the agent label and stage, then leave cursor on the same line for tokens."""
        self._agent_t = time.monotonic()
        label = agent_id.removeprefix("expert:")
        prefix = self._c(_YELLOW, f"  ◆ {label}") + " " + self._c(_DIM, f"({stage})") + "\n  "
        self._write(prefix)
        self._log_info("AGENT_START agent=%s stage=%s", agent_id, stage)

    def on_token(self, token: str) -> None:
        """Write a single streaming token to the output stream."""
        self._write(token)
        self._log_debug("TOKEN %r", token)

    def agent_done(self, agent_id: str) -> None:
        """Finish the agent's output line and log timing."""
        elapsed = self._agent_elapsed()
        self._write(f"\n  {self._c(_DIM, f'[done in {elapsed}]')}\n")
        self._log_info("AGENT_DONE agent=%s elapsed=%s", agent_id, elapsed)

    def info(self, text: str) -> None:
        """Print a dim informational line."""
        self._write(f"  {self._c(_DIM, text)}\n")
        self._log_info("INFO %s", text)

    def selected(self, agent_ids: list[str]) -> None:
        """Print which agents were selected."""
        names = ", ".join(a.removeprefix("expert:") for a in agent_ids)
        self._write(f"  {self._c(_DIM, 'Selected: ')}{self._c(_MAGENTA, names)}\n")
        self._log_info("SELECTED agents=%s", names)

    def phase_done(self, text: str) -> None:
        """Print a green completion line."""
        line = self._c(_GREEN, f"  ✓ {text}") + " " + self._c(_DIM, f"[{self._elapsed()}]")
        self._write(f"{line}\n")
        self._log_info("PHASE_DONE %s", text)
