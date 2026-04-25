"""LLM clients for paper expert agents."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Callable, Protocol


OPENROUTER_MODEL = "nvidia/nemotron-3-super-120b-a12b:free"


class PaperExpertLLM(Protocol):
    """Completion interface used by paper expert agents."""

    def complete(self, messages: list[dict[str, str]]) -> str:
        """Return the assistant text for a chat-style prompt."""


@dataclass
class OpenRouterLLM:
    """Chat-completions client supporting OpenRouter or a local Ollama endpoint.

    Set LOCAL_LLM_URL (e.g. http://100.123.34.54:11434) to route all inference
    to a local Ollama instance instead of OpenRouter.  Optionally set
    LOCAL_LLM_MODEL to override the model name (defaults to gemma4:31b).
    """

    model: str = OPENROUTER_MODEL
    api_key_env: str = "OPENROUTER_API_KEY"
    base_url: str = "https://openrouter.ai/api/v1/chat/completions"
    timeout: float = 60.0
    max_tokens: int = 700
    temperature: float = 0.2

    def complete(self, messages: list[dict[str, str]]) -> str:
        """Return the full completion text (blocking, no streaming)."""
        return self._request(messages, stream=False)

    def complete_stream(
        self,
        messages: list[dict[str, str]],
        on_token: Callable[[str], None],
    ) -> str:
        """
        Stream the completion token-by-token via on_token callback.
        Returns the full concatenated text when the stream ends.
        """
        return self._request(messages, stream=True, on_token=on_token)

    # ── internals ─────────────────────────────────────────────────────────────

    def _build_request(self, messages: list[dict[str, str]], stream: bool) -> urllib.request.Request:
        local_url = os.environ.get("LOCAL_LLM_URL")
        if local_url:
            url = f"{local_url.rstrip('/')}/v1/chat/completions"
            api_key = "ollama"
            model = os.environ.get("LOCAL_LLM_MODEL", self.model)
        else:
            url = self.base_url
            api_key = os.environ.get(self.api_key_env)
            if not api_key:
                raise RuntimeError(f"{self.api_key_env} is required to call OpenRouter.")
            model = self.model

        payload = {
            "model": model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": stream,
        }
        return urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/local/agentswarm",
                "X-Title": "Agent Swarm",
            },
            method="POST",
        )

    def _request(
        self,
        messages: list[dict[str, str]],
        *,
        stream: bool,
        on_token: Callable[[str], None] | None = None,
    ) -> str:
        request = self._build_request(messages, stream)
        local_url = os.environ.get("LOCAL_LLM_URL")
        timeout = float(os.environ.get("LOCAL_LLM_TIMEOUT", "300")) if local_url else self.timeout
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                if stream:
                    return self._consume_stream(response, on_token)
                body = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"LLM request failed with HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"LLM request failed: {exc.reason}") from exc

        return self._extract_content(json.loads(body))

    @staticmethod
    def _consume_stream(
        response: object,
        on_token: Callable[[str], None] | None,
    ) -> str:
        """Parse Server-Sent Events lines from the streaming response body."""
        parts: list[str] = []
        while True:
            raw = response.readline()  # type: ignore[attr-defined]
            if not raw:
                break
            line = raw.decode("utf-8").rstrip("\r\n")
            if not line or line == "data: [DONE]":
                continue
            if not line.startswith("data: "):
                continue
            try:
                data = json.loads(line[6:])
            except json.JSONDecodeError:
                continue
            delta = (data.get("choices") or [{}])[0].get("delta", {}).get("content") or ""
            if delta:
                if on_token is not None:
                    on_token(delta)
                parts.append(delta)

        text = "".join(parts).strip()
        if not text:
            raise RuntimeError("LLM returned an empty streaming completion.")
        return text

    @staticmethod
    def _extract_content(data: dict) -> str:
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"LLM returned an unexpected response: {data!r}") from exc
        text = str(content).strip()
        if not text:
            raise RuntimeError("LLM returned an empty completion.")
        return text
