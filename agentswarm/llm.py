"""LLM clients for paper expert agents."""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Callable, Protocol


OPENROUTER_MODEL = os.environ.get("OPENROUTER_MODEL", "nvidia/nemotron-3-super-120b-a12b:free")
RETRYABLE_HTTP_STATUSES = {408, 409, 425, 429, 500, 502, 503, 504}


class PaperExpertLLM(Protocol):
    """Completion interface used by paper expert agents."""

    def complete(self, messages: list[dict[str, str]]) -> str:
        """Return the assistant text for a chat-style prompt."""


@dataclass
class OpenRouterLLM:
    """Chat-completions client with switchable backends.

    Set LLM_BACKEND=openrouter (default) to use OpenRouter cloud inference, or
    LLM_BACKEND=gx10 to route all inference to the ASUS GX10 local server.

    GX10 requires LOCAL_LLM_URL (e.g. http://100.123.34.54:11434).
    Optionally set LOCAL_LLM_MODEL to override the model name (defaults to gemma4:31b)
    and LOCAL_LLM_MAX_TOKENS / LOCAL_LLM_TIMEOUT for capacity tuning.
    """

    model: str = OPENROUTER_MODEL
    api_key_env: str = "OPENROUTER_API_KEY"
    base_url: str = "https://openrouter.ai/api/v1/chat/completions"
    timeout: float = 60.0
    max_tokens: int = 700
    temperature: float = 0.2
    max_retries: int = 3
    retry_base_delay: float = 1.0

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
        try:
            return self._request(messages, stream=True, on_token=on_token)
        except RuntimeError as exc:
            if not _should_fallback_to_non_streaming(exc):
                raise
            return self._request(messages, stream=False)

    # ── internals ─────────────────────────────────────────────────────────────

    def _build_request(self, messages: list[dict[str, str]], stream: bool) -> urllib.request.Request:
        backend = os.environ.get("LLM_BACKEND", "openrouter").lower()
        if backend == "gx10":
            gx10_url = os.environ.get("LOCAL_LLM_URL")
            if not gx10_url:
                raise RuntimeError("LOCAL_LLM_URL must be set when LLM_BACKEND=gx10")
            url = f"{gx10_url.rstrip('/')}/v1/chat/completions"
            api_key = "ollama"
            model = os.environ.get("LOCAL_LLM_MODEL", self.model)
            max_tokens = int(os.environ.get("LOCAL_LLM_MAX_TOKENS", "2048"))
        else:
            url = self.base_url
            api_key = os.environ.get(self.api_key_env)
            if not api_key:
                raise RuntimeError(f"{self.api_key_env} is required when LLM_BACKEND=openrouter.")
            model = self.model
            max_tokens = self.max_tokens

        print(f"[LLM] backend={backend} model={model}", flush=True)
        payload = {
            "model": model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": max_tokens,
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
        backend = os.environ.get("LLM_BACKEND", "openrouter").lower()
        timeout = float(os.environ.get("LOCAL_LLM_TIMEOUT", "300")) if backend == "gx10" else self.timeout
        attempts = max(1, self.max_retries + 1)
        last_error: RuntimeError | None = None

        for attempt in range(attempts):
            request = self._build_request(messages, stream)
            try:
                with urllib.request.urlopen(request, timeout=timeout) as response:
                    if stream:
                        return self._consume_stream(response, on_token)
                    body = response.read().decode("utf-8")
                return self._extract_content(json.loads(body))
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")
                last_error = RuntimeError(f"LLM request failed with HTTP {exc.code}: {detail}")
                if exc.code not in RETRYABLE_HTTP_STATUSES or attempt == attempts - 1:
                    raise last_error from exc
                self._sleep_before_retry(attempt)
            except urllib.error.URLError as exc:
                last_error = RuntimeError(f"LLM request failed: {exc.reason}")
                if attempt == attempts - 1:
                    raise last_error from exc
                self._sleep_before_retry(attempt)
            except RuntimeError as exc:
                last_error = exc
                if not _is_retryable_runtime_error(exc) or attempt == attempts - 1:
                    raise
                self._sleep_before_retry(attempt)

        raise last_error or RuntimeError("LLM request failed without a captured error.")

    def _sleep_before_retry(self, attempt: int) -> None:
        delay = self.retry_base_delay * (2 ** attempt)
        if delay > 0:
            time.sleep(delay)

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
            if data.get("error"):
                raise RuntimeError(f"LLM stream failed: {data['error']!r}")
            choice = (data.get("choices") or [{}])[0]
            delta_obj = choice.get("delta") or {}
            delta = (
                delta_obj.get("content")
                or delta_obj.get("reasoning_content")
                or choice.get("text")
                or (choice.get("message") or {}).get("content")
                or ""
            )
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
            msg = data["choices"][0]["message"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"LLM returned an unexpected response: {data!r}") from exc
        text = str(msg.get("content") or msg.get("reasoning") or "").strip()
        if not text:
            raise RuntimeError("LLM returned an empty completion.")
        return text


def _should_fallback_to_non_streaming(exc: RuntimeError) -> bool:
    text = str(exc)
    return (
        "empty streaming completion" in text
        or "LLM stream failed" in text
        or any(f"HTTP {status}" in text for status in RETRYABLE_HTTP_STATUSES)
    )


def _is_retryable_runtime_error(exc: RuntimeError) -> bool:
    text = str(exc)
    return (
        "empty streaming completion" in text
        or "LLM stream failed" in text
        or any(f"HTTP {status}" in text for status in RETRYABLE_HTTP_STATUSES)
    )
