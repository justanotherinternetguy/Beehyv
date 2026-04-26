import io
import urllib.error

from agentswarm.llm import OpenRouterLLM


def test_complete_stream_falls_back_to_non_streaming_on_empty_stream():
    class EmptyStreamThenText(OpenRouterLLM):
        def __init__(self):
            super().__init__()
            self.calls = []

        def _request(self, messages, *, stream, on_token=None):
            self.calls.append(stream)
            if stream:
                raise RuntimeError("LLM returned an empty streaming completion.")
            return "fallback response"

    llm = EmptyStreamThenText()

    assert llm.complete_stream([{"role": "user", "content": "hi"}], lambda token: None) == "fallback response"
    assert llm.calls == [True, False]


def test_complete_stream_falls_back_to_non_streaming_on_retryable_http_error():
    class Stream502ThenText(OpenRouterLLM):
        def __init__(self):
            super().__init__()
            self.calls = []

        def _request(self, messages, *, stream, on_token=None):
            self.calls.append(stream)
            if stream:
                raise RuntimeError("LLM request failed with HTTP 502: provider error")
            return "fallback response"

    llm = Stream502ThenText()

    assert llm.complete_stream([{"role": "user", "content": "hi"}], lambda token: None) == "fallback response"
    assert llm.calls == [True, False]


def test_request_retries_retryable_http_errors(monkeypatch):
    class Response:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return b'{"choices":[{"message":{"content":"ok"}}]}'

    calls = []

    def fake_urlopen(request, timeout):
        calls.append(timeout)
        if len(calls) == 1:
            raise urllib.error.HTTPError(
                request.full_url,
                502,
                "Bad Gateway",
                hdrs=None,
                fp=io.BytesIO(b'{"error":"provider"}'),
            )
        return Response()

    monkeypatch.setenv("OPENROUTER_API_KEY", "test")
    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    llm = OpenRouterLLM(max_retries=1, retry_base_delay=0)

    assert llm.complete([{"role": "user", "content": "hi"}]) == "ok"
    assert len(calls) == 2


def test_consume_stream_accepts_reasoning_content_chunks():
    response = io.BytesIO(
        b'data: {"choices":[{"delta":{"reasoning_content":"hello"}}]}\n\n'
        b'data: {"choices":[{"delta":{"content":" world"}}]}\n\n'
        b"data: [DONE]\n\n"
    )
    tokens = []

    text = OpenRouterLLM._consume_stream(response, tokens.append)

    assert text == "hello world"
    assert tokens == ["hello", " world"]
